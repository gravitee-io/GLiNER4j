/*
 * Copyright © 2015 The Gravitee team (http://gravitee.io)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package io.gravitee.lab.gliner4j.llamacpp;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

/**
 * The GLiNER2 (fastino) span model as ggml graphs: DeBERTa encoder ({@link GgmlDebertaV3},
 * {@code enc.*}) → first-subtoken word rows → {@code SpanMarkerV0} span representations; the
 * unit's {@code [P]} row → {@code count_pred} logits; the field-marker rows → GRU count embedding
 * (unrolled {@code count} steps, {@code h₀} = field rows, input = position embeddings) → the
 * {@code count_lstm} projector MLP or the {@code count_lstm_v2} downscaled transformer over the
 * count axis → dot with the span representations → sigmoid. Plus the {@code [L]} classifier MLP.
 * Weights: {@code gguf/model.gguf} from {@code export_llamacpp_gliner2.py}.
 *
 * <p>Layouts match the ONNX {@code ner_full} graph: span scores are
 * {@code [count][field][word][width]}, already sigmoided.
 */
public final class GgmlGliner2Model implements AutoCloseable {

  /** One unit's scores: {@code countLogits[maxCount]} and {@code spans[count][field][word][width]} (sigmoid). */
  public record UnitScores(float[] countLogits, float[][][][] spans) {}

  private final GgmlWeights w;
  private final GgmlDebertaV3 encoder;
  private final int hidden;
  private final int maxCount;
  private final boolean v2;
  private final int tfDim;
  private final int tfHeads = 4;
  private final float tfEps = 1e-5f;

  public GgmlGliner2Model(Path gguf, boolean useGpu, int cpuThreads) {
    this.w = new GgmlWeights(gguf, useGpu, cpuThreads, "ggml GLiNER2");
    this.encoder = new GgmlDebertaV3(w, "enc.");
    this.hidden = encoder.hidden();
    this.maxCount = (int) Ggml.ne(w.get("count_embed.pos_embedding.weight"), 1);
    this.v2 = w.has("count_tf.in_projector.weight");
    this.tfDim = v2
      ? (int) Ggml.ne(w.get("count_tf.in_projector.weight"), 1)
      : 0;
  }

  public int hiddenSize() {
    return hidden;
  }

  public int maxCount() {
    return maxCount;
  }

  /**
   * Scores one schema unit of one input.
   *
   * @param ids the full multi-unit prompt + text token ids
   * @param wordPositions first-subtoken position of every text word
   * @param pPosition position of the unit's {@code [P]} marker
   * @param fieldPositions positions of the unit's field markers ({@code [E]}/{@code [C]}/{@code [R]})
   * @param count how many count steps to score (1 for entities, {@code maxCount} for relations / structures)
   * @param maxWidth span width
   */
  /**
   * Longest (padded) row for which {@link #scoreUnitBatch} beats per-row {@link #scoreUnit} on
   * CUDA. Short rows are launch-bound, so one padded graph keeps the GPU busy; long rows already
   * saturate the card and the padding plus 4-D attention layouts cost more than they save there.
   * Mind the prompt: a 42-entity schema with descriptions is ~900 tokens before any text, so its
   * rows run 1 400+ tokens and never batch under this default — raise the system property
   * {@code gliner4j.ggml.batchMaxTokens} (2048) on cards with headroom, or drop the descriptions
   * (rows shrink to ~870 tokens, and get cheaper with them).
   */
  private static final int BATCH_MAX_TOKENS = Integer.getInteger(
    "gliner4j.ggml.batchMaxTokens",
    384
  );

  /**
   * Whether {@link #scoreUnitBatch} should be preferred over per-row {@link #scoreUnit} for rows
   * padded to {@code n} tokens. Never on the CPU backend or Metal, where the batched graph
   * measured slower at every length.
   */
  public boolean prefersBatchedScoring(int n) {
    return w.gpu && !w.metal && n <= BATCH_MAX_TOKENS;
  }

  public synchronized UnitScores scoreUnit(
    long[] ids,
    int[] wordPositions,
    int pPosition,
    int[] fieldPositions,
    int count,
    int maxWidth
  ) {
    int n = ids.length;
    int words = wordPositions.length;
    int fields = fieldPositions.length;
    int spans = words * maxWidth;
    count = Math.max(0, Math.min(count, maxCount));
    var spanStart = new int[spans];
    var spanEnd = new int[spans];
    for (int s = 0; s < words; s++) {
      for (int k = 0; k < maxWidth; k++) {
        int idx = s * maxWidth + k;
        boolean valid = s + k < words;
        spanStart[idx] = valid ? s : 0;
        spanEnd[idx] = valid ? s + k : 0;
      }
    }
    long nodes =
      encoder.graphNodes() + 128L + 48L * maxCount + (v2 ? 256 : 16) + 64;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, nodes);
      try {
        var built = encoder.build(ctx, n);
        var wordIdx = Ggml.newTensor1d(ctx, w.typeI32, Math.max(1, words));
        var pIdx = Ggml.newTensor1d(ctx, w.typeI32, 1);
        var fieldIdx = Ggml.newTensor1d(ctx, w.typeI32, Math.max(1, fields));
        var startIdx = Ggml.newTensor1d(ctx, w.typeI32, Math.max(1, spans));
        var endIdx = Ggml.newTensor1d(ctx, w.typeI32, Math.max(1, spans));
        for (var t : new MemorySegment[] {
          wordIdx,
          pIdx,
          fieldIdx,
          startIdx,
          endIdx,
        }) {
          Ggml.setInput(t);
        }
        var h = built.output();
        var pRow = Ggml.getRows(ctx, h, pIdx); // (hidden, 1)
        var countLogits = w.linear(
          ctx,
          Ggml.relu(ctx, w.linear(ctx, pRow, "count_pred.0")),
          "count_pred.2"
        ); // (maxCount, 1)

        MemorySegment scores = null;
        if (count > 0 && words > 0 && fields > 0) {
          var wordRows = Ggml.getRows(ctx, h, wordIdx); // (hidden, W)
          var startRep = w.projection(ctx, wordRows, "span_rep.project_start");
          var endRep = w.projection(ctx, wordRows, "span_rep.project_end");
          var cat = Ggml.relu(
            ctx,
            Ggml.concat(
              ctx,
              Ggml.getRows(ctx, startRep, startIdx),
              Ggml.getRows(ctx, endRep, endIdx),
              0
            )
          );
          var spanRep = w.projection(ctx, cat, "span_rep.out_project"); // (hidden, spans)
          var fieldRows = Ggml.getRows(ctx, h, fieldIdx); // (hidden, M)
          // Like the ONNX graph: unroll all max_count steps (the count transformer attends across
          // them), then keep the first `count` columns.
          var structAll = countEmbed(ctx, fieldRows, fields, maxCount); // (hidden, M·maxCount)
          var struct = count == maxCount
            ? structAll
            : Ggml.cont(
              ctx,
              Ggml.view2d(
                ctx,
                structAll,
                hidden,
                (long) fields * count,
                Ggml.nb(structAll, 1),
                0
              )
            );
          scores = Ggml.sigmoid(ctx, Ggml.mulMat(ctx, spanRep, struct)); // (spans, M·count)
        }

        var graph = Ggml.newGraph(ctx, nodes);
        Ggml.setOutput(countLogits);
        Ggml.buildForwardExpand(graph, countLogits);
        if (scores != null) {
          Ggml.setOutput(scores);
          Ggml.buildForwardExpand(graph, scores);
        }
        w.alloc(graph);
        encoder.feed(built, call, ids);
        Ggml.setInts(
          wordIdx,
          call,
          words > 0 ? wordPositions : new int[] { 0 }
        );
        Ggml.setInts(pIdx, call, new int[] { pPosition });
        Ggml.setInts(
          fieldIdx,
          call,
          fields > 0 ? fieldPositions : new int[] { 0 }
        );
        Ggml.setInts(startIdx, call, spans > 0 ? spanStart : new int[] { 0 });
        Ggml.setInts(endIdx, call, spans > 0 ? spanEnd : new int[] { 0 });
        w.run(graph);

        var cl = Ggml.getFloats(countLogits, call, maxCount);
        var out = new float[count][fields][words][maxWidth];
        if (scores != null) {
          var flat = Ggml.getFloats(scores, call, spans * fields * count);
          for (int t = 0; t < count; t++) {
            for (int m = 0; m < fields; m++) {
              int col = t * fields + m;
              for (int s = 0; s < words; s++) {
                for (int k = 0; k < maxWidth; k++) {
                  int idx = s * maxWidth + k;
                  out[t][m][s][k] = s + k < words
                    ? flat[col * spans + idx]
                    : 0f;
                }
              }
            }
          }
        }
        return new UnitScores(cl, out);
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /**
   * Batched {@link #scoreUnit}: rows share the schema prefix, so {@code pPosition} and
   * {@code fieldPositions} are the same token offsets in every row — but the marker states are
   * contextual, so each row's own {@code [P]} / field rows are gathered (unlike the ONNX
   * {@code ner_full} graph, which reuses row 0's and mis-scores the other rows of a mixed batch).
   * Rows are padded to the longest and encoded in one graph; the count embedding runs over all
   * {@code rows × fields} at once and every row's spans are scored in one matmul.
   */
  public synchronized UnitScores[] scoreUnitBatch(
    long[][] idsRows,
    int[][] wordPositionsRows,
    int pPosition,
    int[] fieldPositions,
    int count,
    int maxWidth
  ) {
    int batch = idsRows.length;
    if (batch == 1) {
      return new UnitScores[] {
        scoreUnit(
          idsRows[0],
          wordPositionsRows[0],
          pPosition,
          fieldPositions,
          count,
          maxWidth
        ),
      };
    }
    int n = 0;
    for (var r : idsRows) n = Math.max(n, r.length);
    int fields = fieldPositions.length;
    count = Math.max(0, Math.min(count, maxCount));
    // flat word indices across rows (column of row r token i = i + n·r), and spans padded to a
    // per-row block of maxWords·maxWidth so every row's spans sit on a batch axis: the scores are
    // then a batched matmul that pairs each row's fields with ITS spans only. (The earlier
    // (M·count) × spansTotal product scored every row against every row's spans — B× the work,
    // the readback and the host loop, growing with the square of the batch: 600 MB per call for
    // 8 × 350-word rows with 42 fields.)
    var wordOffsets = new int[batch + 1];
    int maxWords = 0;
    for (int r = 0; r < batch; r++) {
      wordOffsets[r + 1] = wordOffsets[r] + wordPositionsRows[r].length;
      maxWords = Math.max(maxWords, wordPositionsRows[r].length);
    }
    int wordsTotal = wordOffsets[batch];
    int maxSpans = maxWords * maxWidth;
    int spansTotal = maxSpans * batch;
    var wordIdxAll = new int[Math.max(1, wordsTotal)];
    var spanStart = new int[Math.max(1, spansTotal)];
    var spanEnd = new int[Math.max(1, spansTotal)];
    for (int r = 0; r < batch; r++) {
      int words = wordPositionsRows[r].length;
      for (int i = 0; i < words; i++) wordIdxAll[wordOffsets[r] + i] =
        wordPositionsRows[r][i] + r * n;
      for (int s = 0; s < maxWords; s++) {
        for (int k = 0; k < maxWidth; k++) {
          int idx = r * maxSpans + s * maxWidth + k;
          boolean valid = s + k < words;
          spanStart[idx] = wordOffsets[r] + (valid ? s : 0);
          spanEnd[idx] = wordOffsets[r] + (valid ? s + k : 0);
        }
      }
    }
    long nodes =
      encoder.graphNodes() + 128L + 48L * maxCount + (v2 ? 256 : 16) + 64;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, nodes);
      try {
        var built = encoder.build(ctx, n, batch);
        var wordIdx = Ggml.newTensor1d(ctx, w.typeI32, wordIdxAll.length);
        var pIdx = Ggml.newTensor1d(ctx, w.typeI32, batch);
        var fieldIdx = Ggml.newTensor1d(
          ctx,
          w.typeI32,
          Math.max(1, fields * batch)
        );
        var startIdx = Ggml.newTensor1d(ctx, w.typeI32, spanStart.length);
        var endIdx = Ggml.newTensor1d(ctx, w.typeI32, spanEnd.length);
        for (var t : new MemorySegment[] {
          wordIdx,
          pIdx,
          fieldIdx,
          startIdx,
          endIdx,
        })
          Ggml.setInput(t);
        var h = built.output(); // (hidden, n·B)
        var pRow = Ggml.getRows(ctx, h, pIdx);
        var countLogits = w.linear(
          ctx,
          Ggml.relu(ctx, w.linear(ctx, pRow, "count_pred.0")),
          "count_pred.2"
        );
        MemorySegment scores = null;
        if (count > 0 && wordsTotal > 0 && fields > 0) {
          var wordRows = Ggml.getRows(ctx, h, wordIdx); // (hidden, wordsTotal)
          var startRep = w.projection(ctx, wordRows, "span_rep.project_start");
          var endRep = w.projection(ctx, wordRows, "span_rep.project_end");
          var cat = Ggml.relu(
            ctx,
            Ggml.concat(
              ctx,
              Ggml.getRows(ctx, startRep, startIdx),
              Ggml.getRows(ctx, endRep, endIdx),
              0
            )
          );
          var spanRep = w.projection(ctx, cat, "span_rep.out_project"); // (hidden, maxSpans·B)
          var fieldRows = Ggml.getRows(ctx, h, fieldIdx); // (hidden, B·M): column r·M + m
          var structAll = countEmbed(ctx, fieldRows, fields * batch, maxCount); // (hidden, B·M·maxCount)
          var struct = count == maxCount
            ? structAll
            : Ggml.cont(
              ctx,
              Ggml.view2d(
                ctx,
                structAll,
                hidden,
                (long) fields * batch * count,
                Ggml.nb(structAll, 1),
                0
              )
            ); // columns t·(M·B) + r·M + m: count-major
          // (hidden, M, B, count) → (hidden, M, count, B): the batch on the last axis for the batched matmul
          var struct3 = Ggml.reshape3d(
            ctx,
            Ggml.cont(
              ctx,
              Ggml.permute(
                ctx,
                Ggml.reshape4d(ctx, struct, hidden, fields, batch, count),
                0,
                1,
                3,
                2
              )
            ),
            hidden,
            (long) fields * count,
            batch
          );
          var spanRep3 = Ggml.reshape3d(ctx, spanRep, hidden, maxSpans, batch);
          scores = Ggml.sigmoid(ctx, Ggml.mulMat(ctx, struct3, spanRep3)); // (M·count, maxSpans, B): [m + M·t + M·count·s + M·count·maxSpans·r]
        }
        var graph = Ggml.newGraph(ctx, nodes);
        Ggml.setOutput(countLogits);
        Ggml.buildForwardExpand(graph, countLogits);
        if (scores != null) {
          Ggml.setOutput(scores);
          Ggml.buildForwardExpand(graph, scores);
        }
        w.alloc(graph);
        encoder.feed(built, call, idsRows);
        Ggml.setInts(wordIdx, call, wordIdxAll);
        var pAll = new int[batch];
        var fieldAll = new int[Math.max(1, fields * batch)];
        for (int r = 0; r < batch; r++) {
          pAll[r] = pPosition + r * n;
          for (int m2 = 0; m2 < fields; m2++) fieldAll[r * fields + m2] =
            fieldPositions[m2] + r * n;
        }
        Ggml.setInts(pIdx, call, pAll);
        Ggml.setInts(fieldIdx, call, fieldAll);
        Ggml.setInts(startIdx, call, spanStart);
        Ggml.setInts(endIdx, call, spanEnd);
        w.run(graph);
        var clAll = Ggml.getFloats(countLogits, call, maxCount * batch);
        float[] flat = scores != null
          ? Ggml.getFloats(scores, call, fields * count * spansTotal)
          : null;
        var out = new UnitScores[batch];
        int mc = fields * count;
        for (int r = 0; r < batch; r++) {
          int words = wordPositionsRows[r].length;
          var spans = new float[count][fields][words][maxWidth];
          if (flat != null) {
            long rowBase = (long) mc * maxSpans * r;
            for (int s = 0; s < words; s++) {
              for (int k = 0; k < maxWidth && s + k < words; k++) {
                int base = (int) (rowBase + (long) mc * (s * maxWidth + k));
                for (int t = 0; t < count; t++) {
                  int tb = base + fields * t;
                  for (int m = 0; m < fields; m++) spans[t][m][s][k] = flat[tb +
                  m];
                }
              }
            }
          }
          out[r] = new UnitScores(
            java.util.Arrays.copyOfRange(
              clAll,
              r * maxCount,
              (r + 1) * maxCount
            ),
            spans
          );
        }
        return out;
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /** Batched {@link #classify}: rows padded and encoded together; {@code labelPositions} are row-relative. */
  public synchronized float[][] classifyBatch(
    long[][] idsRows,
    int[] labelPositions
  ) {
    int batch = idsRows.length;
    int l = labelPositions.length;
    if (l == 0) return new float[batch][0];
    int n = 0;
    for (var r : idsRows) n = Math.max(n, r.length);
    var flatIdx = new int[l * batch];
    for (int r = 0; r < batch; r++) for (int i = 0; i < l; i++) flatIdx[r * l +
      i] =
      labelPositions[i] + r * n;
    long nodes = encoder.graphNodes() + 32;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, nodes);
      try {
        var built = encoder.build(ctx, n, batch);
        var lIdx = Ggml.newTensor1d(ctx, w.typeI32, flatIdx.length);
        Ggml.setInput(lIdx);
        var rows = Ggml.getRows(ctx, built.output(), lIdx);
        var logits = w.linear(
          ctx,
          Ggml.relu(ctx, w.linear(ctx, rows, "classifier.0")),
          "classifier.2"
        );
        var graph = Ggml.newGraph(ctx, nodes);
        w.compute(ctx, graph, logits);
        encoder.feed(built, call, idsRows);
        Ggml.setInts(lIdx, call, flatIdx);
        w.run(graph);
        var flat = Ggml.getFloats(logits, call, flatIdx.length);
        var out = new float[batch][l];
        for (int r = 0; r < batch; r++) System.arraycopy(
          flat,
          r * l,
          out[r],
          0,
          l
        );
        return out;
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /** Diagnostic: per-row gathered word rows {@code [r][W][H]} and span reps {@code [r][spans][H]} from the batched graph. */
  public synchronized float[][][][] debugBatchReps(
    long[][] idsRows,
    int[][] wordPositionsRows,
    int maxWidth
  ) {
    int batch = idsRows.length;
    int n = 0;
    for (var r : idsRows) n = Math.max(n, r.length);
    var wordOffsets = new int[batch + 1];
    var spanOffsets = new int[batch + 1];
    for (int r = 0; r < batch; r++) {
      wordOffsets[r + 1] = wordOffsets[r] + wordPositionsRows[r].length;
      spanOffsets[r + 1] =
        spanOffsets[r] + wordPositionsRows[r].length * maxWidth;
    }
    int wordsTotal = wordOffsets[batch];
    int spansTotal = spanOffsets[batch];
    var wordIdxAll = new int[wordsTotal];
    var spanStart = new int[spansTotal];
    var spanEnd = new int[spansTotal];
    for (int r = 0; r < batch; r++) {
      int words = wordPositionsRows[r].length;
      for (int i = 0; i < words; i++) wordIdxAll[wordOffsets[r] + i] =
        wordPositionsRows[r][i] + r * n;
      for (int sIdx = 0; sIdx < words; sIdx++) {
        for (int k = 0; k < maxWidth; k++) {
          int idx = spanOffsets[r] + sIdx * maxWidth + k;
          boolean valid = sIdx + k < words;
          spanStart[idx] = wordOffsets[r] + (valid ? sIdx : 0);
          spanEnd[idx] = wordOffsets[r] + (valid ? sIdx + k : 0);
        }
      }
    }
    long nodes = encoder.graphNodes() + 64;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, nodes);
      try {
        var built = encoder.build(ctx, n, batch);
        var wordIdx = Ggml.newTensor1d(ctx, w.typeI32, wordsTotal);
        var startIdx = Ggml.newTensor1d(ctx, w.typeI32, spansTotal);
        var endIdx = Ggml.newTensor1d(ctx, w.typeI32, spansTotal);
        for (var t : new MemorySegment[] { wordIdx, startIdx, endIdx })
          Ggml.setInput(t);
        var wordRows = Ggml.getRows(ctx, built.output(), wordIdx);
        var startRep = w.projection(ctx, wordRows, "span_rep.project_start");
        var endRep = w.projection(ctx, wordRows, "span_rep.project_end");
        var cat = Ggml.relu(
          ctx,
          Ggml.concat(
            ctx,
            Ggml.getRows(ctx, startRep, startIdx),
            Ggml.getRows(ctx, endRep, endIdx),
            0
          )
        );
        var spanRep = w.projection(ctx, cat, "span_rep.out_project");
        var graph = Ggml.newGraph(ctx, nodes);
        Ggml.setOutput(wordRows);
        Ggml.buildForwardExpand(graph, wordRows);
        Ggml.setOutput(spanRep);
        Ggml.buildForwardExpand(graph, spanRep);
        w.alloc(graph);
        encoder.feed(built, call, idsRows);
        Ggml.setInts(wordIdx, call, wordIdxAll);
        Ggml.setInts(startIdx, call, spanStart);
        Ggml.setInts(endIdx, call, spanEnd);
        w.run(graph);
        var wr = Ggml.getFloats(wordRows, call, hidden * wordsTotal);
        var sr = Ggml.getFloats(spanRep, call, hidden * spansTotal);
        var out = new float[batch][2][][];
        for (int r = 0; r < batch; r++) {
          int words = wordPositionsRows[r].length;
          out[r][0] = new float[words][hidden];
          out[r][1] = new float[words * maxWidth][hidden];
          for (int i = 0; i < words; i++) System.arraycopy(
            wr,
            (wordOffsets[r] + i) * hidden,
            out[r][0][i],
            0,
            hidden
          );
          for (int i = 0; i < words * maxWidth; i++) System.arraycopy(
            sr,
            (spanOffsets[r] + i) * hidden,
            out[r][1][i],
            0,
            hidden
          );
        }
        return out;
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /** Classifier logits for the {@code [L]} marker rows of one input (one per choice). */
  public synchronized float[] classify(long[] ids, int[] labelPositions) {
    int n = ids.length;
    int l = labelPositions.length;
    if (l == 0) {
      return new float[0];
    }
    long nodes = encoder.graphNodes() + 32;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, nodes);
      try {
        var built = encoder.build(ctx, n);
        var lIdx = Ggml.newTensor1d(ctx, w.typeI32, l);
        Ggml.setInput(lIdx);
        var rows = Ggml.getRows(ctx, built.output(), lIdx);
        var logits = w.linear(
          ctx,
          Ggml.relu(ctx, w.linear(ctx, rows, "classifier.0")),
          "classifier.2"
        ); // (1, L)
        var graph = Ggml.newGraph(ctx, nodes);
        w.compute(ctx, graph, logits);
        encoder.feed(built, call, ids);
        Ggml.setInts(lIdx, call, labelPositions);
        w.run(graph);
        return Ggml.getFloats(logits, call, l);
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  // ---- count embedding -------------------------------------------------------

  /**
   * gliner2's {@code CountLSTM} / {@code CountLSTMv2}: GRU over the count axis with {@code h₀} =
   * field rows, then the variant projector. Returns {@code (hidden, M·count)} with column
   * {@code t·M + m}.
   */
  private MemorySegment countEmbed(
    MemorySegment ctx,
    MemorySegment fields,
    int m,
    int count
  ) {
    var gi = w.get("count_embed.gi"); // (3·hidden, maxCount): pos_emb·W_ihᵀ + b_ih
    var whh = w.get("count_embed.gru.weight_hh_l0");
    var bhh = w.get("count_embed.gru.bias_hh_l0");
    long giNb1 = Ggml.nb(gi, 1);
    long hBytes = (long) hidden * Float.BYTES;
    var hState = fields;
    MemorySegment outs = null; // (hidden, M·count)
    for (int t = 0; t < count; t++) {
      var gh = Ggml.add(ctx, Ggml.mulMat(ctx, whh, hState), bhh); // (3·hidden, M)
      long ghNb1 = Ggml.nb(gh, 1);
      var giT = Ggml.view2d(ctx, gi, hidden * 3L, 1, giNb1, t * giNb1); // (3·hidden, 1)
      var gir = Ggml.view2d(ctx, giT, hidden, 1, giNb1, 0);
      var giz = Ggml.view2d(ctx, giT, hidden, 1, giNb1, hBytes);
      var gin = Ggml.view2d(ctx, giT, hidden, 1, giNb1, 2 * hBytes);
      var ghr = Ggml.cont(ctx, Ggml.view2d(ctx, gh, hidden, m, ghNb1, 0));
      var ghz = Ggml.cont(ctx, Ggml.view2d(ctx, gh, hidden, m, ghNb1, hBytes));
      var ghn = Ggml.cont(
        ctx,
        Ggml.view2d(ctx, gh, hidden, m, ghNb1, 2 * hBytes)
      );
      var r = Ggml.sigmoid(ctx, Ggml.add(ctx, ghr, gir));
      var z = Ggml.sigmoid(ctx, Ggml.add(ctx, ghz, giz));
      var nn = Ggml.tanh(ctx, Ggml.add(ctx, Ggml.mul(ctx, r, ghn), gin));
      // h = (1 − z)·n + z·h = n + z·(h − n)
      hState = Ggml.add(ctx, nn, Ggml.mul(ctx, z, Ggml.sub(ctx, hState, nn)));
      outs = outs == null ? hState : Ggml.concat(ctx, outs, hState, 1);
    }
    var fieldsRep = count == 1 ? fields : Ggml.repeat(ctx, fields, outs); // (hidden, M·count)
    if (!v2) {
      var x = Ggml.concat(ctx, outs, fieldsRep, 0); // (2·hidden, M·count)
      return w.linear(
        ctx,
        Ggml.relu(ctx, w.linear(ctx, x, "count_embed.projector.0")),
        "count_embed.projector.2"
      );
    }
    return downscaledTransformer(ctx, Ggml.add(ctx, outs, fieldsRep), m, count);
  }

  /** gliner2's {@code DownscaledTransformer}: in_proj → 2 post-LN encoder layers over the count axis → concat(orig) → out_projector. */
  private MemorySegment downscaledTransformer(
    MemorySegment ctx,
    MemorySegment x,
    int m,
    int count
  ) {
    // x: (hidden, M·count), column t·M + m  →  (d, count, M): ne1 = t (sequence), ne2 = m (batch)
    var xIn = w.linear(ctx, x, "count_tf.in_projector"); // (d, M·count)
    var seq = Ggml.cont(
      ctx,
      Ggml.permute(ctx, Ggml.reshape3d(ctx, xIn, tfDim, m, count), 0, 2, 1, 3)
    ); // (d, count, M)
    int hd = tfDim / tfHeads;
    float scale = (float) (1.0 / Math.sqrt(hd));
    for (int l = 0; l < 2; l++) {
      var p = "count_tf.layers." + l + ".";
      var flat = Ggml.reshape2d(ctx, seq, tfDim, (long) count * m);
      var qkv = Ggml.add(
        ctx,
        Ggml.mulMat(ctx, w.get(p + "self_attn.in_proj_weight"), flat),
        w.get(p + "self_attn.in_proj_bias")
      ); // (3d, count·M)
      long nb1 = Ggml.nb(qkv, 1);
      long dBytes = (long) tfDim * Float.BYTES;
      var q = heads(
        ctx,
        Ggml.cont(ctx, Ggml.view2d(ctx, qkv, tfDim, (long) count * m, nb1, 0)),
        hd,
        count,
        m
      );
      var k = heads(
        ctx,
        Ggml.cont(
          ctx,
          Ggml.view2d(ctx, qkv, tfDim, (long) count * m, nb1, dBytes)
        ),
        hd,
        count,
        m
      );
      var v = heads(
        ctx,
        Ggml.cont(
          ctx,
          Ggml.view2d(ctx, qkv, tfDim, (long) count * m, nb1, 2 * dBytes)
        ),
        hd,
        count,
        m
      );
      var att = Ggml.softMax(
        ctx,
        Ggml.scale(ctx, Ggml.mulMat(ctx, k, q), scale)
      ); // (count_k, count_q, heads, M)
      var vT = Ggml.cont(ctx, Ggml.permute(ctx, v, 1, 0, 2, 3)); // (count_k, hd, heads, M)
      var ctxv = Ggml.mulMat(ctx, vT, att); // (hd, count_q, heads, M)
      var merged = Ggml.reshape2d(
        ctx,
        Ggml.cont(ctx, Ggml.permute(ctx, ctxv, 0, 2, 1, 3)), // (hd, heads, count, M)
        tfDim,
        (long) count * m
      );
      var attnOut = w.linear(ctx, merged, p + "self_attn.out_proj");
      var x1 = layerNorm(ctx, Ggml.add(ctx, flat, attnOut), p + "norm1");
      var ff = w.linear(
        ctx,
        Ggml.relu(ctx, w.linear(ctx, x1, p + "linear1")),
        p + "linear2"
      );
      var x2 = layerNorm(ctx, Ggml.add(ctx, x1, ff), p + "norm2");
      seq = Ggml.reshape3d(ctx, x2, tfDim, count, m);
    }
    // back to column t·M + m, concat with the original 768-d input, out_projector
    var y = Ggml.reshape2d(
      ctx,
      Ggml.cont(ctx, Ggml.permute(ctx, seq, 0, 2, 1, 3)),
      tfDim,
      (long) m * count
    ); // (d, M·count)
    var catd = Ggml.concat(ctx, y, x, 0); // (d + hidden, M·count)
    var o = Ggml.relu(ctx, w.linear(ctx, catd, "count_tf.out_projector.0"));
    o = Ggml.relu(ctx, w.linear(ctx, o, "count_tf.out_projector.2"));
    return w.linear(ctx, o, "count_tf.out_projector.4");
  }

  /** (d, count·M) → (hd, count, heads, M) */
  private MemorySegment heads(
    MemorySegment ctx,
    MemorySegment t,
    int hd,
    int count,
    int m
  ) {
    return Ggml.cont(
      ctx,
      Ggml.permute(
        ctx,
        Ggml.reshape4d(ctx, t, hd, tfHeads, count, m),
        0,
        2,
        1,
        3
      )
    );
  }

  private MemorySegment layerNorm(
    MemorySegment ctx,
    MemorySegment t,
    String prefix
  ) {
    return Ggml.add(
      ctx,
      Ggml.mul(ctx, Ggml.norm(ctx, t, tfEps), w.get(prefix + ".weight")),
      w.get(prefix + ".bias")
    );
  }

  @Override
  public synchronized void close() {
    w.close();
  }
}
