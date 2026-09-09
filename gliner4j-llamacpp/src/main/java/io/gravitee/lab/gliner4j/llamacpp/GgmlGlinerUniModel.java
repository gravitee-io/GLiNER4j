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
 * The original-GLiNER uni-encoder span model ({@code UniEncoderSpanModel}, {@code span_mode=markerV0})
 * as one ggml graph: DeBERTa encoder ({@link GgmlDebertaV3}, tensors {@code enc.*}) → first-subtoken
 * word rows and {@code <<ENT>>} prompt rows → optional 1-layer bidirectional LSTM over the words
 * (unrolled) → {@code SpanMarkerV0} (project_start / project_end, gather by span, ReLU(concat),
 * out_project) and {@code prompt_rep_layer} → dot-product logits {@code [words][width][classes]}.
 * Weights: {@code gguf/model.gguf} from {@code export_llamacpp.py gliner-uni}.
 */
public final class GgmlGlinerUniModel implements AutoCloseable {

  private final GgmlWeights w;
  private final GgmlDebertaV3 encoder;
  private final int hidden;
  private final boolean hasRnn;
  private final int lstmHidden;
  private final boolean hasProjection;
  private final boolean hasLabelsProjection;
  private final boolean tokenLevel;

  public GgmlGlinerUniModel(Path gguf, boolean useGpu, int cpuThreads) {
    this.w = new GgmlWeights(
      gguf,
      useGpu,
      cpuThreads,
      "ggml GLiNER uni-encoder"
    );
    this.encoder = new GgmlDebertaV3(w, "enc.");
    this.hidden = encoder.hidden();
    this.hasRnn = w.has("lstm.weight_ih_l0");
    this.lstmHidden = hasRnn ? (int) Ggml.ne(w.get("lstm.weight_hh_l0"), 0) : 0;
    // gliner's Encoder.projection (encoder hidden ≠ model hidden, e.g. deberta-v3-large → 512)
    this.hasProjection = w.has("proj.weight");
    // bi-encoder: label-encoder hidden ≠ model hidden
    this.hasLabelsProjection = w.has("labels_projection.weight");
    this.tokenLevel = "token_level".equals(
      w.metaString("gliner.span_mode", "markerV0")
    );
  }

  /** Model hidden size after the optional projection (the span / token heads' width). */
  public int modelHidden() {
    return hasProjection ? (int) Ggml.ne(w.get("proj.weight"), 1) : hidden;
  }

  public boolean isTokenLevel() {
    return tokenLevel;
  }

  public int hiddenSize() {
    return hidden;
  }

  /**
   * Span logits {@code [W][maxWidth][C]} for one prompt. {@code wordPositions} are the token
   * positions of each text word's first subtoken, {@code promptPositions} those of the
   * {@code <<ENT>>} markers (one per class). Spans running past the text get {@code -1e4}.
   */
  public synchronized float[][][] score(
    long[] ids,
    int[] wordPositions,
    int[] promptPositions,
    int maxWidth
  ) {
    return score(ids, wordPositions, promptPositions, null, maxWidth);
  }

  /**
   * Span logits with the class representations supplied externally (bi-encoder: mean-pooled
   * label-encoder states, {@code [C][labelHidden]}) instead of gathered at {@code promptPositions}.
   */
  public synchronized float[][][] score(
    long[] ids,
    int[] wordPositions,
    int[] promptPositions,
    float[][] labelEmbeddings,
    int maxWidth
  ) {
    int n = ids.length;
    int words = wordPositions.length;
    int classes = labelEmbeddings != null
      ? labelEmbeddings.length
      : promptPositions.length;
    int labelHidden = labelEmbeddings != null && classes > 0
      ? labelEmbeddings[0].length
      : 0;
    int spans = words * maxWidth;
    var spanStart = new int[spans];
    var spanEnd = new int[spans];
    for (int s = 0; s < words; s++) {
      for (int k = 0; k < maxWidth; k++) {
        int idx = s * maxWidth + k;
        int end = s + k;
        spanStart[idx] = end < words ? s : 0;
        spanEnd[idx] = end < words ? end : 0;
      }
    }
    long nodes = encoder.graphNodes() + 64L + (hasRnn ? 40L * words : 0) + 24;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, nodes);
      try {
        var built = encoder.build(ctx, n);
        var wordIdx = Ggml.newTensor1d(ctx, w.typeI32, words);
        var promptIdx = labelEmbeddings == null
          ? Ggml.newTensor1d(ctx, w.typeI32, classes)
          : null;
        var labelsIn = labelEmbeddings != null
          ? Ggml.newTensor2d(ctx, w.typeF32, labelHidden, classes)
          : null;
        var startIdx = Ggml.newTensor1d(ctx, w.typeI32, spans);
        var endIdx = Ggml.newTensor1d(ctx, w.typeI32, spans);
        for (var t : new MemorySegment[] {
          wordIdx,
          promptIdx,
          startIdx,
          endIdx,
          labelsIn,
        }) {
          if (t != null) Ggml.setInput(t);
        }
        var h = built.output(); // (hidden, n)
        var wordRows = project(ctx, Ggml.getRows(ctx, h, wordIdx)); // (H', W)
        MemorySegment promptRows;
        if (labelsIn != null) {
          // bi-encoder: mean-pooled label-encoder states (+ labels_projection when hidden sizes differ)
          promptRows = hasLabelsProjection
            ? w.linear(ctx, labelsIn, "labels_projection")
            : labelsIn;
        } else {
          promptRows = project(ctx, Ggml.getRows(ctx, h, promptIdx)); // (H', C)
        }
        // gliner's BaseBiEncoderModel.get_representations never applies the (present) LSTM.
        if (hasRnn && labelsIn == null) {
          wordRows = biLstm(ctx, wordRows, words);
        }
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
        ); // (2·hidden, spans)
        var spanRep = w.projection(ctx, cat, "span_rep.out_project"); // (hidden, spans)
        var promptRep = w.projection(ctx, promptRows, "prompt_rep"); // (hidden, C)
        var logits = Ggml.mulMat(ctx, promptRep, spanRep); // (C, spans)

        var graph = Ggml.newGraph(ctx, nodes);
        w.compute(ctx, graph, logits);
        encoder.feed(built, call, ids);
        Ggml.setInts(wordIdx, call, wordPositions);
        if (promptIdx != null) Ggml.setInts(promptIdx, call, promptPositions);
        if (labelsIn != null) {
          var flatLabels = new float[labelHidden * classes];
          for (int c = 0; c < classes; c++) System.arraycopy(
            labelEmbeddings[c],
            0,
            flatLabels,
            c * labelHidden,
            labelHidden
          );
          Ggml.setFloats(labelsIn, call, flatLabels);
        }
        Ggml.setInts(startIdx, call, spanStart);
        Ggml.setInts(endIdx, call, spanEnd);
        w.run(graph);
        var flat = Ggml.getFloats(logits, call, classes * spans);
        var out = new float[words][maxWidth][classes];
        for (int s = 0; s < words; s++) {
          for (int k = 0; k < maxWidth; k++) {
            int idx = s * maxWidth + k;
            boolean valid = s + k < words;
            for (int c = 0; c < classes; c++) {
              out[s][k][c] = valid ? flat[idx * classes + c] : -1e4f;
            }
          }
        }
        return out;
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /**
   * Token-level logits {@code [W][C][3]} (start, end, inside) for a {@code span_mode=token_level}
   * checkpoint: gliner's {@code Scorer} over the (LSTM-refined) word rows and the {@code <<ENT>>}
   * rows.
   */
  public synchronized float[][][] scoreTokens(
    long[] ids,
    int[] wordPositions,
    int[] promptPositions
  ) {
    int n = ids.length;
    int words = wordPositions.length;
    int classes = promptPositions.length;
    int hp = modelHidden();
    long nodes = encoder.graphNodes() + 64L + (hasRnn ? 40L * words : 0) + 32;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, nodes);
      try {
        var built = encoder.build(ctx, n);
        var wordIdx = Ggml.newTensor1d(ctx, w.typeI32, words);
        var promptIdx = Ggml.newTensor1d(ctx, w.typeI32, classes);
        Ggml.setInput(wordIdx);
        Ggml.setInput(promptIdx);
        var h = built.output();
        var wordRows = project(ctx, Ggml.getRows(ctx, h, wordIdx)); // (H', W)
        var promptRows = project(ctx, Ggml.getRows(ctx, h, promptIdx)); // (H', C)
        if (hasRnn) {
          wordRows = biLstm(ctx, wordRows, words);
        }
        var tokP = w.linear(ctx, wordRows, "scorer.proj_token"); // (2H', W)
        var labP = w.linear(ctx, promptRows, "scorer.proj_label"); // (2H', C)
        long hpBytes = (long) hp * Float.BYTES;
        var tok0 = Ggml.cont(
          ctx,
          Ggml.view2d(ctx, tokP, hp, words, Ggml.nb(tokP, 1), 0)
        );
        var tok1 = Ggml.cont(
          ctx,
          Ggml.view2d(ctx, tokP, hp, words, Ggml.nb(tokP, 1), hpBytes)
        );
        var lab0 = Ggml.cont(
          ctx,
          Ggml.view2d(ctx, labP, hp, classes, Ggml.nb(labP, 1), 0)
        );
        var lab1 = Ggml.cont(
          ctx,
          Ggml.view2d(ctx, labP, hp, classes, Ggml.nb(labP, 1), hpBytes)
        );
        var tmpl = Ggml.newTensor3d(ctx, w.typeF32, hp, words, classes);
        var tok0r = Ggml.repeat(
          ctx,
          Ggml.reshape3d(ctx, tok0, hp, words, 1),
          tmpl
        );
        var lab0r = Ggml.repeat(
          ctx,
          Ggml.reshape3d(ctx, lab0, hp, 1, classes),
          tmpl
        );
        var prod = Ggml.mul(
          ctx,
          Ggml.repeat(ctx, Ggml.reshape3d(ctx, tok1, hp, words, 1), tmpl),
          Ggml.reshape3d(ctx, lab1, hp, 1, classes)
        );
        var cat = Ggml.concat(ctx, Ggml.concat(ctx, tok0r, lab0r, 0), prod, 0); // (3H', W, C)
        var flat = Ggml.reshape2d(ctx, cat, 3L * hp, (long) words * classes);
        var logits = w.linear(
          ctx,
          Ggml.relu(ctx, w.linear(ctx, flat, "scorer.out_mlp.0")),
          "scorer.out_mlp.3"
        ); // (3, W·C)
        var graph = Ggml.newGraph(ctx, nodes);
        w.compute(ctx, graph, logits);
        encoder.feed(built, call, ids);
        Ggml.setInts(wordIdx, call, wordPositions);
        Ggml.setInts(promptIdx, call, promptPositions);
        w.run(graph);
        var out3 = Ggml.getFloats(logits, call, 3 * words * classes);
        var out = new float[words][classes][3];
        for (int c = 0; c < classes; c++) {
          for (int t = 0; t < words; t++) {
            for (int k = 0; k < 3; k++) out[t][c][k] = out3[k +
            3 * (t + words * c)];
          }
        }
        return out;
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /** Diagnostic: the class representations the scorer sees for external label embeddings ({@code [C][H']}). */
  public synchronized float[][] promptRepresentations(
    float[][] labelEmbeddings
  ) {
    int classes = labelEmbeddings.length;
    int lh = labelEmbeddings[0].length;
    int hp = modelHidden();
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, 64);
      try {
        var labelsIn = Ggml.newTensor2d(ctx, w.typeF32, lh, classes);
        Ggml.setInput(labelsIn);
        var rows = hasLabelsProjection
          ? w.linear(ctx, labelsIn, "labels_projection")
          : labelsIn;
        var rep = w.projection(ctx, rows, "prompt_rep");
        var graph = Ggml.newGraph(ctx, 64);
        w.compute(ctx, graph, rep);
        var flat = new float[lh * classes];
        for (int c = 0; c < classes; c++) System.arraycopy(
          labelEmbeddings[c],
          0,
          flat,
          c * lh,
          lh
        );
        Ggml.setFloats(labelsIn, call, flat);
        w.run(graph);
        var out = Ggml.getFloats(rep, call, hp * classes);
        var res = new float[classes][hp];
        for (int c = 0; c < classes; c++) System.arraycopy(
          out,
          c * hp,
          res[c],
          0,
          hp
        );
        return res;
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /** Diagnostic: raw encoder hidden states {@code [n][hidden]}. */
  public synchronized float[][] encoderRows(long[] ids) {
    int n = ids.length;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, encoder.graphNodes() + 8);
      try {
        var built = encoder.build(ctx, n);
        var graph = Ggml.newGraph(ctx, encoder.graphNodes() + 8);
        w.compute(ctx, graph, built.output());
        encoder.feed(built, call, ids);
        w.run(graph);
        var flat = Ggml.getFloats(built.output(), call, n * hidden);
        var rows = new float[n][hidden];
        for (int i = 0; i < n; i++) System.arraycopy(
          flat,
          i * hidden,
          rows[i],
          0,
          hidden
        );
        return rows;
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /** gliner's {@code Encoder.projection} when the checkpoint has one. */
  private MemorySegment project(MemorySegment ctx, MemorySegment rows) {
    return hasProjection ? w.linear(ctx, rows, "proj") : rows;
  }

  /** gliner's {@code LstmSeq2SeqEncoder}: 1-layer bidirectional LSTM, output = [forward ; backward]. */
  private MemorySegment biLstm(MemorySegment ctx, MemorySegment x, int words) {
    var fwd = direction(ctx, x, words, "", false);
    var bwd = direction(ctx, x, words, "_reverse", true);
    return Ggml.concat(ctx, fwd, bwd, 0); // (2·lstmHidden, W)
  }

  private MemorySegment direction(
    MemorySegment ctx,
    MemorySegment x,
    int words,
    String suffix,
    boolean reverse
  ) {
    int g = 4 * lstmHidden;
    // Input contribution for every step at once: (4·lstmHidden, W), biases folded in.
    var xw = Ggml.add(
      ctx,
      Ggml.add(
        ctx,
        Ggml.mulMat(ctx, w.get("lstm.weight_ih_l0" + suffix), x),
        w.get("lstm.bias_ih_l0" + suffix)
      ),
      w.get("lstm.bias_hh_l0" + suffix)
    );
    var whh = w.get("lstm.weight_hh_l0" + suffix);
    long nb1 = Ggml.nb(xw, 1);
    long gateBytes = (long) lstmHidden * Float.BYTES;
    MemorySegment h = null;
    MemorySegment c = null;
    MemorySegment out = null;
    for (int step = 0; step < words; step++) {
      int t = reverse ? words - 1 - step : step;
      MemorySegment gates = Ggml.view2d(ctx, xw, g, 1, nb1, t * nb1);
      if (h != null) {
        gates = Ggml.add(ctx, gates, Ggml.mulMat(ctx, whh, h));
      } else {
        gates = Ggml.cont(ctx, gates);
      }
      var i = Ggml.sigmoid(ctx, Ggml.view1d(ctx, gates, lstmHidden, 0));
      var f = Ggml.sigmoid(ctx, Ggml.view1d(ctx, gates, lstmHidden, gateBytes));
      var gg = Ggml.tanh(
        ctx,
        Ggml.view1d(ctx, gates, lstmHidden, 2 * gateBytes)
      );
      var o = Ggml.sigmoid(
        ctx,
        Ggml.view1d(ctx, gates, lstmHidden, 3 * gateBytes)
      );
      var ig = Ggml.mul(ctx, i, gg);
      c = c == null ? ig : Ggml.add(ctx, Ggml.mul(ctx, f, c), ig);
      h = Ggml.mul(ctx, o, Ggml.tanh(ctx, c)); // (lstmHidden)
      var hCol = Ggml.reshape2d(ctx, h, lstmHidden, 1);
      if (out == null) {
        out = hCol;
      } else {
        out = reverse
          ? Ggml.concat(ctx, hCol, out, 1)
          : Ggml.concat(ctx, out, hCol, 1);
      }
    }
    return out;
  }

  @Override
  public synchronized void close() {
    w.close();
  }
}
