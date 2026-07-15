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
package io.gravitee.lab.gliner4j.processor;

import io.gravitee.lab.gliner4j.runtime.FlatBatchScoringResult;
import io.gravitee.lab.gliner4j.runtime.FloatBufferPool;
import io.gravitee.lab.gliner4j.runtime.FloatTensorView;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
import io.gravitee.lab.gliner4j.runtime.PinnedTensorLease;
import java.nio.FloatBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Shared encoder → span_rep batch stage used by the NER and multi-unit batch paths.
 *
 * <p>Slots are grouped into length-homogeneous sub-batches by {@link BatchBucketer}, each
 * sub-batch is encoded and span-repped padded only to its own max lengths, and the results
 * are written back slot-indexed so downstream per-text fanout code is order-independent.
 *
 * <p>Tensor data stays flat throughout: hidden states stay pinned to pooled direct buffers, text
 * embeddings are assembled into a pooled direct buffer fed zero-copy into span_rep, and each
 * bucket's span representations come back flat. Consumers that still need per-slot nested
 * arrays (the per-text scoring fallback and the multi-unit fanout) opt in via
 * {@code materializePerSlot}.
 *
 * <p>Buckets run sequentially on the calling thread, so the runtime's reusable batch buffers
 * are never used concurrently.
 */
public final class BatchSpanPipeline {

  private static final FloatBufferPool TEXT_EMBS_POOL = new FloatBufferPool();

  private BatchSpanPipeline() {}

  /**
   * Writes one slot's per-word text embeddings from the bucket's flat hidden states into the
   * flat target buffer.
   *
   * <p>Implementations copy each word's embedding row (hiddenSize floats) from
   * {@code hidden[row][sourcePos]} to {@code target[targetBase + word*hiddenSize]} and MUST
   * write (or zero) every word row in {@code [0, textLen)} — the target is a pooled buffer
   * carrying stale content, and the pipeline only zeroes the padding rows beyond textLen.
   */
  @FunctionalInterface
  public interface TextEmbeddingExtractor {
    void extract(
      FloatTensorView hidden,
      int row,
      int slot,
      FloatBuffer target,
      int targetBase
    );
  }

  /**
   * Scores one bucket through the merged {@code span_scoring.onnx} graph. Invoked in place of
   * the span_rep session while the bucket's text embeddings are still assembled; the
   * representative hidden state is passed so the first invocation can extract the schema
   * embeddings (identical for all buckets).
   */
  @FunctionalInterface
  public interface MergedScorer {
    FlatBatchScoringResult score(
      FloatBuffer textEmbs,
      int batchSize,
      int maxTextLen,
      long[] spanIdxFlat,
      float[][] repHiddenState,
      int repSlot
    );
  }

  /**
   * Scores one bucket through the full merged graph ({@code ner_full.onnx}): raw padded
   * token ids plus gather positions in, span scores out.
   */
  @FunctionalInterface
  public interface FullGraphScorer {
    FlatBatchScoringResult score(
      long[][] inputIds,
      long[][] attentionMask,
      int maxSeqLen,
      long[] wordPositionsFlat,
      int batchSize,
      int maxTextLen,
      long[] spanIdxFlat
    );
  }

  /**
   * One executed sub-batch: the slots it contains and either their span representations
   * ({@code spanRep} — pinned, shape {@code [bucketSize][maxTextLen][maxWidth][hiddenSize]},
   * to be closed by the batched-scoring consumer) or, on the merged span_scoring path, the
   * bucket's scoring result ({@code scoring} — to be closed after decoding). Exactly one of
   * the two is non-null unless {@code materializePerSlot} was requested (then both are null
   * after the per-slot arrays are materialized).
   */
  public record Bucket(
    int[] slots,
    PinnedTensorLease spanRep,
    FlatBatchScoringResult scoring
  ) {}

  /**
   * Slot-indexed pipeline output.
   *
   * @param spanRepBySlot  per-slot span representations sliced to that slot's {@code textLen}
   *                       ({@code [textLen][maxWidth][hiddenSize]}) — only populated when
   *                       {@code materializePerSlot} was requested, {@code null} otherwise
   * @param repHiddenState the full encoder hidden state of one representative slot — the schema
   *                       prefix is identical across slots, so schema/unit embeddings can be
   *                       extracted from it once
   * @param repSlot        the slot {@code repHiddenState} belongs to
   * @param buckets        the executed sub-batches, for consumers that score per bucket
   */
  public record Result(
    float[][][][] spanRepBySlot,
    float[][] repHiddenState,
    int repSlot,
    List<Bucket> buckets
  ) {}

  /**
   * Full-merged-graph variant of {@link #run}: same bucketing, but each bucket is a single
   * {@code ner_full.onnx} run — no encoder round trip, no Java-side gather. Only the
   * per-bucket scoring results are populated on the returned {@link Result}.
   *
   * @param batchInputIds      packed token IDs per slot
   * @param batchAttentionMask packed attention masks per slot
   * @param textLens           per-slot word counts
   * @param wordPositions      per-slot first-subword position of each word (length textLen)
   * @param maxWidth           max span width
   * @param lengthRatio        bucketing ratio (see {@link BatchBucketer#bucketize})
   * @param maxSubBatchSize    max slots per sub-batch (null = unbounded)
   * @param scorer             one full-graph session run per bucket
   */
  public static Result runFullGraph(
    long[][] batchInputIds,
    long[][] batchAttentionMask,
    int[] textLens,
    java.util.function.IntFunction<int[]> wordPositions,
    int maxWidth,
    double lengthRatio,
    Integer maxSubBatchSize,
    FullGraphScorer scorer
  ) {
    int slots = batchInputIds.length;
    var seqLens = new int[slots];
    for (int s = 0; s < slots; s++) {
      seqLens[s] = batchInputIds[s].length;
    }
    var buckets = BatchBucketer.bucketize(
      seqLens,
      lengthRatio,
      maxSubBatchSize
    );

    var executedBuckets = new ArrayList<Bucket>(buckets.size());
    for (var bucket : buckets) {
      int bSize = bucket.length;
      var ids = new long[bSize][];
      var mask = new long[bSize][];
      int maxSeqLen = 0;
      int maxTextLen = 0;
      for (int j = 0; j < bSize; j++) {
        int s = bucket[j];
        ids[j] = batchInputIds[s];
        mask[j] = batchAttentionMask[s];
        maxSeqLen = Math.max(maxSeqLen, seqLens[s]);
        maxTextLen = Math.max(maxTextLen, textLens[s]);
      }

      int maxNumSpans = maxTextLen * maxWidth;
      var wordPosFlat = new long[bSize * maxTextLen];
      java.util.Arrays.fill(wordPosFlat, -1L);
      var spanIdxFlat = new long[bSize * maxNumSpans * 2];
      for (int j = 0; j < bSize; j++) {
        int s = bucket[j];
        var wp = wordPositions.apply(s);
        for (int w = 0; w < wp.length; w++) {
          wordPosFlat[j * maxTextLen + w] = wp[w];
        }
        var textSpanIdx = SpanIndexCache.flatSpanIdx(textLens[s], maxWidth);
        System.arraycopy(
          textSpanIdx,
          0,
          spanIdxFlat,
          j * maxNumSpans * 2,
          textSpanIdx.length
        );
      }

      var scoring = scorer.score(
        ids,
        mask,
        maxSeqLen,
        wordPosFlat,
        bSize,
        maxTextLen,
        spanIdxFlat
      );
      executedBuckets.add(new Bucket(bucket, null, scoring));
    }
    return new Result(null, null, -1, executedBuckets);
  }

  /**
   * Runs the bucketed encoder + span_rep stage over a packed batch.
   *
   * @param runtime            the NER runtime
   * @param batchInputIds      packed token IDs per slot
   * @param batchAttentionMask packed attention masks per slot
   * @param textLens           per-slot word counts
   * @param hiddenSize         encoder hidden size
   * @param maxWidth           max span width
   * @param lengthRatio        bucketing ratio (see {@link BatchBucketer#bucketize})
   * @param maxSubBatchSize    max slots per sub-batch (null = unbounded)
   * @param materializePerSlot whether to materialize per-slot nested span representations
   *                           (needed by per-text scoring paths; skip for batched scoring)
   * @param repRowPositions    hidden-state row positions the caller reads from the
   *                           representative slot (schema/unit marker rows) — only these rows
   *                           are materialized in {@code repHiddenState}, the rest stay null
   * @param extractor          per-slot text-embedding extraction
   * @return slot-indexed span representations plus a representative hidden state
   */
  public static Result run(
    GLiNER4jNERRuntime runtime,
    long[][] batchInputIds,
    long[][] batchAttentionMask,
    int[] textLens,
    int hiddenSize,
    int maxWidth,
    double lengthRatio,
    Integer maxSubBatchSize,
    boolean materializePerSlot,
    java.util.function.IntFunction<int[]> repRowPositions,
    MergedScorer mergedScorer,
    TextEmbeddingExtractor extractor
  ) {
    int slots = batchInputIds.length;
    var seqLens = new int[slots];
    for (int s = 0; s < slots; s++) {
      seqLens[s] = batchInputIds[s].length;
    }
    var buckets = BatchBucketer.bucketize(
      seqLens,
      lengthRatio,
      maxSubBatchSize
    );

    var spanRepBySlot = materializePerSlot ? new float[slots][][][] : null;
    var executedBuckets = new ArrayList<Bucket>(buckets.size());
    float[][] repHiddenState = null;
    int repSlot = -1;

    for (var bucket : buckets) {
      int bSize = bucket.length;
      var ids = new long[bSize][];
      var mask = new long[bSize][];
      int maxSeqLen = 0;
      int maxTextLen = 0;
      for (int j = 0; j < bSize; j++) {
        int s = bucket[j];
        ids[j] = batchInputIds[s];
        mask[j] = batchAttentionMask[s];
        maxSeqLen = Math.max(maxSeqLen, seqLens[s]);
        maxTextLen = Math.max(maxTextLen, textLens[s]);
      }

      int maxNumSpans = maxTextLen * maxWidth;
      int slotFloats = maxTextLen * hiddenSize;
      var spanIdxFlat = new long[bSize * maxNumSpans * 2];

      var embHolder = TEXT_EMBS_POOL.acquire();
      PinnedTensorLease bucketSpanRep = null;
      FlatBatchScoringResult bucketScoring = null;
      try {
        embHolder.ensureCapacity(bSize * slotFloats, 4 * slotFloats);
        var embBuf = embHolder.buf();
        embBuf.clear().limit(bSize * slotFloats);

        // The encoder output stays pinned to a pooled direct buffer for exactly the time it
        // takes to gather the per-word rows into the span_rep input.
        try (
          var hidden = runtime.runEncoderBatchPinned(
            ids,
            mask,
            maxSeqLen,
            hiddenSize
          )
        ) {
          if (repHiddenState == null) {
            repSlot = bucket[0];
            repHiddenState = materializeRows(
              hidden,
              0,
              repRowPositions.apply(repSlot)
            );
          }

          for (int j = 0; j < bSize; j++) {
            int s = bucket[j];
            int textLen = textLens[s];
            // The extractor writes every word row in [0, textLen); only the padding rows
            // beyond textLen carry stale pooled-buffer content and need zeroing.
            zero(
              embBuf,
              j * slotFloats + textLen * hiddenSize,
              (maxTextLen - textLen) * hiddenSize
            );
            extractor.extract(hidden, j, s, embBuf, j * slotFloats);

            // Memoized per-text span indices share the flat layout of the batch row's
            // first textLen*maxWidth pairs; padding beyond stays zero.
            var textSpanIdx = SpanIndexCache.flatSpanIdx(textLen, maxWidth);
            System.arraycopy(
              textSpanIdx,
              0,
              spanIdxFlat,
              j * maxNumSpans * 2,
              textSpanIdx.length
            );
          }
        }

        if (mergedScorer != null) {
          bucketScoring = mergedScorer.score(
            embBuf.rewind(),
            bSize,
            maxTextLen,
            spanIdxFlat,
            repHiddenState,
            repSlot
          );
        } else {
          bucketSpanRep = runtime.runSpanRepBatchPinned(
            embBuf.rewind(),
            bSize,
            maxTextLen,
            hiddenSize,
            maxWidth,
            spanIdxFlat
          );
        }
      } finally {
        TEXT_EMBS_POOL.release(embHolder);
      }

      if (materializePerSlot) {
        // Per-text consumers read the nested arrays; the pinned tensor is no longer needed.
        try (var lease = bucketSpanRep) {
          for (int j = 0; j < bSize; j++) {
            int s = bucket[j];
            spanRepBySlot[s] = materializeSlotSpanRep(
              bucketSpanRep,
              j,
              textLens[s]
            );
          }
        }
      }
      executedBuckets.add(new Bucket(bucket, bucketSpanRep, bucketScoring));
    }

    return new Result(spanRepBySlot, repHiddenState, repSlot, executedBuckets);
  }

  /**
   * Materializes only the requested rows of one batch entry's hidden states as a sparse
   * {@code [seqLen][hidden]} array (unrequested rows stay {@code null}) — schema/unit
   * embeddings read a few dozen marker rows, not the full sequence.
   */
  private static float[][] materializeRows(
    FloatTensorView hidden,
    int row,
    int[] positions
  ) {
    int seqLen = hidden.dim(1);
    int hiddenSize = hidden.dim(2);
    var out = new float[seqLen][];
    long base = (long) row * seqLen * hiddenSize;
    for (int pos : positions) {
      if (pos < 0 || pos >= seqLen || out[pos] != null) {
        continue;
      }
      var rowArr = new float[hiddenSize];
      hidden.copyRowAsFloats(
        base + (long) pos * hiddenSize,
        rowArr,
        0,
        hiddenSize
      );
      out[pos] = rowArr;
    }
    return out;
  }

  /**
   * Materializes one slot's span representations, sliced to its own text length:
   * {@code [textLen][maxWidth][hiddenSize]}.
   */
  private static float[][][] materializeSlotSpanRep(
    PinnedTensorLease spanRep,
    int row,
    int textLen
  ) {
    int maxTextLen = spanRep.dim(1);
    int maxWidth = spanRep.dim(2);
    int hiddenSize = spanRep.dim(3);
    var out = new float[textLen][maxWidth][hiddenSize];
    long rowBase = (long) row * maxTextLen * maxWidth * hiddenSize;
    for (int t = 0; t < textLen; t++) {
      long tBase = rowBase + (long) t * maxWidth * hiddenSize;
      for (int w = 0; w < maxWidth; w++) {
        spanRep.copyRowAsFloats(
          tBase + (long) w * hiddenSize,
          out[t][w],
          0,
          hiddenSize
        );
      }
    }
    return out;
  }

  private static final float[] ZERO_CHUNK = new float[4096];

  private static void zero(FloatBuffer buf, int offset, int length) {
    int pos = offset;
    int remaining = length;
    while (remaining > 0) {
      int n = Math.min(remaining, ZERO_CHUNK.length);
      buf.put(pos, ZERO_CHUNK, 0, n);
      pos += n;
      remaining -= n;
    }
  }
}
