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
import java.util.ArrayList;
import java.util.List;

/**
 * Shared bucketed driver for the merged GLiNER2 graph ({@code ner_full.onnx}).
 *
 * <p>Slots are grouped into length-homogeneous sub-batches by {@link BatchBucketer}; each
 * sub-batch is one full-graph session run — raw token ids and gather positions in, span
 * scores out. Results are written back slot-indexed so downstream decode code is
 * order-independent.
 */
public final class BatchSpanPipeline {

  private BatchSpanPipeline() {}

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
   * One executed sub-batch: the slots it contains and the bucket's scoring result
   * (to be closed after decoding).
   */
  public record Bucket(int[] slots, FlatBatchScoringResult scoring) {}

  /**
   * The executed sub-batches, for consumers that decode per bucket. Every bucket's
   * {@link Bucket#scoring()} must be closed after decoding.
   */
  public record Result(List<Bucket> buckets) {}

  /**
   * Runs the bucketed full-merged-graph stage over a packed batch: same bucketing as the
   * historical split pipeline, but each bucket is a single {@code ner_full.onnx} run — no
   * encoder round trip, no Java-side gather.
   *
   * <p>If a later bucket's session run fails, the scoring results already produced by
   * earlier buckets are closed before the failure propagates, so repeated inference
   * failures cannot retain native tensors.
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
    try {
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
        executedBuckets.add(new Bucket(bucket, scoring));
      }
    } catch (Throwable t) {
      for (var executed : executedBuckets) {
        try {
          executed.scoring().close();
        } catch (RuntimeException suppressed) {
          t.addSuppressed(suppressed);
        }
      }
      throw t;
    }
    return new Result(executedBuckets);
  }
}
