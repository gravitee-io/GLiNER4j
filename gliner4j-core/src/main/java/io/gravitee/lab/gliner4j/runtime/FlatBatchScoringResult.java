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
package io.gravitee.lab.gliner4j.runtime;

/**
 * Batched scoring-head output with flat span scores.
 *
 * <p>{@link #close()} releases the pinned span-scores buffer back to its pool (a no-op for
 * the first, unpinned run); call it once decoding is done.
 *
 * @param countLogits shared count logits [1][maxCount+1]
 * @param spanScores  flat span scores, shape [batch][cnt][numFields][textLen][maxWidth]
 * @param release     releases pinned resources; idempotent
 */
public record FlatBatchScoringResult(
  float[][] countLogits,
  FloatTensorView spanScores,
  Runnable release
) implements AutoCloseable {
  @Override
  public void close() {
    release.run();
  }

  /**
   * Materializes one batch row's span scores as nested arrays, sliced to that text's own
   * word count: {@code [count][numFields][textLen][maxWidth]}. Bridges the flat batched
   * output to the multi-instance decoders (relations, structures) that consume nested
   * arrays.
   *
   * @param batchRow the text's row within the batch
   * @param textLen  the text's word count (may be shorter than the padded tensor textLen)
   */
  public float[][][][] materializeSlot(int batchRow, int textLen) {
    int cnt = spanScores.dim(1);
    int numFields = spanScores.dim(2);
    int paddedTextLen = spanScores.dim(3);
    int maxWidth = spanScores.dim(4);
    int effTextLen = Math.min(textLen, paddedTextLen);

    var out = new float[cnt][numFields][effTextLen][maxWidth];
    long rowBase = (long) batchRow * spanScores.stride(0);
    for (int c = 0; c < cnt; c++) {
      long cBase = rowBase + (long) c * spanScores.stride(1);
      for (int f = 0; f < numFields; f++) {
        long fBase = cBase + (long) f * spanScores.stride(2);
        for (int t = 0; t < effTextLen; t++) {
          spanScores.copyRowAsFloats(
            fBase + (long) t * spanScores.stride(3),
            out[c][f][t],
            0,
            maxWidth
          );
        }
      }
    }
    return out;
  }
}
