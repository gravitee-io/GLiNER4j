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
}
