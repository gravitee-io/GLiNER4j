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
 * Raw batched scoring head output tensors. Count logits are text-independent (they derive
 * from the shared schema embedding), so they are not batched.
 *
 * @param countLogits count prediction logits [1, maxCount]
 * @param spanScores  span classification scores [batch, count, numFields, textLen, maxWidth]
 */
public record BatchScoringResult(
  float[][] countLogits,
  float[][][][][] spanScores
) {}
