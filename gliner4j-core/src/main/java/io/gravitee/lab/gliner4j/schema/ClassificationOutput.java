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
package io.gravitee.lab.gliner4j.schema;

import java.util.List;

/**
 * Classification output for one input text, with the number of tokens the model read for it.
 *
 * <p>{@code inputTokens} counts the non-padding token ids of the encoder row fed to the model for
 * this text: special tokens and the label prompt (schema) tokens are included, padding added to
 * batch rows of unequal length is excluded. Each input text is encoded as exactly one row (no
 * truncation, no windowing). A text that is null, blank, has no words, or is not run through the
 * model (for example because no labels are requested) reports {@code 0}. The exact layout per
 * architecture is documented on
 * {@link io.gravitee.lab.gliner4j.strategy.ClassificationStrategy#classifyWithUsage(String, float)}.
 *
 * @param results labels above threshold, sorted by confidence descending, identical to what the
 *                plain {@code classify} methods return for the same call
 * @param inputTokens number of non-padding tokens the model read for this text
 */
public record ClassificationOutput(
  List<ClassificationResult> results,
  int inputTokens
) {
  /** The output of a text that was not run through the model. */
  public static ClassificationOutput empty() {
    return new ClassificationOutput(List.of(), 0);
  }
}
