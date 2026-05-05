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

import java.util.List;

/**
 * Complete preprocessed model input for the multi-unit prompt path.
 *
 * <p>Schema unit special-token positions are resolved at assembly time into {@link UnitLayout}
 * records so per-unit embedding extraction is a direct index into encoder hidden states.
 *
 * @param inputIds token IDs for the full sequence
 * @param attentionMask attention mask (1 for real tokens, 0 for padding)
 * @param wordStartChars character start offsets per text word
 * @param wordEndChars character end offsets per text word
 * @param wordFirstSubwordPos absolute position of each word's first subword in inputIds
 * @param textLen number of text words
 * @param textTokenStartPos absolute position where text subwords begin in inputIds
 * @param unitLayouts per-unit token positions
 * @param originalText the original input text
 */
public record MultiSchemaInput(
  long[] inputIds,
  long[] attentionMask,
  int[] wordStartChars,
  int[] wordEndChars,
  int[] wordFirstSubwordPos,
  int textLen,
  int textTokenStartPos,
  List<UnitLayout> unitLayouts,
  String originalText
) {}
