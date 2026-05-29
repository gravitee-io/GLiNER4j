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

import io.gravitee.lab.gliner4j.tokenizer.TokenMapping;
import java.util.List;

/**
 * Complete preprocessed model input ready for ONNX inference.
 *
 * @param inputIds token IDs for the full sequence (schema + sep + text)
 * @param attentionMask attention mask (1 for real tokens, 0 for padding)
 * @param mappings token-to-segment mappings for the full sequence
 * @param wordStartChars character start offsets for each text word
 * @param wordEndChars character end offsets for each text word
 * @param textLen number of text words
 * @param numFields number of entity fields
 * @param fieldNames ordered list of entity field names
 * @param schemaTokenPositions positions in {@code inputIds} of {@code [P]} (index 0) and each
 *                             special-marker token (indices {@code 1..numFields}); resolved
 *                             at assembler construction time so per-call embedding extraction
 *                             is O(numFields) instead of an O(seqLen) token-id walk
 * @param originalText the original input text
 */
public record PreprocessedInput(
  long[] inputIds,
  long[] attentionMask,
  TokenMapping[] mappings,
  int[] wordStartChars,
  int[] wordEndChars,
  int textLen,
  int numFields,
  List<String> fieldNames,
  int[] schemaTokenPositions,
  String originalText
) {}
