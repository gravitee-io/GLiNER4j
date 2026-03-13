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

import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.tokenizer.TokenMapping;
import io.gravitee.lab.gliner4j.tokenizer.TokenMapping.SegmentType;
import java.util.ArrayList;
import java.util.List;

/**
 * Merges schema tokens and text words into a single token sequence,
 * tokenizes each to subwords, and builds the full model input.
 */
public class InputAssembler {

  private final DjlTokenizerWrapper tokenizer;

  /**
   * Creates an InputAssembler with the given tokenizer.
   *
   * @param tokenizer the subword tokenizer
   */
  public InputAssembler(DjlTokenizerWrapper tokenizer) {
    this.tokenizer = tokenizer;
  }

  /**
   * Assembles schema and text encodings into a complete model input.
   *
   * @param schemaEncoder the pre-computed schema encoding
   * @param textEncoder the per-request text encoding
   * @return a PreprocessedInput ready for the ONNX model
   */
  public PreprocessedInput assemble(
    SchemaEncoder schemaEncoder,
    TextEncoder textEncoder
  ) {
    var allSubwordIds = new ArrayList<Long>();
    var allMappings = new ArrayList<TokenMapping>();

    // Track which schema index each token maps to (-1 for non-entity tokens)
    int entityIdx = -1;

    // 1. Tokenize schema tokens
    for (int i = 0; i < schemaEncoder.getSchemaTokens().size(); i++) {
      var token = schemaEncoder.getSchemaTokens().get(i);
      // Track entity index: incremented each time we see [E]
      if ("[E]".equals(token)) {
        entityIdx++;
      }
      var subwords = tokenizer.tokenize(token);
      var subwordIds = tokenizer.convertTokensToIds(subwords);
      for (long id : subwordIds) {
        allSubwordIds.add(id);
        allMappings.add(new TokenMapping(SegmentType.SCHEMA, i, entityIdx));
      }
    }

    // 2. Add [SEP_TEXT] separator
    var sepSubwords = tokenizer.tokenize("[SEP_TEXT]");
    var sepIds = tokenizer.convertTokensToIds(sepSubwords);
    for (long id : sepIds) {
      allSubwordIds.add(id);
      allMappings.add(new TokenMapping(SegmentType.SEP, -1, -1));
    }

    // 3. Tokenize text words
    for (int w = 0; w < textEncoder.getTextLen(); w++) {
      var word = textEncoder.getWords().get(w);
      var subwords = tokenizer.tokenize(word);
      var subwordIds = tokenizer.convertTokensToIds(subwords);
      for (long id : subwordIds) {
        allSubwordIds.add(id);
        allMappings.add(new TokenMapping(SegmentType.TEXT, w, -1));
      }
    }

    // Build arrays
    var inputIds = allSubwordIds.stream().mapToLong(Long::longValue).toArray();
    var attentionMask = new long[inputIds.length];
    java.util.Arrays.fill(attentionMask, 1L);

    return new PreprocessedInput(
      inputIds,
      attentionMask,
      allMappings.toArray(new TokenMapping[0]),
      textEncoder.getWordStartChars(),
      textEncoder.getWordEndChars(),
      textEncoder.getTextLen(),
      schemaEncoder.getNumFields(),
      schemaEncoder.getFieldNames(),
      textEncoder.getOriginalText()
    );
  }
}
