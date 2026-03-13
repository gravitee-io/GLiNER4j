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
 *
 * <p>Schema and separator tokens are tokenized once at construction time
 * and cached, since they are immutable across requests.
 */
public class InputAssembler {

  private final DjlTokenizerWrapper tokenizer;
  private final long[] cachedSchemaIds;
  private final TokenMapping[] cachedSchemaMappings;
  private final long[] cachedSepIds;
  private final int numFields;
  private final List<String> fieldNames;

  /**
   * Creates an InputAssembler that pre-computes schema and separator tokenization.
   *
   * @param tokenizer the subword tokenizer
   * @param schemaEncoder the immutable schema encoding
   */
  public InputAssembler(
    DjlTokenizerWrapper tokenizer,
    SchemaEncoder schemaEncoder
  ) {
    this.tokenizer = tokenizer;
    this.numFields = schemaEncoder.getNumFields();
    this.fieldNames = schemaEncoder.getFieldNames();

    // Pre-tokenize schema tokens
    var schemaIdsList = new ArrayList<Long>();
    var schemaMappingsList = new ArrayList<TokenMapping>();
    int entityIdx = -1;

    for (int i = 0; i < schemaEncoder.getSchemaTokens().size(); i++) {
      var token = schemaEncoder.getSchemaTokens().get(i);
      if ("[E]".equals(token)) {
        entityIdx++;
      }
      var result = tokenizer.tokenizeWithIds(token);
      for (long id : result.ids()) {
        schemaIdsList.add(id);
        schemaMappingsList.add(
          new TokenMapping(SegmentType.SCHEMA, i, entityIdx)
        );
      }
    }

    this.cachedSchemaIds =
      schemaIdsList.stream().mapToLong(Long::longValue).toArray();
    this.cachedSchemaMappings = schemaMappingsList.toArray(new TokenMapping[0]);

    // Pre-tokenize [SEP_TEXT] separator
    var sepResult = tokenizer.tokenizeWithIds("[SEP_TEXT]");
    this.cachedSepIds = sepResult.ids();
  }

  /**
   * Assembles cached schema/separator tokens with per-request text encoding into a complete model input.
   *
   * @param textEncoder the per-request text encoding
   * @return a PreprocessedInput ready for the ONNX model
   */
  public PreprocessedInput assemble(TextEncoder textEncoder) {
    // Estimate capacity: schema + sep + text subwords
    int estimatedSize =
      cachedSchemaIds.length +
      cachedSepIds.length +
      textEncoder.getTextLen() *
      2;
    var allSubwordIds = new ArrayList<Long>(estimatedSize);
    var allMappings = new ArrayList<TokenMapping>(estimatedSize);

    // 1. Append cached schema tokens
    for (int i = 0; i < cachedSchemaIds.length; i++) {
      allSubwordIds.add(cachedSchemaIds[i]);
      allMappings.add(cachedSchemaMappings[i]);
    }

    // 2. Append cached [SEP_TEXT] separator
    for (long id : cachedSepIds) {
      allSubwordIds.add(id);
      allMappings.add(new TokenMapping(SegmentType.SEP, -1, -1));
    }

    // 3. Tokenize text words (per-request)
    for (int w = 0; w < textEncoder.getTextLen(); w++) {
      var word = textEncoder.getWords().get(w);
      var result = tokenizer.tokenizeWithIds(word);
      for (long id : result.ids()) {
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
      numFields,
      fieldNames,
      textEncoder.getOriginalText()
    );
  }
}
