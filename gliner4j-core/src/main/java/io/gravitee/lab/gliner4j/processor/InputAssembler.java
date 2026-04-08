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
import java.util.Arrays;
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
  private final long[] schemaPrefixIds;
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

    // Pre-tokenize schema tokens into primitive arrays
    var specialToken = schemaEncoder.getSpecialToken();
    int schemaCapacity = schemaEncoder.getSchemaTokens().size() * 2;
    var schemaIds = new long[schemaCapacity];
    var schemaMappings = new TokenMapping[schemaCapacity];
    int schemaPos = 0;
    int fieldIdx = -1;

    for (int i = 0; i < schemaEncoder.getSchemaTokens().size(); i++) {
      var token = schemaEncoder.getSchemaTokens().get(i);
      if (specialToken.equals(token)) {
        fieldIdx++;
      }
      var result = tokenizer.tokenizeWithIds(token);
      for (long id : result.ids()) {
        if (schemaPos >= schemaIds.length) {
          schemaIds = Arrays.copyOf(schemaIds, schemaIds.length * 2);
          schemaMappings = Arrays.copyOf(
            schemaMappings,
            schemaMappings.length * 2
          );
        }
        schemaIds[schemaPos] = id;
        schemaMappings[schemaPos] = new TokenMapping(
          SegmentType.SCHEMA,
          i,
          fieldIdx
        );
        schemaPos++;
      }
    }

    this.cachedSchemaIds = Arrays.copyOf(schemaIds, schemaPos);
    this.cachedSchemaMappings = Arrays.copyOf(schemaMappings, schemaPos);

    // Pre-tokenize [SEP_TEXT] separator
    var sepResult = tokenizer.tokenizeWithIds("[SEP_TEXT]");
    this.cachedSepIds = sepResult.ids();

    // Build concatenated prefix for buffer reuse in runtime
    this.schemaPrefixIds = new long[cachedSchemaIds.length +
    cachedSepIds.length];
    System.arraycopy(
      cachedSchemaIds,
      0,
      schemaPrefixIds,
      0,
      cachedSchemaIds.length
    );
    System.arraycopy(
      cachedSepIds,
      0,
      schemaPrefixIds,
      cachedSchemaIds.length,
      cachedSepIds.length
    );
  }

  /**
   * Returns the concatenated schema + separator token IDs that form the constant prefix of every input.
   *
   * @return the prefix token IDs
   */
  public long[] getSchemaPrefixIds() {
    return schemaPrefixIds;
  }

  /**
   * Assembles cached schema/separator tokens with per-request text encoding into a complete model input.
   *
   * @param textEncoder the per-request text encoding
   * @return a PreprocessedInput ready for the ONNX model
   */
  public PreprocessedInput assemble(TextEncoder textEncoder) {
    // Estimate capacity: schema + sep + text subwords (avg ~2 subwords per word)
    int estimatedSize =
      cachedSchemaIds.length +
      cachedSepIds.length +
      textEncoder.getTextLen() * 2;
    var ids = new long[estimatedSize];
    var mappings = new TokenMapping[estimatedSize];
    int pos = 0;

    // 1. Append cached schema tokens
    System.arraycopy(cachedSchemaIds, 0, ids, 0, cachedSchemaIds.length);
    System.arraycopy(
      cachedSchemaMappings,
      0,
      mappings,
      0,
      cachedSchemaMappings.length
    );
    pos = cachedSchemaIds.length;

    // 2. Append cached [SEP_TEXT] separator
    var sepMapping = new TokenMapping(SegmentType.SEP, -1, -1);
    for (long id : cachedSepIds) {
      ids[pos] = id;
      mappings[pos] = sepMapping;
      pos++;
    }

    // 3. Tokenize text words (per-request)
    for (int w = 0; w < textEncoder.getTextLen(); w++) {
      var word = textEncoder.getWords().get(w);
      var result = tokenizer.tokenizeWithIds(word);
      var textMapping = new TokenMapping(SegmentType.TEXT, w, -1);
      for (long id : result.ids()) {
        if (pos >= ids.length) {
          ids = Arrays.copyOf(ids, ids.length * 2);
          mappings = Arrays.copyOf(mappings, mappings.length * 2);
        }
        ids[pos] = id;
        mappings[pos] = textMapping;
        pos++;
      }
    }

    // Trim to exact size
    var inputIds = pos == ids.length ? ids : Arrays.copyOf(ids, pos);
    var finalMappings = pos == mappings.length
      ? mappings
      : Arrays.copyOf(mappings, pos);
    var attentionMask = new long[pos];
    Arrays.fill(attentionMask, 1L);

    return new PreprocessedInput(
      inputIds,
      attentionMask,
      finalMappings,
      textEncoder.getWordStartChars(),
      textEncoder.getWordEndChars(),
      textEncoder.getTextLen(),
      numFields,
      fieldNames,
      textEncoder.getOriginalText()
    );
  }
}
