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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Assembles a multi-unit GLiNER schema prompt: concatenates per-unit blocks with
 * {@code [SEP_STRUCT]} separators, ends with {@code [SEP_TEXT]}, then per-request appends
 * the tokenized text.
 *
 * <p>Schema and separator tokenization happens once at construction time; per-request
 * {@link #assemble(TextEncoder)} only tokenizes the text and concatenates onto the cached prefix.
 *
 * <p>Resolved per-unit positions of {@code [P]} and child marker subwords are exposed via
 * {@link MultiSchemaInput#unitLayouts()} so decoders can read embeddings directly from encoder
 * hidden states without re-scanning the input.
 */
public class MultiSchemaInputAssembler {

  private final DjlTokenizerWrapper tokenizer;
  private final long[] cachedPrefixIds;
  private final List<UnitLayout> unitLayouts;
  private final int textTokenStartPos;

  /**
   * Constructs the assembler from an ordered list of schema units.
   *
   * @param tokenizer the subword tokenizer
   * @param schemaUnits the units to encode (must be non-empty)
   */
  public MultiSchemaInputAssembler(
    DjlTokenizerWrapper tokenizer,
    List<SchemaUnit> schemaUnits
  ) {
    if (schemaUnits.isEmpty()) {
      throw new IllegalArgumentException(
        "MultiSchemaInputAssembler requires at least one schema unit"
      );
    }
    this.tokenizer = tokenizer;

    var sepStructIds = tokenizer.tokenizeWithIds("[SEP_STRUCT]").ids();
    var sepTextIds = tokenizer.tokenizeWithIds("[SEP_TEXT]").ids();

    var ids = new ArrayList<Long>(64);
    var layouts = new ArrayList<UnitLayout>(schemaUnits.size());

    for (int u = 0; u < schemaUnits.size(); u++) {
      var unit = schemaUnits.get(u);
      var encoder = new SchemaEncoder(
        unit.parentLabel(),
        unit.childMarker(),
        unit.childNames(),
        unit.childDescriptions()
      );

      int parentTokenPos = -1;
      var childMarkerPositions = new ArrayList<Integer>(
        unit.childNames().size()
      );

      for (var token : encoder.getSchemaTokens()) {
        var subwordIds = tokenizer.tokenizeWithIds(token).ids();
        if (
          "[P]".equals(token) && parentTokenPos == -1 && subwordIds.length > 0
        ) {
          parentTokenPos = ids.size();
        }
        if (unit.childMarker().equals(token) && subwordIds.length > 0) {
          childMarkerPositions.add(ids.size());
        }
        for (long id : subwordIds) {
          ids.add(id);
        }
      }

      if (parentTokenPos < 0) {
        throw new IllegalStateException(
          "No [P] token found in unit " + unit.parentLabel()
        );
      }
      if (childMarkerPositions.size() != unit.childNames().size()) {
        throw new IllegalStateException(
          "Child marker count mismatch in unit " +
            unit.parentLabel() +
            ": expected " +
            unit.childNames().size() +
            ", got " +
            childMarkerPositions.size()
        );
      }

      layouts.add(
        new UnitLayout(
          unit,
          parentTokenPos,
          childMarkerPositions.stream().mapToInt(Integer::intValue).toArray()
        )
      );

      // Append [SEP_STRUCT] between units (not after the last unit)
      if (u < schemaUnits.size() - 1) {
        for (long id : sepStructIds) {
          ids.add(id);
        }
      }
    }

    // Append [SEP_TEXT] at the end of the schema prefix
    for (long id : sepTextIds) {
      ids.add(id);
    }

    this.cachedPrefixIds = new long[ids.size()];
    for (int i = 0; i < ids.size(); i++) {
      this.cachedPrefixIds[i] = ids.get(i);
    }
    this.textTokenStartPos = cachedPrefixIds.length;
    this.unitLayouts = layouts;
  }

  /**
   * Returns the cached schema + separator token IDs that form the constant prefix.
   *
   * @return the prefix token IDs
   */
  public long[] getSchemaPrefixIds() {
    return cachedPrefixIds;
  }

  /**
   * Returns the resolved per-unit layouts.
   *
   * @return the unit layouts in declaration order
   */
  public List<UnitLayout> getUnitLayouts() {
    return unitLayouts;
  }

  /**
   * Assembles the cached schema prefix with per-request text encoding into a complete model input.
   *
   * @param textEncoder the per-request text encoding
   * @return assembled multi-unit input
   */
  public MultiSchemaInput assemble(TextEncoder textEncoder) {
    int textLen = textEncoder.getTextLen();
    int estimated = cachedPrefixIds.length + Math.max(textLen * 2, 4);
    var ids = Arrays.copyOf(cachedPrefixIds, estimated);
    int pos = cachedPrefixIds.length;

    var wordFirstSubwordPos = new int[textLen];

    // Warm the tokenizer cache for all text words in one JNI call, then read per-word
    // tokens inline (every tokenizeWithIds below is now a pure cache hit).
    var words = textEncoder.getWords();
    tokenizer.prefetchTokens(words);
    for (int w = 0; w < textLen; w++) {
      var result = tokenizer.tokenizeWithIds(words.get(w));
      wordFirstSubwordPos[w] = pos;
      for (long id : result.ids()) {
        if (pos >= ids.length) {
          ids = Arrays.copyOf(ids, ids.length * 2);
        }
        ids[pos++] = id;
      }
    }

    var inputIds = pos == ids.length ? ids : Arrays.copyOf(ids, pos);
    var attentionMask = new long[pos];
    Arrays.fill(attentionMask, 1L);

    return new MultiSchemaInput(
      inputIds,
      attentionMask,
      textEncoder.getWordStartChars(),
      textEncoder.getWordEndChars(),
      wordFirstSubwordPos,
      textLen,
      textTokenStartPos,
      unitLayouts,
      textEncoder.getOriginalText()
    );
  }
}
