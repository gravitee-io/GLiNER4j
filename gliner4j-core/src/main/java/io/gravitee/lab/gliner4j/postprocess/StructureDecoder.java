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
package io.gravitee.lab.gliner4j.postprocess;

import io.gravitee.lab.gliner4j.schema.FieldType;
import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import io.gravitee.lab.gliner4j.schema.StructureField;
import io.gravitee.lab.gliner4j.schema.StructureInstance;
import io.gravitee.lab.gliner4j.schema.StructureValue;
import io.gravitee.lab.gliner4j.schema.StructureValue.ListValue;
import io.gravitee.lab.gliner4j.schema.StructureValue.StringValue;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;

/**
 * Decodes raw span scores from the scoring head into typed {@link StructureInstance}s
 * for a single structure. The facade calls {@link #decode} once per structure, passing
 * that structure's slice of the scoring-head output.
 */
public class StructureDecoder {

  /**
   * Decodes one structure's span scores into instances.
   *
   * @param spanScores raw scores [count][numFields][textLen][maxWidth] for this structure
   * @param predCount predicted instance count for this structure (from countLogits argmax)
   * @param structure the structure definition (provides field types and choices)
   * @param wordStartChars character start offsets per word
   * @param wordEndChars character end offsets per word
   * @param originalText the original input text
   * @param textLen number of text words
   * @param threshold minimum confidence for inclusion
   * @return decoded instances (size {@code min(predCount, spanScores.length)})
   */
  public List<StructureInstance> decode(
    float[][][][] spanScores,
    int predCount,
    StructureDefinition structure,
    int[] wordStartChars,
    int[] wordEndChars,
    String originalText,
    int textLen,
    float threshold
  ) {
    if (predCount <= 0 || spanScores == null || spanScores.length == 0) {
      return List.of();
    }

    int actualCount = Math.min(predCount, spanScores.length);
    var instances = new ArrayList<StructureInstance>(actualCount);
    var fields = structure.fields();
    int numFields = fields.size();

    for (int instIdx = 0; instIdx < actualCount; instIdx++) {
      var instanceScores = spanScores[instIdx]; // [numFields][textLen][maxWidth]
      var fieldValues = new LinkedHashMap<String, StructureValue>();

      for (int f = 0; f < numFields && f < instanceScores.length; f++) {
        var field = fields.get(f);
        var candidates = collectCandidates(
          instanceScores[f],
          wordStartChars,
          wordEndChars,
          originalText,
          textLen,
          field,
          threshold
        );

        if (candidates.isEmpty()) continue;

        StructureValue value;
        if (field.type() == FieldType.STRING) {
          candidates.sort(
            Comparator.comparingDouble((StringValue v) ->
              v.confidence()
            ).reversed()
          );
          value = candidates.get(0);
        } else {
          var nonOverlapping = greedyNonOverlap(candidates);
          nonOverlapping.sort(Comparator.comparingInt(StringValue::start));
          value = new ListValue(nonOverlapping);
        }
        fieldValues.put(field.name(), value);
      }

      instances.add(new StructureInstance(fieldValues));
    }

    return instances;
  }

  private static List<StringValue> collectCandidates(
    float[][] scoresForField,
    int[] wordStartChars,
    int[] wordEndChars,
    String originalText,
    int textLen,
    StructureField field,
    float threshold
  ) {
    int maxWidth = scoresForField.length > 0 ? scoresForField[0].length : 0;
    var raw = new ArrayList<StringValue>();

    for (
      int start = 0;
      start < textLen && start < scoresForField.length;
      start++
    ) {
      for (int w = 0; w < maxWidth; w++) {
        float score = scoresForField[start][w];
        if (score < threshold) continue;
        int endWord = start + w;
        if (endWord >= textLen) continue;

        int charStart = wordStartChars[start];
        int charEnd = wordEndChars[endWord];
        var text = originalText.substring(charStart, charEnd);
        raw.add(new StringValue(text, score, charStart, charEnd));
      }
    }

    if (field.choices().isEmpty()) {
      return raw;
    }

    // Snap each candidate to the longest matching choice (case-insensitive).
    // Match if the choice equals, contains, or is contained in the extracted text — this handles the
    // common case of the model extracting "outdoor seating" instead of just "outdoor".
    var lowerChoices = new ArrayList<String>(field.choices().size());
    for (var choice : field.choices()) {
      lowerChoices.add(choice.toLowerCase(Locale.ROOT));
    }
    var snapped = new ArrayList<StringValue>();
    for (var v : raw) {
      String lowerText = v.text().toLowerCase(Locale.ROOT);
      String bestChoice = null;
      int bestLen = -1;
      for (int i = 0; i < lowerChoices.size(); i++) {
        var lc = lowerChoices.get(i);
        if (
          lowerText.equals(lc) ||
          lowerText.contains(lc) ||
          lc.contains(lowerText)
        ) {
          if (lc.length() > bestLen) {
            bestLen = lc.length();
            bestChoice = field.choices().get(i);
          }
        }
      }
      if (bestChoice != null) {
        snapped.add(
          new StringValue(bestChoice, v.confidence(), v.start(), v.end())
        );
      }
    }
    return snapped;
  }

  private static List<StringValue> greedyNonOverlap(
    List<StringValue> candidates
  ) {
    var sorted = new ArrayList<>(candidates);
    sorted.sort(Comparator.comparingDouble(StringValue::confidence).reversed());
    var kept = new ArrayList<StringValue>();
    for (var c : sorted) {
      boolean overlaps = false;
      for (var k : kept) {
        if (c.start() < k.end() && c.end() > k.start()) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps) kept.add(c);
    }
    return kept;
  }
}
