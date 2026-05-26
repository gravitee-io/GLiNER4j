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

import io.gravitee.lab.gliner4j.schema.FieldSpan;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * Decodes a relation unit's scoring tensor into a list of {@link RelationInstance}s.
 *
 * <p>Algorithm: read the predicted instance count from the count head, then for each instance
 * slot {@code i ∈ [0, predCount)} and each field {@code j}, pick the highest-confidence span
 * above threshold. If every field finds a span, emit a {@link RelationInstance} whose overall
 * confidence is the minimum across fields. Duplicate instances (same character offsets across
 * all fields) are collapsed, keeping the highest-confidence one.
 */
public class RelationDecoder {

  /**
   * Decodes one relation unit's outputs into instances for the corresponding relation type.
   *
   * @param relationName the relation type name
   * @param fieldNames ordered field names (e.g. {@code ["head", "tail"]})
   * @param countLogits count head output, shape {@code [1][maxCount]}
   * @param spanScores scoring head output, shape {@code [maxCount][numFields][textLen][maxWidth]}
   * @param wordStartChars character start offsets per text word
   * @param wordEndChars character end offsets per text word
   * @param originalText the original input text
   * @param textLen number of text words
   * @param threshold minimum field confidence required to keep a span
   * @return detected instances (possibly empty)
   */
  public List<RelationInstance> decode(
    String relationName,
    List<String> fieldNames,
    float[][] countLogits,
    float[][][][] spanScores,
    int[] wordStartChars,
    int[] wordEndChars,
    String originalText,
    int textLen,
    float threshold
  ) {
    if (countLogits.length == 0 || countLogits[0].length == 0) {
      return List.of();
    }
    int predCount = argmax(countLogits[0]);
    if (predCount <= 0) {
      return List.of();
    }
    int maxAvailable = spanScores.length;
    int instances = Math.min(predCount, maxAvailable);
    int numFields = fieldNames.size();

    var result = new ArrayList<RelationInstance>(instances);
    var seenKeys = new HashSet<String>();

    for (int inst = 0; inst < instances; inst++) {
      var fieldSpans = new LinkedHashMap<String, FieldSpan>(numFields);
      float minConfidence = Float.MAX_VALUE;
      boolean allFieldsFound = true;

      for (int f = 0; f < numFields; f++) {
        var fieldName = fieldNames.get(f);
        var bestSpan = findBestSpan(
          spanScores[inst][f],
          wordStartChars,
          wordEndChars,
          originalText,
          textLen,
          threshold,
          fieldName
        );
        if (bestSpan == null) {
          allFieldsFound = false;
          break;
        }
        fieldSpans.put(fieldName, bestSpan);
        minConfidence = Math.min(minConfidence, bestSpan.confidence());
      }

      if (!allFieldsFound) {
        continue;
      }

      var dedupKey = buildDedupKey(fieldNames, fieldSpans);
      if (!seenKeys.add(dedupKey)) {
        continue;
      }

      result.add(new RelationInstance(relationName, fieldSpans, minConfidence));
    }

    return result;
  }

  private static FieldSpan findBestSpan(
    float[][] fieldScores,
    int[] wordStartChars,
    int[] wordEndChars,
    String originalText,
    int textLen,
    float threshold,
    String fieldName
  ) {
    if (fieldScores.length == 0) {
      return null;
    }
    int maxWidth = fieldScores[0].length;
    float bestScore = Float.NEGATIVE_INFINITY;
    int bestStart = -1;
    int bestEnd = -1;

    for (
      int start = 0;
      start < textLen && start < fieldScores.length;
      start++
    ) {
      var startScores = fieldScores[start];
      for (int w = 0; w < maxWidth; w++) {
        float score = startScores[w];
        if (score < threshold) {
          continue;
        }
        int endWord = start + w;
        if (endWord >= textLen) {
          continue;
        }
        if (score > bestScore) {
          bestScore = score;
          bestStart = start;
          bestEnd = endWord;
        }
      }
    }

    if (bestStart < 0) {
      return null;
    }
    int charStart = wordStartChars[bestStart];
    int charEnd = wordEndChars[bestEnd];
    return new FieldSpan(
      fieldName,
      originalText.substring(charStart, charEnd),
      bestScore,
      charStart,
      charEnd
    );
  }

  private static String buildDedupKey(
    List<String> fieldNames,
    LinkedHashMap<String, FieldSpan> fieldSpans
  ) {
    var sb = new StringBuilder();
    for (var name : fieldNames) {
      var span = fieldSpans.get(name);
      sb.append(span.start()).append('-').append(span.end()).append('|');
    }
    return sb.toString();
  }

  private static int argmax(float[] values) {
    int maxIdx = 0;
    float maxVal = values[0];
    for (int i = 1; i < values.length; i++) {
      if (values[i] > maxVal) {
        maxVal = values[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }
}
