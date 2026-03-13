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

import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Decodes raw span scores into entity spans.
 * Applies thresholding, character mapping, and greedy non-overlapping removal.
 */
public class SpanDecoder {

  /**
   * Decodes span scores into a list of entity spans.
   *
   * @param spanScores raw scores [count][numFields][textLen][maxWidth]
   * @param fieldNames ordered entity type names
   * @param wordStartChars character start offsets per word
   * @param wordEndChars character end offsets per word
   * @param originalText the original input text
   * @param textLen number of text words
   * @param threshold minimum confidence for inclusion
   * @return list of detected entity spans, after greedy non-overlapping removal
   */
  public List<EntitySpan> decode(
    float[][][][] spanScores,
    List<String> fieldNames,
    int[] wordStartChars,
    int[] wordEndChars,
    String originalText,
    int textLen,
    float threshold
  ) {
    var candidates = new ArrayList<EntitySpan>();

    // Use the first count instance (spanScores[0]) for entities
    if (spanScores.length == 0) {
      return List.of();
    }
    var scores = spanScores[0]; // [numFields][textLen][maxWidth]
    int numFields = scores.length;
    int maxWidth = scores.length > 0 && scores[0].length > 0
      ? scores[0][0].length
      : 0;

    for (int f = 0; f < numFields; f++) {
      var fieldName = fieldNames.get(f);
      for (int start = 0; start < textLen; start++) {
        for (int w = 0; w < maxWidth; w++) {
          float score = scores[f][start][w];
          if (score >= threshold) {
            int endWord = start + w; // inclusive end word index
            if (endWord >= textLen) continue;

            int charStart = wordStartChars[start];
            int charEnd = wordEndChars[endWord];
            var text = originalText.substring(charStart, charEnd);
            candidates.add(
              new EntitySpan(fieldName, text, score, charStart, charEnd)
            );
          }
        }
      }
    }

    // Greedy non-overlapping removal: sort by confidence desc, skip overlaps
    candidates.sort(
      Comparator.comparingDouble(EntitySpan::confidence).reversed()
    );
    var result = new ArrayList<EntitySpan>();
    for (var candidate : candidates) {
      boolean overlaps = result
        .stream()
        .anyMatch(existing ->
          candidate.start() < existing.end() &&
          candidate.end() > existing.start()
        );
      if (!overlaps) {
        result.add(candidate);
      }
    }

    // Sort by position for consistent output
    result.sort(Comparator.comparingInt(EntitySpan::start));
    return result;
  }
}
