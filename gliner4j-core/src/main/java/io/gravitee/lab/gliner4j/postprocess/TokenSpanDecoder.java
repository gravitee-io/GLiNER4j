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
import io.gravitee.lab.gliner4j.utils.LinAlg;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Decodes original-GLiNER <em>token-level</em> (BIO-style) logits into entity spans.
 *
 * <p>The model emits {@code logits[word][class][3]} whose three channels are start / end / inside
 * scores. Mirroring the gliner {@code TokenDecoder}: sigmoid each channel, then for every class pair
 * each above-threshold start with each above-threshold end at or after it where every inside score
 * in between also passes; the span score is the minimum of (start, end, all insides). Greedy
 * non-overlap removal then yields flat (non-nested) spans.
 */
public class TokenSpanDecoder extends AbstractDecoder {

  private static final int START = 0;
  private static final int END = 1;
  private static final int INSIDE = 2;

  /**
   * @param logits         token-level logits {@code [words][numClasses][3]}
   * @param fieldNames     ordered class/entity names (index = class)
   * @param wordStartChars char start offset per word
   * @param wordEndChars   char end offset per word
   * @param originalText   the original input text
   * @param textLen        number of text words
   * @param threshold      minimum confidence for start, end and every inside score
   * @return non-overlapping entity spans, sorted by position
   */
  public List<EntitySpan> decode(
    float[][][] logits,
    List<String> fieldNames,
    int[] wordStartChars,
    int[] wordEndChars,
    String originalText,
    int textLen,
    float threshold
  ) {
    int words = Math.min(textLen, logits.length);
    int numClasses = fieldNames.size();
    if (words == 0 || numClasses == 0) {
      return List.of();
    }

    // Sigmoid the three channels once.
    var startSig = new float[words][numClasses];
    var endSig = new float[words][numClasses];
    var insideSig = new float[words][numClasses];
    for (int w = 0; w < words; w++) {
      for (int c = 0; c < numClasses && c < logits[w].length; c++) {
        startSig[w][c] = LinAlg.sigmoid(logits[w][c][START]);
        endSig[w][c] = LinAlg.sigmoid(logits[w][c][END]);
        insideSig[w][c] = LinAlg.sigmoid(logits[w][c][INSIDE]);
      }
    }

    var candidates = new ArrayList<EntitySpan>();
    for (int st = 0; st < words; st++) {
      for (int c = 0; c < numClasses; c++) {
        if (startSig[st][c] < threshold) {
          continue;
        }
        for (int ed = st; ed < words; ed++) {
          if (endSig[ed][c] < threshold) {
            continue;
          }
          // Every inside score from st..ed (inclusive) must pass; span score is the minimum.
          float minScore = Math.min(startSig[st][c], endSig[ed][c]);
          boolean valid = true;
          for (int pos = st; pos <= ed; pos++) {
            float ins = insideSig[pos][c];
            if (ins < threshold) {
              valid = false;
              break;
            }
            if (ins < minScore) {
              minScore = ins;
            }
          }
          if (valid) {
            int charStart = wordStartChars[st];
            int charEnd = wordEndChars[ed];
            candidates.add(
              new EntitySpan(
                fieldNames.get(c),
                originalText.substring(charStart, charEnd),
                minScore,
                charStart,
                charEnd
              )
            );
          }
        }
      }
    }

    // Greedy non-overlapping removal: sort by confidence desc, skip overlaps (flat NER).
    return getEntitySpans(candidates);
  }
}
