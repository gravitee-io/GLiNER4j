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

import io.gravitee.lab.gliner4j.runtime.FloatTensorView;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.util.ArrayList;
import java.util.List;
import lombok.extern.slf4j.Slf4j;

/**
 * Decodes raw span scores into entity spans.
 * Applies thresholding, character mapping, and greedy non-overlapping removal.
 */
@Slf4j
public class SpanDecoder extends AbstractDecoder {

  // Fires the span_scores buffer-overrun diagnostic only once to avoid log spam under load.
  private static final java.util.concurrent.atomic.AtomicBoolean LOGGED_OOB =
    new java.util.concurrent.atomic.AtomicBoolean(false);

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
    return getEntitySpans(candidates);
  }

  /**
   * Decodes one batch slot's span scores from a flat batched scoring-head output — no
   * nested-array materialization, scores are read straight from the buffer via strides.
   *
   * @param spanScores flat batched scores, shape [batch][count][numFields][textLen][maxWidth]
   * @param batchRow this text's row within the batch
   * @param fieldNames ordered entity type names
   * @param wordStartChars character start offsets per word
   * @param wordEndChars character end offsets per word
   * @param originalText the original input text
   * @param textLen number of text words (may be shorter than the padded tensor textLen)
   * @param threshold minimum confidence for inclusion
   * @return list of detected entity spans, after greedy non-overlapping removal
   */
  public List<EntitySpan> decode(
    FloatTensorView spanScores,
    int batchRow,
    List<String> fieldNames,
    int[] wordStartChars,
    int[] wordEndChars,
    String originalText,
    int textLen,
    float threshold
  ) {
    var candidates = new ArrayList<EntitySpan>();

    int numFields = spanScores.dim(2);
    int paddedTextLen = spanScores.dim(3);
    int maxWidth = spanScores.dim(4);

    int effTextLen = Math.min(textLen, paddedTextLen);

    // Diagnose the observed out-of-bounds read: the decode indices are derived from the tensor's
    // own dims, so they can only overrun if the backing buffer is smaller than the shape implies
    // (or batchRow is out of range). Compute the highest index we would touch and compare it to
    // the real capacity; if it overruns, log the full shape/stride/capacity once and bound the
    // reads so we degrade to a partial decode instead of failing the whole batch.
    long rowBase = batchRow * spanScores.stride(0);
    long cap = spanScores.capacity();
    long maxIndex = rowBase + (long) numFields * paddedTextLen * maxWidth - 1;
    if (maxIndex >= cap && LOGGED_OOB.compareAndSet(false, true)) {
      log.warn(
        "SpanDecoder span_scores buffer overrun: shape={}, capacity={}, batchRow={}, stride0={}, numFields={}, paddedTextLen={}, maxWidth={}, textLen={}, wouldReadUpTo={}",
        java.util.Arrays.toString(spanScores.shape()),
        cap,
        batchRow,
        spanScores.stride(0),
        numFields,
        paddedTextLen,
        maxWidth,
        textLen,
        maxIndex
      );
    }

    // Bulk-copy each width-row into this scratch array once, then read plain float[] elements —
    // one bounds check per (field, start) row instead of one per element. The element-wise
    // FloatBuffer.get() path (Preconditions.checkIndex) was ~42% of on-CPU Java time in profiling.
    float[] rowScores = new float[maxWidth];

    // Use the first count instance for entities (mirrors spanScores[row][0] on the nested path).
    for (int f = 0; f < numFields; f++) {
      var fieldName = fieldNames.get(f);
      long fieldBase = rowBase + (long) f * paddedTextLen * maxWidth;
      for (int start = 0; start < effTextLen; start++) {
        long startBase = fieldBase + (long) start * maxWidth;
        // Bound the copy by the real buffer capacity (mirrors the per-element guard above).
        int rowLen = (int) Math.min(maxWidth, cap - startBase);
        if (rowLen <= 0) continue;
        spanScores.copyRowAsFloats(startBase, rowScores, 0, rowLen);
        for (int w = 0; w < rowLen; w++) {
          float score = rowScores[w];
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

    return getEntitySpans(candidates);
  }
}
