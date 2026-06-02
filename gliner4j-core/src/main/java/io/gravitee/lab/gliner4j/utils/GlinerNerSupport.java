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
package io.gravitee.lab.gliner4j.utils;

import io.gravitee.lab.gliner4j.postprocess.SpanDecoder;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.utils.WhitespaceWordSplitter.Word;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Shared post-tokenization plumbing for the original-GLiNER NER strategies. The markerV0 uni- and
 * bi-encoder strategies differ only in their {@code runtime.run(...)} call; everything around it —
 * the {@code span_idx}/{@code span_mask} grid, the sigmoid+transpose into the {@link SpanDecoder}
 * layout, char offsets, and grouping the decoded spans by type — is identical and lives here.
 */
public final class GlinerNerSupport {

  private GlinerNerSupport() {}

  /** Enumerated spans up to {@code maxWidth}: {@code span_idx[start,end]} + validity mask. */
  public record SpanGrid(long[][] spanIdx, boolean[] spanMask) {}

  /** Per-word inclusive-start / exclusive-end char offsets into the source text. */
  public record CharOffsets(int[] starts, int[] ends) {}

  /**
   * Build the markerV0 {@code span_idx}/{@code span_mask} grid: for each start word, every width up
   * to {@code maxWidth}; a span is valid when its (inclusive) end word is within the text.
   */
  public static SpanGrid buildSpanGrid(int textLen, int maxWidth) {
    int numSpans = textLen * maxWidth;
    var spanIdx = new long[numSpans][2];
    var spanMask = new boolean[numSpans];
    for (int s = 0; s < textLen; s++) {
      for (int wd = 0; wd < maxWidth; wd++) {
        int idx = s * maxWidth + wd;
        int end = s + wd;
        spanIdx[idx][0] = s;
        spanIdx[idx][1] = end;
        spanMask[idx] = end < textLen;
      }
    }
    return new SpanGrid(spanIdx, spanMask);
  }

  /** Extract the per-word char offsets needed to map decoded spans back to the source text. */
  public static CharOffsets charOffsets(List<Word> words) {
    int textLen = words.size();
    int[] starts = new int[textLen];
    int[] ends = new int[textLen];
    for (int i = 0; i < textLen; i++) {
      starts[i] = words.get(i).start();
      ends[i] = words.get(i).end();
    }
    return new CharOffsets(starts, ends);
  }

  /** Group decoded spans by entity type, preserving first-seen order. */
  public static Map<String, List<EntitySpan>> groupByType(
    List<EntitySpan> spans
  ) {
    return spans
      .stream()
      .collect(
        Collectors.groupingBy(
          EntitySpan::type,
          LinkedHashMap::new,
          Collectors.toList()
        )
      );
  }

  /**
   * Decode markerV0 logits {@code [words][width][class]} into a flat span list: sigmoid + transpose
   * into the {@link SpanDecoder} layout {@code [1][class][word][width]}, then decode. The uni- and
   * bi-encoder strategies share this verbatim; the caller groups (and counts) the returned spans.
   */
  public static List<EntitySpan> decodeMarkerV0(
    SpanDecoder decoder,
    float[][][] logits,
    List<Word> words,
    List<EntityDefinition> labels,
    String text,
    int maxWidth,
    float threshold
  ) {
    int textLen = words.size();
    int numClasses = labels.size();
    int kDim = logits.length > 0 && logits[0].length > 0
      ? logits[0].length
      : maxWidth;
    var scores = new float[1][numClasses][textLen][kDim];
    for (int s = 0; s < textLen && s < logits.length; s++) {
      for (int wd = 0; wd < kDim && wd < logits[s].length; wd++) {
        for (int c = 0; c < numClasses && c < logits[s][wd].length; c++) {
          scores[0][c][s][wd] = LinAlg.sigmoid(logits[s][wd][c]);
        }
      }
    }

    var labelNames = labels.stream().map(EntityDefinition::name).toList();
    var offsets = charOffsets(words);
    return decoder.decode(
      scores,
      labelNames,
      offsets.starts(),
      offsets.ends(),
      text,
      textLen,
      threshold
    );
  }
}
