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

import java.util.ArrayList;
import java.util.List;

/**
 * Whitespace word splitter shared by the GLiNER NER strategies (uni markerV0, uni token_level,
 * bi-encoder). Each word keeps its character offsets in the original text so spans can be mapped
 * back after decoding.
 *
 * <p>APPROXIMATION: valid only for space-separated languages. CJK (Chinese/Japanese/Thai) is not
 * supported and punctuation boundaries may drift — see the {@code TODO(gliner-x)} notes in the
 * strategies. Matches the {@code words_splitter_type=whitespace} export path.
 */
public final class WhitespaceWordSplitter {

  /** A text word with its inclusive-start / exclusive-end character offsets in the source text. */
  public record Word(String text, int start, int end) {}

  private WhitespaceWordSplitter() {}

  public static List<Word> split(String text) {
    var out = new ArrayList<Word>();
    int i = 0;
    int n = text.length();
    while (i < n) {
      while (i < n && Character.isWhitespace(text.charAt(i))) i++;
      if (i >= n) break;
      int start = i;
      while (i < n && !Character.isWhitespace(text.charAt(i))) i++;
      out.add(new Word(text.substring(start, i), start, i));
    }
    return out;
  }
}
