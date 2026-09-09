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
 * <p>Mirrors gliner's {@code WhitespaceTokenSplitter} ({@code words_splitter_type=whitespace}):
 * the regex {@code \w+(?:[-_]\w+)*|\S}, i.e. words (with inner hyphens/underscores) and every other
 * non-space character as its own token — so {@code "London,"} is the two words {@code London} and
 * {@code ,}, exactly as the model saw them in training.
 *
 * <p>APPROXIMATION: valid only for space-separated languages. CJK (Chinese/Japanese/Thai) is not
 * supported — see the {@code TODO(gliner-x)} notes in the strategies.
 */
public final class WhitespaceWordSplitter {

  /** A text word with its inclusive-start / exclusive-end character offsets in the source text. */
  public record Word(String text, int start, int end) {}

  private static final java.util.regex.Pattern TOKEN =
    java.util.regex.Pattern.compile(
      "\\w+(?:[-_]\\w+)*|\\S",
      java.util.regex.Pattern.UNICODE_CHARACTER_CLASS
    );

  private WhitespaceWordSplitter() {}

  public static List<Word> split(String text) {
    var out = new ArrayList<Word>();
    var m = TOKEN.matcher(text);
    while (m.find()) {
      out.add(new Word(m.group(), m.start(), m.end()));
    }
    return out;
  }
}
