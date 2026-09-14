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

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import java.util.regex.Pattern;

/**
 * How a model family turns raw text into words: the split pattern plus the lower-case and
 * trailing-period conventions. One instance is shared by every task of a family so NER, relation
 * and classification see identical model inputs.
 *
 * @param pattern the word-split regex
 * @param lowercaseWords lower-case each word (offsets still index the original text)
 * @param appendPeriod append a final {@code "."} word when the text does not end in {@code . ! ?}
 */
public record TextPreprocessing(
  Pattern pattern,
  boolean lowercaseWords,
  boolean appendPeriod
) {
  /** The original GLiNER splitter, text as is. */
  public static final TextPreprocessing DEFAULT = new TextPreprocessing(
    TextEncoder.TOKEN_PATTERN,
    false,
    false
  );

  /**
   * The GLiNER2 processor ({@code WhitespaceTokenSplitter} with {@code lower=True}, period appended
   * at collate), with the bundle's {@code lowercase_words} / {@code append_period} switches.
   */
  public static TextPreprocessing whitespaceSplitter(GLiNER4jConfig config) {
    return new TextPreprocessing(
      TextEncoder.WHITESPACE_SPLITTER_PATTERN,
      config.archBoolean("lowercase_words", true),
      config.archBoolean("append_period", true)
    );
  }

  public TextEncoder encode(String text) {
    return new TextEncoder(text, pattern, lowercaseWords, appendPeriod);
  }
}
