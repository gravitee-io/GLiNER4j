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

import io.gravitee.lab.gliner4j.tokenizer.WhitespaceTokenSplitter;
import java.util.List;
import lombok.Getter;

/**
 * Per-request text encoder: splits text into words with character offsets.
 * Subword tokenization is handled by InputAssembler.
 */
@Getter
public class TextEncoder {

  private final List<String> words;
  private final int[] wordStartChars;
  private final int[] wordEndChars;
  private final int textLen;
  private final String originalText;

  /**
   * Splits a text string into word-level tokens with character positions.
   *
   * @param text the input text to encode
   * @param splitter the whitespace-level tokenizer
   */
  public TextEncoder(String text, WhitespaceTokenSplitter splitter) {
    this.originalText = text;
    var tokens = splitter.tokenize(text);
    this.textLen = tokens.size();
    this.words = tokens
      .stream()
      .map(WhitespaceTokenSplitter.Token::text)
      .toList();
    this.wordStartChars = tokens
      .stream()
      .mapToInt(WhitespaceTokenSplitter.Token::start)
      .toArray();
    this.wordEndChars = tokens
      .stream()
      .mapToInt(WhitespaceTokenSplitter.Token::end)
      .toArray();
  }
}
