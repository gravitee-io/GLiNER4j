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
package io.gravitee.lab.gliner4j.tokenizer;

import java.util.ArrayList;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Splits text into word-level tokens using whitespace and punctuation boundaries.
 * Port of the Python GLiNER tokenizer regex.
 */
public class WhitespaceTokenSplitter {

  private static final Pattern TOKEN_PATTERN = Pattern.compile(
    "\\w+(?:[-_]\\w+)*|\\S"
  );

  /**
   * A single whitespace-level token with character offsets.
   *
   * @param text the token text
   * @param start character start offset (inclusive)
   * @param end character end offset (exclusive)
   */
  public record Token(String text, int start, int end) {}

  /**
   * Tokenizes the input text into word-level tokens with character positions.
   *
   * @param text the input text to tokenize
   * @return list of tokens with their character offsets
   */
  public List<Token> tokenize(String text) {
    var tokens = new ArrayList<Token>();
    Matcher matcher = TOKEN_PATTERN.matcher(text);
    while (matcher.find()) {
      tokens.add(new Token(matcher.group(), matcher.start(), matcher.end()));
    }
    return tokens;
  }
}
