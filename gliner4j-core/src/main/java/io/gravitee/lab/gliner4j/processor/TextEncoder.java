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

import java.util.Arrays;
import java.util.List;
import java.util.Locale;
import java.util.regex.Pattern;
import lombok.Getter;

/**
 * Per-request text encoder: splits text into words with character offsets.
 * Subword tokenization is handled by InputAssembler.
 */
@Getter
public class TextEncoder {

  /**
   * The compiled regex used to split text into word-level tokens — port of the upstream
   * GLiNER tokenizer regex.
   */
  public static final Pattern TOKEN_PATTERN = Pattern.compile(
    "\\w+(?:[-_]\\w+)*|\\S"
  );

  private final List<String> words;
  private final int[] wordStartChars;
  private final int[] wordEndChars;
  private final int textLen;
  private final String originalText;

  /**
   * Splits a text string into word-level tokens with character positions.
   *
   * <p>Runs a single matcher loop that writes directly into primitive arrays — no intermediate
   * {@code List<Token>} and no per-field stream pipeline.
   *
   * @param text the input text to encode
   */
  public TextEncoder(String text) {
    this(text, false, false);
  }

  /**
   * Splits {@code text} into words with optional GLiNER2-processor parity options.
   *
   * @param text the input text
   * @param lowercaseWords lower-case each word before subword tokenization (offsets still index
   *                       the original text) — what fastino's {@code WhitespaceTokenSplitter} does
   * @param appendPeriod append a final {@code "."} word when the text does not end in
   *                     {@code . ! ?} — what fastino's collate step does; the extra word maps to
   *                     the empty char range {@code [len, len)} so spans ending on it stay in bounds
   */
  public TextEncoder(
    String text,
    boolean lowercaseWords,
    boolean appendPeriod
  ) {
    this.originalText = text;

    // Capacity estimate: ~1 token per 5 chars (typical English). Grows by doubling on overflow.
    int initCap = Math.max(8, text.length() / 5);
    var wordsArr = new String[initCap];
    var starts = new int[initCap];
    var ends = new int[initCap];
    int n = 0;

    var matcher = TOKEN_PATTERN.matcher(text);
    while (matcher.find()) {
      if (n == wordsArr.length) {
        int newCap = wordsArr.length << 1;
        wordsArr = Arrays.copyOf(wordsArr, newCap);
        starts = Arrays.copyOf(starts, newCap);
        ends = Arrays.copyOf(ends, newCap);
      }
      var word = matcher.group();
      wordsArr[n] = lowercaseWords ? word.toLowerCase(Locale.ROOT) : word;
      starts[n] = matcher.start();
      ends[n] = matcher.end();
      n++;
    }
    if (
      appendPeriod &&
      n > 0 &&
      !text.endsWith(".") &&
      !text.endsWith("!") &&
      !text.endsWith("?")
    ) {
      if (n == wordsArr.length) {
        wordsArr = Arrays.copyOf(wordsArr, n + 1);
        starts = Arrays.copyOf(starts, n + 1);
        ends = Arrays.copyOf(ends, n + 1);
      }
      wordsArr[n] = ".";
      starts[n] = text.length();
      ends[n] = text.length();
      n++;
    }

    this.textLen = n;
    // Arrays.asList returns a RandomAccess view — O(1) get(w) for InputAssembler.assemble.
    this.words = Arrays.asList(
      n == wordsArr.length ? wordsArr : Arrays.copyOf(wordsArr, n)
    );
    this.wordStartChars = n == starts.length
      ? starts
      : Arrays.copyOf(starts, n);
    this.wordEndChars = n == ends.length ? ends : Arrays.copyOf(ends, n);
  }
}
