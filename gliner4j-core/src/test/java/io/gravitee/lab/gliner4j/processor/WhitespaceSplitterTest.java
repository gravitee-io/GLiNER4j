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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * {@link TextPreprocessing#whitespaceSplitter} against the upstream {@code WhitespaceTokenSplitter(text,
 * lower=True)}. Expected tokens and offsets were produced by the upstream Python splitter
 * ({@code gliner2/processing/word_splitter.py}).
 */
class WhitespaceSplitterTest {

  private static final TextPreprocessing SPLIT_ONLY = new TextPreprocessing(
    TextEncoder.WHITESPACE_SPLITTER_PATTERN,
    true,
    false
  );

  record Tok(String word, int start, int end) {}

  private static Tok t(String word, int start, int end) {
    return new Tok(word, start, end);
  }

  static Stream<Arguments> upstreamCases() {
    return Stream.of(
      Arguments.of(
        "Élodie jane.doe@example.com visited https://example.com in Montréal @café",
        List.of(
          t("élodie", 0, 6),
          t("jane.doe@example.com", 7, 27),
          t("visited", 28, 35),
          t("https://example.com", 36, 55),
          t("in", 56, 58),
          t("montréal", 59, 67),
          t("@caf", 68, 72),
          t("é", 72, 73)
        )
      ),
      Arguments.of(
        "Contact John.Smith@Example.COM or see www.Example.org/path?q=1.",
        List.of(
          t("contact", 0, 7),
          t("john.smith@example.com", 8, 30),
          t("or", 31, 33),
          t("see", 34, 37),
          t("www.example.org/path?q=1.", 38, 63)
        )
      ),
      Arguments.of(
        "state-of-the-art snake_case über-cool",
        List.of(
          t("state-of-the-art", 0, 16),
          t("snake_case", 17, 27),
          t("über-cool", 28, 37)
        )
      ),
      Arguments.of(
        "Tokyo, 東京 — naïve résumé!",
        List.of(
          t("tokyo", 0, 5),
          t(",", 5, 6),
          t("東京", 7, 9),
          t("—", 10, 11),
          t("naïve", 12, 17),
          t("résumé", 18, 24),
          t("!", 24, 25)
        )
      ),
      Arguments.of(
        "@user_1 said: hi",
        List.of(
          t("@user_1", 0, 7),
          t("said", 8, 12),
          t(":", 12, 13),
          t("hi", 14, 16)
        )
      ),
      Arguments.of(
        "ÉCOLE Straße",
        List.of(t("école", 0, 5), t("straße", 6, 12))
      )
    );
  }

  @ParameterizedTest
  @MethodSource("upstreamCases")
  void matchesUpstreamTokensAndOffsets(String text, List<Tok> expected) {
    assertThat(tokens(SPLIT_ONLY.encode(text))).containsExactlyElementsOf(
      expected
    );
  }

  @Test
  void offsetsIndexTheOriginalText() {
    var text = "Élodie visited Montréal";
    var enc = SPLIT_ONLY.encode(text);
    for (int i = 0; i < enc.getTextLen(); i++) {
      var original = text.substring(
        enc.getWordStartChars()[i],
        enc.getWordEndChars()[i]
      );
      assertThat(original.toLowerCase(java.util.Locale.ROOT)).isEqualTo(
        enc.getWords().get(i)
      );
    }
  }

  @Test
  void appendsEmptyRangePeriodWhenTextLacksTerminalPunctuation() {
    var preprocessing = new TextPreprocessing(
      TextEncoder.WHITESPACE_SPLITTER_PATTERN,
      true,
      true
    );
    assertThat(
      tokens(preprocessing.encode("Visit https://example.com"))
    ).containsExactly(
      t("visit", 0, 5),
      t("https://example.com", 6, 25),
      t(".", 25, 25)
    );
    assertThat(tokens(preprocessing.encode("Done!"))).containsExactly(
      t("done", 0, 4),
      t("!", 4, 5)
    );
  }

  private static List<Tok> tokens(TextEncoder enc) {
    var out = new ArrayList<Tok>();
    for (int i = 0; i < enc.getTextLen(); i++) {
      out.add(
        t(
          enc.getWords().get(i),
          enc.getWordStartChars()[i],
          enc.getWordEndChars()[i]
        )
      );
    }
    return out;
  }
}
