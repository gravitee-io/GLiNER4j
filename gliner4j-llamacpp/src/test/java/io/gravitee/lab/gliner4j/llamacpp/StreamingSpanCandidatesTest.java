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
package io.gravitee.lab.gliner4j.llamacpp;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

class StreamingSpanCandidatesTest {

  private static String[] spans(long[] keys) {
    return Arrays.stream(keys)
      .mapToObj(k -> (k >>> 32) + "-" + (k & 0xffffffffL))
      .toArray(String[]::new);
  }

  @Test
  void coldPassEnumeratesEverySpanUpToMaxWidth() {
    var keys = StreamingSpanSession.candidates(0, 3, 2, 12);
    assertThat(spans(keys)).containsExactly("0-0", "0-1", "1-1", "1-2", "2-2");
  }

  @Test
  void appendRescoresSpansEndingInNewWordsAndTheRightContextWindow() {
    // 5 past words, 2 new, width 3, right context 1: minimum end = min(5, 6-1=5) = 5 → spans
    // ending at word 5 or 6 only, starting no earlier than 5-2 = 3.
    var keys = StreamingSpanSession.candidates(5, 2, 3, 1);
    assertThat(spans(keys)).containsExactly(
      "3-5",
      "4-5",
      "4-6",
      "5-5",
      "5-6",
      "6-6"
    );
  }

  @Test
  void wideRightContextRevisitsOlderSpans() {
    // 10 past, 1 new, width 2, right context 12 → minimum end = min(10, max(0, 10-12)=0) = 0.
    var keys = StreamingSpanSession.candidates(10, 1, 2, 12);
    assertThat(keys).hasSize(11 + 10); // 11 singletons + 10 pairs
  }

  @Test
  void nothingToScoreWithoutNewWords() {
    assertThat(StreamingSpanSession.candidates(4, 0, 12, 12)).isEmpty();
  }
}
