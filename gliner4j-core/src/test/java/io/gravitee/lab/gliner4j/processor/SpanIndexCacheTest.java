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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

import org.junit.jupiter.api.Test;

class SpanIndexCacheTest {

  /** The original per-call span index builder, kept as the reference implementation. */
  private static long[] reference(int textLen, int maxWidth) {
    var flat = new long[textLen * maxWidth * 2];
    for (int i = 0; i < textLen; i++) {
      for (int w = 0; w < maxWidth; w++) {
        int endPos = i + w;
        if (endPos < textLen) {
          int flatIdx = (i * maxWidth + w) * 2;
          flat[flatIdx] = i;
          flat[flatIdx + 1] = endPos;
        }
      }
    }
    return flat;
  }

  @Test
  void matchesReferenceImplementation() {
    for (int textLen : new int[] { 1, 2, 5, 17, 128 }) {
      for (int maxWidth : new int[] { 1, 8, 12 }) {
        assertArrayEquals(
          reference(textLen, maxWidth),
          SpanIndexCache.flatSpanIdx(textLen, maxWidth),
          "textLen=" + textLen + " maxWidth=" + maxWidth
        );
      }
    }
  }

  @Test
  void cachesByTextLenAndWidth() {
    assertSame(
      SpanIndexCache.flatSpanIdx(33, 12),
      SpanIndexCache.flatSpanIdx(33, 12)
    );
  }

  @Test
  void expectedLength() {
    assertEquals(7 * 12 * 2, SpanIndexCache.flatSpanIdx(7, 12).length);
  }
}
