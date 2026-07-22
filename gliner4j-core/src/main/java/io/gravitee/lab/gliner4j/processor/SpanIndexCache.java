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

import java.util.concurrent.ConcurrentHashMap;

/**
 * Memoizes flat span-index arrays: they depend only on {@code (textLen, maxWidth)} and were
 * previously rebuilt on every call. Cached arrays are shared and must be treated as read-only
 * (they are only ever copied into ONNX input buffers).
 */
public final class SpanIndexCache {

  private static final int MAX_ENTRIES = 4096;
  private static final ConcurrentHashMap<Long, long[]> CACHE =
    new ConcurrentHashMap<>();

  private SpanIndexCache() {}

  /**
   * Returns the flat span index array for a text: {@code textLen * maxWidth} (start, end)
   * pairs, with out-of-range spans left as (0, 0).
   *
   * @param textLen  number of words
   * @param maxWidth max span width
   * @return read-only flat array of length {@code textLen * maxWidth * 2}
   */
  public static long[] flatSpanIdx(int textLen, int maxWidth) {
    long key = ((long) textLen << 20) | maxWidth;
    var cached = CACHE.get(key);
    if (cached != null) {
      return cached;
    }
    var built = build(textLen, maxWidth);
    if (CACHE.size() < MAX_ENTRIES) {
      CACHE.putIfAbsent(key, built);
    }
    return built;
  }

  private static long[] build(int textLen, int maxWidth) {
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
}
