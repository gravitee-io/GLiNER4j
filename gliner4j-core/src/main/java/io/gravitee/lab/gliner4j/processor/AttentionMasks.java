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
 * Shares all-ones attention-mask arrays by length. Masks are all-ones for every assembled
 * input and are only ever read (copied into ONNX input buffers), so one shared read-only
 * array per length replaces a fresh allocation + fill per call.
 */
final class AttentionMasks {

  private static final int MAX_ENTRIES = 4096;
  private static final ConcurrentHashMap<Integer, long[]> CACHE =
    new ConcurrentHashMap<>();

  private AttentionMasks() {}

  /** Returns a shared, read-only all-ones array of {@code len} elements. */
  static long[] ones(int len) {
    var cached = CACHE.get(len);
    if (cached != null) {
      return cached;
    }
    var built = new long[len];
    java.util.Arrays.fill(built, 1L);
    if (CACHE.size() < MAX_ENTRIES) {
      CACHE.putIfAbsent(len, built);
    }
    return built;
  }
}
