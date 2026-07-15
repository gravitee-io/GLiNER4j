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

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.function.Function;

/**
 * Bounded, thread-safe LRU cache for per-call override assemblers.
 *
 * <p>Building an assembler re-tokenizes the entire schema prompt, so override-path calls that
 * reuse a label set they've sent before skip straight to the cached, fully tokenized assembler.
 * Definition types are records, so the definition list itself is the cache key.
 *
 * <p>Two threads racing on a cache miss may both build the assembler; the loser's instance is
 * simply dropped. Cached values must be safe to share across threads (assemblers snapshot all
 * tokenized state at construction and are stateless per call).
 *
 * @param <K> the cache key (typically a {@code List} of definition records)
 * @param <A> the cached assembler type
 */
public final class AssemblerCache<K, A> {

  private final Map<K, A> cache;
  private final Function<K, A> loader;

  /**
   * @param maxSize max cached assemblers; ≤ 0 disables caching (every call builds fresh)
   * @param loader  builds the assembler for a key on cache miss
   */
  public AssemblerCache(int maxSize, Function<K, A> loader) {
    this.loader = loader;
    this.cache = maxSize <= 0
      ? null
      : Collections.synchronizedMap(
        new LinkedHashMap<>(16, 0.75f, true) {
          @Override
          protected boolean removeEldestEntry(Map.Entry<K, A> eldest) {
            return size() > maxSize;
          }
        }
      );
  }

  /**
   * Returns the cached assembler for {@code key}, building and caching it on miss.
   *
   * @param key the cache key
   * @return the assembler
   */
  public A get(K key) {
    if (cache == null) {
      return loader.apply(key);
    }
    var existing = cache.get(key);
    if (existing != null) {
      return existing;
    }
    var built = loader.apply(key);
    cache.put(key, built);
    return built;
  }
}
