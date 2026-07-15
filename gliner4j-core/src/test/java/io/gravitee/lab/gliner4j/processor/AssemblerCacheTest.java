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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;

class AssemblerCacheTest {

  @Test
  void hitReturnsSameInstance() {
    var builds = new AtomicInteger();
    var cache = new AssemblerCache<List<String>, Object>(4, k -> {
      builds.incrementAndGet();
      return new Object();
    });
    var a = cache.get(List.of("person", "org"));
    var b = cache.get(List.of("person", "org"));
    assertSame(a, b);
    assertEquals(1, builds.get());
  }

  @Test
  void evictsLeastRecentlyUsed() {
    var builds = new AtomicInteger();
    var cache = new AssemblerCache<String, Object>(2, k -> {
      builds.incrementAndGet();
      return new Object();
    });
    var a = cache.get("a");
    cache.get("b");
    cache.get("a"); // touch a → b is LRU
    cache.get("c"); // evicts b
    assertSame(a, cache.get("a"));
    assertEquals(3, builds.get());
    cache.get("b"); // rebuilt
    assertEquals(4, builds.get());
  }

  @Test
  void zeroSizeDisablesCaching() {
    var builds = new AtomicInteger();
    var cache = new AssemblerCache<String, Object>(0, k -> {
      builds.incrementAndGet();
      return new Object();
    });
    assertNotSame(cache.get("a"), cache.get("a"));
    assertEquals(2, builds.get());
  }

  @Test
  void concurrentGetsAreSafe() throws Exception {
    var cache = new AssemblerCache<Integer, Object>(8, k -> new Object());
    int threads = 16;
    var latch = new CountDownLatch(1);
    try (var executor = Executors.newFixedThreadPool(threads)) {
      var futures = new java.util.ArrayList<java.util.concurrent.Future<?>>();
      for (int t = 0; t < threads; t++) {
        futures.add(
          executor.submit(() -> {
            latch.await();
            for (int i = 0; i < 1000; i++) {
              assertTrue(cache.get(i % 12) != null);
            }
            return null;
          })
        );
      }
      latch.countDown();
      for (var f : futures) {
        f.get();
      }
    }
  }
}
