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
package io.gravitee.lab.gliner4j.runtime;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

@Timeout(30)
class MicroBatcherTest {

  @Test
  void singleSubmitPassesThrough() {
    try (
      var batcher = new MicroBatcher<String>(8, 1000, (texts, threshold) ->
        texts
          .stream()
          .map(t -> t + "@" + threshold)
          .toList()
      )
    ) {
      assertEquals("a@0.5", batcher.extract("a", 0.5f));
    }
  }

  @Test
  void concurrentSubmitsCoalesceIntoBatches() throws Exception {
    var batchSizes = new CopyOnWriteArrayList<Integer>();
    int callers = 12;
    var latch = new CountDownLatch(1);
    try (
      var batcher = new MicroBatcher<String>(16, 50_000, (texts, threshold) -> {
        batchSizes.add(texts.size());
        return texts
          .stream()
          .map(t -> "r:" + t)
          .toList();
      });
      var executor = Executors.newFixedThreadPool(callers)
    ) {
      var futures = new ArrayList<Future<String>>();
      for (int i = 0; i < callers; i++) {
        final int id = i;
        futures.add(
          executor.submit(() -> {
            latch.await();
            return batcher.extract("t" + id, 0.5f);
          })
        );
      }
      latch.countDown();
      for (int i = 0; i < callers; i++) {
        assertEquals("r:t" + i, futures.get(i).get());
      }
    }
    int total = batchSizes.stream().mapToInt(Integer::intValue).sum();
    assertEquals(callers, total);
    assertTrue(
      batchSizes.size() < callers,
      "expected coalescing, got batches " + batchSizes
    );
  }

  @Test
  void differentThresholdsAreBatchedSeparately() throws Exception {
    var calls = new CopyOnWriteArrayList<Float>();
    var latch = new CountDownLatch(1);
    try (
      var batcher = new MicroBatcher<String>(16, 50_000, (texts, threshold) -> {
        calls.add(threshold);
        return texts
          .stream()
          .map(t -> t + "@" + threshold)
          .toList();
      });
      var executor = Executors.newFixedThreadPool(4)
    ) {
      var futures = new ArrayList<Future<String>>();
      for (int i = 0; i < 4; i++) {
        final float threshold = i % 2 == 0 ? 0.3f : 0.7f;
        final int id = i;
        futures.add(
          executor.submit(() -> {
            latch.await();
            return batcher.extract("t" + id, threshold);
          })
        );
      }
      latch.countDown();
      assertEquals("t0@0.3", futures.get(0).get());
      assertEquals("t1@0.7", futures.get(1).get());
      assertEquals("t2@0.3", futures.get(2).get());
      assertEquals("t3@0.7", futures.get(3).get());
    }
  }

  @Test
  void batchFailurePropagatesToAllCallersInBatch() {
    try (
      var batcher = new MicroBatcher<String>(8, 1000, (texts, threshold) -> {
        throw new IllegalStateException("boom");
      })
    ) {
      var e = assertThrows(IllegalStateException.class, () ->
        batcher.extract("a", 0.5f)
      );
      assertEquals("boom", e.getMessage());
    }
  }

  @Test
  void extractAfterCloseThrows() {
    var batcher = new MicroBatcher<String>(8, 1000, (texts, threshold) ->
      List.of("x")
    );
    batcher.close();
    assertThrows(IllegalStateException.class, () -> batcher.extract("a", 0.5f));
  }

  @Test
  void nullTextsFlowThrough() {
    var count = new AtomicInteger();
    try (
      var batcher = new MicroBatcher<String>(8, 1000, (texts, threshold) -> {
        count.incrementAndGet();
        return texts
          .stream()
          .map(t -> t == null ? "empty" : t)
          .toList();
      })
    ) {
      assertEquals("empty", batcher.extract(null, 0.5f));
      assertEquals(1, count.get());
    }
  }
}
