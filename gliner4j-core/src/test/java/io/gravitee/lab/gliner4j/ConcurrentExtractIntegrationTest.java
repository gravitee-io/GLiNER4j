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
package io.gravitee.lab.gliner4j;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Verifies that a single facade instance can serve concurrent single-text extract() calls:
 * results under concurrency must be identical to sequential results.
 */
@EnabledIf("modelDirExists")
class ConcurrentExtractIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/ner_full.onnx"));
  }

  @Test
  void concurrentExtractWithMicroBatchingMatchesSequential() throws Exception {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );
    var texts = List.of(
      "John works at Google.",
      "Marie Curie worked at the University of Paris.",
      "Elon Musk founded SpaceX and leads Tesla."
    );
    var runtimeConfig = io.gravitee.lab.gliner4j.runtime.RuntimeConfig.builder()
      .microBatchingEnabled(true)
      .build();

    try (
      var gliner = GLiNER4jNER.load(MODEL_DIR, entities, "onnx", runtimeConfig)
    ) {
      var expected = new ArrayList<Map<String, List<EntitySpan>>>();
      try (var reference = GLiNER4jNER.load(MODEL_DIR, entities)) {
        for (var text : texts) {
          expected.add(reference.extract(text));
        }
      }

      var latch = new CountDownLatch(1);
      try (var executor = Executors.newFixedThreadPool(6)) {
        var futures = new ArrayList<Future<?>>();
        for (int t = 0; t < 6; t++) {
          final int offset = t;
          futures.add(
            executor.submit(() -> {
              latch.await();
              for (int i = 0; i < 3 * texts.size(); i++) {
                int idx = (i + offset) % texts.size();
                assertSameSpans(
                  gliner.extract(texts.get(idx)),
                  expected.get(idx)
                );
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

  /**
   * Asserts the same entity spans (type/text/offsets), tolerating tiny confidence drift. The
   * micro-batched path pads several texts to a common length, which introduces ~1e-6 numerical
   * differences in the encoder vs a single unpadded text — inherent to batched transformer
   * inference, not a correctness difference.
   */
  private static void assertSameSpans(
    Map<String, List<EntitySpan>> actual,
    Map<String, List<EntitySpan>> expected
  ) {
    assertThat(actual.keySet()).isEqualTo(expected.keySet());
    for (var type : expected.keySet()) {
      var a = actual.get(type);
      var e = expected.get(type);
      assertThat(a).hasSameSizeAs(e);
      for (int i = 0; i < e.size(); i++) {
        assertThat(a.get(i).type()).isEqualTo(e.get(i).type());
        assertThat(a.get(i).text()).isEqualTo(e.get(i).text());
        assertThat(a.get(i).start()).isEqualTo(e.get(i).start());
        assertThat(a.get(i).end()).isEqualTo(e.get(i).end());
        assertThat(a.get(i).confidence()).isCloseTo(
          e.get(i).confidence(),
          org.assertj.core.data.Offset.offset(1e-4f)
        );
      }
    }
  }

  @Test
  void concurrentExtractMatchesSequential() throws Exception {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );
    var texts = List.of(
      "John works at Google.",
      "Marie Curie worked at the University of Paris.",
      "Elon Musk founded SpaceX and leads Tesla.",
      "Tim Cook leads Apple in Cupertino, together with senior staff.",
      "Satya Nadella runs Microsoft from Redmond while visiting LinkedIn."
    );

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      var sequential = new ArrayList<Map<String, List<EntitySpan>>>();
      for (var text : texts) {
        sequential.add(gliner.extract(text));
      }

      int threads = 8;
      int iterations = 5;
      var latch = new CountDownLatch(1);
      try (var executor = Executors.newFixedThreadPool(threads)) {
        var futures = new ArrayList<Future<?>>();
        for (int t = 0; t < threads; t++) {
          final int offset = t;
          futures.add(
            executor.submit(() -> {
              latch.await();
              for (int i = 0; i < iterations * texts.size(); i++) {
                int idx = (i + offset) % texts.size();
                var result = gliner.extract(texts.get(idx));
                assertThat(result)
                  .as("text %d under concurrency", idx)
                  .isEqualTo(sequential.get(idx));
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
}
