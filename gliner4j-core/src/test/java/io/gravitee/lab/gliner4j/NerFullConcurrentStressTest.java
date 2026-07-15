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

import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Reproduces the production ner_full crash: two lanes calling extractBatch concurrently on a
 * many-entity schema with long, length-varied texts (multiple buckets). Mirrors the ai-server
 * two-lane MicroBatcher hitting GLiNER4jNER.extractBatch.
 */
@EnabledIf("modelDirExists")
class NerFullConcurrentStressTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return (
      Files.exists(MODEL_DIR.resolve("onnx/ner_full.onnx")) &&
      Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"))
    );
  }

  @Test
  void concurrentExtractBatchDoesNotCrash() throws Exception {
    var entities = new ArrayList<EntityDefinition>();
    for (int i = 0; i < 42; i++) {
      entities.add(new EntityDefinition("entity_type_" + i));
    }

    var shortText = "John works at Google in London.";
    var longSb = new StringBuilder();
    for (int i = 0; i < 40; i++) {
      longSb.append(
        "Alice met Bob at the Paris office of Initech on Tuesday. "
      );
    }
    var longText = longSb.toString();
    // length-varied so bucketing produces multiple sub-batches per call
    var batch = List.of(shortText, longText, shortText, longText);

    var err = new AtomicReference<Throwable>();
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      var latch = new CountDownLatch(1);
      try (var executor = Executors.newFixedThreadPool(3)) {
        var tasks = new ArrayList<java.util.concurrent.Future<?>>();
        for (int t = 0; t < 3; t++) {
          tasks.add(
            executor.submit(() -> {
              try {
                latch.await();
                for (int i = 0; i < 12; i++) {
                  gliner.extractBatch(batch, 0.5f);
                }
              } catch (Throwable e) {
                err.compareAndSet(null, e);
              }
            })
          );
        }
        latch.countDown();
        for (var task : tasks) {
          task.get();
        }
      }
    }
    if (err.get() != null) {
      throw new AssertionError("concurrent extractBatch failed", err.get());
    }
  }
}
