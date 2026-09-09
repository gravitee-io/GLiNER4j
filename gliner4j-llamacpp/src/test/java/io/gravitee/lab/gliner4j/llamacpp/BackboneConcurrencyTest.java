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
import static org.assertj.core.api.Assertions.within;

import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Concurrent callers must get the same answers as sequential ones while the backbone packs their
 * decodes into shared {@code llama_decode} calls. Skipped unless both llama.cpp bundles exist.
 */
@EnabledIf("bundlesExist")
class BackboneConcurrencyTest {

  private static final Path ROUTER_DIR = Path.of("../models/scx-router-onnx");
  private static final Path STREAM_DIR = Path.of(
    "../models/gliner-stream-pii-onnx"
  );

  static boolean bundlesExist() {
    return (
      Files.exists(ROUTER_DIR.resolve("gguf/backbone-q8_0.gguf")) &&
      Files.exists(STREAM_DIR.resolve("gguf/backbone-q8_0.gguf"))
    );
  }

  private static final List<String> TEXTS = List.of(
    "Write a Python function that merges two sorted linked lists.",
    "Compare these two vendor contracts and flag the riskier clauses.",
    "What's a good one-line commit message for this typo fix?",
    "Derive the closed-form solution for this second-order linear recurrence and prove convergence.",
    "Translate the following product description into French and German.",
    "The new mid-range EV just posted a 480-mile range on a single charge."
  );

  private static <T> List<T> runAll(List<Callable<T>> jobs) throws Exception {
    try (var pool = Executors.newFixedThreadPool(jobs.size())) {
      var futures = new ArrayList<Future<T>>();
      for (var job : jobs) futures.add(pool.submit(job));
      var out = new ArrayList<T>();
      for (var f : futures) out.add(f.get());
      return out;
    }
  }

  @Test
  void concurrentRouterCallsMatchSequential() throws Exception {
    try (var router = DecoderKvRouter.load(ROUTER_DIR)) {
      var labels = RouterDemo.MODELS;
      var sequential = TEXTS.stream()
        .map(t -> router.classify(t, labels, 0.0f))
        .toList();
      var jobs = new ArrayList<Callable<List<ClassificationResult>>>();
      for (int round = 0; round < 3; round++) {
        for (var text : TEXTS)
          jobs.add(() -> router.classify(text, labels, 0.0f));
      }
      var concurrent = runAll(jobs);
      for (int i = 0; i < jobs.size(); i++) {
        var expected = sequential.get(i % TEXTS.size());
        var got = concurrent.get(i);
        assertThat(got)
          .extracting(ClassificationResult::label)
          .containsExactlyElementsOf(
            expected.stream().map(ClassificationResult::label).toList()
          );
        for (int k = 0; k < expected.size(); k++) {
          // Packing changes the float reduction order slightly; scores must stay within noise.
          assertThat(got.get(k).confidence())
            .as("text %d label %s", i, expected.get(k).label())
            .isCloseTo(expected.get(k).confidence(), within(5e-3f));
        }
      }
    }
  }

  @Test
  void concurrentStreamingSessionsMatchSequential() throws Exception {
    var labels = List.of(
      "person",
      "email address",
      "phone number",
      "credit card number"
    );
    var chunksA = List.of(
      "Customer Alice Johnson ",
      "can be reached at alice@example.com ",
      "or +1 202-555-0147."
    );
    var chunksB = List.of(
      "Jane",
      " Doe asked us to call",
      " +1 (415) 555-0132.",
      " Her card is 4111 1111 1111 1111."
    );
    try (var ner = StreamingSpanNer.load(STREAM_DIR)) {
      var expectedA = run(ner, "seq-a", labels, chunksA);
      var expectedB = run(ner, "seq-b", labels, chunksB);
      var jobs = new ArrayList<Callable<List<EntitySpan>>>();
      for (int i = 0; i < 3; i++) {
        final int n = i;
        jobs.add(() -> run(ner, "par-a-" + n, labels, chunksA));
        jobs.add(() -> run(ner, "par-b-" + n, labels, chunksB));
      }
      var results = runAll(jobs);
      for (int i = 0; i < results.size(); i++) {
        var expected = i % 2 == 0 ? expectedA : expectedB;
        var got = results.get(i);
        assertThat(got)
          .extracting(EntitySpan::text)
          .containsExactlyElementsOf(
            expected.stream().map(EntitySpan::text).toList()
          );
        assertThat(got)
          .extracting(EntitySpan::type)
          .containsExactlyElementsOf(
            expected.stream().map(EntitySpan::type).toList()
          );
        for (int k = 0; k < expected.size(); k++) {
          assertThat(got.get(k).confidence()).isCloseTo(
            expected.get(k).confidence(),
            within(5e-3f)
          );
        }
      }
    }
  }

  private static List<EntitySpan> run(
    StreamingSpanNer ner,
    String id,
    List<String> labels,
    List<String> chunks
  ) {
    try (var session = ner.openSession(id, labels)) {
      List<EntitySpan> last = List.of();
      for (var chunk : chunks) last = session.append(chunk, 0.5f);
      return last;
    }
  }
}
