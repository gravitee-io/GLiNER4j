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

import com.fasterxml.jackson.databind.ObjectMapper;
import io.gravitee.lab.gliner4j.GLiNER4jClassifier;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Parity of the llama.cpp + ONNX pipeline with the gliclass Python pipeline on the reference
 * scores the export script wrote into the bundle ({@code reference.json}). Skipped unless
 * {@code task scx-router} has produced {@code models/scx-router-onnx}.
 */
@EnabledIf("bundleExists")
class DecoderKvRouterIntegrationTest {

  private static final Path MODEL_DIR = Path.of("../models/scx-router-onnx");
  // q8_0 backbone vs fp32 PyTorch reference
  private static final float SCORE_TOLERANCE = 0.05f;

  static boolean bundleExists() {
    return (
      Files.exists(MODEL_DIR.resolve("gguf/backbone-q8_0.gguf")) &&
      Files.exists(MODEL_DIR.resolve("onnx/scorer.onnx"))
    );
  }

  record Entry(
    String text,
    String family,
    List<String> labels,
    String classification_type,
    List<Double> scores
  ) {}

  private static DecoderKvRouter router;
  private static List<Entry> reference;

  @BeforeAll
  static void load() throws Exception {
    router = DecoderKvRouter.load(MODEL_DIR);
    var mapper = new ObjectMapper();
    reference = List.of(
      mapper.readValue(
        MODEL_DIR.resolve("reference.json").toFile(),
        Entry[].class
      )
    );
  }

  @AfterAll
  static void close() {
    if (router != null) router.close();
  }

  @Test
  void matchesPythonReferenceScores() {
    assertThat(reference).isNotEmpty();
    for (var entry : reference) {
      List<ClassificationResult> results = entry
          .classification_type()
          .equals("single-label")
        ? router.classifySingleLabel(entry.text(), entry.labels())
        : router.classify(entry.text(), entry.labels(), 0.0f);
      var byLabel = results
        .stream()
        .collect(
          java.util.stream.Collectors.toMap(
            ClassificationResult::label,
            ClassificationResult::confidence
          )
        );
      int best = 0;
      for (int i = 1; i < entry.scores().size(); i++) {
        if (entry.scores().get(i) > entry.scores().get(best)) best = i;
      }
      assertThat(results.get(0).label())
        .as("%s / %s top label", entry.family(), entry.text())
        .isEqualTo(entry.labels().get(best));
      for (int i = 0; i < entry.labels().size(); i++) {
        assertThat(byLabel.get(entry.labels().get(i)))
          .as(
            "%s / %s / %s",
            entry.family(),
            entry.text(),
            entry.labels().get(i)
          )
          .isCloseTo(
            entry.scores().get(i).floatValue(),
            within(SCORE_TOLERANCE)
          );
      }
    }
  }

  @Test
  void sessionOverChunksEqualsOneShotClassification() {
    var chunks = List.of(
      "I need help refactoring some Rust code.",
      " Specifically the borrow checker keeps rejecting this function.",
      " fn parse(&mut self, buf: &[u8]) -> Result<Token, Error> { ... }"
    );
    var full = String.join("", chunks);
    var labels = RouterDemo.MODELS;
    var oneShot = router.classify(full, labels, 0.0f);
    try (var session = router.openSession("t-1")) {
      for (var chunk : chunks) session.append(chunk);
      var streamed = session.classify(labels, 0.0f);
      assertThat(streamed)
        .extracting(ClassificationResult::label)
        .containsExactlyElementsOf(
          oneShot.stream().map(ClassificationResult::label).toList()
        );
      for (int i = 0; i < labels.size(); i++) {
        assertThat(streamed.get(i).confidence()).isCloseTo(
          oneShot.get(i).confidence(),
          within(2e-3f)
        );
      }
    }
  }

  @Test
  void labelsDoNotLeakBetweenSessionCalls() {
    var text =
      "Compare these two vendor contracts and flag the riskier clauses.";
    try (var session = router.openSession("t-2")) {
      session.append(text);
      session.classifySingleLabel(RouterDemo.TASKS);
      var second = session.classifySingleLabel(RouterDemo.REASONING);
      var fresh = router.classifySingleLabel(text, RouterDemo.REASONING);
      assertThat(second.get(0).label()).isEqualTo(fresh.get(0).label());
      assertThat(second.get(0).confidence()).isCloseTo(
        fresh.get(0).confidence(),
        within(2e-3f)
      );
    }
  }

  @Test
  void facadeDispatchesThroughServiceLoader() {
    var labels = List.of(
      new ClassificationLabel("travel"),
      new ClassificationLabel("finance"),
      new ClassificationLabel("technology")
    );
    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      var texts = List.of(
        "The new mid-range EV just posted a 480-mile range on a single charge.",
        "Quarterly earnings beat estimates on strong subscription revenue."
      );
      var batch = classifier.classifyBatch(texts, 0.0f);
      assertThat(batch).hasSize(2);
      assertThat(batch.get(0).get(0).label()).isEqualTo("technology");
      assertThat(batch.get(1).get(0).label()).isEqualTo("finance");
      assertThat(batch.get(0)).isEqualTo(
        classifier.classify(texts.get(0), 0.0f)
      );
    }
  }
}
