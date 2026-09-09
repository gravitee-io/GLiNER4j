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
import io.gravitee.lab.gliner4j.runtime.ExecutionProvider;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * GLiClass on the llama.cpp engine vs the PyTorch reference scores the export script wrote into
 * the bundle, and vs the ONNX engine on the same bundle family when it is present. Skipped unless
 * {@code task gliclass-edge:llamacpp} has produced {@code models/gliclass-edge-llamacpp}.
 */
@EnabledIf("bundleExists")
class LlamaGliclassIntegrationTest {

  private static final Path MODEL_DIR = Path.of(
    "../models/gliclass-edge-llamacpp"
  );
  private static final Path ONNX_DIR = Path.of("../models/gliclass-edge-onnx");
  // q8_0 encoder (Metal f16 activations) vs fp32 PyTorch
  private static final float SCORE_TOLERANCE = 0.05f;
  // f16 encoder vs fp32 PyTorch
  private static final float F16_TOLERANCE = 0.01f;

  static boolean bundleExists() {
    return Files.exists(MODEL_DIR.resolve("gguf/heads.gguf"));
  }

  record Entry(String text, List<String> labels, List<Double> scores) {}

  private static GLiNER4jClassifier classifier;
  private static List<Entry> reference;

  @BeforeAll
  static void load() throws Exception {
    classifier = GLiNER4jClassifier.load(
      MODEL_DIR,
      labels(List.of("technology", "sports"))
    );
    reference = List.of(
      new ObjectMapper().readValue(
        MODEL_DIR.resolve("reference.json").toFile(),
        Entry[].class
      )
    );
  }

  @AfterAll
  static void close() {
    if (classifier != null) classifier.close();
  }

  static List<ClassificationLabel> labels(List<String> names) {
    return names
      .stream()
      .map(n -> new ClassificationLabel(n, ""))
      .toList();
  }

  static Map<String, Float> byLabel(List<ClassificationResult> results) {
    return results
      .stream()
      .collect(
        Collectors.toMap(
          ClassificationResult::label,
          ClassificationResult::confidence
        )
      );
  }

  @Test
  void matchesPythonReferenceScores() {
    assertThat(reference).isNotEmpty();
    for (var entry : reference) {
      var scores = byLabel(
        classifier.classify(entry.text(), labels(entry.labels()), 0.0f)
      );
      for (int i = 0; i < entry.labels().size(); i++) {
        assertThat(scores.get(entry.labels().get(i)))
          .as("%s / %s", entry.text(), entry.labels().get(i))
          .isCloseTo(
            entry.scores().get(i).floatValue(),
            within(SCORE_TOLERANCE)
          );
      }
    }
  }

  @Test
  void f16BackboneMatchesPythonReferenceTightly() {
    try (
      var f16 = GLiNER4jClassifier.load(MODEL_DIR, labels(List.of("a")), "f16")
    ) {
      float maxDiff = 0f;
      for (var entry : reference) {
        var scores = byLabel(
          f16.classify(entry.text(), labels(entry.labels()), 0.0f)
        );
        for (int i = 0; i < entry.labels().size(); i++) {
          float diff = Math.abs(
            scores.get(entry.labels().get(i)) -
              entry.scores().get(i).floatValue()
          );
          maxDiff = Math.max(maxDiff, diff);
          assertThat(scores.get(entry.labels().get(i)))
            .as("%s / %s", entry.text(), entry.labels().get(i))
            .isCloseTo(
              entry.scores().get(i).floatValue(),
              within(F16_TOLERANCE)
            );
        }
      }
      System.out.println("f16 max |score diff| vs PyTorch = " + maxDiff);
    }
  }

  @Test
  void batchEqualsSingleCalls() {
    var texts = reference.stream().map(Entry::text).toList();
    var batch = classifier.classifyBatch(texts, 0.0f);
    for (int i = 0; i < texts.size(); i++) {
      var single = byLabel(classifier.classify(texts.get(i), 0.0f));
      var packed = byLabel(batch.get(i));
      assertThat(packed.keySet()).isEqualTo(single.keySet());
      for (var label : single.keySet()) {
        assertThat(packed.get(label)).isCloseTo(
          single.get(label),
          within(0.02f)
        );
      }
    }
  }

  @Test
  void cpuBackendMatchesPythonReference() {
    // CPU and Metal/CUDA quantize activations differently for q8_0 matmuls; compare each to the
    // fp32 reference rather than to each other.
    var rc = RuntimeConfig.builder()
      .executionProvider(ExecutionProvider.CPU)
      .build();
    try (
      var cpu = GLiNER4jClassifier.load(MODEL_DIR, labels(List.of("a")), rc)
    ) {
      for (var entry : reference) {
        var scores = byLabel(
          cpu.classify(entry.text(), labels(entry.labels()), 0.0f)
        );
        for (int i = 0; i < entry.labels().size(); i++) {
          assertThat(scores.get(entry.labels().get(i)))
            .as("cpu %s / %s", entry.text(), entry.labels().get(i))
            .isCloseTo(
              entry.scores().get(i).floatValue(),
              within(SCORE_TOLERANCE)
            );
        }
      }
    }
  }

  @Test
  @EnabledIf("onnxBundleExists")
  void matchesOnnxEngine() {
    try (var onnx = GLiNER4jClassifier.load(ONNX_DIR, labels(List.of("a")))) {
      for (var entry : reference) {
        var a = byLabel(
          classifier.classify(entry.text(), labels(entry.labels()), 0.0f)
        );
        var b = byLabel(
          onnx.classify(entry.text(), labels(entry.labels()), 0.0f)
        );
        for (var label : a.keySet()) {
          assertThat(a.get(label))
            .as("%s / %s", entry.text(), label)
            .isCloseTo(b.get(label), within(SCORE_TOLERANCE));
        }
      }
    }
  }

  static boolean onnxBundleExists() {
    return Files.exists(ONNX_DIR.resolve("onnx/encoder.onnx"));
  }
}
