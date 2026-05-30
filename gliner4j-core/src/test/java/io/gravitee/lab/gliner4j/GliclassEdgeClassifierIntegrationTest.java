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

import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Proves a GLiClass <em>variant</em> with a different backbone and a different scorer
 * (gliclass-edge-v3.0: ettin-32m/ModernBERT, {@code mlp} scorer) loads through the same
 * {@link GLiNER4jClassifier} facade with <em>zero code changes</em> — the score head graph
 * encapsulates the scorer, so simple/mlp are interchangeable. Confirms the GLiClass variants are
 * export-and-load config-swaps.
 *
 * <p>Skipped unless the bundle is present (export with gliclass-edge-v3.0).
 */
@EnabledIf("modelDirExists")
class GliclassEdgeClassifierIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliclass-edge-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  @Test
  void mlpScorerVariantClassifiesThroughSameFacade() {
    var labels = List.of(
      new ClassificationLabel("technology"),
      new ClassificationLabel("sports"),
      new ClassificationLabel("cooking")
    );
    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      var results = classifier.classify(
        "The new GPU doubles neural network training throughput.",
        0.5f
      );
      assertThat(results).isNotEmpty();
      assertThat(results.get(0).label()).isEqualTo("technology");
    }
  }
}
