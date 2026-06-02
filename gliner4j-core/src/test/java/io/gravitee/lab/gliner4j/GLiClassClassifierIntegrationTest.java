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
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Integration test for the GLiClass family (knowledgator/gliclass-modern-base-v3.0, ModernBERT
 * backbone) flowing through the public {@link GLiNER4jClassifier} facade — proving a second,
 * non-GLiNER2 family is auto-dispatched from the bundle's {@code architecture=gliclass} key with
 * no facade changes.
 *
 * <p>Skipped unless the exported ONNX bundle is present (run
 * {@code uv run scripts/export_gliclass_onnx.py ...}).
 */
@EnabledIf("modelDirExists")
class GLiClassClassifierIntegrationTest {

  private static final Path MODEL_DIR = Path.of(
    "models/gliclass-modern-base-onnx"
  );

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  @Test
  void classifiesSentimentMultiLabel() {
    var labels = List.of(
      new ClassificationLabel("positive"),
      new ClassificationLabel("negative"),
      new ClassificationLabel("neutral")
    );
    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      var results = classifier.classify(
        "I really loved the cinematography and the soundtrack of this film.",
        0.5f
      );
      // Strongly positive text: "positive" should win and be the top (only) result above 0.5.
      assertThat(results).isNotEmpty();
      assertThat(results.get(0).label()).isEqualTo("positive");
      assertThat(results.get(0).confidence()).isGreaterThan(0.9f);
      assertThat(results).noneMatch(
        r -> r.label().equals("negative") && r.confidence() > 0.5f
      );
    }
  }

  @Test
  void perCallLabelOverrideWorks() {
    var loadLabels = List.of(new ClassificationLabel("spam"));
    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, loadLabels)) {
      var results = classifier.classify(
        "The match ended 3-1 and the striker scored a hat-trick.",
        List.of(
          new ClassificationLabel("sports"),
          new ClassificationLabel("politics"),
          new ClassificationLabel("cooking")
        ),
        0.5f
      );
      assertThat(results).isNotEmpty();
      assertThat(results.get(0).label()).isEqualTo("sports");
    }
  }
}
