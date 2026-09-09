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
 * GLiNER2.5 zero-shot classification through the shared {@code classifier_full.onnx} contract
 * (same [L]-marker prompt and classifier MLP as GLiNER2). Skipped unless the small bundle is
 * exported ({@code task gliner2dot5-small}).
 */
@EnabledIf("modelDirExists")
class Gliner2dot5ClassifierIntegrationTest {

  private static final Path MODEL_DIR = Path.of(
    "models/gliner2dot5-small-onnx"
  );

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/classifier_full.onnx"));
  }

  @Test
  void classifiesSentimentZeroShot() {
    var labels = List.of(
      new ClassificationLabel("positive"),
      new ClassificationLabel("negative")
    );
    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      var results = classifier.classify(
        "This movie is absolutely fantastic and I loved every minute of it!",
        0.3f
      );
      assertThat(results).isNotEmpty();
      assertThat(results.get(0).label()).isEqualTo("positive");
      assertThat(results.get(0).confidence()).isBetween(0.0f, 1.0f);
    }
  }
}
