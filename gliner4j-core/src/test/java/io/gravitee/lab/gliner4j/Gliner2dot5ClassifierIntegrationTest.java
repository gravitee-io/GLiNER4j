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
import static org.assertj.core.api.Assertions.within;

import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * GLiNER2.5 zero-shot classification through the shared {@code classifier_full.onnx} contract
 * (same [L]-marker prompt and classifier MLP as GLiNER2), fed with the GLiNER2 processor text conventions.
 *
 * <p>Expected top labels and confidences come from the upstream reference implementation
 * ({@code AutoExtractor.classify_text}, gliner2 package) on {@code fastino/gliner2.5-small-v1}. The
 * cases cover upper-case text, a missing terminal period, accents, a URL, an email and a mention.
 * Skipped unless the small bundle is exported ({@code task gliner2dot5-small}).
 */
@EnabledIf("modelDirExists")
class Gliner2dot5ClassifierIntegrationTest {

  private static final Path MODEL_DIR = Path.of(
    "models/gliner2dot5-small-onnx"
  );

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/classifier_full.onnx"));
  }

  static Stream<Arguments> upstreamReference() {
    return Stream.of(
      Arguments.of(
        "This movie is absolutely fantastic and I loved every minute of it!",
        List.of("positive", "negative"),
        "positive",
        0.9818537f
      ),
      Arguments.of(
        "The Service Was TERRIBLE and the food arrived cold",
        List.of("positive", "negative"),
        "negative",
        0.9617798f
      ),
      Arguments.of(
        "Élodie jane.doe@example.com visited https://example.com in Montréal @café",
        List.of("travel", "contact information", "sports"),
        "contact information",
        0.9697773f
      )
    );
  }

  @ParameterizedTest
  @MethodSource("upstreamReference")
  void matchesUpstreamScores(
    String text,
    List<String> labelNames,
    String expectedLabel,
    float expectedConfidence
  ) {
    var labels = labelNames.stream().map(ClassificationLabel::new).toList();
    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      var results = classifier.classify(text, 0.0f);
      assertThat(results).isNotEmpty();
      assertThat(results.get(0).label()).isEqualTo(expectedLabel);
      assertThat(results.get(0).confidence()).isCloseTo(
        expectedConfidence,
        within(0.01f)
      );
      // Batch path shares the same preprocessing.
      var batch = classifier.classifyBatch(List.of(text), 0.0f).get(0);
      assertThat(batch.get(0).confidence()).isCloseTo(
        results.get(0).confidence(),
        within(1e-4f)
      );
    }
  }
}
