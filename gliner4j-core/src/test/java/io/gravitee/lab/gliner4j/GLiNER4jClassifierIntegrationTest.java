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
 * Integration tests for text classification via GLiNER4jClassifier.
 * Requires actual ONNX model files including classifier_full.onnx — skipped when not present.
 */
@EnabledIf("modelDirExists")
class GLiNER4jClassifierIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return (Files.exists(MODEL_DIR.resolve("onnx/classifier_full.onnx")));
  }

  @Test
  void singleLabelClassification() {
    var labels = List.of(
      new ClassificationLabel("positive"),
      new ClassificationLabel("negative")
    );

    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      List<ClassificationResult> results = classifier.classify(
        "This movie is absolutely fantastic and I loved every minute of it!",
        0.3f
      );

      assertThat(results).isNotEmpty();
      // At least one result should be returned
      assertThat(results.get(0).label()).isIn("positive", "negative");
      assertThat(results.get(0).confidence()).isBetween(0.0f, 1.0f);
    }
  }

  @Test
  void multiLabelClassification() {
    var labels = List.of(
      new ClassificationLabel("technology"),
      new ClassificationLabel("business"),
      new ClassificationLabel("sports"),
      new ClassificationLabel("entertainment")
    );

    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      List<ClassificationResult> results = classifier.classify(
        "Apple Inc. announced a new MacBook Pro at their annual tech conference.",
        0.3f
      );

      assertThat(results).isNotEmpty();
      // Results should be sorted by confidence descending
      for (int i = 1; i < results.size(); i++) {
        assertThat(results.get(i).confidence()).isLessThanOrEqualTo(
          results.get(i - 1).confidence()
        );
      }
    }
  }

  @Test
  void classificationWithDescriptions() {
    var labels = List.of(
      new ClassificationLabel(
        "positive",
        "Expresses positive sentiment, approval, or satisfaction"
      ),
      new ClassificationLabel(
        "negative",
        "Expresses negative sentiment, disapproval, or dissatisfaction"
      )
    );

    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      List<ClassificationResult> results = classifier.classify(
        "The food was terrible and the service was even worse.",
        0.3f
      );

      assertThat(results).isNotEmpty();
      assertThat(results.get(0).confidence()).isBetween(0.0f, 1.0f);
    }
  }

  @Test
  void batchClassification() {
    var labels = List.of(
      new ClassificationLabel("positive"),
      new ClassificationLabel("negative")
    );

    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      var texts = List.of(
        "I love this product, it's amazing!",
        "This is the worst experience I've ever had.",
        "The weather is nice today."
      );

      List<List<ClassificationResult>> results = classifier.classifyBatch(
        texts,
        0.3f
      );

      assertThat(results).hasSize(3);

      // Each text should produce results
      for (var textResults : results) {
        assertThat(textResults).isNotEmpty();
      }
    }
  }

  @Test
  void batchWithEmptyTextReturnsEmptyForThatEntry() {
    var labels = List.of(
      new ClassificationLabel("positive"),
      new ClassificationLabel("negative")
    );

    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      var texts = List.of("This is great!", "", "This is terrible.");

      List<List<ClassificationResult>> results = classifier.classifyBatch(
        texts,
        0.3f
      );

      assertThat(results).hasSize(3);
      assertThat(results.get(0)).isNotEmpty();
      assertThat(results.get(1)).isEmpty();
      assertThat(results.get(2)).isNotEmpty();
    }
  }

  @Test
  void perCallLabelOverride() {
    var defaultLabels = List.of(
      new ClassificationLabel("positive"),
      new ClassificationLabel("negative")
    );

    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, defaultLabels)) {
      // Override with different labels
      var topicLabels = List.of(
        new ClassificationLabel("technology"),
        new ClassificationLabel("sports"),
        new ClassificationLabel("politics")
      );

      List<ClassificationResult> results = classifier.classify(
        "The new iPhone features an improved processor and better camera.",
        topicLabels,
        0.3f
      );

      assertThat(results).isNotEmpty();
      // Should only return labels from the override set
      for (var result : results) {
        assertThat(result.label()).isIn("technology", "sports", "politics");
      }
    }
  }

  @Test
  void defaultLabelsStillWorkAfterOverrideCall() {
    var defaultLabels = List.of(
      new ClassificationLabel("positive"),
      new ClassificationLabel("negative")
    );

    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, defaultLabels)) {
      // First call with override
      var topicLabels = List.of(new ClassificationLabel("technology"));
      classifier.classify("New iPhone announced.", topicLabels);

      // Second call with default labels should still work
      List<ClassificationResult> results = classifier.classify(
        "I love this!",
        0.3f
      );

      assertThat(results).isNotEmpty();
      for (var result : results) {
        assertThat(result.label()).isIn("positive", "negative");
      }
    }
  }

  @Test
  void emptyTextReturnsEmpty() {
    var labels = List.of(
      new ClassificationLabel("positive"),
      new ClassificationLabel("negative")
    );

    try (var classifier = GLiNER4jClassifier.load(MODEL_DIR, labels)) {
      List<ClassificationResult> results = classifier.classify("");
      assertThat(results).isEmpty();
    }
  }
}
