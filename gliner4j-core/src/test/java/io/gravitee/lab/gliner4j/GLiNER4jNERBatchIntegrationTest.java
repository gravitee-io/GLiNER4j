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
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Integration tests for batch extraction via extractBatch().
 * Skipped automatically when model directory is not present.
 */
@EnabledIf("modelDirExists")
class GLiNER4jNERBatchIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  @Test
  void batchExtractionReturnsCorrectEntities() {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      var texts = List.of(
        "John works at Google.",
        "Marie Curie worked at the University of Paris."
      );

      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(texts);

      assertThat(results).hasSize(2);

      // First text
      assertThat(results.get(0)).containsKey("person");
      assertThat(results.get(0).get("person"))
        .anyMatch(span -> span.text().equals("John"));
      assertThat(results.get(0)).containsKey("organization");
      assertThat(results.get(0).get("organization"))
        .anyMatch(span -> span.text().equals("Google"));

      // Second text
      assertThat(results.get(1)).containsKey("person");
      assertThat(results.get(1).get("person"))
        .anyMatch(span -> span.text().contains("Marie Curie"));
    }
  }

  @Test
  void batchResultsMatchSingleTextResults() {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      var texts = List.of(
        "John works at Google.",
        "Elon Musk founded SpaceX and leads Tesla."
      );

      // Batch extraction
      List<Map<String, List<EntitySpan>>> batchResults = gliner.extractBatch(
        texts
      );

      // Single-text extraction for each
      var singleResult0 = gliner.extract(texts.get(0));
      var singleResult1 = gliner.extract(texts.get(1));

      // Same entity types detected
      assertThat(batchResults.get(0).keySet())
        .isEqualTo(singleResult0.keySet());
      assertThat(batchResults.get(1).keySet())
        .isEqualTo(singleResult1.keySet());

      // Same entity texts detected
      for (var type : singleResult0.keySet()) {
        var batchTexts = batchResults
          .get(0)
          .get(type)
          .stream()
          .map(EntitySpan::text)
          .toList();
        var singleTexts = singleResult0
          .get(type)
          .stream()
          .map(EntitySpan::text)
          .toList();
        assertThat(batchTexts).containsExactlyElementsOf(singleTexts);
      }
      for (var type : singleResult1.keySet()) {
        var batchTexts = batchResults
          .get(1)
          .get(type)
          .stream()
          .map(EntitySpan::text)
          .toList();
        var singleTexts = singleResult1
          .get(type)
          .stream()
          .map(EntitySpan::text)
          .toList();
        assertThat(batchTexts).containsExactlyElementsOf(singleTexts);
      }
    }
  }

  @Test
  void singleElementBatchMatchesSingleExtract() {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      var text = "John works at Google.";

      var singleResult = gliner.extract(text);
      var batchResults = gliner.extractBatch(List.of(text));

      assertThat(batchResults).hasSize(1);
      assertThat(batchResults.get(0).keySet()).isEqualTo(singleResult.keySet());

      for (var type : singleResult.keySet()) {
        var batchTexts = batchResults
          .get(0)
          .get(type)
          .stream()
          .map(EntitySpan::text)
          .toList();
        var singleTexts = singleResult
          .get(type)
          .stream()
          .map(EntitySpan::text)
          .toList();
        assertThat(batchTexts).containsExactlyElementsOf(singleTexts);
      }
    }
  }

  @Test
  void batchWithEmptyTextReturnsEmptyForThatEntry() {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      var texts = List.of(
        "John works at Google.",
        "",
        "Marie Curie worked at the University of Paris."
      );

      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(texts);

      assertThat(results).hasSize(3);

      // First text has entities
      assertThat(results.get(0)).isNotEmpty();

      // Empty text returns empty map
      assertThat(results.get(1)).isEmpty();

      // Third text has entities
      assertThat(results.get(2)).isNotEmpty();
    }
  }

  @Test
  void batchWithCustomThreshold() {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      var texts = List.of(
        "John works at Google.",
        "Elon Musk founded SpaceX and leads Tesla."
      );

      // High threshold should still find high-confidence entities
      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(
        texts,
        0.7f
      );

      assertThat(results).hasSize(2);
    }
  }
}
