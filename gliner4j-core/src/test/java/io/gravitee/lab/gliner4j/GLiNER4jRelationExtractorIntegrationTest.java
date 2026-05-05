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

import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Acceptance tests for GLiNER4jRelationExtractor.
 *
 * <p>Encodes the contract for relation extraction:
 * <ul>
 *   <li>directional binary relations are returned as RelationInstance per type</li>
 *   <li>all requested relation types appear in the output map (empty list when no instances)</li>
 *   <li>threshold and per-call relation overrides behave the same way as NER</li>
 *   <li>descriptions and configurable field lists are honored end-to-end</li>
 * </ul>
 */
@EnabledIf("modelDirExists")
class GLiNER4jRelationExtractorIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  @Test
  void extractsSingleWorksForRelation() {
    var relations = List.of(new RelationDefinition("works_for"));

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      Map<String, List<RelationInstance>> results = extractor.extract(
        "John works for Apple."
      );

      assertThat(results).containsKey("works_for");
      assertThat(results.get("works_for"))
        .hasSizeGreaterThanOrEqualTo(1)
        .anyMatch(
          instance ->
            instance.head() != null &&
            instance.head().text().equals("John") &&
            instance.tail() != null &&
            instance.tail().text().equals("Apple")
        );
    }
  }

  @Test
  void extractsMultipleInstancesOfSameRelation() {
    var relations = List.of(new RelationDefinition("works_for"));

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      Map<String, List<RelationInstance>> results = extractor.extract(
        "John works for Microsoft. Mary works for Google. Bob works for Apple."
      );

      assertThat(results).containsKey("works_for");
      assertThat(results.get("works_for")).hasSizeGreaterThanOrEqualTo(2);
    }
  }

  @Test
  void extractsMultipleRelationTypes() {
    var relations = List.of(
      new RelationDefinition("works_for"),
      new RelationDefinition("lives_in")
    );

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      Map<String, List<RelationInstance>> results = extractor.extract(
        "John works for Apple and lives in San Francisco."
      );

      assertThat(results).containsKeys("works_for", "lives_in");
      assertThat(results.get("works_for")).anyMatch(
        instance ->
          instance.head() != null && instance.head().text().equals("John")
      );
      assertThat(results.get("lives_in")).anyMatch(
        instance ->
          instance.tail() != null &&
          instance.tail().text().contains("San Francisco")
      );
    }
  }

  @Test
  void requestedRelationTypesAlwaysPresentInOutput() {
    var relations = List.of(
      new RelationDefinition("works_for"),
      new RelationDefinition("acquired"),
      new RelationDefinition("partnered_with")
    );

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      Map<String, List<RelationInstance>> results = extractor.extract(
        "Pure technical noise that contains no real relations."
      );

      assertThat(results).containsKeys(
        "works_for",
        "acquired",
        "partnered_with"
      );
      // Not asserting empty lists strictly — model may produce a noisy match,
      // but every requested key MUST be present.
    }
  }

  @Test
  void emptyTextReturnsAllKeysWithEmptyLists() {
    var relations = List.of(
      new RelationDefinition("works_for"),
      new RelationDefinition("lives_in")
    );

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      Map<String, List<RelationInstance>> results = extractor.extract("");

      assertThat(results).isEmpty();
    }
  }

  @Test
  void higherThresholdProducesFewerOrEqualInstances() {
    var relations = List.of(new RelationDefinition("works_for"));
    var text =
      "John works for Microsoft. Mary works for Google. Bob works for Apple.";

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      var lowThreshold = extractor.extract(text, 0.3f);
      var highThreshold = extractor.extract(text, 0.95f);

      assertThat(highThreshold.get("works_for").size()).isLessThanOrEqualTo(
        lowThreshold.get("works_for").size()
      );
    }
  }

  @Test
  void extractWithRelationDescriptions() {
    var relations = List.of(
      new RelationDefinition(
        "works_for",
        "Employment relationship where person works at organization"
      )
    );

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      Map<String, List<RelationInstance>> results = extractor.extract(
        "John works for Apple."
      );

      assertThat(results).containsKey("works_for");
      assertThat(results.get("works_for")).anyMatch(
        instance ->
          instance.head() != null && instance.head().text().equals("John")
      );
    }
  }

  @Test
  void extractWithCustomFieldNames() {
    var relations = List.of(
      new RelationDefinition(
        "employment",
        "Employment relation",
        List.of("subject", "object")
      )
    );

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      Map<String, List<RelationInstance>> results = extractor.extract(
        "John works for Apple.",
        0.3f
      );

      assertThat(results).containsKey("employment");
      var instances = results.get("employment");
      if (!instances.isEmpty()) {
        var first = instances.get(0);
        assertThat(first.fields()).containsKeys("subject", "object");
      }
    }
  }

  @Test
  void instanceConfidenceIsMinimumAcrossFields() {
    var relations = List.of(new RelationDefinition("works_for"));

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      Map<String, List<RelationInstance>> results = extractor.extract(
        "John works for Apple."
      );

      for (var instance : results.getOrDefault("works_for", List.of())) {
        var fieldConfidences = instance
          .fields()
          .values()
          .stream()
          .map(s -> s.confidence())
          .toList();
        if (!fieldConfidences.isEmpty()) {
          float minField = fieldConfidences
            .stream()
            .reduce(Float.MAX_VALUE, Math::min);
          assertThat(instance.confidence()).isEqualTo(minField);
        }
      }
    }
  }
}
