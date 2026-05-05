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
 * Acceptance tests for per-call relation overrides.
 *
 * <p>Mirrors the per-call entity override behavior in {@link GLiNER4jNERPerCallEntityOverrideTest}:
 * callers can swap the relation set on a single call without reloading the model.
 */
@EnabledIf("modelDirExists")
class GLiNER4jRelationExtractorPerCallOverrideTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  @Test
  void perCallOverrideReplacesRelationSet() {
    var defaults = List.of(new RelationDefinition("works_for"));

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, defaults)) {
      var overrideRelations = List.of(new RelationDefinition("lives_in"));
      Map<String, List<RelationInstance>> results = extractor.extract(
        "Mary lives in Paris.",
        overrideRelations
      );

      assertThat(results).containsKey("lives_in");
      assertThat(results).doesNotContainKey("works_for");
    }
  }

  @Test
  void defaultRelationsStillWorkAfterOverrideCall() {
    var defaults = List.of(new RelationDefinition("works_for"));

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, defaults)) {
      // First call: override
      extractor.extract(
        "Mary lives in Paris.",
        List.of(new RelationDefinition("lives_in"))
      );

      // Second call: default relations
      Map<String, List<RelationInstance>> results = extractor.extract(
        "John works for Apple."
      );
      assertThat(results).containsKey("works_for");
    }
  }

  @Test
  void perCallOverrideWithThreshold() {
    var defaults = List.of(new RelationDefinition("works_for"));

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, defaults)) {
      var override = List.of(new RelationDefinition("lives_in"));
      Map<String, List<RelationInstance>> results = extractor.extract(
        "Mary lives in Paris.",
        override,
        0.3f
      );

      assertThat(results).containsKey("lives_in");
    }
  }
}
