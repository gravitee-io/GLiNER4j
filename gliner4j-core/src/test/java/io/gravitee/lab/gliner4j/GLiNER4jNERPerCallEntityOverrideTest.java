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
 * Tests for per-call entity override in extract().
 * Requires actual ONNX model files — skipped when not present.
 */
@EnabledIf("modelDirExists")
class GLiNER4jNERPerCallEntityOverrideTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  @Test
  void extractWithPerCallEntityOverride() {
    var defaultEntities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, defaultEntities)) {
      var locationOnly = List.of(new EntityDefinition("location"));
      Map<String, List<EntitySpan>> results = gliner.extract(
        "John works at Google in New York.",
        locationOnly
      );

      // Should only contain location entities, not person or organization
      assertThat(results).doesNotContainKey("person");
      assertThat(results).doesNotContainKey("organization");
      assertThat(results).containsKey("location");
      assertThat(results.get("location")).anyMatch(span ->
        span.text().contains("New York")
      );
    }
  }

  @Test
  void extractWithPerCallEntityOverrideAndThreshold() {
    var defaultEntities = List.of(new EntityDefinition("person"));

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, defaultEntities)) {
      var overrideEntities = List.of(
        new EntityDefinition("person"),
        new EntityDefinition("organization")
      );
      Map<String, List<EntitySpan>> results = gliner.extract(
        "John works at Google.",
        overrideEntities,
        0.3f
      );

      assertThat(results).containsKey("person");
      assertThat(results.get("person")).anyMatch(span ->
        span.text().equals("John")
      );
      assertThat(results).containsKey("organization");
      assertThat(results.get("organization")).anyMatch(span ->
        span.text().equals("Google")
      );
    }
  }

  @Test
  void extractWithPerCallEntityOverrideEmptyText() {
    var defaultEntities = List.of(new EntityDefinition("person"));

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, defaultEntities)) {
      var overrideEntities = List.of(new EntityDefinition("location"));
      Map<String, List<EntitySpan>> results = gliner.extract(
        "",
        overrideEntities
      );
      assertThat(results).isEmpty();
    }
  }

  @Test
  void defaultEntitiesStillWorkAfterOverrideCall() {
    var defaultEntities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, defaultEntities)) {
      // First call with override
      var locationOnly = List.of(new EntityDefinition("location"));
      gliner.extract("John works at Google in New York.", locationOnly);

      // Second call with default entities should still work
      Map<String, List<EntitySpan>> results = gliner.extract(
        "John works at Google."
      );
      assertThat(results).containsKey("person");
      assertThat(results.get("person")).anyMatch(span ->
        span.text().equals("John")
      );
    }
  }
}
