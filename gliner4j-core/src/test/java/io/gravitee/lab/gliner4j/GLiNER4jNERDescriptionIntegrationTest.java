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
 * Integration tests for entity extraction with descriptions.
 * Requires actual ONNX model files — skipped when not present.
 */
@EnabledIf("modelDirExists")
class GLiNER4jNERDescriptionIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  @Test
  void extractWithEntityDescriptions() {
    var entities = List.of(
      new EntityDefinition("person", "Names of individuals"),
      new EntityDefinition("organization", "Company or institution names")
    );

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      Map<String, List<EntitySpan>> results = gliner.extract(
        "John works at Google."
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
  void extractWithPerCallDescriptionOverride() {
    var defaultEntities = List.of(new EntityDefinition("person"));

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, defaultEntities)) {
      var overrideEntities = List.of(
        new EntityDefinition("person", "Names of individuals"),
        new EntityDefinition("organization", "Company or institution names")
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
}
