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
import io.gravitee.lab.gliner4j.schema.ExtractionResult;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.Schema;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Acceptance tests for the unified {@link GLiNER4j} facade.
 *
 * <p>Verifies that a single forward pass over a {@link Schema} produces an
 * {@link ExtractionResult} with both entity and relation outputs.
 */
@EnabledIf("modelDirExists")
class GLiNER4jCombinedExtractionIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  @Test
  void extractsEntitiesAndRelationsTogether() {
    var schema = Schema.builder()
      .entities(
        List.of(
          new EntityDefinition("person"),
          new EntityDefinition("organization")
        )
      )
      .relations(
        List.of(new RelationDefinition("works_for", "Employment relationship"))
      )
      .build();

    try (var gliner = GLiNER4j.load(MODEL_DIR)) {
      ExtractionResult result = gliner.extract("John works for Apple.", schema);

      assertThat(result.entities())
        .as("entity types present")
        .containsKeys("person", "organization");

      assertThat(result.entities().get("person")).anyMatch(span ->
        span.text().equals("John")
      );
      assertThat(result.entities().get("organization")).anyMatch(span ->
        span.text().equals("Apple")
      );

      assertThat(result.relations())
        .as("relation types present")
        .containsKey("works_for");
      assertThat(result.relations().get("works_for")).anyMatch(
        instance ->
          instance.head() != null && instance.head().text().equals("John")
      );
    }
  }

  @Test
  void entitiesOnlySchemaWorks() {
    var schema = Schema.builder()
      .entities(List.of(new EntityDefinition("person")))
      .build();

    try (var gliner = GLiNER4j.load(MODEL_DIR)) {
      ExtractionResult result = gliner.extract("John works at Google.", schema);

      assertThat(result.entities()).containsKey("person");
      assertThat(result.relations()).isEmpty();
    }
  }

  @Test
  void relationsOnlySchemaWorks() {
    var schema = Schema.builder()
      .relations(List.of(new RelationDefinition("works_for")))
      .build();

    try (var gliner = GLiNER4j.load(MODEL_DIR)) {
      ExtractionResult result = gliner.extract("John works for Apple.", schema);

      assertThat(result.entities()).isEmpty();
      assertThat(result.relations()).containsKey("works_for");
    }
  }

  @Test
  void emptyTextReturnsEmptyResult() {
    var schema = Schema.builder()
      .entities(List.of(new EntityDefinition("person")))
      .relations(List.of(new RelationDefinition("works_for")))
      .build();

    try (var gliner = GLiNER4j.load(MODEL_DIR)) {
      ExtractionResult result = gliner.extract("", schema);

      assertThat(result.entities()).isEmpty();
      assertThat(result.relations()).isEmpty();
    }
  }
}
