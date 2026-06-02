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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Integration test for the original-GLiNER uni-encoder family (knowledgator/gliner-pii-base-v1.0,
 * deberta-v3-small, span_mode=markerV0) flowing through the public {@link GLiNER4jNER} facade —
 * proving a third, non-GLiNER2 family auto-dispatches from {@code architecture=gliner-uni} with no
 * facade changes, reusing the shared SpanDecoder + span-idx machinery.
 *
 * <p>Skipped unless the exported bundle is present (run
 * {@code uv run scripts/export_gliner_uni_onnx.py ...} or {@code task gliner-pii}).
 */
@EnabledIf("modelDirExists")
class GlinerUniNerIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner-pii-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/model.onnx"));
  }

  @Test
  void extractsPersonsZeroShot() {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("location")
    );
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      var results = gliner.extract(
        "Barack Obama visited Berlin with Angela Merkel.",
        0.5f
      );
      assertThat(results).isNotEmpty();
      var persons = results.getOrDefault("person", List.of());
      assertThat(persons)
        .as("person spans: %s", persons)
        .extracting(EntitySpan::text)
        .anyMatch(t -> t.contains("Obama"));
    }
  }

  @Test
  void perCallLabelOverrideWorks() {
    try (
      var gliner = GLiNER4jNER.load(
        MODEL_DIR,
        List.of(new EntityDefinition("misc"))
      )
    ) {
      var results = gliner.extract(
        "Contact Jane Doe at jane.doe@example.com or call 555-0123.",
        List.of(
          new EntityDefinition("person"),
          new EntityDefinition("email"),
          new EntityDefinition("phone number")
        ),
        0.5f
      );
      assertThat(results.values().stream().mapToLong(List::size).sum())
        .as("total entities: %s", results)
        .isGreaterThan(0);
    }
  }
}
