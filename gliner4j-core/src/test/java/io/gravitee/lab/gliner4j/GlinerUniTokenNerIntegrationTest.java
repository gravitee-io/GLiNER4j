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
 * Integration test for the original-GLiNER <em>token-level</em> uni-encoder family
 * (knowledgator/gliner-multitask-large-v0.5, deberta-v3-large, span_mode=token_level) flowing
 * through the public {@link GLiNER4jNER} facade — proving the same {@code gliner-uni} architecture
 * dispatches to the BIO {@link io.gravitee.lab.gliner4j.postprocess.TokenSpanDecoder} when
 * {@code span_mode == token_level}, with no facade change.
 *
 * <p>Skipped unless the bundle is present (export with gliner-multitask-large-v0.5).
 */
@EnabledIf("modelDirExists")
class GlinerUniTokenNerIntegrationTest {

  private static final Path MODEL_DIR = Path.of(
    "models/gliner-multitask-large-onnx"
  );

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/model.onnx"));
  }

  @Test
  void extractsEntitiesTokenLevel() {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("location"),
      new EntityDefinition("date")
    );
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      var results = gliner.extract(
        "Barack Obama visited Berlin in July 2015 with Angela Merkel.",
        0.5f
      );
      assertThat(results).isNotEmpty();
      var persons = results.getOrDefault("person", List.of());
      assertThat(persons)
        .as("person spans: %s", persons)
        .extracting(EntitySpan::text)
        .anyMatch(t -> t.contains("Obama"));
      var locations = results.getOrDefault("location", List.of());
      assertThat(locations)
        .as("location spans: %s", locations)
        .extracting(EntitySpan::text)
        .anyMatch(t -> t.contains("Berlin"));
    }
  }
}
