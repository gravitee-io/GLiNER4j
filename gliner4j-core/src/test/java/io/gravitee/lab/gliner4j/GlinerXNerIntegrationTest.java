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
 * Integration test for GLiNER-X (knowledgator/gliner-x-base, mT5-base, span_mode=markerV0) flowing
 * through the public {@link GLiNER4jNER} facade — multilingual zero-shot NER via the {@code
 * gliner-uni} architecture on an mT5 backbone, whitespace-split (see the TODO in
 * {@link io.gravitee.lab.gliner4j.strategy.glineruni.GlinerUniNerStrategy}: EU/space-separated
 * languages only, no CJK).
 *
 * <p>Skipped unless the bundle is present (export with gliner-x-base).
 */
@EnabledIf("modelDirExists")
class GlinerXNerIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner-x-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/model.onnx"));
  }

  @Test
  void extractsEntitiesAcrossLanguages() {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("location")
    );
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      // English
      var en = gliner.extract("Barack Obama visited Berlin.", 0.5f);
      assertThat(en.getOrDefault("person", List.of()))
        .extracting(EntitySpan::text)
        .anyMatch(t -> t.contains("Obama"));
      // French
      var fr = gliner.extract("Emmanuel Macron est allé à Paris.", 0.5f);
      assertThat(fr.getOrDefault("person", List.of()))
        .as("fr persons: %s", fr.get("person"))
        .extracting(EntitySpan::text)
        .anyMatch(t -> t.contains("Macron"));
    }
  }
}
