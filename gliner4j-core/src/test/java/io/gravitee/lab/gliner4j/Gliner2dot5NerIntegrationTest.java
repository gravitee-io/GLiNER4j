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
import static org.assertj.core.api.Assertions.within;

import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Integration test for the GLiNER2.5 boundary family (fastino/gliner2.5-small-v1) through the
 * public {@link GLiNER4jNER} facade — auto-dispatched from {@code architecture=gliner2dot5}.
 *
 * <p>Expected spans and confidences come from the fastino reference implementation
 * ({@code AutoExtractor.extract_entities}) on the same checkpoint. Skipped unless the exported
 * bundle is present ({@code task gliner2dot5-small}).
 */
@EnabledIf("modelDirExists")
class Gliner2dot5NerIntegrationTest {

  private static final Path MODEL_DIR = Path.of(
    "models/gliner2dot5-small-onnx"
  );

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/ner_full.onnx"));
  }

  private static final List<EntityDefinition> ENTITIES = List.of(
    new EntityDefinition("person"),
    new EntityDefinition("location"),
    new EntityDefinition("organization")
  );

  @Test
  void matchesFastinoReferenceOutput() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var results = gliner.extract(
        "Barack Obama visited Berlin with Angela Merkel and met Google executives.",
        0.5f
      );
      assertThat(results.get("person"))
        .extracting(EntitySpan::text)
        .containsExactly("Barack Obama", "Angela Merkel");
      assertThat(results.get("location"))
        .extracting(EntitySpan::text)
        .containsExactly("Berlin");
      assertThat(results.get("organization"))
        .extracting(EntitySpan::text)
        .containsExactly("Google");

      // Reference confidences: 0.9955 / 0.9874 / 0.9975 / 0.9894.
      assertThat(results.get("person").get(0).confidence()).isCloseTo(
        0.9955f,
        within(0.01f)
      );
      assertThat(results.get("person").get(1).confidence()).isCloseTo(
        0.9874f,
        within(0.01f)
      );
      assertThat(results.get("location").get(0).confidence()).isCloseTo(
        0.9975f,
        within(0.01f)
      );
      assertThat(results.get("organization").get(0).confidence()).isCloseTo(
        0.9894f,
        within(0.01f)
      );

      var obama = results.get("person").get(0);
      assertThat(obama.start()).isZero();
      assertThat(obama.end()).isEqualTo(12);
    }
  }

  @Test
  void batchWithPaddingMatchesSingleTextResults() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var texts = List.of(
        "Marie Curie worked in Paris.",
        "",
        "Barack Obama visited Berlin with Angela Merkel and met Google executives."
      );
      var batch = gliner.extractBatch(texts, 0.5f);
      assertThat(batch).hasSize(3);
      assertThat(batch.get(1)).isEmpty();
      assertThat(batch.get(0).get("person"))
        .extracting(EntitySpan::text)
        .containsExactly("Marie Curie");
      assertThat(batch.get(0).get("location"))
        .extracting(EntitySpan::text)
        .containsExactly("Paris");
      assertThat(batch.get(2)).isEqualTo(gliner.extract(texts.get(2), 0.5f));
    }
  }

  @Test
  void fp16AndQuantizedVariantsAgreeWithFp32() {
    var text =
      "Barack Obama visited Berlin with Angela Merkel and met Google executives.";
    for (var variant : List.of("onnx_fp16", "onnx_quantized")) {
      if (!Files.exists(MODEL_DIR.resolve(variant).resolve("ner_full.onnx"))) {
        continue;
      }
      try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES, variant)) {
        var results = gliner.extract(text, 0.5f);
        assertThat(results.get("person"))
          .as("%s person spans: %s", variant, results)
          .extracting(EntitySpan::text)
          .containsExactly("Barack Obama", "Angela Merkel");
        assertThat(results.get("location"))
          .extracting(EntitySpan::text)
          .containsExactly("Berlin");
        assertThat(results.get("organization"))
          .extracting(EntitySpan::text)
          .containsExactly("Google");
      }
    }
  }

  @Test
  void perCallOverrideAndDescriptionsWork() {
    try (
      var gliner = GLiNER4jNER.load(
        MODEL_DIR,
        List.of(new EntityDefinition("misc"))
      )
    ) {
      var results = gliner.extract(
        "Contact Jane Doe at jane.doe@example.com or call 555-0123.",
        List.of(
          new EntityDefinition("person", "full names of people"),
          new EntityDefinition("email", "email addresses"),
          new EntityDefinition("phone number")
        ),
        0.5f
      );
      assertThat(results.getOrDefault("person", List.of()))
        .extracting(EntitySpan::text)
        .contains("Jane Doe");
      assertThat(results.getOrDefault("email", List.of()))
        .extracting(EntitySpan::text)
        .contains("jane.doe@example.com");
    }
  }
}
