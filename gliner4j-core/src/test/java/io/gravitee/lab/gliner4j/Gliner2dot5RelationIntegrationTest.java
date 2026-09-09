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

import io.gravitee.lab.gliner4j.extractor.RelationExtractor;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * GLiNER2.5 relation extraction through the public {@link RelationExtractor} facade, dispatched
 * from {@code architecture=gliner2dot5} to the merged {@code relation_full.onnx}. Expected edges
 * and confidences come from the fastino reference implementation on the same checkpoint.
 * Skipped unless the small bundle is exported ({@code task gliner2dot5-small}).
 */
@EnabledIf("modelDirExists")
class Gliner2dot5RelationIntegrationTest {

  private static final Path MODEL_DIR = Path.of(
    "models/gliner2dot5-small-onnx"
  );

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/relation_full.onnx"));
  }

  private static final List<RelationDefinition> RELATIONS = List.of(
    new RelationDefinition("works_for"),
    new RelationDefinition("lives_in"),
    new RelationDefinition("born_in"),
    new RelationDefinition("married_to")
  );

  @Test
  void matchesFastinoReferenceOutput() {
    try (var extractor = RelationExtractor.load(MODEL_DIR, RELATIONS)) {
      var results = extractor.extract(
        "Barack Obama was born in Honolulu and married Michelle Obama.",
        0.5f
      );
      assertThat(results).containsOnlyKeys(
        "works_for",
        "lives_in",
        "born_in",
        "married_to"
      );
      assertThat(results.get("works_for")).isEmpty();

      var born = results.get("born_in");
      assertThat(born).hasSize(1);
      assertThat(born.get(0).head().text()).isEqualTo("Barack Obama");
      assertThat(born.get(0).tail().text()).isEqualTo("Honolulu");
      assertThat(born.get(0).confidence()).isCloseTo(0.9068f, within(0.01f));
      assertThat(born.get(0).head().start()).isZero();
      assertThat(born.get(0).tail().end()).isEqualTo(33);

      var married = results.get("married_to");
      assertThat(married).hasSize(1);
      assertThat(married.get(0).head().text()).isEqualTo("Barack Obama");
      assertThat(married.get(0).tail().text()).isEqualTo("Michelle Obama");
      assertThat(married.get(0).confidence()).isCloseTo(0.8042f, within(0.01f));

      var lives = results.get("lives_in");
      assertThat(lives).hasSize(1);
      assertThat(lives.get(0).tail().text()).isEqualTo("Honolulu");
      assertThat(lives.get(0).confidence()).isCloseTo(0.7347f, within(0.01f));
    }
  }

  @Test
  void batchAndOverrideWork() {
    try (
      var extractor = RelationExtractor.load(
        MODEL_DIR,
        List.of(new RelationDefinition("unused"))
      )
    ) {
      var override = List.of(
        new RelationDefinition("works_for", "employment relation"),
        new RelationDefinition("lives_in")
      );
      var results = extractor.extract(
        "John works for Apple and lives in San Francisco. Mary works for Google.",
        override,
        0.4f
      );
      assertThat(results).containsOnlyKeys("works_for", "lives_in");
      assertThat(results.get("works_for")).anyMatch(
        r -> r.head().text().equals("John") && r.tail().text().equals("Apple")
      );
      assertThat(results.get("lives_in")).anyMatch(r ->
        r.tail().text().equals("San Francisco")
      );

      var batch = extractor.extractBatch(
        List.of(
          "",
          "Barack Obama was born in Honolulu and married Michelle Obama."
        ),
        0.5f
      );
      assertThat(batch).hasSize(2);
      assertThat(batch.get(0)).containsOnlyKeys("unused");
      assertThat(batch.get(0).get("unused")).isEmpty();
      assertThat(batch.get(1)).containsOnlyKeys("unused");
    }
  }
}
