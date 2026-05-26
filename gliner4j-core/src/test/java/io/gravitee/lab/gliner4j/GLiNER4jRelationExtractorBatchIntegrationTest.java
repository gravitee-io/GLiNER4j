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
 * Acceptance tests for batched relation extraction.
 *
 * <p>The contract is that {@code extractBatch} produces the same map shape per text as serial
 * {@code extract} calls, with the encoder shared across the batch.
 */
@EnabledIf("modelDirExists")
class GLiNER4jRelationExtractorBatchIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  @Test
  void batchPreservesPerTextResults() {
    var relations = List.of(
      new RelationDefinition("works_for"),
      new RelationDefinition("lives_in")
    );
    var texts = List.of(
      "John works for Apple.",
      "Mary lives in Paris.",
      "Bob works for Google and lives in Berlin."
    );

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      List<Map<String, List<RelationInstance>>> batchResults =
        extractor.extractBatch(texts);

      assertThat(batchResults).hasSize(texts.size());
      for (var result : batchResults) {
        assertThat(result).containsKeys("works_for", "lives_in");
      }
    }
  }

  @Test
  void batchHandlesEmptyAndBlankInputs() {
    var relations = List.of(new RelationDefinition("works_for"));
    var texts = List.of(
      "John works for Apple.",
      "",
      "   ",
      "Mary works for Google."
    );

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      List<Map<String, List<RelationInstance>>> batchResults =
        extractor.extractBatch(texts);

      assertThat(batchResults).hasSize(4);
      assertThat(batchResults.get(1)).isEmpty();
      assertThat(batchResults.get(2)).isEmpty();
      assertThat(batchResults.get(0)).containsKey("works_for");
      assertThat(batchResults.get(3)).containsKey("works_for");
    }
  }

  @Test
  void batchMatchesSerialResults() {
    var relations = List.of(new RelationDefinition("works_for"));
    var texts = List.of("John works for Apple.", "Mary works for Google.");

    try (var extractor = GLiNER4jRelationExtractor.load(MODEL_DIR, relations)) {
      var serial = texts.stream().map(extractor::extract).toList();
      var batch = extractor.extractBatch(texts);

      // Serial and batch must agree on which keys are present per text
      for (int i = 0; i < texts.size(); i++) {
        assertThat(batch.get(i).keySet())
          .as("text %d keys", i)
          .containsExactlyInAnyOrderElementsOf(serial.get(i).keySet());
      }
    }
  }
}
