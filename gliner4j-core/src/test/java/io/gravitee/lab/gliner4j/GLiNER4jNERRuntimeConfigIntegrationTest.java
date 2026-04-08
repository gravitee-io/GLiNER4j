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

import ai.onnxruntime.OrtSession;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Integration tests for RuntimeConfig — verifies that custom ORT resource
 * settings are applied without breaking inference correctness.
 */
@EnabledIf("modelDirExists")
class GLiNER4jNERRuntimeConfigIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  private static List<EntityDefinition> defaultEntities() {
    return List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );
  }

  @Test
  void loadWithDefaultConfig_producesIdenticalResults() {
    var entities = defaultEntities();
    var config = RuntimeConfig.defaults();

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities, config)) {
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
  void loadWithCustomThreadCounts_succeeds() {
    var entities = defaultEntities();
    var config = RuntimeConfig.builder()
      .encoderIntraOpThreads(2)
      .encoderInterOpThreads(1)
      .scoringIntraOpThreads(1)
      .scoringInterOpThreads(1)
      .build();

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities, config)) {
      Map<String, List<EntitySpan>> results = gliner.extract(
        "John works at Google."
      );

      assertThat(results).containsKey("person");
      assertThat(results.get("person")).anyMatch(span ->
        span.text().equals("John")
      );
    }
  }

  @Test
  void loadWithCacheDisabled_succeeds() {
    var entities = defaultEntities();
    var config = RuntimeConfig.builder()
      .optimizedModelCacheEnabled(false)
      .build();

    // Use a unique variant dir suffix to avoid interference with cached models
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities, config)) {
      Map<String, List<EntitySpan>> results = gliner.extract(
        "Marie Curie worked at the University of Paris."
      );

      assertThat(results).containsKey("person");
      assertThat(results.get("person")).anyMatch(span ->
        span.text().contains("Marie")
      );
    }
  }

  @Test
  void loadWithReducedOptLevel_succeeds() {
    var entities = defaultEntities();
    var config = RuntimeConfig.builder()
      .optimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
      .optimizedModelCacheEnabled(false)
      .build();

    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities, config)) {
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
