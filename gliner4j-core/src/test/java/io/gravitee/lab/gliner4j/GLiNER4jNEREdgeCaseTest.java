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
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Edge-case tests for null, empty, and blank inputs across all extract and extractBatch overloads.
 * Skipped automatically when model directory is not present.
 */
@EnabledIf("modelDirExists")
class GLiNER4jNEREdgeCaseTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");
  private static final List<EntityDefinition> ENTITIES = List.of(
    new EntityDefinition("person"),
    new EntityDefinition("organization")
  );

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/encoder.onnx"));
  }

  // ── extract(String) ──

  @Test
  void extract_nullText_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      Map<String, List<EntitySpan>> results = gliner.extract((String) null);
      assertThat(results).isEmpty();
    }
  }

  @Test
  void extract_emptyText_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      Map<String, List<EntitySpan>> results = gliner.extract("");
      assertThat(results).isEmpty();
    }
  }

  @Test
  void extract_blankText_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      Map<String, List<EntitySpan>> results = gliner.extract("   ");
      assertThat(results).isEmpty();
    }
  }

  // ── extract(String, List<EntityDefinition>) ──

  @Test
  void extractWithOverride_nullText_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var override = List.of(new EntityDefinition("location"));
      Map<String, List<EntitySpan>> results = gliner.extract(null, override);
      assertThat(results).isEmpty();
    }
  }

  @Test
  void extractWithOverride_emptyText_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var override = List.of(new EntityDefinition("location"));
      Map<String, List<EntitySpan>> results = gliner.extract("", override);
      assertThat(results).isEmpty();
    }
  }

  @Test
  void extractWithOverride_blankText_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var override = List.of(new EntityDefinition("location"));
      Map<String, List<EntitySpan>> results = gliner.extract("   ", override);
      assertThat(results).isEmpty();
    }
  }

  // ── extract(String, List<EntityDefinition>, float) ──

  @Test
  void extractWithOverrideAndThreshold_nullText_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var override = List.of(new EntityDefinition("location"));
      Map<String, List<EntitySpan>> results = gliner.extract(
        null,
        override,
        0.5f
      );
      assertThat(results).isEmpty();
    }
  }

  @Test
  void extractWithOverrideAndThreshold_emptyText_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var override = List.of(new EntityDefinition("location"));
      Map<String, List<EntitySpan>> results = gliner.extract(
        "",
        override,
        0.5f
      );
      assertThat(results).isEmpty();
    }
  }

  @Test
  void extractWithOverrideAndThreshold_blankText_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var override = List.of(new EntityDefinition("location"));
      Map<String, List<EntitySpan>> results = gliner.extract(
        "   ",
        override,
        0.5f
      );
      assertThat(results).isEmpty();
    }
  }

  // ── extractBatch(null list) ──

  @Test
  void extractBatch_nullList_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(null);
      assertThat(results).isEmpty();
    }
  }

  // ── extractBatch with uniform edge-case lists ──

  @Test
  void extractBatch_allNullTexts_returnsEmptyMaps() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var texts = new ArrayList<String>();
      texts.add(null);
      texts.add(null);

      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(texts);

      assertThat(results).hasSize(2);
      assertThat(results.get(0)).isEmpty();
      assertThat(results.get(1)).isEmpty();
    }
  }

  @Test
  void extractBatch_allEmptyTexts_returnsEmptyMaps() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var texts = List.of("", "");

      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(texts);

      assertThat(results).hasSize(2);
      assertThat(results.get(0)).isEmpty();
      assertThat(results.get(1)).isEmpty();
    }
  }

  @Test
  void extractBatch_allBlankTexts_returnsEmptyMaps() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var texts = List.of("   ", "  \t  ");

      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(texts);

      assertThat(results).hasSize(2);
      assertThat(results.get(0)).isEmpty();
      assertThat(results.get(1)).isEmpty();
    }
  }

  // ── extractBatch mixed ──

  @Test
  void extractBatch_mixedNullEmptyBlankNormal_returnsCorrectResults() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var texts = new ArrayList<String>();
      texts.add(null);
      texts.add("");
      texts.add("   ");
      texts.add("John works at Google.");

      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(texts);

      assertThat(results).hasSize(4);

      // null → empty map
      assertThat(results.get(0)).isEmpty();
      // empty → empty map
      assertThat(results.get(1)).isEmpty();
      // blank → empty map
      assertThat(results.get(2)).isEmpty();

      // normal text → has entities
      assertThat(results.get(3)).containsKey("person");
      assertThat(results.get(3).get("person"))
        .anyMatch(span -> span.text().equals("John"));
      assertThat(results.get(3)).containsKey("organization");
      assertThat(results.get(3).get("organization"))
        .anyMatch(span -> span.text().equals("Google"));
    }
  }

  // ── extractBatch(List<String>, float) threshold overload edge cases ──

  @Test
  void extractBatchWithThreshold_nullList_returnsEmpty() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(
        null,
        0.5f
      );
      assertThat(results).isEmpty();
    }
  }

  @Test
  void extractBatchWithThreshold_allEmptyTexts_returnsEmptyMaps() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var texts = List.of("", "");

      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(
        texts,
        0.5f
      );

      assertThat(results).hasSize(2);
      assertThat(results.get(0)).isEmpty();
      assertThat(results.get(1)).isEmpty();
    }
  }

  @Test
  void extractBatchWithThreshold_mixedInputs_returnsCorrectResults() {
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, ENTITIES)) {
      var texts = new ArrayList<String>();
      texts.add(null);
      texts.add("   ");
      texts.add("John works at Google.");

      List<Map<String, List<EntitySpan>>> results = gliner.extractBatch(
        texts,
        0.3f
      );

      assertThat(results).hasSize(3);

      // null → empty map
      assertThat(results.get(0)).isEmpty();
      // blank → empty map
      assertThat(results.get(1)).isEmpty();

      // normal text → has entities
      assertThat(results.get(2)).isNotEmpty();
    }
  }
}
