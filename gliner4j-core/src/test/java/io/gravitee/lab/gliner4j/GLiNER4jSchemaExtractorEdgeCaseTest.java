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

import io.gravitee.lab.gliner4j.extractor.SchemaExtractor;
import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import io.gravitee.lab.gliner4j.schema.StructureInstance;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Edge-case tests for null, empty, and blank inputs across schema extract overloads.
 * Skipped automatically when the model directory is not present.
 */
@EnabledIf("modelDirExists")
class GLiNER4jSchemaExtractorEdgeCaseTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");
  private static final List<StructureDefinition> STRUCTURES = List.of(
    StructureDefinition.builder("product")
      .string("name")
      .list("features")
      .build()
  );

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/ner_full.onnx"));
  }

  // ── extract(String) ──

  @Test
  void extract_nullText_returnsEmpty() {
    try (var extractor = SchemaExtractor.load(MODEL_DIR, STRUCTURES)) {
      Map<String, List<StructureInstance>> result = extractor.extract(
        (String) null
      );
      assertThat(result).isEmpty();
    }
  }

  @Test
  void extract_emptyText_returnsEmpty() {
    try (var extractor = SchemaExtractor.load(MODEL_DIR, STRUCTURES)) {
      Map<String, List<StructureInstance>> result = extractor.extract("");
      assertThat(result).isEmpty();
    }
  }

  @Test
  void extract_blankText_returnsEmpty() {
    try (var extractor = SchemaExtractor.load(MODEL_DIR, STRUCTURES)) {
      Map<String, List<StructureInstance>> result = extractor.extract("   ");
      assertThat(result).isEmpty();
    }
  }

  // ── extract(String, List<StructureDefinition>, float) ──

  @Test
  void extractWithOverride_blankText_returnsEmpty() {
    try (var extractor = SchemaExtractor.load(MODEL_DIR, STRUCTURES)) {
      var override = List.of(
        StructureDefinition.builder("contact").string("name").build()
      );
      Map<String, List<StructureInstance>> result = extractor.extract(
        "  ",
        override,
        0.5f
      );
      assertThat(result).isEmpty();
    }
  }

  // ── extractBatch ──

  @Test
  void extractBatch_nullList_returnsEmpty() {
    try (var extractor = SchemaExtractor.load(MODEL_DIR, STRUCTURES)) {
      List<Map<String, List<StructureInstance>>> result =
        extractor.extractBatch(null);
      assertThat(result).isEmpty();
    }
  }

  @Test
  void extractBatch_allBlankTexts_returnsEmptyMaps() {
    try (var extractor = SchemaExtractor.load(MODEL_DIR, STRUCTURES)) {
      var texts = List.of("", " ", "  \t  ");
      List<Map<String, List<StructureInstance>>> result =
        extractor.extractBatch(texts);
      assertThat(result).hasSize(3);
      assertThat(result).allSatisfy(m -> assertThat(m).isEmpty());
    }
  }

  @Test
  void extractBatch_mixedNullAndNormal_returnsCorrectShape() {
    try (var extractor = SchemaExtractor.load(MODEL_DIR, STRUCTURES)) {
      var texts = new ArrayList<String>();
      texts.add(null);
      texts.add("  ");
      texts.add(
        "The MacBook Pro features M3 chip, 16GB RAM, and 512GB storage."
      );

      var result = extractor.extractBatch(texts, 0.3f);

      assertThat(result).hasSize(3);
      assertThat(result.get(0)).isEmpty();
      assertThat(result.get(1)).isEmpty();
      assertThat(result.get(2)).containsKey("product");
    }
  }
}
