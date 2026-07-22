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
import io.gravitee.lab.gliner4j.schema.FieldType;
import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import io.gravitee.lab.gliner4j.schema.StructureInstance;
import io.gravitee.lab.gliner4j.schema.StructureValue;
import io.gravitee.lab.gliner4j.schema.StructureValue.ListValue;
import io.gravitee.lab.gliner4j.schema.StructureValue.StringValue;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Acceptance tests for schema/structure extraction. Examples are ported verbatim
 * from the upstream GLiNER2 tutorial (tutorial/3-json_extraction.md).
 *
 * <p>Skipped automatically when the model directory is not present.
 */
@EnabledIf("modelDirExists")
class GLiNER4jSchemaExtractorIntegrationTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/ner_full.onnx"));
  }

  @Test
  void extractsSimpleProduct() {
    var product = StructureDefinition.builder("product")
      .string("name")
      .list("price")
      .list("features")
      .build();

    try (var extractor = SchemaExtractor.load(MODEL_DIR, List.of(product))) {
      Map<String, List<StructureInstance>> result = extractor.extract(
        "The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage."
      );

      assertThat(result).containsKey("product");
      assertThat(result.get("product")).hasSize(1);

      var instance = result.get("product").get(0);
      assertThat(instance.fields()).containsKey("name");
      assertThat(asString(instance, "name")).isEqualTo("MacBook Pro");

      assertThat(asListTexts(instance, "price")).contains("$1999");
      assertThat(asListTexts(instance, "features")).contains(
        "M3 chip",
        "16GB RAM",
        "512GB storage"
      );
    }
  }

  @Test
  void extractsWithChoices() {
    var reservation = StructureDefinition.builder("reservation")
      .string("restaurant", "Restaurant name")
      .string("date")
      .string("time")
      .choice(
        "seating",
        FieldType.STRING,
        List.of("indoor", "outdoor", "bar"),
        "Seating preference"
      )
      .choice(
        "dietary",
        FieldType.LIST,
        List.of("vegetarian", "vegan", "gluten-free", "none"),
        "Dietary restrictions"
      )
      .build();

    try (
      var extractor = SchemaExtractor.load(MODEL_DIR, List.of(reservation))
    ) {
      var result = extractor.extract(
        "Reservation at Le Bernardin for 4 people on March 15th at 7:30 PM. " +
          "We'd prefer outdoor seating. Two guests are vegetarian and one is gluten-free."
      );

      assertThat(result.get("reservation")).hasSize(1);
      var instance = result.get("reservation").get(0);

      assertThat(asString(instance, "seating")).isEqualTo("outdoor");
      assertThat(asListTexts(instance, "dietary"))
        .contains("vegetarian", "gluten-free")
        .doesNotContain("vegan", "none");
    }
  }

  @Test
  void extractsMultipleInstances() {
    var transaction = StructureDefinition.builder("transaction")
      .string("date")
      .string("merchant")
      .string("amount")
      .choice(
        "category",
        FieldType.STRING,
        List.of("food", "transport", "shopping", "utilities")
      )
      .build();

    try (
      var extractor = SchemaExtractor.load(MODEL_DIR, List.of(transaction))
    ) {
      var result = extractor.extract(
        """
        Recent transactions:
        - Jan 5: Starbucks $5.50 (food)
        - Jan 5: Uber $23.00 (transport)
        - Jan 6: Amazon $156.99 (shopping)
        """
      );

      assertThat(result.get("transaction")).hasSize(3);
      var merchants = result
        .get("transaction")
        .stream()
        .map(i -> asString(i, "merchant"))
        .toList();
      assertThat(merchants).contains("Starbucks", "Uber", "Amazon");

      var categories = result
        .get("transaction")
        .stream()
        .map(i -> asString(i, "category"))
        .toList();
      assertThat(categories).contains("food", "transport", "shopping");
    }
  }

  @Test
  void extractsAcrossMultipleStructures() {
    var patient = StructureDefinition.builder("patient")
      .string("name", "Patient full name")
      .string("age", "Patient age")
      .list("symptoms", "Reported symptoms")
      .build();
    var prescription = StructureDefinition.builder("prescription")
      .string("medication", "Drug name")
      .string("dosage", "Dosage amount")
      .string("frequency", "How often to take")
      .build();

    try (
      var extractor = SchemaExtractor.load(
        MODEL_DIR,
        List.of(patient, prescription)
      )
    ) {
      var result = extractor.extract(
        """
        Patient: Sarah Johnson, 34, presented with chest pain.
        Prescribed: Lisinopril 10mg daily, Metoprolol 25mg twice daily.
        Follow-up scheduled for next Tuesday.
        """
      );

      assertThat(result).containsKeys("patient", "prescription");

      assertThat(result.get("patient")).hasSize(1);
      var patientInstance = result.get("patient").get(0);
      assertThat(asString(patientInstance, "name")).isEqualTo("Sarah Johnson");
      assertThat(asString(patientInstance, "age")).isEqualTo("34");
      assertThat(asListTexts(patientInstance, "symptoms")).contains(
        "chest pain"
      );

      assertThat(result.get("prescription")).hasSize(2);
      var medications = result
        .get("prescription")
        .stream()
        .map(i -> asString(i, "medication"))
        .toList();
      assertThat(medications).contains("Lisinopril", "Metoprolol");
    }
  }

  @Test
  void respectsCustomThreshold() {
    var product = StructureDefinition.builder("product")
      .string("name")
      .list("features")
      .build();

    try (var extractor = SchemaExtractor.load(MODEL_DIR, List.of(product))) {
      var lowThreshold = extractor.extract(
        "The MacBook Pro features M3 chip, 16GB RAM.",
        0.1f
      );
      var highThreshold = extractor.extract(
        "The MacBook Pro features M3 chip, 16GB RAM.",
        0.95f
      );

      // Low threshold should pick up at least as many features as high threshold
      var lowFeatures = lowThreshold.get("product").get(0);
      assertThat(asListTexts(lowFeatures, "features")).isNotEmpty();
      // High threshold may yield no instances or fewer features — assert it's a subset
      if (
        highThreshold.containsKey("product") &&
        !highThreshold.get("product").isEmpty()
      ) {
        var highFeatures = highThreshold.get("product").get(0);
        assertThat(
          asListTexts(highFeatures, "features").size()
        ).isLessThanOrEqualTo(asListTexts(lowFeatures, "features").size());
      }
    }
  }

  @Test
  void perCallStructureOverride() {
    var defaultStructure = StructureDefinition.builder("product")
      .string("name")
      .build();

    try (
      var extractor = SchemaExtractor.load(MODEL_DIR, List.of(defaultStructure))
    ) {
      var override = StructureDefinition.builder("contact")
        .string("name", "Contact name")
        .string("email", "Email address")
        .build();

      var result = extractor.extract(
        "Contact: John Smith, email john@example.com",
        List.of(override),
        0.3f
      );

      assertThat(result).containsKey("contact").doesNotContainKey("product");
      var contact = result.get("contact").get(0);
      assertThat(asString(contact, "name")).isEqualTo("John Smith");
      assertThat(asString(contact, "email")).isEqualTo("john@example.com");
    }
  }

  // ── helpers ───────────────────────────────────────────────────────────

  private static String asString(StructureInstance instance, String fieldName) {
    StructureValue value = instance.fields().get(fieldName);
    if (value instanceof StringValue sv) {
      return sv.text();
    }
    if (value instanceof ListValue lv && !lv.items().isEmpty()) {
      return lv.items().get(0).text();
    }
    return null;
  }

  private static List<String> asListTexts(
    StructureInstance instance,
    String fieldName
  ) {
    StructureValue value = instance.fields().get(fieldName);
    if (value instanceof ListValue lv) {
      return lv.items().stream().map(StringValue::text).toList();
    }
    if (value instanceof StringValue sv) {
      return List.of(sv.text());
    }
    return List.of();
  }
}
