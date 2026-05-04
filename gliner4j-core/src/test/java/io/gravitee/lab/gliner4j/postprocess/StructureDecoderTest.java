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
package io.gravitee.lab.gliner4j.postprocess;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.lab.gliner4j.schema.FieldType;
import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import io.gravitee.lab.gliner4j.schema.StructureInstance;
import io.gravitee.lab.gliner4j.schema.StructureValue.ListValue;
import io.gravitee.lab.gliner4j.schema.StructureValue.StringValue;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link StructureDecoder}. Synthetic span-score arrays — no ONNX needed.
 *
 * <p>Test scoring layout: {@code spanScores[count][numFields][textLen][maxWidth]}
 * where each entry is the score for the field f at start-word=start with width w (inclusive end = start + w).
 */
class StructureDecoderTest {

  // 4 words: ["The", "MacBook", "Pro", "$1999"]
  private static final int[] WORD_STARTS = { 0, 4, 12, 16 };
  private static final int[] WORD_ENDS = { 3, 11, 15, 21 };
  private static final String TEXT = "The MacBook Pro $1999";
  private static final int MAX_WIDTH = 3;

  @Test
  void stringField_returnsHighestScoringSpan() {
    var product = StructureDefinition.builder("product").string("name").build();

    // 1 instance, 1 field, 4 words, maxWidth 3
    var scores = zeros(1, 1, 4, MAX_WIDTH);
    // High score for "MacBook Pro" (start=1, width=2 → words 1-2)
    scores[0][0][1][1] = 0.9f;
    // Lower score for "MacBook" alone (start=1, width=1 → word 1)
    scores[0][0][1][0] = 0.6f;

    var decoder = new StructureDecoder();
    List<StructureInstance> instances = decoder.decode(
      scores,
      1, // predCount
      product,
      WORD_STARTS,
      WORD_ENDS,
      TEXT,
      4, // textLen
      0.5f // threshold
    );

    assertThat(instances).hasSize(1);
    var name = (StringValue) instances.get(0).fields().get("name");
    assertThat(name.text()).isEqualTo("MacBook Pro");
    assertThat(name.confidence()).isEqualTo(0.9f);
  }

  @Test
  void listField_returnsAllAboveThreshold() {
    var product = StructureDefinition
      .builder("product")
      .list("features")
      .build();

    var scores = zeros(1, 1, 4, MAX_WIDTH);
    // Three non-overlapping high-scoring single-word spans
    scores[0][0][0][0] = 0.7f; // "The"
    scores[0][0][1][0] = 0.8f; // "MacBook"
    scores[0][0][3][0] = 0.6f; // "$1999"
    scores[0][0][2][0] = 0.3f; // "Pro" — below threshold

    var decoder = new StructureDecoder();
    var instances = decoder.decode(
      scores,
      1,
      product,
      WORD_STARTS,
      WORD_ENDS,
      TEXT,
      4,
      0.5f
    );

    var features = (ListValue) instances.get(0).fields().get("features");
    assertThat(features.items()).hasSize(3);
    assertThat(features.items().stream().map(StringValue::text).toList())
      .containsExactly("The", "MacBook", "$1999");
  }

  @Test
  void choiceField_dropsValuesNotInAllowedSet() {
    // 4 words: ["food", "transport", "shopping", "other"]
    int[] starts = { 0, 5, 15, 24 };
    int[] ends = { 4, 14, 23, 29 };
    String text = "food transport shopping other";

    var transaction = StructureDefinition
      .builder("transaction")
      .choice(
        "category",
        FieldType.STRING,
        List.of("food", "transport", "shopping")
      )
      .build();

    var scores = zeros(1, 1, 4, MAX_WIDTH);
    // "other" has the highest score, but is not in the allowed choices
    scores[0][0][3][0] = 0.95f;
    scores[0][0][1][0] = 0.85f; // "transport"

    var decoder = new StructureDecoder();
    var instances = decoder.decode(
      scores,
      1,
      transaction,
      starts,
      ends,
      text,
      4,
      0.5f
    );

    var category = (StringValue) instances.get(0).fields().get("category");
    assertThat(category.text()).isEqualTo("transport");
  }

  @Test
  void multipleInstances_producesOnePerCount() {
    var transaction = StructureDefinition
      .builder("transaction")
      .string("merchant")
      .build();

    // 3 words: ["Starbucks", "Uber", "Amazon"], 3 instances expected
    int[] starts = { 0, 10, 15 };
    int[] ends = { 9, 14, 21 };
    String text = "Starbucks Uber Amazon";

    var scores = zeros(3, 1, 3, MAX_WIDTH);
    scores[0][0][0][0] = 0.9f; // instance 0 → Starbucks
    scores[1][0][1][0] = 0.85f; // instance 1 → Uber
    scores[2][0][2][0] = 0.88f; // instance 2 → Amazon

    var decoder = new StructureDecoder();
    var instances = decoder.decode(
      scores,
      3, // predCount
      transaction,
      starts,
      ends,
      text,
      3,
      0.5f
    );

    assertThat(instances).hasSize(3);
    var merchants = instances
      .stream()
      .map(i -> ((StringValue) i.fields().get("merchant")).text())
      .toList();
    assertThat(merchants).containsExactly("Starbucks", "Uber", "Amazon");
  }

  @Test
  void zeroCount_returnsEmptyList() {
    var product = StructureDefinition.builder("product").string("name").build();
    var scores = zeros(1, 1, 4, MAX_WIDTH);

    var decoder = new StructureDecoder();
    var instances = decoder.decode(
      scores,
      0, // predCount = 0
      product,
      WORD_STARTS,
      WORD_ENDS,
      TEXT,
      4,
      0.5f
    );

    assertThat(instances).isEmpty();
  }

  private static float[][][][] zeros(int a, int b, int c, int d) {
    return new float[a][b][c][d];
  }
}
