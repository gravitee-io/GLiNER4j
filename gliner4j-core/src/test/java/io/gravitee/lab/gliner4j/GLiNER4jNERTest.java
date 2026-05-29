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

import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import org.junit.jupiter.api.Test;

class GLiNER4jNERTest {

  @Test
  void entityDefinitionWithDefaults() {
    var entity = new EntityDefinition("person");
    assertThat(entity.name()).isEqualTo("person");
    assertThat(entity.description()).isEmpty();
  }

  @Test
  void entityDefinitionWithDescription() {
    var entity = new EntityDefinition(
      "organization",
      "A company or institution"
    );
    assertThat(entity.name()).isEqualTo("organization");
    assertThat(entity.description()).isEqualTo("A company or institution");
  }

  @Test
  void entitySpanCreation() {
    var span = new EntitySpan("person", "John Doe", 0.95f, 0, 8);
    assertThat(span.type()).isEqualTo("person");
    assertThat(span.text()).isEqualTo("John Doe");
    assertThat(span.confidence()).isEqualTo(0.95f);
    assertThat(span.start()).isZero();
    assertThat(span.end()).isEqualTo(8);
  }

  @Test
  void textEncoderBasic() {
    var enc = new TextEncoder("Hello world");
    assertThat(enc.getTextLen()).isEqualTo(2);
    assertThat(enc.getWords()).containsExactly("Hello", "world");
    assertThat(enc.getWordStartChars()).containsExactly(0, 6);
    assertThat(enc.getWordEndChars()).containsExactly(5, 11);
  }

  @Test
  void textEncoderWithPunctuation() {
    var enc = new TextEncoder("John works at Google.");
    assertThat(enc.getTextLen()).isEqualTo(5);
    assertThat(enc.getWords()).containsExactly(
      "John",
      "works",
      "at",
      "Google",
      "."
    );
  }

  @Test
  void textEncoderEmptyInput() {
    var enc = new TextEncoder("");
    assertThat(enc.getTextLen()).isZero();
    assertThat(enc.getWords()).isEmpty();
    assertThat(enc.getWordStartChars()).isEmpty();
    assertThat(enc.getWordEndChars()).isEmpty();
  }

  @Test
  void textEncoderHyphenatedWords() {
    var enc = new TextEncoder("well-known state-of-the-art");
    assertThat(enc.getTextLen()).isEqualTo(2);
    assertThat(enc.getWords()).containsExactly(
      "well-known",
      "state-of-the-art"
    );
  }

  // --- Classification records ---

  @Test
  void classificationLabelWithDefaults() {
    var label = new ClassificationLabel("positive");
    assertThat(label.name()).isEqualTo("positive");
    assertThat(label.description()).isEmpty();
  }

  @Test
  void classificationLabelWithDescription() {
    var label = new ClassificationLabel(
      "positive",
      "Expresses positive sentiment"
    );
    assertThat(label.name()).isEqualTo("positive");
    assertThat(label.description()).isEqualTo("Expresses positive sentiment");
  }

  @Test
  void classificationResultCreation() {
    var result = new ClassificationResult("positive", 0.87f);
    assertThat(result.label()).isEqualTo("positive");
    assertThat(result.confidence()).isEqualTo(0.87f);
  }

  @Test
  void configDefaults() {
    var config = GLiNER4jConfig.builder()
      .modelPath(java.nio.file.Path.of("/tmp/model"))
      .build();
    assertThat(config.getDefaultThreshold()).isEqualTo(0.5f);
    assertThat(config.getMaxWidth()).isEqualTo(12);
  }
}
