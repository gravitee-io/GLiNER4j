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

import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.tokenizer.WhitespaceTokenSplitter;
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
  void whitespaceTokenSplitterBasic() {
    var splitter = new WhitespaceTokenSplitter();
    var tokens = splitter.tokenize("Hello world");
    assertThat(tokens).hasSize(2);
    assertThat(tokens.get(0).text()).isEqualTo("Hello");
    assertThat(tokens.get(0).start()).isZero();
    assertThat(tokens.get(0).end()).isEqualTo(5);
    assertThat(tokens.get(1).text()).isEqualTo("world");
    assertThat(tokens.get(1).start()).isEqualTo(6);
    assertThat(tokens.get(1).end()).isEqualTo(11);
  }

  @Test
  void whitespaceTokenSplitterWithPunctuation() {
    var splitter = new WhitespaceTokenSplitter();
    var tokens = splitter.tokenize("John works at Google.");
    assertThat(tokens).hasSize(5);
    assertThat(tokens.get(0).text()).isEqualTo("John");
    assertThat(tokens.get(3).text()).isEqualTo("Google");
    assertThat(tokens.get(4).text()).isEqualTo(".");
  }

  @Test
  void whitespaceTokenSplitterEmptyInput() {
    var splitter = new WhitespaceTokenSplitter();
    var tokens = splitter.tokenize("");
    assertThat(tokens).isEmpty();
  }

  @Test
  void whitespaceTokenSplitterHyphenatedWords() {
    var splitter = new WhitespaceTokenSplitter();
    var tokens = splitter.tokenize("well-known state-of-the-art");
    assertThat(tokens).hasSize(2);
    assertThat(tokens.get(0).text()).isEqualTo("well-known");
    assertThat(tokens.get(1).text()).isEqualTo("state-of-the-art");
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
