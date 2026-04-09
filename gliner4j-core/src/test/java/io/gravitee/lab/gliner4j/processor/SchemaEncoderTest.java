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
package io.gravitee.lab.gliner4j.processor;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

class SchemaEncoderTest {

  @Test
  void nerSchemaTokensWithoutDescriptions() {
    var encoder = new SchemaEncoder(
      "entities",
      "[E]",
      List.of("person", "organization"),
      List.of("", "")
    );

    assertThat(encoder.getSchemaTokens()).containsExactly(
      "(",
      "[P]",
      "entities",
      "(",
      "[E]",
      "person",
      "[E]",
      "organization",
      ")",
      ")"
    );
    assertThat(encoder.getFieldNames()).containsExactly(
      "person",
      "organization"
    );
    assertThat(encoder.getNumFields()).isEqualTo(2);
    assertThat(encoder.getSpecialToken()).isEqualTo("[E]");
  }

  @Test
  void nerSchemaTokensWithDescriptions() {
    var encoder = new SchemaEncoder(
      "entities",
      "[E]",
      List.of("person", "org"),
      List.of("Names of individuals", "Organization names")
    );

    assertThat(encoder.getSchemaTokens()).containsExactly(
      "(",
      "[P]",
      "entities",
      "[DESCRIPTION]",
      "person:",
      "Names of individuals",
      "[DESCRIPTION]",
      "org:",
      "Organization names",
      "(",
      "[E]",
      "person",
      "[E]",
      "org",
      ")",
      ")"
    );
  }

  @Test
  void nerSchemaTokensWithMixedDescriptions() {
    var encoder = new SchemaEncoder(
      "entities",
      "[E]",
      List.of("person", "org"),
      List.of("Names of individuals", "")
    );

    assertThat(encoder.getSchemaTokens()).containsExactly(
      "(",
      "[P]",
      "entities",
      "[DESCRIPTION]",
      "person:",
      "Names of individuals",
      "(",
      "[E]",
      "person",
      "[E]",
      "org",
      ")",
      ")"
    );
  }

  @Test
  void nerSchemaTokensWithBlankDescriptionTreatedAsNoDescription() {
    var encoder = new SchemaEncoder(
      "entities",
      "[E]",
      List.of("person", "org"),
      List.of("   ", "")
    );

    assertThat(encoder.getSchemaTokens()).containsExactly(
      "(",
      "[P]",
      "entities",
      "(",
      "[E]",
      "person",
      "[E]",
      "org",
      ")",
      ")"
    );
  }

  @Test
  void classificationSchemaTokensWithoutDescriptions() {
    var encoder = new SchemaEncoder(
      "classify",
      "[L]",
      List.of("positive", "negative"),
      List.of("", "")
    );

    assertThat(encoder.getSchemaTokens()).containsExactly(
      "(",
      "[P]",
      "classify",
      "(",
      "[L]",
      "positive",
      "[L]",
      "negative",
      ")",
      ")"
    );
    assertThat(encoder.getFieldNames()).containsExactly("positive", "negative");
    assertThat(encoder.getNumFields()).isEqualTo(2);
    assertThat(encoder.getSpecialToken()).isEqualTo("[L]");
  }

  @Test
  void classificationSchemaTokensWithDescriptions() {
    var encoder = new SchemaEncoder(
      "classify",
      "[L]",
      List.of("positive", "negative"),
      List.of("Expresses positive sentiment", "Expresses negative sentiment")
    );

    assertThat(encoder.getSchemaTokens()).containsExactly(
      "(",
      "[P]",
      "classify",
      "[DESCRIPTION]",
      "positive:",
      "Expresses positive sentiment",
      "[DESCRIPTION]",
      "negative:",
      "Expresses negative sentiment",
      "(",
      "[L]",
      "positive",
      "[L]",
      "negative",
      ")",
      ")"
    );
  }

  @Test
  void classificationSchemaTokensWithMixedDescriptions() {
    var encoder = new SchemaEncoder(
      "classify",
      "[L]",
      List.of("positive", "negative"),
      List.of("Expresses positive sentiment", "")
    );

    assertThat(encoder.getSchemaTokens()).containsExactly(
      "(",
      "[P]",
      "classify",
      "[DESCRIPTION]",
      "positive:",
      "Expresses positive sentiment",
      "(",
      "[L]",
      "positive",
      "[L]",
      "negative",
      ")",
      ")"
    );
  }

  @Test
  void classificationSchemaTokensWithBlankDescriptionTreatedAsNoDescription() {
    var encoder = new SchemaEncoder(
      "classify",
      "[L]",
      List.of("positive", "negative"),
      List.of("   ", "")
    );

    assertThat(encoder.getSchemaTokens()).containsExactly(
      "(",
      "[P]",
      "classify",
      "(",
      "[L]",
      "positive",
      "[L]",
      "negative",
      ")",
      ")"
    );
  }
}
