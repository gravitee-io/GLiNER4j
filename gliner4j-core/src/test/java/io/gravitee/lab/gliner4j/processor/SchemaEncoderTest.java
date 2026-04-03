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

import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import java.util.List;
import org.junit.jupiter.api.Test;

class SchemaEncoderTest {

  @Test
  void schemaTokensWithoutDescriptions() {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization")
    );
    var encoder = new SchemaEncoder(entities);

    assertThat(encoder.getSchemaTokens())
      .containsExactly(
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
    assertThat(encoder.getFieldNames())
      .containsExactly("person", "organization");
    assertThat(encoder.getNumFields()).isEqualTo(2);
  }

  @Test
  void schemaTokensWithDescriptions() {
    var entities = List.of(
      new EntityDefinition("person", "Names of individuals"),
      new EntityDefinition("org", "Organization names")
    );
    var encoder = new SchemaEncoder(entities);

    assertThat(encoder.getSchemaTokens())
      .containsExactly(
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
  void schemaTokensWithMixedDescriptions() {
    var entities = List.of(
      new EntityDefinition("person", "Names of individuals"),
      new EntityDefinition("org")
    );
    var encoder = new SchemaEncoder(entities);

    // Only person has a description; org has no description token
    assertThat(encoder.getSchemaTokens())
      .containsExactly(
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
  void schemaTokensWithBlankDescriptionTreatedAsNoDescription() {
    var entities = List.of(
      new EntityDefinition("person", "   "),
      new EntityDefinition("org", "")
    );
    var encoder = new SchemaEncoder(entities);

    // Blank/empty descriptions should produce the same output as no descriptions
    assertThat(encoder.getSchemaTokens())
      .containsExactly(
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
}
