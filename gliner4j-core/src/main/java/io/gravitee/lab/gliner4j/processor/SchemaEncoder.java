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

import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import java.util.ArrayList;
import java.util.List;
import lombok.Getter;

/**
 * Builds the schema token list from entity definitions.
 * Schema format: ["(", "[P]", "entities", "(", "[E]", "person", "[E]", "org", ")", ")"]
 * Immutable after construction — safe to share across threads.
 */
@Getter
public class SchemaEncoder {

  private final List<String> schemaTokens;
  private final List<String> fieldNames;
  private final int numFields;

  /**
   * Builds schema token list from entity definitions.
   *
   * @param entities the entity definitions to encode
   */
  public SchemaEncoder(List<EntityDefinition> entities) {
    this.fieldNames = entities.stream().map(EntityDefinition::name).toList();
    this.numFields = entities.size();
    this.schemaTokens = buildSchemaTokens(entities);
  }

  private static List<String> buildSchemaTokens(
    List<EntityDefinition> entities
  ) {
    var tokens = new ArrayList<String>();
    tokens.add("(");
    tokens.add("[P]");
    tokens.add("entities");
    tokens.add("(");
    for (var entity : entities) {
      tokens.add("[E]");
      tokens.add(entity.name());
    }
    tokens.add(")");
    tokens.add(")");
    return List.copyOf(tokens);
  }
}
