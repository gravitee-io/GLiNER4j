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
package io.gravitee.lab.gliner4j.schema;

import java.util.List;

/**
 * Defines a relation type to extract.
 *
 * <p>Each relation has a name (the relation type, e.g. "works_for"), an optional description,
 * and an ordered list of field names. Field names default to {@code ["head", "tail"]} —
 * directional binary relations matching the GLiNER2 reference behavior.
 *
 * @param name the relation type name (e.g. "works_for", "lives_in")
 * @param description optional description used as additional context for the model
 * @param fields ordered field names (typically {@code ["head", "tail"]})
 */
public record RelationDefinition(
  String name,
  String description,
  List<String> fields
) {
  private static final List<String> DEFAULT_FIELDS = List.of("head", "tail");

  public RelationDefinition {
    if (fields.isEmpty()) {
      throw new IllegalArgumentException(
        "RelationDefinition '" + name + "' must declare at least one field"
      );
    }
  }

  public RelationDefinition(String name) {
    this(name, "", DEFAULT_FIELDS);
  }

  public RelationDefinition(String name, String description) {
    this(name, description, DEFAULT_FIELDS);
  }
}
