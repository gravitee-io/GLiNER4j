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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * A structure (parent entity) the extractor should pull from text.
 * Each instance found in text becomes one {@link StructureInstance} keyed by {@link #name()}.
 *
 * <p>Built via {@link #builder(String)}:
 * <pre>{@code
 * var product = StructureDefinition.builder("product")
 *     .string("name", "Product name")
 *     .string("price")
 *     .list("features")
 *     .choice("category", FieldType.STRING, List.of("electronics", "software"))
 *     .build();
 * }</pre>
 */
public record StructureDefinition(String name, List<StructureField> fields) {
  public StructureDefinition {
    Objects.requireNonNull(name, "name");
    fields = List.copyOf(fields);
  }

  /**
   * Starts a builder for a structure with the given name.
   *
   * @param name the structure name (becomes a top-level JSON key in the result)
   * @return a new builder
   */
  public static Builder builder(String name) {
    return new Builder(name);
  }

  public static final class Builder {

    private final String name;
    private final List<StructureField> fields = new ArrayList<>();

    private Builder(String name) {
      this.name = Objects.requireNonNull(name, "name");
    }

    /** Adds a STRING field with no description. */
    public Builder string(String fieldName) {
      return field(new StructureField(fieldName, FieldType.STRING, null, null));
    }

    /** Adds a STRING field with a description token. */
    public Builder string(String fieldName, String description) {
      return field(
        new StructureField(fieldName, FieldType.STRING, description, null)
      );
    }

    /** Adds a LIST field with no description. */
    public Builder list(String fieldName) {
      return field(new StructureField(fieldName, FieldType.LIST, null, null));
    }

    /** Adds a LIST field with a description token. */
    public Builder list(String fieldName, String description) {
      return field(
        new StructureField(fieldName, FieldType.LIST, description, null)
      );
    }

    /** Adds a field constrained to one of the given choices. */
    public Builder choice(
      String fieldName,
      FieldType type,
      List<String> choices
    ) {
      return field(new StructureField(fieldName, type, null, choices));
    }

    /** Adds a field constrained to one of the given choices, with a description token. */
    public Builder choice(
      String fieldName,
      FieldType type,
      List<String> choices,
      String description
    ) {
      return field(new StructureField(fieldName, type, description, choices));
    }

    /** Adds a pre-built field. */
    public Builder field(StructureField field) {
      fields.add(Objects.requireNonNull(field, "field"));
      return this;
    }

    public StructureDefinition build() {
      if (fields.isEmpty()) {
        throw new IllegalStateException(
          "StructureDefinition '" + name + "' must have at least one field"
        );
      }
      return new StructureDefinition(name, fields);
    }
  }
}
