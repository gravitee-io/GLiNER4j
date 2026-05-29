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
import java.util.Objects;

/**
 * A single field of a {@link StructureDefinition}.
 *
 * @param name field name (becomes a JSON key in the result)
 * @param type cardinality — STRING for single value, LIST for array
 * @param description optional description token added to the prompt; may be null or blank
 * @param choices optional allowed values; non-matching extracted values are dropped. Empty means no restriction.
 */
public record StructureField(
  String name,
  FieldType type,
  String description,
  List<String> choices
) {
  public StructureField {
    Objects.requireNonNull(name, "name");
    Objects.requireNonNull(type, "type");
    choices = choices == null ? List.of() : choices;
  }
}
