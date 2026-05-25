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
package io.gravitee.lab.gliner4j.demo.profile;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import io.gravitee.lab.gliner4j.schema.FieldType;
import io.gravitee.lab.gliner4j.schema.StructureField;
import java.util.List;

public record StructureFieldDef(String name, FieldType type, String description, List<String> choices) {
  @JsonCreator
  public StructureFieldDef(
    @JsonProperty("name") String name,
    @JsonProperty("type") FieldType type,
    @JsonProperty("description") String description,
    @JsonProperty("choices") List<String> choices
  ) {
    this.name = name;
    this.type = type == null ? FieldType.STRING : type;
    this.description = description;
    this.choices = choices == null ? List.of() : List.copyOf(choices);
  }

  public StructureField toField() {
    return new StructureField(name, type, description, choices);
  }
}
