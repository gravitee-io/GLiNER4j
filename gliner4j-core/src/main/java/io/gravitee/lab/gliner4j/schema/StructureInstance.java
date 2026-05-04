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

import java.util.Map;

/**
 * A single instance of a {@link StructureDefinition} extracted from text.
 * Fields with no extracted value are absent from the map (use {@link Map#getOrDefault}).
 *
 * @param fields field-name → extracted value
 */
public record StructureInstance(Map<String, StructureValue> fields) {
  public StructureInstance {
    fields = Map.copyOf(fields);
  }
}
