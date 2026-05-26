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

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A detected relation instance with one {@link FieldSpan} per declared field.
 *
 * <p>For directional binary relations the fields map will contain {@code "head"} and {@code "tail"} keys.
 * The {@code confidence} is the minimum confidence across all fields — every field must clear the
 * threshold for the instance to be emitted.
 *
 * @param type the relation type name (matches {@link RelationDefinition#name()})
 * @param fields map of field name to detected span (insertion-ordered to preserve field order)
 * @param confidence overall instance confidence (minimum across fields)
 */
public record RelationInstance(
  String type,
  Map<String, FieldSpan> fields,
  float confidence
) {
  public RelationInstance {
    fields = Map.copyOf(new LinkedHashMap<>(fields));
  }

  /**
   * Convenience accessor for the conventional "head" field.
   *
   * @return the head span, or null if no field is named "head"
   */
  public FieldSpan head() {
    return fields.get("head");
  }

  /**
   * Convenience accessor for the conventional "tail" field.
   *
   * @return the tail span, or null if no field is named "tail"
   */
  public FieldSpan tail() {
    return fields.get("tail");
  }
}
