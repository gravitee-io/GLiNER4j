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
 * Extracted value for a {@link StructureField}, carrying confidence and character offsets.
 * Two shapes: {@link StringValue} for STRING fields, {@link ListValue} for LIST fields.
 */
public sealed interface StructureValue
  permits StructureValue.StringValue, StructureValue.ListValue {
  /**
   * Single extracted value with confidence and character offsets in the source text.
   *
   * @param text the extracted text
   * @param confidence model score in [0, 1]
   * @param start inclusive character offset in the source text
   * @param end exclusive character offset in the source text
   */
  record StringValue(
    String text,
    float confidence,
    int start,
    int end
  ) implements StructureValue {}

  /**
   * Multiple extracted values for a LIST field.
   *
   * @param items extracted values, ordered by character position
   */
  record ListValue(List<StringValue> items) implements StructureValue {
    public ListValue {
      items = List.copyOf(items);
    }
  }
}
