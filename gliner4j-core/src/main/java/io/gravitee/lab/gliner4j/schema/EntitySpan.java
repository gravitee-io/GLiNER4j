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

/**
 * A detected entity span in the input text.
 *
 * @param type the entity label that matched
 * @param text the matched substring
 * @param confidence model confidence score (0..1)
 * @param start character start offset (inclusive) in the original text
 * @param end character end offset (exclusive) in the original text
 */
public record EntitySpan(
  String type,
  String text,
  float confidence,
  int start,
  int end
) {}
