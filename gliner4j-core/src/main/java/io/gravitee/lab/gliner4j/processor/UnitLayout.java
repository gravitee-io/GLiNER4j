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

/**
 * Resolved positions of a single unit's special tokens within the assembled input sequence.
 *
 * <p>Computed once at assembly time so per-unit embedding extraction can index directly into
 * encoder hidden states without re-scanning the token stream.
 *
 * @param unit the original schema unit description
 * @param parentTokenPos absolute position of the {@code [P]} subword in the input sequence
 * @param childMarkerPositions absolute positions of each {@code [E]}/{@code [L]}/{@code [R]} subword (one per child)
 */
public record UnitLayout(
  SchemaUnit unit,
  int parentTokenPos,
  int[] childMarkerPositions
) {}
