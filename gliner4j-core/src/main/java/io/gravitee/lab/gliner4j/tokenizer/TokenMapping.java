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
package io.gravitee.lab.gliner4j.tokenizer;

/**
 * Maps a subword token position to its origin segment and index.
 *
 * @param type the segment this token belongs to
 * @param origIdx original word index (for TEXT) or entity index (for SCHEMA)
 * @param schemaIdx entity index when type is SCHEMA, -1 otherwise
 */
public record TokenMapping(SegmentType type, int origIdx, int schemaIdx) {
  public enum SegmentType {
    SCHEMA,
    TEXT,
    SEP,
  }
}
