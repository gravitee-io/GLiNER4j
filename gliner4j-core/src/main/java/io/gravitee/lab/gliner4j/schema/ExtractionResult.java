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
import java.util.Map;

/**
 * Combined output of a single forward pass over a {@link Schema}.
 *
 * <p>All requested entity types and relation types are present as keys, even when no spans are
 * detected (matching the documented GLiNER2 contract). The {@code classifications} list contains
 * labels above threshold sorted by confidence descending, or is empty when no classification block
 * was requested.
 *
 * @param entities map of entity type name to detected spans (empty list when type was requested but no spans found)
 * @param relations map of relation type name to detected instances (empty list when type was requested but no instances found)
 * @param classifications labels above threshold, sorted by confidence descending
 */
public record ExtractionResult(
  Map<String, List<EntitySpan>> entities,
  Map<String, List<RelationInstance>> relations,
  List<ClassificationResult> classifications
) {}
