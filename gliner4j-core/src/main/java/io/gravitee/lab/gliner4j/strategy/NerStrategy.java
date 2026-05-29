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
package io.gravitee.lab.gliner4j.strategy;

import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.util.List;
import java.util.Map;

/**
 * Per-family NER implementation: owns {@code assemble → forward → decode} for one model family.
 *
 * <p>The public {@link io.gravitee.lab.gliner4j.GLiNER4jNER} facade selects an implementation at
 * load time from the bundle's {@code architecture} (via
 * {@link io.gravitee.lab.gliner4j.arch.ModelArchitecture}) and delegates to it. The facade applies
 * the default threshold and null-guards; the strategy receives an explicit threshold.
 */
public interface NerStrategy extends AutoCloseable {
  /**
   * Extracts entities using the load-time entity schema.
   *
   * @param text the input text
   * @param threshold minimum confidence (0..1) for span inclusion
   * @return map of entity type to detected spans
   */
  Map<String, List<EntitySpan>> extract(String text, float threshold);

  /**
   * Extracts entities using per-call entity definitions (overriding the load-time schema).
   *
   * @param text the input text
   * @param entities the entity types to extract for this call
   * @param threshold minimum confidence (0..1) for span inclusion
   * @return map of entity type to detected spans
   */
  Map<String, List<EntitySpan>> extract(
    String text,
    List<EntityDefinition> entities,
    float threshold
  );

  /**
   * Extracts entities from multiple texts in one batched pass.
   *
   * @param texts the input texts
   * @param threshold minimum confidence (0..1) for span inclusion
   * @return per-text results in input order
   */
  List<Map<String, List<EntitySpan>>> extractBatch(
    List<String> texts,
    float threshold
  );

  @Override
  void close();
}
