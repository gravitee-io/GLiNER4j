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

import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import java.util.List;
import java.util.Map;

/**
 * Per-family relation-extraction implementation. The public
 * {@link io.gravitee.lab.gliner4j.extractor.RelationExtractor} facade selects an implementation
 * at load time from the bundle's {@code architecture} and delegates to it; the facade applies the
 * default threshold, the strategy receives an explicit one.
 */
public interface RelationStrategy extends AutoCloseable {
  /** Extracts relations using the load-time relation schema. */
  Map<String, List<RelationInstance>> extract(String text, float threshold);

  /** Extracts relations using per-call relation definitions (overriding the load-time schema). */
  Map<String, List<RelationInstance>> extract(
    String text,
    List<RelationDefinition> relations,
    float threshold
  );

  /** Extracts relations from multiple texts in one batched pass, results in input order. */
  List<Map<String, List<RelationInstance>>> extractBatch(
    List<String> texts,
    float threshold
  );

  @Override
  void close();
}
