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
package io.gravitee.lab.gliner4j.llamacpp;

import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.utils.GlinerNerSupport;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * {@link NerStrategy} over a {@link StreamingSpanEngine}: the stateless path of the
 * {@link io.gravitee.lab.gliner4j.GLiNER4jNER} facade for {@code gliner-streaming-span} bundles.
 * Entity descriptions are not part of this prompt format and are ignored.
 */
public final class StreamingSpanNerStrategy implements NerStrategy {

  private final StreamingSpanEngine engine;
  private final List<String> labels;
  private final GLiNER4jTelemetry telemetry = new GLiNER4jTelemetry("extract");

  StreamingSpanNerStrategy(
    StreamingSpanEngine engine,
    List<EntityDefinition> entities
  ) {
    this.engine = engine;
    this.labels = entities.stream().map(EntityDefinition::name).toList();
  }

  @Override
  public Map<String, List<EntitySpan>> extract(String text, float threshold) {
    return extractWith(text, labels, threshold);
  }

  @Override
  public Map<String, List<EntitySpan>> extract(
    String text,
    List<EntityDefinition> entities,
    float threshold
  ) {
    return extractWith(
      text,
      entities.stream().map(EntityDefinition::name).toList(),
      threshold
    );
  }

  @Override
  public List<Map<String, List<EntitySpan>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    var out = new ArrayList<Map<String, List<EntitySpan>>>(texts.size());
    for (var text : texts) out.add(extractWith(text, labels, threshold));
    return out;
  }

  private Map<String, List<EntitySpan>> extractWith(
    String text,
    List<String> activeLabels,
    float threshold
  ) {
    long start = System.nanoTime();
    if (text == null || text.isBlank() || activeLabels.isEmpty()) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }
    List<EntitySpan> spans;
    try (
      var session = new StreamingSpanSession(engine, "stateless", activeLabels)
    ) {
      spans = session.append(text, threshold);
    }
    telemetry.record(
      (System.nanoTime() - start) / 1_000_000.0,
      1,
      spans.size()
    );
    return GlinerNerSupport.groupByType(spans);
  }

  @Override
  public void close() {
    engine.close();
  }
}
