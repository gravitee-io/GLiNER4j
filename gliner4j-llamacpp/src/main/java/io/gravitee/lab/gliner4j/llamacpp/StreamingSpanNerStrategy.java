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
import io.gravitee.lab.gliner4j.schema.NerOutput;
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
 *
 * <p>{@link NerOutput#inputTokens()}: every call runs a fresh stateless session, so the decoder
 * evaluates, in one pass on an empty KV cache, the label prompt
 * {@code label1 <<LABEL>> label2 <<LABEL>> ... <<SEP>>} followed by the text's word subtokens
 * (no BOS/EOS). The count is that sequence's length, the tokens actually decoded for the text.
 * The prompt's tokenization is cached across calls, but its tokens are decoded again for every
 * text (nothing is reused from the KV cache across texts), so each text counts it. A null or
 * blank text, a text with no words, or an empty label list decodes nothing and reports
 * {@code 0}.
 */
public final class StreamingSpanNerStrategy implements NerStrategy {

  private final StreamingSpanBackend engine;
  private final Runnable onClose;
  private final List<String> labels;
  private final GLiNER4jTelemetry telemetry = new GLiNER4jTelemetry("extract");

  StreamingSpanNerStrategy(
    StreamingSpanEngine engine,
    List<EntityDefinition> entities
  ) {
    this(engine, entities, engine::close);
  }

  /** Over any backend; {@code onClose} releases it. The seam lets usage be tested without a model. */
  StreamingSpanNerStrategy(
    StreamingSpanBackend engine,
    List<EntityDefinition> entities,
    Runnable onClose
  ) {
    this.engine = engine;
    this.onClose = onClose;
    this.labels = entities.stream().map(EntityDefinition::name).toList();
  }

  @Override
  public Map<String, List<EntitySpan>> extract(String text, float threshold) {
    return extractWith(text, labels, threshold).spans();
  }

  @Override
  public Map<String, List<EntitySpan>> extract(
    String text,
    List<EntityDefinition> entities,
    float threshold
  ) {
    return extractWithUsage(text, entities, threshold).spans();
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
    for (var text : texts)
      out.add(extractWith(text, labels, threshold).spans());
    return out;
  }

  @Override
  public NerOutput extractWithUsage(String text, float threshold) {
    return extractWith(text, labels, threshold);
  }

  @Override
  public NerOutput extractWithUsage(
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
  public List<NerOutput> extractBatchWithUsage(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    var out = new ArrayList<NerOutput>(texts.size());
    for (var text : texts) out.add(extractWith(text, labels, threshold));
    return out;
  }

  private NerOutput extractWith(
    String text,
    List<String> activeLabels,
    float threshold
  ) {
    long start = System.nanoTime();
    if (text == null || text.isBlank() || activeLabels.isEmpty()) {
      telemetry.record(0.0, 1, 0);
      return NerOutput.empty();
    }
    List<EntitySpan> spans;
    int inputTokens;
    try (
      var session = new StreamingSpanSession(engine, "stateless", activeLabels)
    ) {
      spans = session.append(text, threshold);
      // A fresh session decodes prompt + words once, from position 0: its cache length is
      // exactly the number of tokens evaluated (0 when the text has no words).
      inputTokens = session.cachedTokens();
    }
    telemetry.record(
      (System.nanoTime() - start) / 1_000_000.0,
      1,
      spans.size()
    );
    return new NerOutput(GlinerNerSupport.groupByType(spans), inputTokens);
  }

  @Override
  public void close() {
    onClose.run();
  }
}
