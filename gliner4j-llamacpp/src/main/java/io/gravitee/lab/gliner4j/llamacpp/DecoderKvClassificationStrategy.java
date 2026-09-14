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

import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.strategy.ClassificationStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import java.util.ArrayList;
import java.util.List;

/**
 * {@link ClassificationStrategy} over a {@link DecoderKvEngine}: the multi-label path of the
 * {@link io.gravitee.lab.gliner4j.GLiNER4jClassifier} facade for {@code gliclass-decoder-kv}
 * bundles. Label descriptions are not part of the decoder-kv prompt and are ignored.
 */
public final class DecoderKvClassificationStrategy
  implements ClassificationStrategy {

  private final DecoderKvEngine engine;
  private final List<String> labels;
  private final GLiNER4jTelemetry telemetry = new GLiNER4jTelemetry("classify");

  DecoderKvClassificationStrategy(
    DecoderKvEngine engine,
    List<ClassificationLabel> labels
  ) {
    this.engine = engine;
    this.labels = names(labels);
  }

  @Override
  public List<ClassificationResult> classify(String text, float threshold) {
    return classifyWith(text, labels, threshold);
  }

  @Override
  public List<ClassificationResult> classify(
    String text,
    List<ClassificationLabel> overrideLabels,
    float threshold
  ) {
    return classifyWith(text, names(overrideLabels), threshold);
  }

  @Override
  public List<List<ClassificationResult>> classifyBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    var out = new ArrayList<List<ClassificationResult>>(texts.size());
    for (var text : texts) {
      out.add(classifyWith(text, labels, threshold));
    }
    return out;
  }

  private List<ClassificationResult> classifyWith(
    String text,
    List<String> activeLabels,
    float threshold
  ) {
    long start = System.nanoTime();
    if (text == null || text.isBlank() || activeLabels.isEmpty()) {
      telemetry.record(0.0, 1, 0);
      return List.of();
    }
    var ids = engine.encodeText(text);
    if (ids.length == 0) {
      telemetry.record(0.0, 1, 0);
      return List.of();
    }
    var logits = engine.score(ids, engine.section(activeLabels));
    var results = DecoderKvScores.multiLabel(logits, activeLabels, threshold);
    telemetry.record(
      (System.nanoTime() - start) / 1_000_000.0,
      1,
      results.size()
    );
    return results;
  }

  private static List<String> names(List<ClassificationLabel> labels) {
    return labels.stream().map(ClassificationLabel::name).toList();
  }

  @Override
  public void close() {
    engine.close();
  }
}
