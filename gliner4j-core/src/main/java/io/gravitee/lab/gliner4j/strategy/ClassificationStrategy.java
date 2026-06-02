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

import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import java.util.List;

/**
 * Per-family text-classification implementation: owns {@code assemble → forward → decode} for one
 * model family (GLiNER2's classifier head, GLiClass's scaled dot-product head, …).
 *
 * <p>The public {@link io.gravitee.lab.gliner4j.GLiNER4jClassifier} facade selects an implementation
 * at load time from the bundle's {@code architecture} and delegates to it, applying the default
 * threshold and null-guards.
 */
public interface ClassificationStrategy extends AutoCloseable {
  /**
   * Classifies text against the load-time labels.
   *
   * @param text the input text
   * @param threshold minimum confidence (0..1) for a label to be returned
   * @return labels above threshold, sorted by confidence descending
   */
  List<ClassificationResult> classify(String text, float threshold);

  /**
   * Classifies text against per-call labels (overriding the load-time labels).
   *
   * @param text the input text
   * @param labels the labels for this call
   * @param threshold minimum confidence (0..1) for a label to be returned
   * @return labels above threshold, sorted by confidence descending
   */
  List<ClassificationResult> classify(
    String text,
    List<ClassificationLabel> labels,
    float threshold
  );

  /**
   * Classifies multiple texts in one batched pass.
   *
   * @param texts the input texts
   * @param threshold minimum confidence (0..1) for a label to be returned
   * @return per-text results in input order
   */
  List<List<ClassificationResult>> classifyBatch(
    List<String> texts,
    float threshold
  );

  @Override
  void close();
}
