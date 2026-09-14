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
package io.gravitee.lab.gliner4j;

import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.arch.ModelArchitectures;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.strategy.ClassificationStrategy;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Path;
import java.util.List;
import lombok.extern.slf4j.Slf4j;

/**
 * Text classification facade for GLiNER4j — classifies text into label categories via ONNX Runtime.
 *
 * <p>The bundle's model family is detected automatically from the {@code architecture} key in
 * {@code gliner4j_config.json} (absent ⇒ GLiNER2). This facade resolves the family to a
 * {@link ClassificationStrategy} at load time and delegates to it, applying the default threshold
 * and null-guards.
 *
 * <p>Supports single-label (highest confidence) and multi-label (threshold-based) classification,
 * with optional per-call label override.
 *
 * <p>Usage:
 * <pre>{@code
 * var labels = List.of(new ClassificationLabel("positive"), new ClassificationLabel("negative"));
 * try (var classifier = GLiNER4jClassifier.load(modelDir, labels)) {
 *     List<ClassificationResult> results = classifier.classify("Great product!", 0.5f);
 * }
 * }</pre>
 */
@Slf4j
public final class GLiNER4jClassifier implements AutoCloseable {

  private final GLiNER4jConfig config;
  private final ClassificationStrategy strategy;

  private GLiNER4jClassifier(
    GLiNER4jConfig config,
    ClassificationStrategy strategy
  ) {
    this.config = config;
    this.strategy = strategy;
  }

  /**
   * Loads a classifier using the default "onnx" variant and auto-detected resources.
   *
   * @param modelDir path to the root model directory
   * @param labels the classification labels
   * @return a ready-to-use GLiNER4jClassifier instance
   */
  public static GLiNER4jClassifier load(
    Path modelDir,
    List<ClassificationLabel> labels
  ) {
    return load(
      modelDir,
      labels,
      BaseRuntime.DEFAULT_VARIANT,
      RuntimeConfig.defaults()
    );
  }

  /**
   * Loads a classifier with a specific ONNX variant.
   *
   * @param modelDir path to the root model directory
   * @param labels the classification labels
   * @param variant ONNX variant folder name
   * @return a ready-to-use GLiNER4jClassifier instance
   */
  public static GLiNER4jClassifier load(
    Path modelDir,
    List<ClassificationLabel> labels,
    String variant
  ) {
    return load(modelDir, labels, variant, RuntimeConfig.defaults());
  }

  /**
   * Loads a classifier with explicit resource control.
   *
   * @param modelDir path to the root model directory
   * @param labels the classification labels
   * @param runtimeConfig resource control configuration
   * @return a ready-to-use GLiNER4jClassifier instance
   */
  public static GLiNER4jClassifier load(
    Path modelDir,
    List<ClassificationLabel> labels,
    RuntimeConfig runtimeConfig
  ) {
    return load(modelDir, labels, BaseRuntime.DEFAULT_VARIANT, runtimeConfig);
  }

  /**
   * Loads a classifier with a specific ONNX variant and explicit resource control.
   *
   * @param modelDir      path to the root model directory
   * @param labels        the classification labels
   * @param variant       ONNX variant folder name
   * @param runtimeConfig resource control configuration
   * @return a ready-to-use GLiNER4jClassifier instance
   */
  public static GLiNER4jClassifier load(
    Path modelDir,
    List<ClassificationLabel> labels,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    var config = GLiNER4jConfig.load(modelDir);
    log.info(
      "Loading GLiNER4jClassifier from {} (variant={}, architecture={}) with {} labels",
      modelDir,
      variant,
      config.getArchitecture().configValue(),
      labels.size()
    );

    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var ctx = new LoadContext(
      modelDir,
      variant,
      runtimeConfig,
      config,
      tokenizer
    );
    var strategy = ModelArchitectures.forConfig(
      config
    ).newClassificationStrategy(ctx, labels);

    log.info("GLiNER4jClassifier loaded successfully");
    return new GLiNER4jClassifier(config, strategy);
  }

  /**
   * Classifies text in multi-label mode using the default threshold.
   *
   * @param text the input text to classify
   * @return list of classification results above threshold
   */
  public List<ClassificationResult> classify(String text) {
    return strategy.classify(text, config.getDefaultThreshold());
  }

  /**
   * Classifies text in multi-label mode with a custom threshold.
   *
   * @param text the input text to classify
   * @param threshold minimum confidence score (0..1) for label inclusion
   * @return list of classification results above threshold, sorted by confidence descending
   */
  public List<ClassificationResult> classify(String text, float threshold) {
    return strategy.classify(text, threshold);
  }

  /**
   * Classifies text using per-call label definitions and default threshold.
   *
   * @param text the input text to classify
   * @param labels the classification labels (overrides load-time labels)
   * @return list of classification results above threshold
   */
  public List<ClassificationResult> classify(
    String text,
    List<ClassificationLabel> labels
  ) {
    return strategy.classify(text, labels, config.getDefaultThreshold());
  }

  /**
   * Classifies text using per-call label definitions and custom threshold.
   *
   * @param text the input text to classify
   * @param labels the classification labels (overrides load-time labels)
   * @param threshold minimum confidence score (0..1) for label inclusion
   * @return list of classification results above threshold, sorted by confidence descending
   */
  public List<ClassificationResult> classify(
    String text,
    List<ClassificationLabel> labels,
    float threshold
  ) {
    return strategy.classify(text, labels, threshold);
  }

  /**
   * Classifies multiple texts in batch using default threshold.
   *
   * @param texts the input texts to classify
   * @return list of results, one per input text (same order)
   */
  public List<List<ClassificationResult>> classifyBatch(List<String> texts) {
    if (texts == null) {
      return List.of();
    }
    return strategy.classifyBatch(texts, config.getDefaultThreshold());
  }

  /**
   * Classifies multiple texts in batch with a custom threshold.
   *
   * @param texts the input texts to classify
   * @param threshold minimum confidence score (0..1) for label inclusion
   * @return list of results, one per input text (same order)
   */
  public List<List<ClassificationResult>> classifyBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    return strategy.classifyBatch(texts, threshold);
  }

  @Override
  public void close() {
    strategy.close();
    log.info("GLiNER4jClassifier closed");
  }
}
