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
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Main facade for GLiNER4j NER via ONNX Runtime.
 *
 * <p>The bundle's model family is detected automatically from the {@code architecture} key in
 * {@code gliner4j_config.json} (absent ⇒ GLiNER2). This facade is family-agnostic: it resolves the
 * family to a {@link NerStrategy} at load time and delegates extraction to it, applying the default
 * threshold and null-guards.
 *
 * <p>Usage:
 * <pre>{@code
 * var entities = List.of(new EntityDefinition("person"), new EntityDefinition("organization"));
 * try (var gliner = GLiNER4jNER.load(modelDir, entities)) {
 *     Map<String, List<EntitySpan>> results = gliner.extract("John works at Google.");
 * }
 * }</pre>
 */
@Slf4j
public final class GLiNER4jNER implements AutoCloseable {

  private final GLiNER4jConfig config;
  private final NerStrategy strategy;

  private GLiNER4jNER(GLiNER4jConfig config, NerStrategy strategy) {
    this.config = config;
    this.strategy = strategy;
  }

  /**
   * Loads a GLiNER model using the default "onnx" variant and auto-detected resources.
   *
   * @param modelDir path to the root model directory
   * @param entities the entity types to extract
   * @return a ready-to-use GLiNER4jNER instance
   */
  public static GLiNER4jNER load(
    Path modelDir,
    List<EntityDefinition> entities
  ) {
    return load(
      modelDir,
      entities,
      BaseRuntime.DEFAULT_VARIANT,
      RuntimeConfig.defaults()
    );
  }

  /**
   * Loads a GLiNER model with a specific ONNX variant and auto-detected resources.
   *
   * @param modelDir path to the root model directory
   * @param entities the entity types to extract
   * @param variant  ONNX variant folder name (e.g. "onnx", "onnx_fp16", "onnx_quantized")
   * @return a ready-to-use GLiNER4jNER instance
   */
  public static GLiNER4jNER load(
    Path modelDir,
    List<EntityDefinition> entities,
    String variant
  ) {
    return load(modelDir, entities, variant, RuntimeConfig.defaults());
  }

  /**
   * Loads a GLiNER model using the default "onnx" variant with explicit resource control.
   *
   * @param modelDir      path to the root model directory
   * @param entities      the entity types to extract
   * @param runtimeConfig resource control configuration for ORT sessions
   * @return a ready-to-use GLiNER4jNER instance
   */
  public static GLiNER4jNER load(
    Path modelDir,
    List<EntityDefinition> entities,
    RuntimeConfig runtimeConfig
  ) {
    return load(modelDir, entities, BaseRuntime.DEFAULT_VARIANT, runtimeConfig);
  }

  /**
   * Loads a GLiNER model with a specific ONNX variant and explicit resource control.
   *
   * <p>The model directory should contain shared files (config, tokenizer) at the root
   * and ONNX model files in variant subfolders (e.g. "onnx", "onnx_fp16", "onnx_quantized").
   *
   * @param modelDir      path to the root model directory
   * @param entities      the entity types to extract
   * @param variant       ONNX variant folder name (e.g. "onnx", "onnx_fp16", "onnx_quantized")
   * @param runtimeConfig resource control configuration for ORT sessions
   * @return a ready-to-use GLiNER4jNER instance
   */
  public static GLiNER4jNER load(
    Path modelDir,
    List<EntityDefinition> entities,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    var config = GLiNER4jConfig.load(modelDir);
    log.info(
      "Loading GLiNER4jNER model from {} (variant={}, architecture={}) with {} entities",
      modelDir,
      variant,
      config.getArchitecture().configValue(),
      entities.size()
    );

    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var ctx = new LoadContext(
      modelDir,
      variant,
      runtimeConfig,
      config,
      tokenizer
    );
    var strategy = ModelArchitectures.forConfig(config).newNerStrategy(
      ctx,
      entities
    );

    log.info("GLiNER4jNER model loaded successfully");
    return new GLiNER4jNER(config, strategy);
  }

  /**
   * Extracts entities from the given text using the default threshold.
   *
   * @param text the input text to analyze
   * @return map of entity type to list of detected spans
   */
  public Map<String, List<EntitySpan>> extract(String text) {
    return strategy.extract(text, config.getDefaultThreshold());
  }

  /**
   * Extracts entities from the given text using the specified threshold.
   *
   * @param text the input text to analyze
   * @param threshold minimum confidence score (0..1) for span inclusion
   * @return map of entity type to list of detected spans
   */
  public Map<String, List<EntitySpan>> extract(String text, float threshold) {
    return strategy.extract(text, threshold);
  }

  /**
   * Extracts entities from the given text using per-call entity definitions and the default threshold.
   *
   * @param text the input text to analyze
   * @param entities the entity types to extract (overrides the entities provided at load time)
   * @return map of entity type to list of detected spans
   */
  public Map<String, List<EntitySpan>> extract(
    String text,
    List<EntityDefinition> entities
  ) {
    return strategy.extract(text, entities, config.getDefaultThreshold());
  }

  /**
   * Extracts entities from the given text using per-call entity definitions and a custom threshold.
   *
   * @param text the input text to analyze
   * @param entities the entity types to extract (overrides the entities provided at load time)
   * @param threshold minimum confidence score (0..1) for span inclusion
   * @return map of entity type to list of detected spans
   */
  public Map<String, List<EntitySpan>> extract(
    String text,
    List<EntityDefinition> entities,
    float threshold
  ) {
    return strategy.extract(text, entities, threshold);
  }

  /**
   * Extracts entities from multiple texts in batch, using a single batched encoder call.
   *
   * @param texts the input texts to analyze
   * @return list of results, one per input text (same order)
   */
  public List<Map<String, List<EntitySpan>>> extractBatch(List<String> texts) {
    if (texts == null) {
      return List.of();
    }
    return strategy.extractBatch(texts, config.getDefaultThreshold());
  }

  /**
   * Extracts entities from multiple texts in batch with a custom threshold.
   *
   * @param texts the input texts to analyze
   * @param threshold minimum confidence score (0..1) for span inclusion
   * @return list of results, one per input text (same order)
   */
  public List<Map<String, List<EntitySpan>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    return strategy.extractBatch(texts, threshold);
  }

  @Override
  public void close() {
    strategy.close();
    log.info("GLiNER4jNER closed");
  }
}
