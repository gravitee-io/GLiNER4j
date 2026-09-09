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
package io.gravitee.lab.gliner4j.extractor;

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.arch.ModelArchitectures;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import io.gravitee.lab.gliner4j.strategy.RelationStrategy;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Zero-shot relation extraction facade. The family is read from the bundle's
 * {@code architecture} at load time and the matching {@link RelationStrategy} does the work:
 * GLiNER2 scores each relation unit through the count-aware span head, GLiNER2.5 runs its typed
 * pair generator + relation scorer inside {@code relation_full.onnx}.
 */
@Slf4j
public final class RelationExtractor implements AutoCloseable {

  private final GLiNER4jConfig config;
  private final RelationStrategy strategy;

  private RelationExtractor(GLiNER4jConfig config, RelationStrategy strategy) {
    this.config = config;
    this.strategy = strategy;
  }

  public static RelationExtractor load(
    Path modelDir,
    List<RelationDefinition> relations
  ) {
    return load(
      modelDir,
      relations,
      BaseRuntime.DEFAULT_VARIANT,
      RuntimeConfig.defaults()
    );
  }

  public static RelationExtractor load(
    Path modelDir,
    List<RelationDefinition> relations,
    String variant
  ) {
    return load(modelDir, relations, variant, RuntimeConfig.defaults());
  }

  public static RelationExtractor load(
    Path modelDir,
    List<RelationDefinition> relations,
    RuntimeConfig runtimeConfig
  ) {
    return load(
      modelDir,
      relations,
      BaseRuntime.DEFAULT_VARIANT,
      runtimeConfig
    );
  }

  public static RelationExtractor load(
    Path modelDir,
    List<RelationDefinition> relations,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    if (relations.isEmpty()) {
      throw new IllegalArgumentException(
        "GLiNER4jRelationExtractor requires at least one relation definition"
      );
    }
    var config = GLiNER4jConfig.load(modelDir);
    log.info(
      "Loading GLiNER4jRelationExtractor from {} (variant={}, architecture={}) with {} relations",
      modelDir,
      variant,
      config.getArchitecture().configValue(),
      relations.size()
    );
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var ctx = new LoadContext(
      modelDir,
      variant,
      runtimeConfig,
      config,
      tokenizer
    );
    var strategy = ModelArchitectures.forConfig(config).newRelationStrategy(
      ctx,
      relations
    );

    log.info("GLiNER4jRelationExtractor loaded successfully");
    return new RelationExtractor(config, strategy);
  }

  public Map<String, List<RelationInstance>> extract(String text) {
    return strategy.extract(text, config.getDefaultThreshold());
  }

  public Map<String, List<RelationInstance>> extract(
    String text,
    float threshold
  ) {
    return strategy.extract(text, threshold);
  }

  public Map<String, List<RelationInstance>> extract(
    String text,
    List<RelationDefinition> overrideRelations
  ) {
    return extract(text, overrideRelations, config.getDefaultThreshold());
  }

  public Map<String, List<RelationInstance>> extract(
    String text,
    List<RelationDefinition> overrideRelations,
    float threshold
  ) {
    if (overrideRelations.isEmpty()) {
      throw new IllegalArgumentException(
        "Override relations list must not be empty"
      );
    }
    return strategy.extract(text, overrideRelations, threshold);
  }

  public List<Map<String, List<RelationInstance>>> extractBatch(
    List<String> texts
  ) {
    return strategy.extractBatch(texts, config.getDefaultThreshold());
  }

  public List<Map<String, List<RelationInstance>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    return strategy.extractBatch(texts, threshold);
  }

  @Override
  public void close() {
    strategy.close();
    log.info("RelationExtractor closed");
  }
}
