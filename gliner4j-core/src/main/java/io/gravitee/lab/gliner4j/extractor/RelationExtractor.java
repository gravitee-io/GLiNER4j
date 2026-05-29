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
import io.gravitee.lab.gliner4j.arch.ModelArchitectures;
import io.gravitee.lab.gliner4j.arch.TaskType;
import io.gravitee.lab.gliner4j.postprocess.RelationDecoder;
import io.gravitee.lab.gliner4j.processor.MultiSchemaEmbeddings;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInput;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInputAssembler;
import io.gravitee.lab.gliner4j.processor.SchemaUnit;
import io.gravitee.lab.gliner4j.processor.UnitLayout;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Standalone facade for GLiNER4j relation extraction.
 *
 * <p>Builds a multi-unit GLiNER prompt where each relation type is its own block joined by
 * {@code [SEP_STRUCT]}, runs the encoder once, then per relation type runs the scoring head
 * and decodes head/tail (or arbitrary fields) spans into {@link RelationInstance}s.
 *
 * <p>Usage:
 * <pre>{@code
 * var relations = List.of(
 *     new RelationDefinition("works_for", "Employment relationship"),
 *     new RelationDefinition("lives_in", "Residence relationship")
 * );
 * try (var extractor = GLiNER4jRelationExtractor.load(modelDir, relations)) {
 *     Map<String, List<RelationInstance>> result = extractor.extract("John works at Google.");
 * }
 * }</pre>
 */
@Slf4j
public final class RelationExtractor
  extends Extractor<RelationDefinition, RelationInstance> {

  private final RelationDecoder relationDecoder;

  private RelationExtractor(
    GLiNER4jConfig config,
    List<RelationDefinition> relations,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jNERRuntime runtime,
    MultiSchemaInputAssembler inputAssembler
  ) {
    super(
      config,
      relations,
      tokenizer,
      runtime,
      inputAssembler,
      new GLiNER4jTelemetry("extract_relations")
    );
    this.relationDecoder = new RelationDecoder();
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
    log.info(
      "Loading GLiNER4jRelationExtractor from {} (variant={}) with {} relations",
      modelDir,
      variant,
      relations.size()
    );
    var config = GLiNER4jConfig.load(modelDir);
    ModelArchitectures.forId(config.getArchitecture()).requireSupported(
      TaskType.RELATION
    );
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var runtime = new GLiNER4jNERRuntime(modelDir, variant, runtimeConfig);
    var assembler = assemblerFor(tokenizer, relations);
    runtime.initEncoderBuffers(assembler.getSchemaPrefixIds());

    log.info("GLiNER4jRelationExtractor loaded successfully");
    return new RelationExtractor(
      config,
      relations,
      tokenizer,
      runtime,
      assembler
    );
  }

  public Map<String, List<RelationInstance>> extract(String text) {
    return doExtractOnce(text, config.getDefaultThreshold());
  }

  public Map<String, List<RelationInstance>> extract(
    String text,
    float threshold
  ) {
    return doExtractOnce(text, threshold);
  }

  public Map<String, List<RelationInstance>> extract(
    String text,
    List<RelationDefinition> overrideRelations
  ) {
    return doExtractOverride(
      text,
      overrideRelations,
      config.getDefaultThreshold()
    );
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
    return doExtractOverride(text, overrideRelations, threshold);
  }

  public List<Map<String, List<RelationInstance>>> extractBatch(
    List<String> texts
  ) {
    return doExtractBatch(texts, config.getDefaultThreshold());
  }

  public List<Map<String, List<RelationInstance>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    return doExtractBatch(texts, threshold);
  }

  // ---- AbstractMultiUnitFacade hooks --------------------------------------

  @Override
  protected MultiSchemaInputAssembler buildAssembler(
    List<RelationDefinition> relations
  ) {
    return assemblerFor(tokenizer, relations);
  }

  @Override
  protected List<RelationInstance> decodeUnit(
    RelationDefinition relation,
    UnitLayout layout,
    MultiSchemaEmbeddings.UnitEmbeddings unitEmbs,
    float[][][] spanRep,
    MultiSchemaInput input,
    String text,
    int textLen,
    float threshold
  ) {
    var scoringResult = runtime.runScoringHead(
      spanRep,
      unitEmbs.schemaEmbP(),
      unitEmbs.schemaEmbFields(),
      (long) config.getMaxCount()
    );
    return relationDecoder.decode(
      layout.unit().parentLabel(),
      layout.unit().childNames(),
      scoringResult.countLogits(),
      scoringResult.spanScores(),
      input.wordStartChars(),
      input.wordEndChars(),
      text,
      textLen,
      threshold
    );
  }

  // ---- helpers -------------------------------------------------------------

  private static MultiSchemaInputAssembler assemblerFor(
    DjlTokenizerWrapper tokenizer,
    List<RelationDefinition> relations
  ) {
    var units = relations.stream().map(SchemaUnit::forRelation).toList();
    return new MultiSchemaInputAssembler(tokenizer, units);
  }
}
