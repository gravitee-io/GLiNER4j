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
import io.gravitee.lab.gliner4j.postprocess.StructureDecoder;
import io.gravitee.lab.gliner4j.processor.MultiSchemaEmbeddings;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInput;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInputAssembler;
import io.gravitee.lab.gliner4j.processor.SchemaUnit;
import io.gravitee.lab.gliner4j.processor.UnitLayout;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import io.gravitee.lab.gliner4j.schema.StructureInstance;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Schema/JSON structure extraction facade for GLiNER4j.
 *
 * <p>Mirrors the upstream GLiNER2 {@code extract_json} API: given one or more
 * {@link StructureDefinition}s, pulls all matching instances out of a text and
 * returns them as typed {@link StructureInstance}s grouped by structure name.
 *
 * <p>Usage:
 * <pre>{@code
 * var product = StructureDefinition.builder("product")
 *     .string("name", "Product name")
 *     .list("price")
 *     .list("features")
 *     .build();
 *
 * try (var extractor = GLiNER4jSchemaExtractor.load(modelDir, List.of(product))) {
 *     Map<String, List<StructureInstance>> result = extractor.extract(
 *         "The MacBook Pro costs $1999 and features M3 chip, 16GB RAM, and 512GB storage."
 *     );
 * }
 * }</pre>
 */
@Slf4j
public final class SchemaExtractor
  extends Extractor<StructureDefinition, StructureInstance> {

  private final StructureDecoder decoder;

  private SchemaExtractor(
    GLiNER4jConfig config,
    List<StructureDefinition> structures,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jNERRuntime runtime,
    MultiSchemaInputAssembler inputAssembler
  ) {
    super(
      config,
      structures,
      tokenizer,
      runtime,
      inputAssembler,
      new GLiNER4jTelemetry("extract_schema")
    );
    this.decoder = new StructureDecoder();
  }

  public static SchemaExtractor load(
    Path modelDir,
    List<StructureDefinition> structures
  ) {
    return load(
      modelDir,
      structures,
      BaseRuntime.DEFAULT_VARIANT,
      RuntimeConfig.defaults()
    );
  }

  public static SchemaExtractor load(
    Path modelDir,
    List<StructureDefinition> structures,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    log.info(
      "Loading GLiNER4jSchemaExtractor model from {} (variant={}) with {} structures",
      modelDir,
      variant,
      structures.size()
    );

    var config = GLiNER4jConfig.load(modelDir);
    ModelArchitectures.forId(config.getArchitecture()).requireSupported(
      TaskType.STRUCTURE
    );
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var runtime = new GLiNER4jNERRuntime(modelDir, variant, runtimeConfig);
    var assembler = assemblerFor(tokenizer, structures);
    runtime.initEncoderBuffers(assembler.getSchemaPrefixIds());

    log.info("GLiNER4jSchemaExtractor loaded successfully");
    return new SchemaExtractor(
      config,
      structures,
      tokenizer,
      runtime,
      assembler
    );
  }

  public Map<String, List<StructureInstance>> extract(String text) {
    return doExtractOnce(text, config.getDefaultThreshold());
  }

  public Map<String, List<StructureInstance>> extract(
    String text,
    float threshold
  ) {
    return doExtractOnce(text, threshold);
  }

  public Map<String, List<StructureInstance>> extract(
    String text,
    List<StructureDefinition> overrideStructures,
    float threshold
  ) {
    return doExtractOverride(text, overrideStructures, threshold);
  }

  public List<Map<String, List<StructureInstance>>> extractBatch(
    List<String> texts
  ) {
    if (texts == null) {
      return List.of();
    }
    return doExtractBatch(texts, config.getDefaultThreshold());
  }

  public List<Map<String, List<StructureInstance>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    return doExtractBatch(texts, threshold);
  }

  // ---- AbstractMultiUnitFacade hooks --------------------------------------

  @Override
  protected MultiSchemaInputAssembler buildAssembler(
    List<StructureDefinition> structures
  ) {
    return assemblerFor(tokenizer, structures);
  }

  @Override
  protected List<StructureInstance> decodeUnit(
    StructureDefinition structure,
    UnitLayout layout,
    MultiSchemaEmbeddings.UnitEmbeddings unitEmbs,
    float[][][] spanRep,
    MultiSchemaInput input,
    String text,
    int textLen,
    float threshold
  ) {
    if (unitEmbs.schemaEmbFields().length == 0) {
      return List.of();
    }

    var scoringResult = runtime.runScoringHead(
      spanRep,
      unitEmbs.schemaEmbP(),
      unitEmbs.schemaEmbFields(),
      config.getMaxCount()
    );

    int predCount = argmax(scoringResult.countLogits()[0]);
    log.debug(
      "Structure '{}' predicted count: {}",
      structure.name(),
      predCount
    );

    return decoder.decode(
      scoringResult.spanScores(),
      predCount,
      structure,
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
    List<StructureDefinition> structures
  ) {
    var units = structures.stream().map(SchemaUnit::forStructure).toList();
    return new MultiSchemaInputAssembler(tokenizer, units);
  }

  private static int argmax(float[] values) {
    int maxIdx = 0;
    float maxVal = values[0];
    for (int i = 1; i < values.length; i++) {
      if (values[i] > maxVal) {
        maxVal = values[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }
}
