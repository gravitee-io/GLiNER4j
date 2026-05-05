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

import io.gravitee.lab.gliner4j.postprocess.RelationDecoder;
import io.gravitee.lab.gliner4j.processor.MultiSchemaEmbeddings;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInput;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInputAssembler;
import io.gravitee.lab.gliner4j.processor.SchemaUnit;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.processor.UnitLayout;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.tokenizer.WhitespaceTokenSplitter;
import java.nio.file.Path;
import java.util.LinkedHashMap;
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
public class GLiNER4jRelationExtractor implements AutoCloseable {

  private final GLiNER4jConfig config;
  private final List<RelationDefinition> relations;
  private final DjlTokenizerWrapper tokenizer;
  private final GLiNER4jNERRuntime runtime;
  private final MultiSchemaInputAssembler inputAssembler;
  private final WhitespaceTokenSplitter splitter;
  private final RelationDecoder relationDecoder;
  private final GLiNER4jTelemetry telemetry;

  private GLiNER4jRelationExtractor(
    GLiNER4jConfig config,
    List<RelationDefinition> relations,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jNERRuntime runtime,
    MultiSchemaInputAssembler inputAssembler
  ) {
    this.config = config;
    this.relations = List.copyOf(relations);
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.inputAssembler = inputAssembler;
    this.splitter = new WhitespaceTokenSplitter();
    this.relationDecoder = new RelationDecoder();
    this.telemetry = new GLiNER4jTelemetry("extract_relations");
  }

  public static GLiNER4jRelationExtractor load(
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

  public static GLiNER4jRelationExtractor load(
    Path modelDir,
    List<RelationDefinition> relations,
    String variant
  ) {
    return load(modelDir, relations, variant, RuntimeConfig.defaults());
  }

  public static GLiNER4jRelationExtractor load(
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

  public static GLiNER4jRelationExtractor load(
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
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var runtime = new GLiNER4jNERRuntime(modelDir, variant, runtimeConfig);
    var assembler = buildAssembler(tokenizer, relations);
    runtime.initEncoderBuffers(assembler.getSchemaPrefixIds());

    log.info("GLiNER4jRelationExtractor loaded successfully");
    return new GLiNER4jRelationExtractor(
      config,
      relations,
      tokenizer,
      runtime,
      assembler
    );
  }

  public Map<String, List<RelationInstance>> extract(String text) {
    return extract(text, config.getDefaultThreshold());
  }

  public Map<String, List<RelationInstance>> extract(
    String text,
    float threshold
  ) {
    long startNanos = System.nanoTime();
    var emptyResult = emptyMapForCurrentRelations();
    if (text == null || text.isBlank()) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    var textEncoder = new TextEncoder(text, splitter);
    if (textEncoder.getTextLen() == 0) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    var input = inputAssembler.assemble(textEncoder);
    var hiddenStates = runtime.runEncoder(
      input.inputIds(),
      input.attentionMask()
    );

    var result = decodeAllUnits(hiddenStates[0], input, text, threshold);
    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    telemetry.record(durationMs, 1, totalInstances(result));
    return result;
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
    long startNanos = System.nanoTime();
    if (text == null || text.isBlank()) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }
    if (overrideRelations.isEmpty()) {
      throw new IllegalArgumentException(
        "Override relations list must not be empty"
      );
    }

    var overrideAssembler = buildAssembler(tokenizer, overrideRelations);

    var textEncoder = new TextEncoder(text, splitter);
    if (textEncoder.getTextLen() == 0) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    var input = overrideAssembler.assemble(textEncoder);
    var hiddenStates = runtime.runEncoderFull(
      input.inputIds(),
      input.attentionMask()
    );

    var result = decodeAllUnits(hiddenStates[0], input, text, threshold);
    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    telemetry.record(durationMs, 1, totalInstances(result));
    return result;
  }

  public List<Map<String, List<RelationInstance>>> extractBatch(
    List<String> texts
  ) {
    return extractBatch(texts, config.getDefaultThreshold());
  }

  public List<Map<String, List<RelationInstance>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    long startNanos = System.nanoTime();
    var results = new java.util.ArrayList<Map<String, List<RelationInstance>>>(
      texts.size()
    );
    long totalInstances = 0;
    for (var text : texts) {
      if (text == null || text.isBlank()) {
        results.add(Map.of());
        continue;
      }
      var perText = extract(text, threshold);
      results.add(perText);
      totalInstances += totalInstances(perText);
    }
    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    telemetry.record(durationMs, texts.size(), totalInstances);
    return results;
  }

  private Map<String, List<RelationInstance>> decodeAllUnits(
    float[][] hiddenState,
    MultiSchemaInput input,
    String text,
    float threshold
  ) {
    int hiddenSize = config.getHiddenSize();
    int maxWidth = config.getMaxWidth();
    int textLen = input.textLen();

    // Extract text embeddings once (shared across all units)
    var textEmbs = new float[textLen][hiddenSize];
    MultiSchemaEmbeddings.extractText(
      hiddenState,
      input.wordFirstSubwordPos(),
      textEmbs
    );

    // Run span_rep once (depends only on text embeddings)
    int numSpans = textLen * maxWidth;
    var spanIdxFlat = buildSpanIdxFlat(textLen, maxWidth, numSpans);
    var textEmbs3d = new float[1][textLen][hiddenSize];
    System.arraycopy(textEmbs, 0, textEmbs3d[0], 0, textLen);
    var spanRep4d = runtime.runSpanRepFlat(textEmbs3d, spanIdxFlat, numSpans);
    var spanRep = spanRep4d[0];

    var result = new LinkedHashMap<String, List<RelationInstance>>();
    for (var layout : input.unitLayouts()) {
      if (layout.unit().kind() != SchemaUnit.Kind.RELATION) {
        continue;
      }
      var instances = decodeRelationUnit(
        layout,
        hiddenState,
        spanRep,
        input,
        text,
        threshold
      );
      result.put(layout.unit().parentLabel(), instances);
    }
    return result;
  }

  private List<RelationInstance> decodeRelationUnit(
    UnitLayout layout,
    float[][] hiddenState,
    float[][][] spanRep,
    MultiSchemaInput input,
    String text,
    float threshold
  ) {
    var unitEmbs = MultiSchemaEmbeddings.extractUnit(hiddenState, layout);
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
      input.textLen(),
      threshold
    );
  }

  private Map<String, List<RelationInstance>> emptyMapForCurrentRelations() {
    var map = new LinkedHashMap<String, List<RelationInstance>>();
    for (var rel : relations) {
      map.put(rel.name(), List.of());
    }
    return map;
  }

  private static long totalInstances(
    Map<String, List<RelationInstance>> result
  ) {
    return result.values().stream().mapToLong(List::size).sum();
  }

  private static long[] buildSpanIdxFlat(
    int textLen,
    int maxWidth,
    int numSpans
  ) {
    var flat = new long[numSpans * 2];
    for (int i = 0; i < textLen; i++) {
      for (int w = 0; w < maxWidth; w++) {
        int endPos = i + w;
        if (endPos < textLen) {
          int flatIdx = (i * maxWidth + w) * 2;
          flat[flatIdx] = i;
          flat[flatIdx + 1] = endPos;
        }
      }
    }
    return flat;
  }

  private static MultiSchemaInputAssembler buildAssembler(
    DjlTokenizerWrapper tokenizer,
    List<RelationDefinition> relations
  ) {
    var units = relations.stream().map(SchemaUnit::forRelation).toList();
    return new MultiSchemaInputAssembler(tokenizer, units);
  }

  @Override
  public void close() {
    runtime.close();
    tokenizer.close();
    log.info("GLiNER4jRelationExtractor closed");
  }
}
