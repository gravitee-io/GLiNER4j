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

import io.gravitee.lab.gliner4j.postprocess.StructureDecoder;
import io.gravitee.lab.gliner4j.processor.InputAssembler;
import io.gravitee.lab.gliner4j.processor.MultiSchemaEncoder;
import io.gravitee.lab.gliner4j.processor.PreprocessedInput;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import io.gravitee.lab.gliner4j.schema.StructureInstance;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.tokenizer.WhitespaceTokenSplitter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
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
public class GLiNER4jSchemaExtractor implements AutoCloseable {

  private final GLiNER4jConfig config;
  private final List<StructureDefinition> structures;
  private final DjlTokenizerWrapper tokenizer;
  private final GLiNER4jNERRuntime runtime;
  private final InputAssembler inputAssembler;
  private final WhitespaceTokenSplitter splitter;
  private final StructureDecoder decoder;
  private final GLiNER4jTelemetry telemetry;

  private GLiNER4jSchemaExtractor(
    GLiNER4jConfig config,
    List<StructureDefinition> structures,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jNERRuntime runtime,
    InputAssembler inputAssembler
  ) {
    this.config = config;
    this.structures = List.copyOf(structures);
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.inputAssembler = inputAssembler;
    this.splitter = new WhitespaceTokenSplitter();
    this.decoder = new StructureDecoder();
    this.telemetry = new GLiNER4jTelemetry("extract_schema");
  }

  public static GLiNER4jSchemaExtractor load(
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

  public static GLiNER4jSchemaExtractor load(
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
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var runtime = new GLiNER4jNERRuntime(modelDir, variant, runtimeConfig);
    var multiEncoder = new MultiSchemaEncoder(structures);
    var inputAssembler = new InputAssembler(
      tokenizer,
      multiEncoder.getCombined()
    );

    runtime.initEncoderBuffers(inputAssembler.getSchemaPrefixIds());

    log.info("GLiNER4jSchemaExtractor loaded successfully");
    return new GLiNER4jSchemaExtractor(
      config,
      structures,
      tokenizer,
      runtime,
      inputAssembler
    );
  }

  public Map<String, List<StructureInstance>> extract(String text) {
    return extract(text, config.getDefaultThreshold());
  }

  public Map<String, List<StructureInstance>> extract(
    String text,
    float threshold
  ) {
    long startNanos = System.nanoTime();
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

    var result = extractFromHiddenStates(
      hiddenStates,
      input,
      structures,
      text,
      threshold
    );
    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    long instanceCount = result.values().stream().mapToLong(List::size).sum();
    telemetry.record(durationMs, 1, instanceCount);
    return result;
  }

  public Map<String, List<StructureInstance>> extract(
    String text,
    List<StructureDefinition> overrideStructures,
    float threshold
  ) {
    long startNanos = System.nanoTime();
    if (text == null || text.isBlank()) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    var overrideEncoder = new MultiSchemaEncoder(overrideStructures);
    var overrideAssembler = new InputAssembler(
      tokenizer,
      overrideEncoder.getCombined()
    );

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

    var result = extractFromHiddenStates(
      hiddenStates,
      input,
      overrideStructures,
      text,
      threshold
    );
    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    long instanceCount = result.values().stream().mapToLong(List::size).sum();
    telemetry.record(durationMs, 1, instanceCount);
    return result;
  }

  public List<Map<String, List<StructureInstance>>> extractBatch(
    List<String> texts
  ) {
    if (texts == null) {
      return List.of();
    }
    return extractBatch(texts, config.getDefaultThreshold());
  }

  public List<Map<String, List<StructureInstance>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    var results = new ArrayList<Map<String, List<StructureInstance>>>(
      texts.size()
    );
    for (var text : texts) {
      results.add(extract(text, threshold));
    }
    return results;
  }

  private Map<String, List<StructureInstance>> extractFromHiddenStates(
    float[][][] hiddenStates,
    PreprocessedInput input,
    List<StructureDefinition> activeStructures,
    String text,
    float threshold
  ) {
    int hiddenSize = config.getHiddenSize();
    int maxWidth = config.getMaxWidth();
    int textLen = input.textLen();
    int numSpans = textLen * maxWidth;

    var hiddenState = hiddenStates[0];

    // Group [P] and [C] embeddings per structure by walking input ids
    var perStructureEmbeddings = groupEmbeddingsByStructure(
      input,
      hiddenState,
      activeStructures.size()
    );

    // Extract text embeddings (shared across structures)
    var textEmbs = new float[textLen][hiddenSize];
    extractTextEmbeddings(hiddenState, input, textEmbs);

    // Run span_rep once on the text-only embeddings
    var textEmbs3d = new float[1][textLen][hiddenSize];
    System.arraycopy(textEmbs, 0, textEmbs3d[0], 0, textLen);
    var spanIdxFlat = buildSpanIdxFlat(textLen, maxWidth, numSpans);
    var spanRep4d = runtime.runSpanRepFlat(textEmbs3d, spanIdxFlat, numSpans);
    var spanRep = spanRep4d[0];

    // Per structure: run scoring head with that structure's embeddings, then decode
    var result = new LinkedHashMap<String, List<StructureInstance>>();
    for (int s = 0; s < activeStructures.size(); s++) {
      var structure = activeStructures.get(s);
      var schemaEmbP = perStructureEmbeddings.pEmbs.get(s);
      var schemaEmbFields = perStructureEmbeddings.cEmbs
        .get(s)
        .toArray(new float[0][]);

      if (schemaEmbFields.length == 0) {
        result.put(structure.name(), List.of());
        continue;
      }

      var scoringResult = runtime.runScoringHead(
        spanRep,
        schemaEmbP,
        schemaEmbFields,
        config.getMaxCount()
      );

      int predCount = argmax(scoringResult.countLogits()[0]);
      log.debug(
        "Structure '{}' predicted count: {}",
        structure.name(),
        predCount
      );

      var instances = decoder.decode(
        scoringResult.spanScores(),
        predCount,
        structure,
        input.wordStartChars(),
        input.wordEndChars(),
        text,
        textLen,
        threshold
      );
      result.put(structure.name(), instances);
    }

    return result;
  }

  private PerStructureEmbeddings groupEmbeddingsByStructure(
    PreprocessedInput input,
    float[][] hiddenState,
    int numStructures
  ) {
    var specialTokenIds = config.getSpecialTokenIds();
    long pTokenId = specialTokenIds.getOrDefault("P", -1L);
    long cTokenId = specialTokenIds.getOrDefault("C", -1L);

    var pEmbs = new ArrayList<float[]>(numStructures);
    var cEmbs = new ArrayList<List<float[]>>(numStructures);
    int currentStructure = -1;

    for (int i = 0; i < input.inputIds().length; i++) {
      long id = input.inputIds()[i];
      if (id == pTokenId) {
        currentStructure++;
        if (currentStructure < numStructures) {
          pEmbs.add(hiddenState[i]);
          cEmbs.add(new ArrayList<>());
        }
      } else if (
        id == cTokenId &&
        currentStructure >= 0 &&
        currentStructure < numStructures
      ) {
        cEmbs.get(currentStructure).add(hiddenState[i]);
      }
    }

    // Pad with empties if any [P] tokens were missing (defensive)
    while (pEmbs.size() < numStructures) {
      pEmbs.add(new float[config.getHiddenSize()]);
      cEmbs.add(new ArrayList<>());
    }

    return new PerStructureEmbeddings(pEmbs, cEmbs);
  }

  private void extractTextEmbeddings(
    float[][] hiddenState,
    PreprocessedInput input,
    float[][] target
  ) {
    var seenWord = new boolean[input.textLen()];
    for (int i = 0; i < input.mappings().length; i++) {
      var mapping = input.mappings()[i];
      if (
        mapping.type() ==
          io.gravitee.lab.gliner4j.tokenizer.TokenMapping.SegmentType.TEXT &&
        !seenWord[mapping.origIdx()]
      ) {
        target[mapping.origIdx()] = hiddenState[i];
        seenWord[mapping.origIdx()] = true;
      }
    }
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

  @Override
  public void close() {
    runtime.close();
    tokenizer.close();
    log.info("GLiNER4jSchemaExtractor closed");
  }

  private record PerStructureEmbeddings(
    List<float[]> pEmbs,
    List<List<float[]>> cEmbs
  ) {}
}
