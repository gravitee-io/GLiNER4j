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
import io.gravitee.lab.gliner4j.postprocess.SpanDecoder;
import io.gravitee.lab.gliner4j.processor.MultiSchemaEmbeddings;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInput;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInputAssembler;
import io.gravitee.lab.gliner4j.processor.SchemaUnit;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.processor.UnitLayout;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.schema.ExtractionResult;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import io.gravitee.lab.gliner4j.schema.Schema;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.tokenizer.WhitespaceTokenSplitter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;

/**
 * Unified facade for GLiNER4j combined extraction — entities and relations in a single forward
 * pass over the encoder, driven by a {@link Schema} description.
 *
 * <p>The schema is rendered as a multi-unit prompt: the entity unit (if any) followed by one
 * relation unit per relation type, joined with {@code [SEP_STRUCT]}. The encoder runs once over
 * the full prompt; the scoring head runs once per unit using that unit's parent and child-marker
 * embeddings.
 *
 * <p>Classifications are not yet wired through this facade — use {@link GLiNER4jClassifier}
 * directly. Calling {@link #extract(String, Schema)} with a classification block in the schema
 * will throw an {@link UnsupportedOperationException}.
 *
 * <p>Usage:
 * <pre>{@code
 * var schema = Schema.builder()
 *     .entities(List.of(new EntityDefinition("person"), new EntityDefinition("organization")))
 *     .relations(List.of(new RelationDefinition("works_for", "Employment relationship")))
 *     .build();
 *
 * try (var gliner = GLiNER4j.load(modelDir)) {
 *     ExtractionResult result = gliner.extract("John works at Google.", schema);
 * }
 * }</pre>
 */
@Slf4j
public class GLiNER4j implements AutoCloseable {

  private final GLiNER4jConfig config;
  private final DjlTokenizerWrapper tokenizer;
  private final GLiNER4jNERRuntime runtime;
  private final WhitespaceTokenSplitter splitter;
  private final SpanDecoder spanDecoder;
  private final RelationDecoder relationDecoder;

  private GLiNER4j(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jNERRuntime runtime
  ) {
    this.config = config;
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.splitter = new WhitespaceTokenSplitter();
    this.spanDecoder = new SpanDecoder();
    this.relationDecoder = new RelationDecoder();
  }

  public static GLiNER4j load(Path modelDir) {
    return load(
      modelDir,
      BaseRuntime.DEFAULT_VARIANT,
      RuntimeConfig.defaults()
    );
  }

  public static GLiNER4j load(Path modelDir, String variant) {
    return load(modelDir, variant, RuntimeConfig.defaults());
  }

  public static GLiNER4j load(
    Path modelDir,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    log.info(
      "Loading GLiNER4j unified model from {} (variant={})",
      modelDir,
      variant
    );
    var config = GLiNER4jConfig.load(modelDir);
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var runtime = new GLiNER4jNERRuntime(modelDir, variant, runtimeConfig);
    log.info("GLiNER4j unified model loaded successfully");
    return new GLiNER4j(config, tokenizer, runtime);
  }

  public ExtractionResult extract(String text, Schema schema) {
    return extract(text, schema, config.getDefaultThreshold());
  }

  public ExtractionResult extract(String text, Schema schema, float threshold) {
    if (schema.hasClassifications()) {
      throw new UnsupportedOperationException(
        "Classifications in combined extraction are not yet supported. " +
          "Use GLiNER4jClassifier directly for classification-only extraction."
      );
    }
    if (text == null || text.isBlank()) {
      return emptyResult();
    }

    var textEncoder = new TextEncoder(text, splitter);
    if (textEncoder.getTextLen() == 0) {
      return emptyResult();
    }

    var assembler = buildAssembler(tokenizer, schema);
    var input = assembler.assemble(textEncoder);
    var hiddenStates = runtime.runEncoderFull(
      input.inputIds(),
      input.attentionMask()
    );

    return decodeAllUnits(hiddenStates[0], input, text, threshold);
  }

  public List<ExtractionResult> extractBatch(
    List<String> texts,
    Schema schema
  ) {
    return extractBatch(texts, schema, config.getDefaultThreshold());
  }

  public List<ExtractionResult> extractBatch(
    List<String> texts,
    Schema schema,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    var results = new ArrayList<ExtractionResult>(texts.size());
    for (var text : texts) {
      results.add(extract(text, schema, threshold));
    }
    return results;
  }

  private ExtractionResult decodeAllUnits(
    float[][] hiddenState,
    MultiSchemaInput input,
    String text,
    float threshold
  ) {
    int hiddenSize = config.getHiddenSize();
    int maxWidth = config.getMaxWidth();
    int textLen = input.textLen();

    var textEmbs = new float[textLen][hiddenSize];
    MultiSchemaEmbeddings.extractText(
      hiddenState,
      input.wordFirstSubwordPos(),
      textEmbs
    );

    int numSpans = textLen * maxWidth;
    var spanIdxFlat = buildSpanIdxFlat(textLen, maxWidth, numSpans);
    var textEmbs3d = new float[1][textLen][hiddenSize];
    System.arraycopy(textEmbs, 0, textEmbs3d[0], 0, textLen);
    var spanRep4d = runtime.runSpanRepFlat(textEmbs3d, spanIdxFlat, numSpans);
    var spanRep = spanRep4d[0];

    var entities = new LinkedHashMap<String, List<EntitySpan>>();
    var relations = new LinkedHashMap<String, List<RelationInstance>>();

    for (var layout : input.unitLayouts()) {
      switch (layout.unit().kind()) {
        case ENTITIES -> entities.putAll(
          decodeEntitiesUnit(
            layout,
            hiddenState,
            spanRep,
            input,
            text,
            threshold
          )
        );
        case RELATION -> relations.put(
          layout.unit().parentLabel(),
          decodeRelationUnit(
            layout,
            hiddenState,
            spanRep,
            input,
            text,
            threshold
          )
        );
        case CLASSIFICATIONS -> {
          // Filtered out earlier — should never reach here
        }
      }
    }

    return new ExtractionResult(
      entities,
      relations,
      List.<ClassificationResult>of()
    );
  }

  private Map<String, List<EntitySpan>> decodeEntitiesUnit(
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

    int predCount = argmax(scoringResult.countLogits()[0]);
    if (predCount == 0) {
      return emptyEntityMap(layout.unit().childNames());
    }

    var spans = spanDecoder.decode(
      scoringResult.spanScores(),
      layout.unit().childNames(),
      input.wordStartChars(),
      input.wordEndChars(),
      text,
      input.textLen(),
      threshold
    );

    var grouped = spans
      .stream()
      .collect(
        Collectors.groupingBy(
          EntitySpan::type,
          LinkedHashMap::new,
          Collectors.toList()
        )
      );

    // Ensure every requested entity type is present in the output map
    var result = emptyEntityMap(layout.unit().childNames());
    result.putAll(grouped);
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

  private static MultiSchemaInputAssembler buildAssembler(
    DjlTokenizerWrapper tokenizer,
    Schema schema
  ) {
    var units = new ArrayList<SchemaUnit>(
      (schema.hasEntities() ? 1 : 0) + schema.relations().size()
    );
    if (schema.hasEntities()) {
      units.add(SchemaUnit.forEntities(schema.entities()));
    }
    for (var rel : schema.relations()) {
      units.add(SchemaUnit.forRelation(rel));
    }
    return new MultiSchemaInputAssembler(tokenizer, units);
  }

  private static Map<String, List<EntitySpan>> emptyEntityMap(
    List<String> entityTypes
  ) {
    var map = new LinkedHashMap<String, List<EntitySpan>>();
    for (var type : entityTypes) {
      map.put(type, List.of());
    }
    return map;
  }

  private static ExtractionResult emptyResult() {
    return new ExtractionResult(
      Map.of(),
      Map.of(),
      List.<ClassificationResult>of()
    );
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
    log.info("GLiNER4j unified model closed");
  }
}
