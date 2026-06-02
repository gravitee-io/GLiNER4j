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

import static io.gravitee.lab.gliner4j.utils.LinAlg.argmax;

import io.gravitee.lab.gliner4j.arch.ModelArchitectures;
import io.gravitee.lab.gliner4j.arch.TaskType;
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
import io.gravitee.lab.gliner4j.utils.GlinerNerSupport;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
    ModelArchitectures.forId(config.getArchitecture()).requireSupported(
      TaskType.RELATION
    );
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

    var textEncoder = new TextEncoder(text);
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

  /**
   * Runs combined entity + relation extraction over multiple texts in a single batched pass.
   *
   * <p>The schema is rendered once into a {@link MultiSchemaInputAssembler}; the encoder runs
   * once over the padded {@code [batchSize, maxSeqLen]} batch; span_rep runs once over the
   * padded batch; then per text we fan out scoring_head + decode across all units on virtual
   * threads. Schema-marker embeddings are read once from a representative slot and shared.
   *
   * <p>Classifications in the schema throw {@link UnsupportedOperationException} — same as
   * the single-text path.
   *
   * @param texts the input texts to analyze
   * @param schema entity and relation definitions to extract
   * @param threshold minimum confidence score (0..1) for span inclusion
   * @return list of per-text extraction results, in the same order as the input
   */
  public List<ExtractionResult> extractBatch(
    List<String> texts,
    Schema schema,
    float threshold
  ) {
    if (schema.hasClassifications()) {
      throw new UnsupportedOperationException(
        "Classifications in combined extraction are not yet supported. " +
          "Use GLiNER4jClassifier directly for classification-only extraction."
      );
    }
    if (texts == null) {
      return List.of();
    }
    int batchSize = texts.size();

    // Initialize all slots with empty result; null/blank/empty-after-split texts keep these
    var results = new ArrayList<ExtractionResult>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(emptyResult());
    }

    // Schema is constant across the batch — build the assembler once
    var assembler = buildAssembler(tokenizer, schema);

    // 1. Preprocess all texts
    var inputs = new MultiSchemaInput[batchSize];
    int maxSeqLen = 0;
    int nonEmptyCount = 0;
    for (int i = 0; i < batchSize; i++) {
      var text = texts.get(i);
      if (text == null || text.isBlank()) continue;
      var textEncoder = new TextEncoder(text);
      if (textEncoder.getTextLen() == 0) continue;
      inputs[i] = assembler.assemble(textEncoder);
      maxSeqLen = Math.max(maxSeqLen, inputs[i].inputIds().length);
      nonEmptyCount++;
    }

    if (nonEmptyCount == 0) {
      return results;
    }

    // 2. Pack non-empty inputs for the encoder
    var batchInputIds = new long[nonEmptyCount][];
    var batchAttentionMask = new long[nonEmptyCount][];
    var batchIndices = new int[nonEmptyCount];
    int slot = 0;
    for (int i = 0; i < batchSize; i++) {
      if (inputs[i] != null) {
        batchInputIds[slot] = inputs[i].inputIds();
        batchAttentionMask[slot] = inputs[i].attentionMask();
        batchIndices[slot] = i;
        slot++;
      }
    }

    // 3. Single batched encoder call
    var batchedHiddenStates = runtime.runEncoderBatch(
      batchInputIds,
      batchAttentionMask,
      maxSeqLen
    );

    // 4. Cache per-unit ([P] + child markers) embeddings from a representative slot
    var layouts = inputs[batchIndices[0]].unitLayouts();
    var unitEmbsCache =
      new MultiSchemaEmbeddings.UnitEmbeddings[layouts.size()];
    for (int u = 0; u < layouts.size(); u++) {
      unitEmbsCache[u] = MultiSchemaEmbeddings.extractUnit(
        batchedHiddenStates[0],
        layouts.get(u)
      );
    }

    // 5. Extract per-text text embeddings and find the batch max
    int hiddenSize = config.getHiddenSize();
    int maxWidth = config.getMaxWidth();
    int maxTextLen = 0;
    var textLens = new int[nonEmptyCount];
    var perTextEmbs = new float[nonEmptyCount][][];
    for (int s = 0; s < nonEmptyCount; s++) {
      var input = inputs[batchIndices[s]];
      int textLen = input.textLen();
      textLens[s] = textLen;
      if (textLen > maxTextLen) maxTextLen = textLen;
      perTextEmbs[s] = new float[textLen][hiddenSize];
      MultiSchemaEmbeddings.extractText(
        batchedHiddenStates[s],
        input.wordFirstSubwordPos(),
        perTextEmbs[s]
      );
    }

    // 6. Pad embeddings and build flat span indices, then run batched span_rep
    int maxNumSpans = maxTextLen * maxWidth;
    var batchTextEmbs = new float[nonEmptyCount][maxTextLen][hiddenSize];
    var batchSpanIdxFlat = new long[nonEmptyCount * maxNumSpans * 2];
    for (int s = 0; s < nonEmptyCount; s++) {
      int textLen = textLens[s];
      System.arraycopy(perTextEmbs[s], 0, batchTextEmbs[s], 0, textLen);
      int batchOffset = s * maxNumSpans * 2;
      for (int i = 0; i < textLen; i++) {
        for (int w = 0; w < maxWidth; w++) {
          int endPos = i + w;
          if (endPos < textLen) {
            int flatIdx = batchOffset + (i * maxWidth + w) * 2;
            batchSpanIdxFlat[flatIdx] = i;
            batchSpanIdxFlat[flatIdx + 1] = endPos;
          }
        }
      }
    }

    var batchSpanRep4d = runtime.runSpanRepBatch(
      batchTextEmbs,
      batchSpanIdxFlat,
      nonEmptyCount,
      maxNumSpans
    );

    // 7. Per-text fanout: scoring_head + decode for each unit (entities or relation)
    @SuppressWarnings("unchecked")
    var futures = new Future[nonEmptyCount];

    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int s = 0; s < nonEmptyCount; s++) {
        final int si = s;
        futures[si] = executor.submit(() -> {
          int origIdx = batchIndices[si];
          var input = inputs[origIdx];
          int textLen = textLens[si];
          var text = texts.get(origIdx);

          // Slice the per-text span_rep view from the padded batch tensor
          var spanRep = Arrays.copyOfRange(batchSpanRep4d[si], 0, textLen);

          var entities = new LinkedHashMap<String, List<EntitySpan>>();
          var relations = new LinkedHashMap<String, List<RelationInstance>>();

          for (int u = 0; u < layouts.size(); u++) {
            var layout = layouts.get(u);
            var unitEmbs = unitEmbsCache[u];
            switch (layout.unit().kind()) {
              case ENTITIES -> entities.putAll(
                decodeEntitiesUnit(
                  layout,
                  unitEmbs,
                  spanRep,
                  input,
                  text,
                  textLen,
                  threshold
                )
              );
              case RELATION -> relations.put(
                layout.unit().parentLabel(),
                decodeRelationUnit(
                  layout,
                  unitEmbs,
                  spanRep,
                  input,
                  text,
                  textLen,
                  threshold
                )
              );
              case CLASSIFICATIONS -> {
                // Filtered out earlier
              }
            }
          }

          return new ExtractionResult(
            entities,
            relations,
            List.<ClassificationResult>of()
          );
        });
      }

      for (int s = 0; s < nonEmptyCount; s++) {
        try {
          int origIdx = batchIndices[s];
          var result = (ExtractionResult) futures[s].get();
          results.set(origIdx, result);
        } catch (ExecutionException e) {
          throw new RuntimeException(
            "Unified scoring failed for batch slot " + s,
            e.getCause()
          );
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("Batch unified scoring interrupted", e);
        }
      }
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
      var unitEmbs = MultiSchemaEmbeddings.extractUnit(hiddenState, layout);
      switch (layout.unit().kind()) {
        case ENTITIES -> entities.putAll(
          decodeEntitiesUnit(
            layout,
            unitEmbs,
            spanRep,
            input,
            text,
            input.textLen(),
            threshold
          )
        );
        case RELATION -> relations.put(
          layout.unit().parentLabel(),
          decodeRelationUnit(
            layout,
            unitEmbs,
            spanRep,
            input,
            text,
            input.textLen(),
            threshold
          )
        );
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
      textLen,
      threshold
    );

    var grouped = GlinerNerSupport.groupByType(spans);

    // Ensure every requested entity type is present in the output map
    var result = emptyEntityMap(layout.unit().childNames());
    result.putAll(grouped);
    return result;
  }

  private List<RelationInstance> decodeRelationUnit(
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

  @Override
  public void close() {
    runtime.close();
    tokenizer.close();
    log.info("GLiNER4j unified model closed");
  }
}
