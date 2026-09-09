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
import io.gravitee.lab.gliner4j.processor.BatchPreprocessor;
import io.gravitee.lab.gliner4j.processor.BatchSpanPipeline;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInputAssembler;
import io.gravitee.lab.gliner4j.processor.SchemaUnit;
import io.gravitee.lab.gliner4j.processor.UnitLayout;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.Gliner2SpanRuntime;
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
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import lombok.extern.slf4j.Slf4j;

/**
 * Unified facade for GLiNER4j combined extraction — entities and relations driven by a
 * {@link Schema} description, scored through the merged {@code ner_full.onnx} graph.
 *
 * <p>The schema is rendered as a multi-unit prompt: the entity unit (if any) followed by one
 * relation unit per relation type, joined with {@code [SEP_STRUCT]}. The merged graph scores
 * one schema unit per session run (single {@code p_position} + {@code field_positions} set),
 * so each unit is one bucketed full-graph pass — units fan out on virtual threads.
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

  // Entity units decode only count-instance 0 (see Gliner2NerStrategy); relation units slice
  // to maxCount and decode up to argmax(count_logits) instances.
  private static final long ENTITY_COUNT_INSTANCES = 1;

  private final GLiNER4jConfig config;
  private final RuntimeConfig runtimeConfig;
  private final DjlTokenizerWrapper tokenizer;
  private final Gliner2SpanRuntime runtime;
  private final SpanDecoder spanDecoder;
  private final RelationDecoder relationDecoder;

  private GLiNER4j(
    GLiNER4jConfig config,
    RuntimeConfig runtimeConfig,
    DjlTokenizerWrapper tokenizer,
    Gliner2SpanRuntime runtime
  ) {
    this.config = config;
    this.runtimeConfig = runtimeConfig;
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
    // The unified facade runs entities, relations and structures as units of one GLiNER2
    // multi-unit prompt through the count-aware span graph — it needs the full GLiNER2 task set.
    var architecture = ModelArchitectures.forConfig(config);
    architecture.requireSupported(TaskType.RELATION);
    architecture.requireSupported(TaskType.STRUCTURE);
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var runtime = architecture.newSpanRuntime(
      new io.gravitee.lab.gliner4j.arch.LoadContext(
        modelDir,
        variant,
        runtimeConfig,
        config,
        tokenizer
      )
    );
    log.info("GLiNER4j unified model loaded successfully");
    return new GLiNER4j(config, runtimeConfig, tokenizer, runtime);
  }

  public ExtractionResult extract(String text, Schema schema) {
    return extract(text, schema, config.getDefaultThreshold());
  }

  public ExtractionResult extract(String text, Schema schema, float threshold) {
    return extractBatch(
      List.of(text == null ? "" : text),
      schema,
      threshold
    ).get(0);
  }

  public List<ExtractionResult> extractBatch(
    List<String> texts,
    Schema schema
  ) {
    return extractBatch(texts, schema, config.getDefaultThreshold());
  }

  /**
   * Runs combined entity + relation extraction over multiple texts through the merged graph.
   *
   * <p>The schema is rendered once into a {@link MultiSchemaInputAssembler}; then per schema
   * unit one bucketed {@code ner_full} pass scores the whole batch, fanned out over units on
   * virtual threads.
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
    var preproc = BatchPreprocessor.preprocess(texts, assembler);
    int nonEmptyCount = preproc.nonEmptyCount();
    if (nonEmptyCount == 0) {
      return results;
    }

    var inputs = preproc.inputs();
    var batchIndices = preproc.batchIndices();

    var textLens = new int[nonEmptyCount];
    for (int s = 0; s < nonEmptyCount; s++) {
      textLens[s] = inputs[batchIndices[s]].textLen();
    }

    // Unit layouts are identical across the batch (shared schema prefix).
    var layouts = inputs[batchIndices[0]].unitLayouts();
    int numUnits = layouts.size();

    // One merged-graph pass per unit, fanned out on virtual threads.
    @SuppressWarnings("unchecked")
    var unitFutures = (Future<List<Object>>[]) new Future[numUnits];

    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int u = 0; u < numUnits; u++) {
        final var layout = layouts.get(u);
        unitFutures[u] = executor.submit(() ->
          scoreUnit(layout, preproc, textLens, texts, threshold, nonEmptyCount)
        );
      }

      var perUnitResults = new ArrayList<List<Object>>(numUnits);
      for (int u = 0; u < numUnits; u++) {
        try {
          perUnitResults.add(unitFutures[u].get());
        } catch (ExecutionException e) {
          throw new RuntimeException(
            "Unified scoring failed for unit " +
              layouts.get(u).unit().parentLabel(),
            e.getCause()
          );
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("Batch unified scoring interrupted", e);
        }
      }

      for (int s = 0; s < nonEmptyCount; s++) {
        var entities = new LinkedHashMap<String, List<EntitySpan>>();
        var relations = new LinkedHashMap<String, List<RelationInstance>>();
        for (int u = 0; u < numUnits; u++) {
          var layout = layouts.get(u);
          var unitResult = perUnitResults.get(u).get(s);
          switch (layout.unit().kind()) {
            case ENTITIES -> {
              @SuppressWarnings("unchecked")
              var perText = (Map<String, List<EntitySpan>>) unitResult;
              entities.putAll(perText);
            }
            case RELATION -> {
              @SuppressWarnings("unchecked")
              var perText = (List<RelationInstance>) unitResult;
              relations.put(layout.unit().parentLabel(), perText);
            }
            case CLASSIFICATIONS -> {
              // Filtered out earlier
            }
          }
        }
        results.set(
          batchIndices[s],
          new ExtractionResult(
            entities,
            relations,
            List.<ClassificationResult>of()
          )
        );
      }
    }

    return results;
  }

  /**
   * Scores one schema unit over the whole batch through {@code ner_full}. Returns one decoded
   * result per slot: a {@code Map<String, List<EntitySpan>>} for ENTITIES units, a
   * {@code List<RelationInstance>} for RELATION units.
   */
  private List<Object> scoreUnit(
    UnitLayout layout,
    BatchPreprocessor.MultiResult preproc,
    int[] textLens,
    List<String> texts,
    float threshold,
    int nonEmptyCount
  ) {
    boolean isEntities = layout.unit().kind() == SchemaUnit.Kind.ENTITIES;
    var slotResults = new ArrayList<Object>(nonEmptyCount);
    for (int s = 0; s < nonEmptyCount; s++) {
      slotResults.add(
        isEntities
          ? emptyEntityMap(layout.unit().childNames())
          : List.<RelationInstance>of()
      );
    }
    if (layout.childMarkerPositions().length == 0) {
      return slotResults;
    }

    var inputs = preproc.inputs();
    var batchIndices = preproc.batchIndices();

    var fieldPositions = new long[layout.childMarkerPositions().length];
    for (int i = 0; i < fieldPositions.length; i++) {
      fieldPositions[i] = layout.childMarkerPositions()[i];
    }
    long count = isEntities ? ENTITY_COUNT_INSTANCES : config.getMaxCount();

    var pipeline = BatchSpanPipeline.runFullGraph(
      preproc.batchInputIds(),
      preproc.batchAttentionMask(),
      textLens,
      slot -> inputs[batchIndices[slot]].wordFirstSubwordPos(),
      config.getMaxWidth(),
      runtimeConfig.getBatchLengthRatio(),
      runtimeConfig.getMaxSubBatchSize(),
      (ids, mask, maxSeqLen, wpFlat, bSize, maxTextLen, spanIdxFlat) ->
        runtime.runNerFullBatch(
          ids,
          mask,
          maxSeqLen,
          wpFlat,
          bSize,
          maxTextLen,
          layout.parentTokenPos(),
          fieldPositions,
          spanIdxFlat,
          config.getMaxWidth(),
          count
        )
    );

    try {
      for (var bucket : pipeline.buckets()) {
        try (var scoring = bucket.scoring()) {
          for (int j = 0; j < bucket.slots().length; j++) {
            if (isEntities && argmax(scoring.countLogitsFor(j)) == 0) {
              continue;
            }
            int si = bucket.slots()[j];
            int origIdx = batchIndices[si];
            var input = inputs[origIdx];
            var text = texts.get(origIdx);
            int textLen = textLens[si];
            if (isEntities) {
              var spans = spanDecoder.decode(
                scoring.spanScores(),
                j,
                layout.unit().childNames(),
                input.wordStartChars(),
                input.wordEndChars(),
                text,
                textLen,
                threshold
              );
              // Ensure every requested entity type is present in the output map
              var perText = emptyEntityMap(layout.unit().childNames());
              perText.putAll(GlinerNerSupport.groupByType(spans));
              slotResults.set(si, perText);
            } else {
              slotResults.set(
                si,
                relationDecoder.decode(
                  layout.unit().parentLabel(),
                  layout.unit().childNames(),
                  new float[][] { scoring.countLogitsFor(j) },
                  scoring.materializeSlot(j, textLen),
                  input.wordStartChars(),
                  input.wordEndChars(),
                  text,
                  textLen,
                  threshold
                )
              );
            }
          }
        }
      }
    } finally {
      // Idempotent: releases anything not yet closed (e.g. on a decode failure).
      for (var bucket : pipeline.buckets()) {
        bucket.scoring().close();
      }
    }
    return slotResults;
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

  @Override
  public void close() {
    runtime.close();
    tokenizer.close();
    log.info("GLiNER4j unified model closed");
  }
}
