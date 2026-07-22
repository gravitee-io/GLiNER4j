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
import io.gravitee.lab.gliner4j.processor.AssemblerCache;
import io.gravitee.lab.gliner4j.processor.BatchPreprocessor;
import io.gravitee.lab.gliner4j.processor.BatchSpanPipeline;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInput;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInputAssembler;
import io.gravitee.lab.gliner4j.processor.UnitLayout;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
import io.gravitee.lab.gliner4j.runtime.MicroBatcher;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import lombok.extern.slf4j.Slf4j;

/**
 * Shared scaffolding for multi-unit task facades (relation extraction, structured/JSON
 * extraction, …) on the merged {@code ner_full.onnx} graph.
 *
 * <p>All tasks in this family build a multi-unit prompt: one {@code [P] / child-marker} block
 * per definition (relation, structure, …) joined by {@code [SEP_STRUCT]}. The merged graph
 * scores one schema unit per session run (a single {@code p_position} + {@code field_positions}
 * set), so each unit is one bucketed full-graph pass over the batch — units fan out on virtual
 * threads (independent {@code OrtSession.run} calls overlap on CUDA).
 *
 * <p>Subclasses only supply:
 * <ul>
 *   <li>how to build a {@link MultiSchemaInputAssembler} from their definition type
 *       ({@link #buildAssembler(List)}), and</li>
 *   <li>how to decode one unit's scoring output into the per-unit instance list
 *       ({@link #decodeUnit}).</li>
 * </ul>
 *
 * @param <D> the definition type (e.g. {@code RelationDefinition}, {@code StructureDefinition})
 * @param <R> the per-unit instance type (e.g. {@code RelationInstance}, {@code StructureInstance})
 */
@Slf4j
public abstract sealed class Extractor<D, R>
  implements AutoCloseable
  permits RelationExtractor, SchemaExtractor {

  protected final GLiNER4jConfig config;
  protected final RuntimeConfig runtimeConfig;
  protected final List<D> definitions;
  protected final DjlTokenizerWrapper tokenizer;
  protected final GLiNER4jNERRuntime runtime;
  protected final MultiSchemaInputAssembler inputAssembler;
  protected final GLiNER4jTelemetry telemetry;
  private final AssemblerCache<
    List<D>,
    MultiSchemaInputAssembler
  > overrideAssemblers;
  private final MicroBatcher<Map<String, List<R>>> microBatcher;

  protected Extractor(
    GLiNER4jConfig config,
    RuntimeConfig runtimeConfig,
    List<D> definitions,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jNERRuntime runtime,
    MultiSchemaInputAssembler inputAssembler,
    GLiNER4jTelemetry telemetry
  ) {
    this.config = config;
    this.runtimeConfig = runtimeConfig;
    this.definitions = definitions;
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.inputAssembler = inputAssembler;
    this.telemetry = telemetry;
    this.overrideAssemblers = new AssemblerCache<>(
      runtimeConfig.effectiveOverrideCacheSize(),
      this::buildAssembler
    );
    this.microBatcher = runtimeConfig.isMicroBatchingEnabled()
      ? new MicroBatcher<>(
        runtimeConfig.getMicroBatchMaxSize(),
        runtimeConfig.getMicroBatchMaxWaitMicros(),
        this::doExtractBatch
      )
      : null;
  }

  // ---- subclass hooks ------------------------------------------------------

  /**
   * Builds a multi-unit assembler from the given definitions. Called once at load time by
   * subclass {@code load(...)} factories, and again on each override-path call.
   */
  protected abstract MultiSchemaInputAssembler buildAssembler(
    List<D> definitions
  );

  /**
   * Decodes one unit's scoring output into a list of typed instances.
   *
   * @param definition the original definition that produced this unit (same index as {@code layout})
   * @param layout     the resolved unit positions (parent + child markers)
   * @param countLogits the unit's count logits, shape {@code [1][maxCount+1]}
   * @param spanScores the unit's span scores for this text,
   *                   shape {@code [count][numFields][textLen][maxWidth]}
   * @param input      the assembled input for this text
   * @param text       the original input text
   * @param textLen    number of words in {@code text}
   * @param threshold  minimum confidence for an instance to be kept
   * @return decoded instances for this unit (possibly empty)
   */
  protected abstract List<R> decodeUnit(
    D definition,
    UnitLayout layout,
    float[][] countLogits,
    float[][][][] spanScores,
    MultiSchemaInput input,
    String text,
    int textLen,
    float threshold
  );

  // ---- shared single-text / override paths ---------------------------------

  /**
   * Runs the load-time assembler against {@code text} (batch of one through the merged graph).
   */
  protected final Map<String, List<R>> doExtractOnce(
    String text,
    float threshold
  ) {
    if (microBatcher != null) {
      return microBatcher.extract(text, threshold);
    }
    return extractMerged(
      List.of(text == null ? "" : text),
      threshold,
      definitions,
      inputAssembler
    ).get(0);
  }

  /**
   * Runs a per-call schema against {@code text} via a fresh assembler. The merged graph embeds
   * the encoder, so the load-time and override assemblers take the exact same path.
   */
  protected final Map<String, List<R>> doExtractOverride(
    String text,
    List<D> overrideDefinitions,
    float threshold
  ) {
    return extractMerged(
      List.of(text == null ? "" : text),
      threshold,
      overrideDefinitions,
      overrideAssemblers.get(overrideDefinitions)
    ).get(0);
  }

  // ---- shared batch path ---------------------------------------------------

  /**
   * Batched multi-text extraction through the merged graph.
   */
  protected final List<Map<String, List<R>>> doExtractBatch(
    List<String> texts,
    float threshold
  ) {
    return extractMerged(texts, threshold, definitions, inputAssembler);
  }

  /**
   * The one extraction path: preprocess the batch once, then score one bucketed
   * {@code ner_full} pass per schema unit, fanned out on virtual threads, and fold the
   * per-unit results back into per-text maps in unit order.
   */
  private List<Map<String, List<R>>> extractMerged(
    List<String> texts,
    float threshold,
    List<D> activeDefs,
    MultiSchemaInputAssembler assembler
  ) {
    if (texts == null) {
      return List.of();
    }
    long startNanos = System.nanoTime();
    int batchSize = texts.size();

    var preproc = BatchPreprocessor.preprocess(texts, assembler);
    int nonEmptyCount = preproc.nonEmptyCount();

    var results = new ArrayList<Map<String, List<R>>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(Map.of());
    }

    if (nonEmptyCount == 0) {
      double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
      telemetry.record(durationMs, batchSize, 0);
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

    // One merged-graph pass per unit, fanned out on virtual threads. OrtSession.run() is
    // thread-safe; on CUDA, independent runs dispatch onto separate streams and overlap.
    @SuppressWarnings("unchecked")
    var unitFutures = (Future<List<List<R>>>[]) new Future[numUnits];

    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int u = 0; u < numUnits; u++) {
        final var layout = layouts.get(u);
        final var def = activeDefs.get(u);
        unitFutures[u] = executor.submit(() ->
          scoreUnit(
            def,
            layout,
            preproc,
            textLens,
            texts,
            threshold,
            nonEmptyCount
          )
        );
      }

      var perUnitResults = new ArrayList<List<List<R>>>(numUnits);
      for (int u = 0; u < numUnits; u++) {
        try {
          perUnitResults.add(unitFutures[u].get());
        } catch (ExecutionException e) {
          throw new RuntimeException(
            getClass().getSimpleName() +
              " merged scoring failed for unit " +
              layouts.get(u).unit().parentLabel(),
            e.getCause()
          );
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException(
            getClass().getSimpleName() + " unit fanout interrupted",
            e
          );
        }
      }

      // Fold per-unit slot results into per-text maps, preserving unit order.
      for (int s = 0; s < nonEmptyCount; s++) {
        var perTextResult = new LinkedHashMap<String, List<R>>();
        for (int u = 0; u < numUnits; u++) {
          perTextResult.put(
            layouts.get(u).unit().parentLabel(),
            perUnitResults.get(u).get(s)
          );
        }
        results.set(batchIndices[s], perTextResult);
      }
    }

    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    long total = results.stream().mapToLong(Extractor::totalInstances).sum();
    telemetry.record(durationMs, batchSize, total);
    return results;
  }

  /**
   * Scores one schema unit over the whole batch: one bucketed {@code ner_full} pass with
   * this unit's {@code [P]} / child-marker positions, decoded slot by slot.
   *
   * @return per-slot instance lists, indexed like {@code textLens}
   */
  private List<List<R>> scoreUnit(
    D def,
    UnitLayout layout,
    BatchPreprocessor.MultiResult preproc,
    int[] textLens,
    List<String> texts,
    float threshold,
    int nonEmptyCount
  ) {
    var slotResults = new ArrayList<List<R>>(nonEmptyCount);
    for (int s = 0; s < nonEmptyCount; s++) {
      slotResults.add(List.of());
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
          config.getMaxCount()
        )
    );

    try {
      for (var bucket : pipeline.buckets()) {
        try (var scoring = bucket.scoring()) {
          for (int j = 0; j < bucket.slots().length; j++) {
            int si = bucket.slots()[j];
            int origIdx = batchIndices[si];
            var input = inputs[origIdx];
            var spanScores = scoring.materializeSlot(j, textLens[si]);
            slotResults.set(
              si,
              decodeUnit(
                def,
                layout,
                scoring.countLogits(),
                spanScores,
                input,
                texts.get(origIdx),
                textLens[si],
                threshold
              )
            );
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

  // ---- lifecycle / helpers -------------------------------------------------

  @Override
  public final void close() {
    if (microBatcher != null) {
      microBatcher.close();
    }
    runtime.close();
    tokenizer.close();
    log.info("{} closed", getClass().getSimpleName());
  }

  private static long totalInstances(Map<String, ? extends List<?>> result) {
    return result.values().stream().mapToLong(List::size).sum();
  }
}
