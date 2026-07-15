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
import io.gravitee.lab.gliner4j.processor.MultiSchemaEmbeddings;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInput;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInputAssembler;
import io.gravitee.lab.gliner4j.processor.SpanIndexCache;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
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
 * extraction, …).
 *
 * <p>All tasks in this family build a multi-unit prompt: one {@code [P] / child-marker} block
 * per definition (relation, structure, …) joined by {@code [SEP_STRUCT]}. The encoder runs once
 * over the full prompt, {@code span_rep} once over the text region, then the scoring head runs
 * per unit with that unit's parent + child-marker embeddings, and the result is decoded into a
 * per-unit list of instances.
 *
 * <p>Subclasses only supply:
 * <ul>
 *   <li>how to build a {@link MultiSchemaInputAssembler} from their definition type
 *       ({@link #buildAssembler(List)}), and</li>
 *   <li>how to decode one unit's scoring result into the per-unit instance list
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
   * Decodes one unit's scoring head output into a list of typed instances.
   *
   * @param definition the original definition that produced this unit (same index as {@code layout})
   * @param layout     the resolved unit positions (parent + child markers)
   * @param unitEmbs   the unit's {@code [P]} and child-marker embeddings
   * @param spanRep    the per-text span representations, shape {@code [textLen][maxWidth][hiddenSize]}
   * @param input      the assembled input for this text
   * @param text       the original input text
   * @param textLen    number of words in {@code text}
   * @param threshold  minimum confidence for an instance to be kept
   * @return decoded instances for this unit (possibly empty)
   */
  protected abstract List<R> decodeUnit(
    D definition,
    UnitLayout layout,
    MultiSchemaEmbeddings.UnitEmbeddings unitEmbs,
    float[][][] spanRep,
    MultiSchemaInput input,
    String text,
    int textLen,
    float threshold
  );

  // ---- shared single-text / override paths ---------------------------------

  /**
   * Runs the load-time assembler against {@code text}. Uses the runtime's prefix-cached encoder.
   */
  protected final Map<String, List<R>> doExtractOnce(
    String text,
    float threshold
  ) {
    if (microBatcher != null) {
      return microBatcher.extract(text, threshold);
    }
    return doExtractWith(text, threshold, definitions, inputAssembler, false);
  }

  /**
   * Runs a per-call schema against {@code text} via a fresh assembler. Uses
   * {@code runEncoderFull} since the prefix cache is keyed on the load-time schema.
   */
  protected final Map<String, List<R>> doExtractOverride(
    String text,
    List<D> overrideDefinitions,
    float threshold
  ) {
    var overrideAssembler = overrideAssemblers.get(overrideDefinitions);
    return doExtractWith(
      text,
      threshold,
      overrideDefinitions,
      overrideAssembler,
      true
    );
  }

  private Map<String, List<R>> doExtractWith(
    String text,
    float threshold,
    List<D> activeDefs,
    MultiSchemaInputAssembler assembler,
    boolean useFullEncoder
  ) {
    long startNanos = System.nanoTime();
    if (text == null || text.isBlank()) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }
    var textEncoder = new TextEncoder(text);
    if (textEncoder.getTextLen() == 0) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    var input = assembler.assemble(textEncoder);
    var hiddenStates = useFullEncoder
      ? runtime.runEncoderFull(input.inputIds(), input.attentionMask())
      : runtime.runEncoder(input.inputIds(), input.attentionMask());

    var result = decodeAllUnits(
      hiddenStates[0],
      input,
      activeDefs,
      text,
      threshold
    );
    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    telemetry.record(durationMs, 1, totalInstances(result));
    return result;
  }

  private Map<String, List<R>> decodeAllUnits(
    float[][] hiddenState,
    MultiSchemaInput input,
    List<D> activeDefs,
    String text,
    float threshold
  ) {
    int hiddenSize = config.getHiddenSize();
    int maxWidth = config.getMaxWidth();
    int textLen = input.textLen();

    // 1. Pull text embeddings out (shared across all units)
    var textEmbs = new float[textLen][hiddenSize];
    MultiSchemaEmbeddings.extractText(
      hiddenState,
      input.wordFirstSubwordPos(),
      textEmbs
    );

    // 2. One span_rep call per text
    int numSpans = textLen * maxWidth;
    var spanIdxFlat = SpanIndexCache.flatSpanIdx(textLen, maxWidth);
    var textEmbs3d = new float[1][textLen][hiddenSize];
    System.arraycopy(textEmbs, 0, textEmbs3d[0], 0, textLen);
    var spanRep4d = runtime.runSpanRepFlat(textEmbs3d, spanIdxFlat, numSpans);
    var spanRep = spanRep4d[0];

    // 3. Per unit: scoring head + decode
    var layouts = input.unitLayouts();
    var result = new LinkedHashMap<String, List<R>>();
    for (int u = 0; u < layouts.size(); u++) {
      var layout = layouts.get(u);
      var def = activeDefs.get(u);
      var unitEmbs = MultiSchemaEmbeddings.extractUnit(hiddenState, layout);
      var instances = decodeUnit(
        def,
        layout,
        unitEmbs,
        spanRep,
        input,
        text,
        textLen,
        threshold
      );
      result.put(layout.unit().parentLabel(), instances);
    }
    return result;
  }

  // ---- shared batch path ---------------------------------------------------

  /**
   * Batched multi-text extraction: encoder + span_rep batched, per-unit embeddings cached
   * once from a representative slot, then per-text fanout over the units on virtual threads.
   */
  protected final List<Map<String, List<R>>> doExtractBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    long startNanos = System.nanoTime();
    int batchSize = texts.size();

    // 1. Preprocess and pack the contiguous batch
    var preproc = BatchPreprocessor.preprocess(texts, inputAssembler);
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

    // 2-4. Bucketed encoder + span_rep (padded per sub-batch, not batch-wide)
    var textLens = new int[nonEmptyCount];
    for (int s = 0; s < nonEmptyCount; s++) {
      textLens[s] = inputs[batchIndices[s]].textLen();
    }

    var pipeline = BatchSpanPipeline.run(
      runtime,
      preproc.batchInputIds(),
      preproc.batchAttentionMask(),
      textLens,
      config.getHiddenSize(),
      config.getMaxWidth(),
      runtimeConfig.getBatchLengthRatio(),
      runtimeConfig.getMaxSubBatchSize(),
      // per-unit fanout scores per slot, so it needs the nested per-slot span reps
      true,
      // only the [P] + child marker rows of the representative slot are read
      slot ->
        inputs[batchIndices[slot]].unitLayouts()
          .stream()
          .flatMapToInt(l ->
            java.util.stream.IntStream.concat(
              java.util.stream.IntStream.of(l.parentTokenPos()),
              java.util.stream.IntStream.of(l.childMarkerPositions())
            )
          )
          .toArray(),
      // multi-unit fanout scores per unit; the merged NER graph does not apply
      null,
      (hidden, row, s, target, targetBase) ->
        MultiSchemaEmbeddings.extractTextFlat(
          hidden,
          row,
          inputs[batchIndices[s]].wordFirstSubwordPos(),
          target,
          targetBase
        )
    );

    // 5. Cache per-unit ([P] + child markers) embeddings from a representative slot
    var layouts = inputs[batchIndices[pipeline.repSlot()]].unitLayouts();
    var unitEmbsCache =
      new MultiSchemaEmbeddings.UnitEmbeddings[layouts.size()];
    for (int u = 0; u < layouts.size(); u++) {
      unitEmbsCache[u] = MultiSchemaEmbeddings.extractUnit(
        pipeline.repHiddenState(),
        layouts.get(u)
      );
    }

    // 6. Per-text fanout: scoring_head + decode for each unit on virtual threads.
    var fanned = new ArrayList<Map<String, List<R>>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      fanned.add(Map.of());
    }

    @SuppressWarnings("unchecked")
    var futures = (Future<Map<String, List<R>>>[]) new Future[nonEmptyCount];

    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int s = 0; s < nonEmptyCount; s++) {
        final int si = s;
        futures[si] = executor.submit(() -> {
          int origIdx = batchIndices[si];
          var input = inputs[origIdx];
          int textLen = textLens[si];
          var text = texts.get(origIdx);

          var spanRep = pipeline.spanRepBySlot()[si];

          var perTextResult = new LinkedHashMap<String, List<R>>();
          for (int u = 0; u < layouts.size(); u++) {
            var layout = layouts.get(u);
            var def = definitions.get(u);
            var unitEmbs = unitEmbsCache[u];
            var instances = decodeUnit(
              def,
              layout,
              unitEmbs,
              spanRep,
              input,
              text,
              textLen,
              threshold
            );
            perTextResult.put(layout.unit().parentLabel(), instances);
          }
          return perTextResult;
        });
      }

      for (int s = 0; s < nonEmptyCount; s++) {
        try {
          fanned.set(batchIndices[s], futures[s].get());
        } catch (ExecutionException e) {
          throw new RuntimeException(
            getClass().getSimpleName() +
              " scoring head failed for batch slot " +
              s,
            e.getCause()
          );
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException(
            getClass().getSimpleName() + " batch fanout interrupted",
            e
          );
        }
      }
    }

    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    long total = fanned.stream().mapToLong(Extractor::totalInstances).sum();
    telemetry.record(durationMs, batchSize, total);
    return fanned;
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
