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
import io.gravitee.lab.gliner4j.processor.BatchPreprocessor;
import io.gravitee.lab.gliner4j.processor.MultiSchemaEmbeddings;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInput;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInputAssembler;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.processor.UnitLayout;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
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
  protected final List<D> definitions;
  protected final DjlTokenizerWrapper tokenizer;
  protected final GLiNER4jNERRuntime runtime;
  protected final MultiSchemaInputAssembler inputAssembler;
  protected final GLiNER4jTelemetry telemetry;

  protected Extractor(
    GLiNER4jConfig config,
    List<D> definitions,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jNERRuntime runtime,
    MultiSchemaInputAssembler inputAssembler,
    GLiNER4jTelemetry telemetry
  ) {
    this.config = config;
    this.definitions = definitions;
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.inputAssembler = inputAssembler;
    this.telemetry = telemetry;
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
    var overrideAssembler = buildAssembler(overrideDefinitions);
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
    var spanIdxFlat = buildSpanIdxFlat(textLen, maxWidth, numSpans);
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

    // 2. Single batched encoder call
    var batchedHiddenStates = runtime.runEncoderBatch(
      preproc.batchInputIds(),
      preproc.batchAttentionMask(),
      preproc.maxSeqLen()
    );

    // 3. Cache per-unit ([P] + child markers) embeddings from a representative slot
    var layouts = inputs[batchIndices[0]].unitLayouts();
    var unitEmbsCache =
      new MultiSchemaEmbeddings.UnitEmbeddings[layouts.size()];
    for (int u = 0; u < layouts.size(); u++) {
      unitEmbsCache[u] = MultiSchemaEmbeddings.extractUnit(
        batchedHiddenStates[0],
        layouts.get(u)
      );
    }

    // 4. Extract per-text text embeddings and find the batch max
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

    // 5. Pad embeddings and build flat span indices, then run batched span_rep
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

          var spanRep = Arrays.copyOfRange(batchSpanRep4d[si], 0, textLen);

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
    runtime.close();
    tokenizer.close();
    log.info("{} closed", getClass().getSimpleName());
  }

  private static long totalInstances(Map<String, ? extends List<?>> result) {
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
}
