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
package io.gravitee.lab.gliner4j.strategy.gliner2;

import static io.gravitee.lab.gliner4j.utils.LinAlg.argmax;

import io.gravitee.lab.gliner4j.AbstractNLP;
import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.postprocess.SpanDecoder;
import io.gravitee.lab.gliner4j.processor.BatchPreprocessor;
import io.gravitee.lab.gliner4j.processor.BatchSpanPipeline;
import io.gravitee.lab.gliner4j.processor.InputAssembler;
import io.gravitee.lab.gliner4j.processor.PreprocessedInput;
import io.gravitee.lab.gliner4j.processor.SchemaEncoder;
import io.gravitee.lab.gliner4j.runtime.Gliner2SpanRuntime;
import io.gravitee.lab.gliner4j.runtime.MicroBatcher;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.tokenizer.TokenMapping;
import io.gravitee.lab.gliner4j.utils.GlinerNerSupport;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * GLiNER2 (fastino) NER strategy on the merged {@code ner_full.onnx} graph: encoder,
 * word/schema gathers and count-aware span scoring in one session run per bucket, decoded
 * into non-overlapping entity spans. Single-text, override and batch requests all score
 * through the same merged path.
 *
 * <p>This holds the GLiNER2-specific NER pipeline; the public {@link io.gravitee.lab.gliner4j.GLiNER4jNER}
 * facade delegates to it. Selected by {@link io.gravitee.lab.gliner4j.arch.gliner2.Gliner2Architecture}.
 */
@Slf4j
public final class Gliner2NerStrategy
  extends AbstractNLP<
    EntityDefinition,
    Map<String, List<EntitySpan>>,
    Gliner2SpanRuntime
  >
  implements NerStrategy {

  // The count-aware scoring head slices span_scores to `count` count-instances, but NER decodes
  // only instance 0 (a single non-overlapping span set; count_logits still drives the
  // empty-vs-nonempty check and is unaffected by this value). Requesting a single instance keeps
  // span_scores 1/maxCount the size — a large, deterministic saving on the padded batched output.
  private static final long SCORING_COUNT_INSTANCES = 1;

  private final SpanDecoder spanDecoder;
  private final MicroBatcher<Map<String, List<EntitySpan>>> microBatcher;

  private Gliner2NerStrategy(
    GLiNER4jConfig config,
    RuntimeConfig runtimeConfig,
    DjlTokenizerWrapper tokenizer,
    Gliner2SpanRuntime runtime,
    InputAssembler inputAssembler
  ) {
    super(
      config,
      runtimeConfig,
      tokenizer,
      runtime,
      inputAssembler,
      new GLiNER4jTelemetry("extract")
    );
    this.spanDecoder = new SpanDecoder();
    this.microBatcher = runtimeConfig.isMicroBatchingEnabled()
      ? new MicroBatcher<>(
        runtimeConfig.getMicroBatchMaxSize(),
        runtimeConfig.getMicroBatchMaxWaitMicros(),
        this::extractBatch
      )
      : null;
  }

  /**
   * Builds the GLiNER2 NER strategy from the family-agnostic load context.
   *
   * @param ctx the load context (model dir, variant, runtime config, parsed config, tokenizer)
   * @param entities the load-time entity schema
   * @return a ready strategy
   */
  public static Gliner2NerStrategy create(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    return create(
      ctx,
      entities,
      new io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime(
        ctx.modelDir(),
        ctx.variant(),
        ctx.runtimeConfig()
      )
    );
  }

  /** Same, over an already-built span runtime (e.g. the ggml one from {@code gliner4j-llamacpp}). */
  public static Gliner2NerStrategy create(
    LoadContext ctx,
    List<EntityDefinition> entities,
    Gliner2SpanRuntime runtime
  ) {
    var schemaEncoder = entitySchemaEncoder(entities);
    var inputAssembler = new InputAssembler(ctx.tokenizer(), schemaEncoder);

    return new Gliner2NerStrategy(
      ctx.config(),
      ctx.runtimeConfig(),
      ctx.tokenizer(),
      runtime,
      inputAssembler
    );
  }

  // ---- NerStrategy ---------------------------------------------------------

  @Override
  public Map<String, List<EntitySpan>> extract(String text, float threshold) {
    if (microBatcher != null) {
      return microBatcher.extract(text, threshold);
    }
    if (text == null || text.isBlank()) {
      return Map.of();
    }
    return extractBatch(List.of(text), threshold).get(0);
  }

  @Override
  protected void onClose() {
    if (microBatcher != null) {
      microBatcher.close();
    }
  }

  @Override
  public Map<String, List<EntitySpan>> extract(
    String text,
    List<EntityDefinition> entities,
    float threshold
  ) {
    if (text == null || text.isBlank()) {
      return Map.of();
    }
    return extractBatchWith(
      List.of(text),
      threshold,
      overrideAssembler(entities)
    ).get(0);
  }

  @Override
  public List<Map<String, List<EntitySpan>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    return extractBatchWith(texts, threshold, inputAssembler);
  }

  /**
   * Merged-graph batch extraction with an explicit assembler — the load-time one for the
   * standard paths, a per-call one for schema overrides.
   */
  private List<Map<String, List<EntitySpan>>> extractBatchWith(
    List<String> texts,
    float threshold,
    InputAssembler assembler
  ) {
    if (texts == null) {
      return List.of();
    }
    long startNanos = System.nanoTime();
    int batchSize = texts.size();

    // 1. Preprocess and pack the contiguous batch (shared with other facades)
    var preproc = BatchPreprocessor.preprocess(texts, assembler);
    int nonEmptyCount = preproc.nonEmptyCount();

    var results = new ArrayList<Map<String, List<EntitySpan>>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(Map.of());
    }

    // Short-circuit: all texts are empty
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

    // 2. Bucketed full-graph scoring: schema gathers happen in-graph from marker positions
    // (identical across the batch — the schema prefix is shared).
    var schemaPositions = inputs[batchIndices[0]].schemaTokenPositions();
    long pPosition = schemaPositions[0];
    var fieldPositions = new long[schemaPositions.length - 1];
    for (int i = 0; i < fieldPositions.length; i++) {
      fieldPositions[i] = schemaPositions[i + 1];
    }
    var pipeline = BatchSpanPipeline.runFullGraph(
      preproc.batchInputIds(),
      preproc.batchAttentionMask(),
      textLens,
      slot -> firstSubwordPositions(inputs[batchIndices[slot]]),
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
          pPosition,
          fieldPositions,
          spanIdxFlat,
          config.getMaxWidth(),
          SCORING_COUNT_INSTANCES
        )
    );

    // 3. Decode per bucket.
    try {
      for (var bucket : pipeline.buckets()) {
        try (var scoring = bucket.scoring()) {
          for (int j = 0; j < bucket.slots().length; j++) {
            if (argmax(scoring.countLogitsFor(j)) == 0) {
              continue;
            }
            int si = bucket.slots()[j];
            int origIdx = batchIndices[si];
            var input = inputs[origIdx];
            var spans = spanDecoder.decode(
              scoring.spanScores(),
              j,
              input.fieldNames(),
              input.wordStartChars(),
              input.wordEndChars(),
              texts.get(origIdx),
              textLens[si],
              threshold
            );
            results.set(origIdx, GlinerNerSupport.groupByType(spans));
          }
        }
      }
    } finally {
      // Idempotent: releases anything not yet closed (e.g. on a decode failure).
      for (var bucket : pipeline.buckets()) {
        bucket.scoring().close();
      }
    }

    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    long totalEntities = results
      .stream()
      .mapToLong(m -> m.values().stream().mapToLong(List::size).sum())
      .sum();
    telemetry.record(durationMs, batchSize, totalEntities);
    return results;
  }

  // ---- AbstractNLP hooks ---------------------------------------------------

  @Override
  protected SchemaEncoder buildSchemaEncoder(
    List<EntityDefinition> definitions
  ) {
    return entitySchemaEncoder(definitions);
  }

  @Override
  protected Map<String, List<EntitySpan>> emptyResult() {
    return Map.of();
  }

  @Override
  protected long resultSize(Map<String, List<EntitySpan>> result) {
    return result.values().stream().mapToLong(List::size).sum();
  }

  @Override
  protected Map<String, List<EntitySpan>> decodeFromHiddenStates(
    float[][][] hiddenStates,
    PreprocessedInput input,
    String text,
    float threshold
  ) {
    throw new UnsupportedOperationException(
      "Gliner2NerStrategy scores through the merged ner_full graph; the split hidden-states path is gone"
    );
  }

  // ---- NER-specific helpers ------------------------------------------------

  private static SchemaEncoder entitySchemaEncoder(
    List<EntityDefinition> entities
  ) {
    return new SchemaEncoder(
      "entities",
      "[E]",
      entities.stream().map(EntityDefinition::name).toList(),
      entities.stream().map(EntityDefinition::description).toList()
    );
  }

  /**
   * Per-word first-subword positions ({@code -1} for words without a TEXT mapping) — the
   * word-gather indices consumed by the full merged graph.
   */
  private static int[] firstSubwordPositions(PreprocessedInput input) {
    var positions = new int[input.textLen()];
    java.util.Arrays.fill(positions, -1);
    var mappings = input.mappings();
    for (int i = 0; i < mappings.length; i++) {
      var mapping = mappings[i];
      if (
        mapping.type() == TokenMapping.SegmentType.TEXT &&
        positions[mapping.origIdx()] == -1
      ) {
        positions[mapping.origIdx()] = i;
      }
    }
    return positions;
  }
}
