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
import io.gravitee.lab.gliner4j.processor.SpanIndexCache;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
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
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import lombok.extern.slf4j.Slf4j;

/**
 * GLiNER2 (fastino) NER strategy: encoder → span_rep → count-aware scoring_head, decoded into
 * non-overlapping entity spans.
 *
 * <p>This holds the GLiNER2-specific NER pipeline; the public {@link io.gravitee.lab.gliner4j.GLiNER4jNER}
 * facade delegates to it. Selected by {@link io.gravitee.lab.gliner4j.arch.gliner2.Gliner2Architecture}.
 */
@Slf4j
public final class Gliner2NerStrategy
  extends AbstractNLP<
    EntityDefinition,
    Map<String, List<EntitySpan>>,
    GLiNER4jNERRuntime
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
    GLiNER4jNERRuntime runtime,
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
    var runtime = new GLiNER4jNERRuntime(
      ctx.modelDir(),
      ctx.variant(),
      ctx.runtimeConfig()
    );
    var schemaEncoder = entitySchemaEncoder(entities);
    var inputAssembler = new InputAssembler(ctx.tokenizer(), schemaEncoder);

    // Pre-allocate encoder buffers with the constant schema prefix
    runtime.initEncoderBuffers(inputAssembler.getSchemaPrefixIds());

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
    // Route single-text through the merged graph too, so every path produces identical scores
    // (the split per-text path uses a differently-fused encoder — same spans, ~1e-6 confidence
    // drift). Falls back to the split single-text path only when ner_full is not shipped.
    if (runtime.isNerFullGraph()) {
      if (text == null || text.isBlank()) {
        return Map.of();
      }
      return extractBatch(java.util.List.of(text), threshold).get(0);
    }
    return doExtractOnce(text, threshold);
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
    return doExtractOverride(text, entities, threshold);
  }

  @Override
  public List<Map<String, List<EntitySpan>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    long startNanos = System.nanoTime();
    int batchSize = texts.size();

    // 1. Preprocess and pack the contiguous batch (shared with other facades)
    var preproc = BatchPreprocessor.preprocess(texts, inputAssembler);
    int nonEmptyCount = preproc.nonEmptyCount();

    // Short-circuit: all texts are empty
    if (nonEmptyCount == 0) {
      var emptyResults = new ArrayList<Map<String, List<EntitySpan>>>(
        batchSize
      );
      for (int i = 0; i < batchSize; i++) {
        emptyResults.add(Map.of());
      }
      double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
      telemetry.record(durationMs, batchSize, 0);
      return emptyResults;
    }

    var inputs = preproc.inputs();
    var batchIndices = preproc.batchIndices();

    // 2-4. Bucketed encoder + span_rep (padded per sub-batch, not batch-wide)
    var textLens = new int[nonEmptyCount];
    for (int s = 0; s < nonEmptyCount; s++) {
      textLens[s] = inputs[batchIndices[s]].textLen();
    }

    // Schema embeddings are batch-invariant; the merged scorer extracts them once, on the
    // first bucket, from the representative hidden state.
    var mergedSchema = new java.util.concurrent.atomic.AtomicReference<
      SchemaEmbeddings
    >();
    BatchSpanPipeline.Result pipeline;
    if (runtime.isNerFullGraph()) {
      // Full merged graph: schema gathers happen in-graph from marker positions.
      var schemaPositions = inputs[batchIndices[0]].schemaTokenPositions();
      long pPosition = schemaPositions[0];
      var fieldPositions = new long[schemaPositions.length - 1];
      for (int i = 0; i < fieldPositions.length; i++) {
        fieldPositions[i] = schemaPositions[i + 1];
      }
      pipeline = BatchSpanPipeline.runFullGraph(
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
    } else {
      pipeline = BatchSpanPipeline.run(
        runtime,
        preproc.batchInputIds(),
        preproc.batchAttentionMask(),
        textLens,
        config.getHiddenSize(),
        config.getMaxWidth(),
        runtimeConfig.getBatchLengthRatio(),
        runtimeConfig.getMaxSubBatchSize(),
        // per-slot nested arrays are only needed by the per-text scoring fallback
        !runtime.isScoringHeadBatched(),
        // only the [P] + [E] marker rows of the representative slot are read
        slot -> inputs[batchIndices[slot]].schemaTokenPositions(),
        // merged span_scoring.onnx: one session run per bucket, span_rep stays on-device
        !runtime.isSpanScoringMerged()
          ? null
          : (textEmbs, bSize, maxTextLen, spanIdxFlat, repHidden, repSlot) -> {
            var schema = mergedSchema.updateAndGet(existing ->
              existing != null
                ? existing
                : extractSchemaEmbeddings(
                  repHidden,
                  inputs[batchIndices[repSlot]]
                )
            );
            return runtime.runSpanScoringBatchFlat(
              textEmbs,
              bSize,
              maxTextLen,
              config.getHiddenSize(),
              config.getMaxWidth(),
              spanIdxFlat,
              schema.schemaEmbP(),
              schema.schemaEmbFields(),
              SCORING_COUNT_INSTANCES
            );
          },
        (hidden, row, s, target, targetBase) ->
          extractTextEmbeddingsFlat(
            hidden,
            row,
            inputs[batchIndices[s]],
            target,
            targetBase
          )
      );
    }

    // 5. Extract schema embeddings once (identical for all texts in the batch). On the merged
    // path the scorer already did it inside the first bucket.
    var schemaEmbs = runtime.isNerFullGraph()
      ? null // in-graph gathers; nothing extracted on the Java side
      : runtime.isSpanScoringMerged()
        ? mergedSchema.get()
        : extractSchemaEmbeddings(
          pipeline.repHiddenState(),
          inputs[batchIndices[pipeline.repSlot()]]
        );

    // 6. Score + decode.
    var results = new ArrayList<Map<String, List<EntitySpan>>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(Map.of());
    }

    // One batched scoring_head call per bucket beats N small session runs: the schema/count
    // leg is computed once instead of per text, and N dispatches collapse into one. Only a
    // batched-capable (rank-4) model artifact supports it; older artifacts fall through to
    // the per-text virtual-thread fanout below.
    if (
      runtime.isScoringHeadBatched() ||
      runtime.isSpanScoringMerged() ||
      runtime.isNerFullGraph()
    ) {
      try {
        for (var bucket : pipeline.buckets()) {
          try (
            var scoring = bucket.scoring() != null
              ? bucket.scoring()
              : runtime.runScoringHeadBatchFlat(
                bucket.spanRep(),
                schemaEmbs.schemaEmbP(),
                schemaEmbs.schemaEmbFields(),
                SCORING_COUNT_INSTANCES
              )
          ) {
            // The scores are pinned to their own buffer; the span_rep lease can go now.
            if (bucket.spanRep() != null) {
              bucket.spanRep().close();
            }
            int predCount = argmax(scoring.countLogits()[0]);
            if (predCount == 0) {
              continue;
            }
            for (int j = 0; j < bucket.slots().length; j++) {
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
        // Idempotent: releases anything not yet closed (e.g. on a scoring failure).
        for (var bucket : pipeline.buckets()) {
          if (bucket.spanRep() != null) {
            bucket.spanRep().close();
          }
          if (bucket.scoring() != null) {
            bucket.scoring().close();
          }
        }
      }

      double bucketedDurationMs =
        (System.nanoTime() - startNanos) / 1_000_000.0;
      long bucketedEntities = results
        .stream()
        .mapToLong(m -> m.values().stream().mapToLong(List::size).sum())
        .sum();
      telemetry.record(bucketedDurationMs, batchSize, bucketedEntities);
      return results;
    }

    // Per-text fanout: scoring_head + decode on virtual threads.
    // OrtSession.run() is thread-safe; on CUDA, independent runs can dispatch onto
    // separate streams and the GPU overlaps their kernels on its SMs.
    @SuppressWarnings("unchecked")
    var futures = (Future<
      Map<String, List<EntitySpan>>
    >[]) new Future[nonEmptyCount];

    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int s = 0; s < nonEmptyCount; s++) {
        final int si = s;
        futures[si] = executor.submit(() -> {
          int origIdx = batchIndices[si];
          var input = inputs[origIdx];
          int textLen = textLens[si];

          var spanRep = pipeline.spanRepBySlot()[si];

          var scoringResult = runtime.runScoringHead(
            spanRep,
            schemaEmbs.schemaEmbP(),
            schemaEmbs.schemaEmbFields(),
            config.getMaxCount()
          );

          int predCount = argmax(scoringResult.countLogits()[0]);
          if (predCount == 0) {
            return Map.<String, List<EntitySpan>>of();
          }

          var spans = spanDecoder.decode(
            scoringResult.spanScores(),
            input.fieldNames(),
            input.wordStartChars(),
            input.wordEndChars(),
            texts.get(origIdx),
            textLen,
            threshold
          );

          return GlinerNerSupport.groupByType(spans);
        });
      }

      for (int s = 0; s < nonEmptyCount; s++) {
        try {
          results.set(batchIndices[s], futures[s].get());
        } catch (ExecutionException e) {
          throw new RuntimeException(
            "NER scoring head failed for batch slot " + s,
            e.getCause()
          );
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("NER batch fanout interrupted", e);
        }
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
    var embeddings = extractEmbeddings(hiddenStates[0], input);

    int maxWidth = config.getMaxWidth();
    int textLen = input.textLen();
    int numSpans = textLen * maxWidth;
    var spanIdxFlat = SpanIndexCache.flatSpanIdx(textLen, maxWidth);

    var textEmbs3d = new float[1][textLen][config.getHiddenSize()];
    System.arraycopy(embeddings.textEmbs, 0, textEmbs3d[0], 0, textLen);
    var spanRep4d = runtime.runSpanRepFlat(textEmbs3d, spanIdxFlat, numSpans);
    var spanRep = spanRep4d[0];

    var scoringResult = runtime.runScoringHead(
      spanRep,
      embeddings.schemaEmbP,
      embeddings.schemaEmbFields,
      (long) config.getMaxCount()
    );

    int predCount = argmax(scoringResult.countLogits()[0]);
    log.debug("Predicted count: {}", predCount);
    if (predCount == 0) {
      return Map.of();
    }

    var spans = spanDecoder.decode(
      scoringResult.spanScores(),
      input.fieldNames(),
      input.wordStartChars(),
      input.wordEndChars(),
      text,
      textLen,
      threshold
    );

    return GlinerNerSupport.groupByType(spans);
  }

  // ---- NER-specific helpers ------------------------------------------------

  private ExtractedEmbeddings extractEmbeddings(
    float[][] hiddenState,
    PreprocessedInput input
  ) {
    var schema = extractSchemaEmbeddings(hiddenState, input);
    var textEmbs = new float[input.textLen()][config.getHiddenSize()];
    extractTextEmbeddings(hiddenState, input, textEmbs);
    return new ExtractedEmbeddings(
      schema.schemaEmbP(),
      schema.schemaEmbFields(),
      textEmbs
    );
  }

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

  private record ExtractedEmbeddings(
    float[] schemaEmbP,
    float[][] schemaEmbFields,
    float[][] textEmbs
  ) {}

  private record SchemaEmbeddings(
    float[] schemaEmbP,
    float[][] schemaEmbFields
  ) {}

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

  private SchemaEmbeddings extractSchemaEmbeddings(
    float[][] hiddenState,
    PreprocessedInput input
  ) {
    // Positions of [P] and each [E] marker are baked into the assembler at construction time —
    // O(numFields) array lookups instead of an O(seqLen) input-id scan.
    var positions = input.schemaTokenPositions();
    float[] schemaEmbP = positions[0] >= 0
      ? hiddenState[positions[0]]
      : new float[config.getHiddenSize()];
    var schemaEmbFields = new float[positions.length - 1][];
    for (int i = 0; i < schemaEmbFields.length; i++) {
      schemaEmbFields[i] = hiddenState[positions[i + 1]];
    }
    return new SchemaEmbeddings(schemaEmbP, schemaEmbFields);
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
        mapping.type() == TokenMapping.SegmentType.TEXT &&
        !seenWord[mapping.origIdx()]
      ) {
        target[mapping.origIdx()] = hiddenState[i];
        seenWord[mapping.origIdx()] = true;
      }
    }
  }

  /**
   * Flat variant of {@link #extractTextEmbeddings}: copies each word's first-subword embedding
   * row from the bucket's flat hidden states into the flat text-embedding buffer.
   */
  private static void extractTextEmbeddingsFlat(
    io.gravitee.lab.gliner4j.runtime.FloatTensorView hidden,
    int row,
    PreprocessedInput input,
    java.nio.FloatBuffer target,
    int targetBase
  ) {
    int hiddenSize = hidden.dim(2);
    long rowBase = (long) row * hidden.dim(1) * hiddenSize;
    var seenWord = new boolean[input.textLen()];
    for (int i = 0; i < input.mappings().length; i++) {
      var mapping = input.mappings()[i];
      if (
        mapping.type() == TokenMapping.SegmentType.TEXT &&
        !seenWord[mapping.origIdx()]
      ) {
        hidden.copyRowAsFloats(
          rowBase + (long) i * hiddenSize,
          target,
          targetBase + mapping.origIdx() * hiddenSize,
          hiddenSize
        );
        seenWord[mapping.origIdx()] = true;
      }
    }
    // Contract with the pipeline: every word row in [0, textLen) must be written — the
    // target is a pooled buffer. Words without a TEXT mapping (shouldn't happen, but
    // guard) get zeroed explicitly.
    float[] zeroRow = null;
    for (int w = 0; w < seenWord.length; w++) {
      if (!seenWord[w]) {
        if (zeroRow == null) {
          zeroRow = new float[hiddenSize];
        }
        target.put(targetBase + w * hiddenSize, zeroRow, 0, hiddenSize);
      }
    }
  }
}
