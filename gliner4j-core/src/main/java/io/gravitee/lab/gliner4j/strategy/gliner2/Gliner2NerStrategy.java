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
import io.gravitee.lab.gliner4j.processor.InputAssembler;
import io.gravitee.lab.gliner4j.processor.PreprocessedInput;
import io.gravitee.lab.gliner4j.processor.SchemaEncoder;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.tokenizer.TokenMapping;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
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

  private final SpanDecoder spanDecoder;

  private Gliner2NerStrategy(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jNERRuntime runtime,
    InputAssembler inputAssembler
  ) {
    super(
      config,
      tokenizer,
      runtime,
      inputAssembler,
      new GLiNER4jTelemetry("extract")
    );
    this.spanDecoder = new SpanDecoder();
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
      ctx.tokenizer(),
      runtime,
      inputAssembler
    );
  }

  // ---- NerStrategy ---------------------------------------------------------

  @Override
  public Map<String, List<EntitySpan>> extract(String text, float threshold) {
    return doExtractOnce(text, threshold);
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

    // 2. Single batched encoder call
    var batchedHiddenStates = runtime.runEncoderBatch(
      preproc.batchInputIds(),
      preproc.batchAttentionMask(),
      preproc.maxSeqLen()
    );

    // 3. Extract schema embeddings once (identical for all texts in the batch)
    var schemaEmbs = extractSchemaEmbeddings(
      batchedHiddenStates[0],
      inputs[batchIndices[0]]
    );

    // 4. Extract text embeddings and find maxTextLen
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
      extractTextEmbeddings(batchedHiddenStates[s], input, perTextEmbs[s]);
    }

    // 5. Build padded batch for span_rep (single ONNX call)
    int maxNumSpans = maxTextLen * maxWidth;
    var batchTextEmbs = new float[nonEmptyCount][maxTextLen][hiddenSize];
    var batchSpanIdxFlat = new long[nonEmptyCount * maxNumSpans * 2];

    for (int s = 0; s < nonEmptyCount; s++) {
      int textLen = textLens[s];
      System.arraycopy(perTextEmbs[s], 0, batchTextEmbs[s], 0, textLen);
      // Zero-padded rows beyond textLen are already 0.0f (default)

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

    // 6. Per-text fanout: scoring_head + decode on virtual threads.
    //    OrtSession.run() is thread-safe; on CUDA, independent runs can dispatch onto
    //    separate streams and the GPU overlaps their kernels on its SMs.
    var results = new ArrayList<Map<String, List<EntitySpan>>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(Map.of());
    }

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

          // Slice the per-text span_rep view [textLen][maxWidth][hiddenSize]
          // from the padded batch tensor. Reference-only copy; inner rows are shared.
          var spanRep = Arrays.copyOfRange(batchSpanRep4d[si], 0, textLen);

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

          return spans
            .stream()
            .collect(
              Collectors.groupingBy(
                EntitySpan::type,
                LinkedHashMap::new,
                Collectors.toList()
              )
            );
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
    var spanIdxFlat = buildSpanIdxFlat(textLen, maxWidth, numSpans);

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

    return spans
      .stream()
      .collect(
        Collectors.groupingBy(
          EntitySpan::type,
          LinkedHashMap::new,
          Collectors.toList()
        )
      );
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
}
