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

import io.gravitee.lab.gliner4j.postprocess.SpanDecoder;
import io.gravitee.lab.gliner4j.processor.InputAssembler;
import io.gravitee.lab.gliner4j.processor.PreprocessedInput;
import io.gravitee.lab.gliner4j.processor.SchemaEncoder;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.tokenizer.TokenMapping;
import io.gravitee.lab.gliner4j.tokenizer.WhitespaceTokenSplitter;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;

/**
 * Main facade for GLiNER4jNER — Java-native GLiNER2 NER via ONNX Runtime.
 *
 * <p>Usage:
 * <pre>{@code
 * var entities = List.of(new EntityDefinition("person"), new EntityDefinition("organization"));
 * try (var gliner = GLiNER4jNER.load(modelDir, entities)) {
 *     Map<String, List<EntitySpan>> results = gliner.extract("John works at Google.");
 * }
 * }</pre>
 */
@Slf4j
public class GLiNER4jNER implements AutoCloseable {

  private final GLiNER4jConfig config;
  private final List<EntityDefinition> entities;
  private final DjlTokenizerWrapper tokenizer;
  private final GLiNER4jNERRuntime runtime;
  private final InputAssembler inputAssembler;
  private final WhitespaceTokenSplitter splitter;
  private final SpanDecoder spanDecoder;
  private final GLiNER4jTelemetry telemetry;

  private GLiNER4jNER(
    GLiNER4jConfig config,
    List<EntityDefinition> entities,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jNERRuntime runtime,
    InputAssembler inputAssembler
  ) {
    this.config = config;
    this.entities = List.copyOf(entities);
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.inputAssembler = inputAssembler;
    this.splitter = new WhitespaceTokenSplitter();
    this.spanDecoder = new SpanDecoder();
    this.telemetry = new GLiNER4jTelemetry("extract");
  }

  /**
   * Loads a GLiNER model using the default "onnx" variant and auto-detected resources.
   *
   * @param modelDir path to the root model directory
   * @param entities the entity types to extract
   * @return a ready-to-use GLiNER4jNER instance
   */
  public static GLiNER4jNER load(
    Path modelDir,
    List<EntityDefinition> entities
  ) {
    return load(
      modelDir,
      entities,
      BaseRuntime.DEFAULT_VARIANT,
      RuntimeConfig.defaults()
    );
  }

  /**
   * Loads a GLiNER model with a specific ONNX variant and auto-detected resources.
   *
   * @param modelDir path to the root model directory
   * @param entities the entity types to extract
   * @param variant  ONNX variant folder name (e.g. "onnx", "onnx_fp16", "onnx_quantized")
   * @return a ready-to-use GLiNER4jNER instance
   */
  public static GLiNER4jNER load(
    Path modelDir,
    List<EntityDefinition> entities,
    String variant
  ) {
    return load(modelDir, entities, variant, RuntimeConfig.defaults());
  }

  /**
   * Loads a GLiNER model using the default "onnx" variant with explicit resource control.
   *
   * @param modelDir      path to the root model directory
   * @param entities      the entity types to extract
   * @param runtimeConfig resource control configuration for ORT sessions
   * @return a ready-to-use GLiNER4jNER instance
   */
  public static GLiNER4jNER load(
    Path modelDir,
    List<EntityDefinition> entities,
    RuntimeConfig runtimeConfig
  ) {
    return load(modelDir, entities, BaseRuntime.DEFAULT_VARIANT, runtimeConfig);
  }

  /**
   * Loads a GLiNER model with a specific ONNX variant and explicit resource control.
   *
   * <p>The model directory should contain shared files (config, tokenizer) at the root
   * and ONNX model files in variant subfolders (e.g. "onnx", "onnx_fp16", "onnx_quantized").
   *
   * @param modelDir      path to the root model directory
   * @param entities      the entity types to extract
   * @param variant       ONNX variant folder name (e.g. "onnx", "onnx_fp16", "onnx_quantized")
   * @param runtimeConfig resource control configuration for ORT sessions
   * @return a ready-to-use GLiNER4jNER instance
   */
  public static GLiNER4jNER load(
    Path modelDir,
    List<EntityDefinition> entities,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    log.info(
      "Loading GLiNER4jNER model from {} (variant={}) with {} entities",
      modelDir,
      variant,
      entities.size()
    );

    var config = GLiNER4jConfig.load(modelDir);
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var runtime = new GLiNER4jNERRuntime(modelDir, variant, runtimeConfig);
    var schemaEncoder = createSchemaEncoder(entities);
    var inputAssembler = new InputAssembler(tokenizer, schemaEncoder);

    // Pre-allocate encoder buffers with the constant schema prefix
    runtime.initEncoderBuffers(inputAssembler.getSchemaPrefixIds());

    log.info("GLiNER4jNER model loaded successfully");
    return new GLiNER4jNER(
      config,
      entities,
      tokenizer,
      runtime,
      inputAssembler
    );
  }

  /**
   * Extracts entities from the given text using the default threshold.
   *
   * @param text the input text to analyze
   * @return map of entity type to list of detected spans
   */
  public Map<String, List<EntitySpan>> extract(String text) {
    return extract(text, config.getDefaultThreshold());
  }

  /**
   * Extracts entities from the given text using per-call entity definitions and the default threshold.
   *
   * @param text the input text to analyze
   * @param entities the entity types to extract (overrides the entities provided at load time)
   * @return map of entity type to list of detected spans
   */
  public Map<String, List<EntitySpan>> extract(
    String text,
    List<EntityDefinition> entities
  ) {
    return extract(text, entities, config.getDefaultThreshold());
  }

  /**
   * Extracts entities from the given text using per-call entity definitions and a custom threshold.
   *
   * @param text the input text to analyze
   * @param entities the entity types to extract (overrides the entities provided at load time)
   * @param threshold minimum confidence score (0..1) for span inclusion
   * @return map of entity type to list of detected spans
   */
  public Map<String, List<EntitySpan>> extract(
    String text,
    List<EntityDefinition> entities,
    float threshold
  ) {
    long startNanos = System.nanoTime();
    if (text == null || text.isBlank()) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    // Build a fresh schema encoder and input assembler for the override entities
    var schemaEncoder = createSchemaEncoder(entities);
    var overrideAssembler = new InputAssembler(tokenizer, schemaEncoder);

    // Split text into words with char offsets
    var textEncoder = new TextEncoder(text, splitter);
    if (textEncoder.getTextLen() == 0) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    // Assemble full token sequence with the override schema
    var input = overrideAssembler.assemble(textEncoder);

    // Run encoder without buffer cache (schema prefix differs from load-time)
    var hiddenStates = runtime.runEncoderFull(
      input.inputIds(),
      input.attentionMask()
    );

    // Steps 4-10 are identical to the default extract path
    var result = extractFromHiddenStates(hiddenStates, input, text, threshold);
    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    long entityCount = result.values().stream().mapToLong(List::size).sum();
    telemetry.record(durationMs, 1, entityCount);
    return result;
  }

  /**
   * Extracts entities from multiple texts in batch, using a single batched encoder call.
   *
   * <p>The encoder (the most expensive stage) processes all texts in one ONNX call with
   * shape [batchSize, maxSeqLen]. The span_rep and scoring_head stages remain per-text.
   *
   * @param texts the input texts to analyze
   * @return list of results, one per input text (same order)
   */
  public List<Map<String, List<EntitySpan>>> extractBatch(List<String> texts) {
    if (texts == null) {
      return List.of();
    }
    return extractBatch(texts, config.getDefaultThreshold());
  }

  /**
   * Extracts entities from multiple texts in batch with a custom threshold.
   *
   * @param texts the input texts to analyze
   * @param threshold minimum confidence score (0..1) for span inclusion
   * @return list of results, one per input text (same order)
   */
  public List<Map<String, List<EntitySpan>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    long startNanos = System.nanoTime();
    int batchSize = texts.size();

    // 1. Preprocess all texts and find max sequence length
    var inputs = new PreprocessedInput[batchSize];
    int maxSeqLen = 0;
    int nonEmptyCount = 0;

    for (int i = 0; i < batchSize; i++) {
      var text = texts.get(i);
      if (text == null || text.isBlank()) {
        inputs[i] = null;
        continue;
      }
      var textEncoder = new TextEncoder(text, splitter);
      if (textEncoder.getTextLen() == 0) {
        inputs[i] = null;
        continue;
      }
      inputs[i] = inputAssembler.assemble(textEncoder);
      maxSeqLen = Math.max(maxSeqLen, inputs[i].inputIds().length);
      nonEmptyCount++;
    }

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

    // 2. Build batched encoder inputs (only non-empty texts)
    var batchInputIds = new long[nonEmptyCount][];
    var batchAttentionMask = new long[nonEmptyCount][];
    var batchIndices = new int[nonEmptyCount]; // maps batch slot -> original index
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

    // 4. Extract schema embeddings once (identical for all texts in the batch)
    var firstInput = inputs[batchIndices[0]];
    var schemaEmbs = extractSchemaEmbeddings(
      batchedHiddenStates[0],
      firstInput
    );

    // 5. Extract text embeddings and find maxTextLen
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

    // 6. Build padded batch for span_rep (single ONNX call)
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

    // 7. Per-text: run scoring_head and decode in parallel via virtual threads
    //    OrtSession.run() is thread-safe; scoring_head creates fresh tensors per call
    @SuppressWarnings("unchecked")
    var futures = new Future[nonEmptyCount];
    var results = new ArrayList<Map<String, List<EntitySpan>>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(Map.of());
    }

    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int s = 0; s < nonEmptyCount; s++) {
        final int si = s;
        futures[si] =
          executor.submit(() -> {
            int origIdx = batchIndices[si];
            var input = inputs[origIdx];
            int textLen = textLens[si];

            // Extract the per-text span_rep slice [textLen][maxWidth][hiddenSize]
            var spanRep = new float[textLen][maxWidth][hiddenSize];
            for (int i = 0; i < textLen; i++) {
              System.arraycopy(
                batchSpanRep4d[si][i],
                0,
                spanRep[i],
                0,
                maxWidth
              );
            }

            var scoringResult = runtime.runScoringHead(
              spanRep,
              schemaEmbs.schemaEmbP(),
              schemaEmbs.schemaEmbFields(),
              (long) config.getMaxCount()
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

      // Collect results preserving original order
      for (int s = 0; s < nonEmptyCount; s++) {
        try {
          int origIdx = batchIndices[s];
          @SuppressWarnings("unchecked")
          var result = (Map<String, List<EntitySpan>>) futures[s].get();
          results.set(origIdx, result);
        } catch (ExecutionException e) {
          throw new RuntimeException(
            "Scoring head failed for batch slot " + s,
            e.getCause()
          );
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("Batch scoring interrupted", e);
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

  /**
   * Extracts entities from the given text using the specified threshold.
   *
   * @param text the input text to analyze
   * @param threshold minimum confidence score (0..1) for span inclusion
   * @return map of entity type to list of detected spans
   */
  public Map<String, List<EntitySpan>> extract(String text, float threshold) {
    long startNanos = System.nanoTime();
    if (text == null || text.isBlank()) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    // 1. Split text into words with char offsets
    var textEncoder = new TextEncoder(text, splitter);
    if (textEncoder.getTextLen() == 0) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    // 2. Assemble full token sequence
    var input = inputAssembler.assemble(textEncoder);

    // 3. Run encoder (uses cached schema prefix buffer)
    var hiddenStates = runtime.runEncoder(
      input.inputIds(),
      input.attentionMask()
    );

    // 4-10. Extract embeddings, score spans, decode, and group
    var result = extractFromHiddenStates(hiddenStates, input, text, threshold);
    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    long entityCount = result.values().stream().mapToLong(List::size).sum();
    telemetry.record(durationMs, 1, entityCount);
    return result;
  }

  private Map<String, List<EntitySpan>> extractFromHiddenStates(
    float[][][] hiddenStates,
    PreprocessedInput input,
    String text,
    float threshold
  ) {
    // 4. Extract embeddings from hidden states
    var embeddings = extractEmbeddings(hiddenStates[0], input);

    // 5. Build flat span indices and run span_rep
    int maxWidth = config.getMaxWidth();
    int textLen = input.textLen();
    int numSpans = textLen * maxWidth;
    var spanIdxFlat = buildSpanIdxFlat(textLen, maxWidth, numSpans);

    var textEmbs3d = new float[1][textLen][config.getHiddenSize()];
    System.arraycopy(embeddings.textEmbs, 0, textEmbs3d[0], 0, textLen);
    var spanRep4d = runtime.runSpanRepFlat(textEmbs3d, spanIdxFlat, numSpans);
    var spanRep = spanRep4d[0];

    // 7. Run scoring head once with maxCount
    var scoringResult = runtime.runScoringHead(
      spanRep,
      embeddings.schemaEmbP,
      embeddings.schemaEmbFields,
      (long) config.getMaxCount()
    );

    // 8. Predict count from logits
    int predCount = argmax(scoringResult.countLogits()[0]);
    log.debug("Predicted count: {}", predCount);
    if (predCount == 0) {
      return Map.of();
    }

    // 9. Decode spans
    var spans = spanDecoder.decode(
      scoringResult.spanScores(),
      input.fieldNames(),
      input.wordStartChars(),
      input.wordEndChars(),
      text,
      textLen,
      threshold
    );

    // 10. Group by type
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

  private static SchemaEncoder createSchemaEncoder(
    List<EntityDefinition> entities
  ) {
    return new SchemaEncoder(
      "entities",
      "[E]",
      entities.stream().map(EntityDefinition::name).toList(),
      entities.stream().map(EntityDefinition::description).toList()
    );
  }

  @Override
  public void close() {
    runtime.close();
    tokenizer.close();
    log.info("GLiNER4jNER closed");
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
    int hiddenSize = config.getHiddenSize();
    var specialTokenIds = config.getSpecialTokenIds();
    long pTokenId = specialTokenIds.getOrDefault("P", -1L);
    long eTokenId = specialTokenIds.getOrDefault("E", -1L);

    var schemaEmbsList = new ArrayList<float[]>();
    for (int i = 0; i < input.inputIds().length; i++) {
      long tokenId = input.inputIds()[i];
      if (tokenId == pTokenId || tokenId == eTokenId) {
        schemaEmbsList.add(hiddenState[i]);
      }
    }

    float[] schemaEmbP = schemaEmbsList.isEmpty()
      ? new float[hiddenSize]
      : schemaEmbsList.get(0);
    float[][] schemaEmbFields = new float[schemaEmbsList.size() -
    1][hiddenSize];
    for (int i = 1; i < schemaEmbsList.size(); i++) {
      schemaEmbFields[i - 1] = schemaEmbsList.get(i);
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
