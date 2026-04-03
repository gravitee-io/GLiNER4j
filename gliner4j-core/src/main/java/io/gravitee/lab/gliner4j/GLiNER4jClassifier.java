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

import io.gravitee.lab.gliner4j.processor.InputAssembler;
import io.gravitee.lab.gliner4j.processor.PreprocessedInput;
import io.gravitee.lab.gliner4j.processor.SchemaEncoder;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jClassifierRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.tokenizer.WhitespaceTokenSplitter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import lombok.extern.slf4j.Slf4j;

/**
 * Text classification facade for GLiNER4j — classifies text into label categories via ONNX Runtime.
 *
 * <p>Supports single-label (highest confidence) and multi-label (threshold-based) classification,
 * with optional per-call label override.
 *
 * <p>Usage:
 * <pre>{@code
 * var labels = List.of(new ClassificationLabel("positive"), new ClassificationLabel("negative"));
 * try (var classifier = GLiNER4jClassifier.load(modelDir, labels)) {
 *     List<ClassificationResult> results = classifier.classify("Great product!", 0.5f);
 * }
 * }</pre>
 */
@Slf4j
public class GLiNER4jClassifier implements AutoCloseable {

  private final GLiNER4jConfig config;
  private final List<ClassificationLabel> labels;
  private final DjlTokenizerWrapper tokenizer;
  private final GLiNER4jClassifierRuntime runtime;
  private final InputAssembler inputAssembler;
  private final WhitespaceTokenSplitter splitter;

  private GLiNER4jClassifier(
    GLiNER4jConfig config,
    List<ClassificationLabel> labels,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jClassifierRuntime runtime,
    InputAssembler inputAssembler
  ) {
    this.config = config;
    this.labels = List.copyOf(labels);
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.inputAssembler = inputAssembler;
    this.splitter = new WhitespaceTokenSplitter();
  }

  /**
   * Loads a classifier using the default "onnx" variant and auto-detected resources.
   *
   * @param modelDir path to the root model directory
   * @param labels the classification labels
   * @return a ready-to-use GLiNER4jClassifier instance
   */
  public static GLiNER4jClassifier load(
    Path modelDir,
    List<ClassificationLabel> labels
  ) {
    return load(
      modelDir,
      labels,
      BaseRuntime.DEFAULT_VARIANT,
      RuntimeConfig.defaults()
    );
  }

  /**
   * Loads a classifier with a specific ONNX variant.
   *
   * @param modelDir path to the root model directory
   * @param labels the classification labels
   * @param variant ONNX variant folder name
   * @return a ready-to-use GLiNER4jClassifier instance
   */
  public static GLiNER4jClassifier load(
    Path modelDir,
    List<ClassificationLabel> labels,
    String variant
  ) {
    return load(modelDir, labels, variant, RuntimeConfig.defaults());
  }

  /**
   * Loads a classifier with explicit resource control.
   *
   * @param modelDir path to the root model directory
   * @param labels the classification labels
   * @param runtimeConfig resource control configuration
   * @return a ready-to-use GLiNER4jClassifier instance
   */
  public static GLiNER4jClassifier load(
    Path modelDir,
    List<ClassificationLabel> labels,
    RuntimeConfig runtimeConfig
  ) {
    return load(modelDir, labels, BaseRuntime.DEFAULT_VARIANT, runtimeConfig);
  }

  /**
   * Loads a classifier with a specific ONNX variant and explicit resource control.
   *
   * @param modelDir      path to the root model directory
   * @param labels        the classification labels
   * @param variant       ONNX variant folder name
   * @param runtimeConfig resource control configuration
   * @return a ready-to-use GLiNER4jClassifier instance
   */
  public static GLiNER4jClassifier load(
    Path modelDir,
    List<ClassificationLabel> labels,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    log.info(
      "Loading GLiNER4jClassifier from {} (variant={}) with {} labels",
      modelDir,
      variant,
      labels.size()
    );

    var config = GLiNER4jConfig.load(modelDir);
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var runtime = new GLiNER4jClassifierRuntime(
      modelDir,
      variant,
      runtimeConfig
    );
    var schemaEncoder = createSchemaEncoder(labels);
    var inputAssembler = new InputAssembler(tokenizer, schemaEncoder);

    runtime.initEncoderBuffers(inputAssembler.getSchemaPrefixIds());

    log.info("GLiNER4jClassifier loaded successfully");
    return new GLiNER4jClassifier(
      config,
      labels,
      tokenizer,
      runtime,
      inputAssembler
    );
  }

  /**
   * Classifies text in multi-label mode using the default threshold.
   *
   * @param text the input text to classify
   * @return list of classification results above threshold
   */
  public List<ClassificationResult> classify(String text) {
    return classify(text, config.getDefaultThreshold());
  }

  /**
   * Classifies text in multi-label mode with a custom threshold.
   *
   * @param text the input text to classify
   * @param threshold minimum confidence score (0..1) for label inclusion
   * @return list of classification results above threshold, sorted by confidence descending
   */
  public List<ClassificationResult> classify(String text, float threshold) {
    if (text == null || text.isBlank()) {
      return List.of();
    }

    var textEncoder = new TextEncoder(text, splitter);
    if (textEncoder.getTextLen() == 0) {
      return List.of();
    }

    var input = inputAssembler.assemble(textEncoder);
    var hiddenStates = runtime.runEncoder(
      input.inputIds(),
      input.attentionMask()
    );

    return classifyFromHiddenStates(hiddenStates, input, threshold);
  }

  /**
   * Classifies text using per-call label definitions and default threshold.
   *
   * @param text the input text to classify
   * @param labels the classification labels (overrides load-time labels)
   * @return list of classification results above threshold
   */
  public List<ClassificationResult> classify(
    String text,
    List<ClassificationLabel> labels
  ) {
    return classify(text, labels, config.getDefaultThreshold());
  }

  /**
   * Classifies text using per-call label definitions and custom threshold.
   *
   * @param text the input text to classify
   * @param labels the classification labels (overrides load-time labels)
   * @param threshold minimum confidence score (0..1) for label inclusion
   * @return list of classification results above threshold, sorted by confidence descending
   */
  public List<ClassificationResult> classify(
    String text,
    List<ClassificationLabel> labels,
    float threshold
  ) {
    if (text == null || text.isBlank()) {
      return List.of();
    }

    var schemaEncoder = createSchemaEncoder(labels);
    var overrideAssembler = new InputAssembler(tokenizer, schemaEncoder);

    var textEncoder = new TextEncoder(text, splitter);
    if (textEncoder.getTextLen() == 0) {
      return List.of();
    }

    var input = overrideAssembler.assemble(textEncoder);
    var hiddenStates = runtime.runEncoderFull(
      input.inputIds(),
      input.attentionMask()
    );

    return classifyFromHiddenStates(hiddenStates, input, threshold);
  }

  /**
   * Classifies multiple texts in batch using default threshold.
   *
   * @param texts the input texts to classify
   * @return list of results, one per input text (same order)
   */
  public List<List<ClassificationResult>> classifyBatch(List<String> texts) {
    return classifyBatch(texts, config.getDefaultThreshold());
  }

  /**
   * Classifies multiple texts in batch with a custom threshold.
   *
   * <p>The encoder processes all texts in one batched ONNX call.
   * The classifier head runs per-text via virtual threads.
   *
   * @param texts the input texts to classify
   * @param threshold minimum confidence score (0..1) for label inclusion
   * @return list of results, one per input text (same order)
   */
  public List<List<ClassificationResult>> classifyBatch(
    List<String> texts,
    float threshold
  ) {
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
      var emptyResults = new ArrayList<List<ClassificationResult>>(batchSize);
      for (int i = 0; i < batchSize; i++) {
        emptyResults.add(List.of());
      }
      return emptyResults;
    }

    // 2. Build batched encoder inputs (only non-empty texts)
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

    // 4. Per-text: extract label embeddings and run classifier head via virtual threads
    @SuppressWarnings("unchecked")
    var futures = new Future[nonEmptyCount];
    var results = new ArrayList<List<ClassificationResult>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(List.of());
    }

    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int s = 0; s < nonEmptyCount; s++) {
        final int si = s;
        futures[si] =
          executor.submit(() -> {
            int origIdx = batchIndices[si];
            var input = inputs[origIdx];

            var labelEmbs = extractLabelEmbeddings(
              batchedHiddenStates[si],
              input
            );
            return applyClassifierHead(labelEmbs, input, threshold);
          });
      }

      for (int s = 0; s < nonEmptyCount; s++) {
        try {
          int origIdx = batchIndices[s];
          @SuppressWarnings("unchecked")
          var result = (List<ClassificationResult>) futures[s].get();
          results.set(origIdx, result);
        } catch (ExecutionException e) {
          throw new RuntimeException(
            "Classifier head failed for batch slot " + s,
            e.getCause()
          );
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("Batch classification interrupted", e);
        }
      }
    }

    return results;
  }

  private List<ClassificationResult> classifyFromHiddenStates(
    float[][][] hiddenStates,
    PreprocessedInput input,
    float threshold
  ) {
    var labelEmbs = extractLabelEmbeddings(hiddenStates[0], input);
    return applyClassifierHead(labelEmbs, input, threshold);
  }

  /**
   * Extracts label embeddings from hidden states at [L] token positions.
   */
  private float[][] extractLabelEmbeddings(
    float[][] hiddenState,
    PreprocessedInput input
  ) {
    int hiddenSize = config.getHiddenSize();
    long lTokenId = config.getSpecialTokenIds().getOrDefault("L", -1L);

    var labelEmbsList = new ArrayList<float[]>();
    for (int i = 0; i < input.inputIds().length; i++) {
      if (input.inputIds()[i] == lTokenId) {
        labelEmbsList.add(hiddenState[i]);
      }
    }

    return labelEmbsList.toArray(new float[0][]);
  }

  /**
   * Runs classifier head and applies threshold/sorting.
   */
  private List<ClassificationResult> applyClassifierHead(
    float[][] labelEmbs,
    PreprocessedInput input,
    float threshold
  ) {
    if (labelEmbs.length == 0) {
      return List.of();
    }

    // Run classifier MLP: [numLabels][hiddenSize] → [numLabels][1]
    var logits = runtime.runClassifierHead(labelEmbs);

    // Apply sigmoid activation and threshold
    var results = new ArrayList<ClassificationResult>();
    var labelNames = input.fieldNames();
    for (int i = 0; i < logits.length && i < labelNames.size(); i++) {
      float score = sigmoid(logits[i][0]);
      if (score >= threshold) {
        results.add(new ClassificationResult(labelNames.get(i), score));
      }
    }

    // Sort by confidence descending
    results.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));
    return results;
  }

  private static SchemaEncoder createSchemaEncoder(
    List<ClassificationLabel> labels
  ) {
    return new SchemaEncoder(
      "classify",
      "[L]",
      labels.stream().map(ClassificationLabel::name).toList(),
      labels.stream().map(ClassificationLabel::description).toList()
    );
  }

  private static float sigmoid(float x) {
    return 1.0f / (1.0f + (float) Math.exp(-x));
  }

  @Override
  public void close() {
    runtime.close();
    tokenizer.close();
    log.info("GLiNER4jClassifier closed");
  }
}
