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

import io.gravitee.lab.gliner4j.arch.ModelArchitectures;
import io.gravitee.lab.gliner4j.arch.TaskType;
import io.gravitee.lab.gliner4j.processor.BatchPreprocessor;
import io.gravitee.lab.gliner4j.processor.InputAssembler;
import io.gravitee.lab.gliner4j.processor.PreprocessedInput;
import io.gravitee.lab.gliner4j.processor.SchemaEncoder;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jClassifierRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
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
public final class GLiNER4jClassifier
  extends AbstractNLP<
    ClassificationLabel,
    List<ClassificationResult>,
    GLiNER4jClassifierRuntime
  > {

  private GLiNER4jClassifier(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jClassifierRuntime runtime,
    InputAssembler inputAssembler
  ) {
    super(
      config,
      tokenizer,
      runtime,
      inputAssembler,
      new GLiNER4jTelemetry("classify")
    );
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
    ModelArchitectures.forId(config.getArchitecture()).requireSupported(
      TaskType.CLASSIFICATION
    );
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var runtime = new GLiNER4jClassifierRuntime(
      modelDir,
      variant,
      runtimeConfig
    );
    var schemaEncoder = labelSchemaEncoder(labels);
    var inputAssembler = new InputAssembler(tokenizer, schemaEncoder);

    runtime.initEncoderBuffers(inputAssembler.getSchemaPrefixIds());

    log.info("GLiNER4jClassifier loaded successfully");
    return new GLiNER4jClassifier(config, tokenizer, runtime, inputAssembler);
  }

  /**
   * Classifies text in multi-label mode using the default threshold.
   *
   * @param text the input text to classify
   * @return list of classification results above threshold
   */
  public List<ClassificationResult> classify(String text) {
    return doExtractOnce(text, config.getDefaultThreshold());
  }

  /**
   * Classifies text in multi-label mode with a custom threshold.
   *
   * @param text the input text to classify
   * @param threshold minimum confidence score (0..1) for label inclusion
   * @return list of classification results above threshold, sorted by confidence descending
   */
  public List<ClassificationResult> classify(String text, float threshold) {
    return doExtractOnce(text, threshold);
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
    return doExtractOverride(text, labels, config.getDefaultThreshold());
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
    return doExtractOverride(text, labels, threshold);
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
    long startNanos = System.nanoTime();
    int batchSize = texts.size();

    // 1. Preprocess and pack the contiguous batch (shared with other facades)
    var preproc = BatchPreprocessor.preprocess(texts, inputAssembler);
    int nonEmptyCount = preproc.nonEmptyCount();

    // Short-circuit: all texts are empty
    if (nonEmptyCount == 0) {
      var emptyResults = new ArrayList<List<ClassificationResult>>(batchSize);
      for (int i = 0; i < batchSize; i++) {
        emptyResults.add(List.of());
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

    // 3. Per-text fanout: extract label embeddings + classifier head MLP on virtual threads.
    var results = new ArrayList<List<ClassificationResult>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(List.of());
    }

    @SuppressWarnings("unchecked")
    var futures = (Future<
      List<ClassificationResult>
    >[]) new Future[nonEmptyCount];

    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      for (int s = 0; s < nonEmptyCount; s++) {
        final int si = s;
        futures[si] = executor.submit(() -> {
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
          results.set(batchIndices[s], futures[s].get());
        } catch (ExecutionException e) {
          throw new RuntimeException(
            "Classifier head failed for batch slot " + s,
            e.getCause()
          );
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("Classifier batch fanout interrupted", e);
        }
      }
    }

    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    long totalLabels = results.stream().mapToLong(List::size).sum();
    telemetry.record(durationMs, batchSize, totalLabels);
    return results;
  }

  // ---- AbstractTaskFacade hooks -------------------------------------------

  @Override
  protected SchemaEncoder buildSchemaEncoder(
    List<ClassificationLabel> definitions
  ) {
    return labelSchemaEncoder(definitions);
  }

  @Override
  protected List<ClassificationResult> emptyResult() {
    return List.of();
  }

  @Override
  protected long resultSize(List<ClassificationResult> result) {
    return result.size();
  }

  @Override
  protected List<ClassificationResult> decodeFromHiddenStates(
    float[][][] hiddenStates,
    PreprocessedInput input,
    String text,
    float threshold
  ) {
    var labelEmbs = extractLabelEmbeddings(hiddenStates[0], input);
    return applyClassifierHead(labelEmbs, input, threshold);
  }

  // ---- classifier-specific helpers ----------------------------------------

  /**
   * Extracts label embeddings from hidden states at the precomputed [L] marker positions.
   * Positions are baked into the assembler at construction time (skips index 0 which holds [P]).
   */
  private float[][] extractLabelEmbeddings(
    float[][] hiddenState,
    PreprocessedInput input
  ) {
    var positions = input.schemaTokenPositions();
    var labelEmbs = new float[positions.length - 1][];
    for (int i = 0; i < labelEmbs.length; i++) {
      labelEmbs[i] = hiddenState[positions[i + 1]];
    }
    return labelEmbs;
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

  private static SchemaEncoder labelSchemaEncoder(
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
}
