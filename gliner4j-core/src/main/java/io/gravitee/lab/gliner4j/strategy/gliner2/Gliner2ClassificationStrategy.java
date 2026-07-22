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

import io.gravitee.lab.gliner4j.AbstractNLP;
import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.processor.BatchPreprocessor;
import io.gravitee.lab.gliner4j.processor.InputAssembler;
import io.gravitee.lab.gliner4j.processor.PreprocessedInput;
import io.gravitee.lab.gliner4j.processor.SchemaEncoder;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jClassifierRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.strategy.ClassificationStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.utils.LinAlg;
import java.util.ArrayList;
import java.util.List;
import lombok.extern.slf4j.Slf4j;

/**
 * GLiNER2 (fastino) text-classification strategy on the merged {@code classifier_full.onnx}
 * graph: encoder, in-graph {@code [L]} label gather and classifier-head MLP in one session
 * run, followed by sigmoid + threshold. Single-text, override and batch requests all score
 * through the same merged path.
 *
 * <p>Holds the GLiNER2-specific classification pipeline; the public
 * {@link io.gravitee.lab.gliner4j.GLiNER4jClassifier} facade delegates to it. Selected by
 * {@link io.gravitee.lab.gliner4j.arch.gliner2.Gliner2Architecture}.
 */
@Slf4j
public final class Gliner2ClassificationStrategy
  extends AbstractNLP<
    ClassificationLabel,
    List<ClassificationResult>,
    GLiNER4jClassifierRuntime
  >
  implements ClassificationStrategy {

  private Gliner2ClassificationStrategy(
    GLiNER4jConfig config,
    RuntimeConfig runtimeConfig,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jClassifierRuntime runtime,
    InputAssembler inputAssembler
  ) {
    super(
      config,
      runtimeConfig,
      tokenizer,
      runtime,
      inputAssembler,
      new GLiNER4jTelemetry("classify")
    );
  }

  /**
   * Builds the GLiNER2 classification strategy from the family-agnostic load context.
   *
   * @param ctx the load context
   * @param labels the load-time classification labels
   * @return a ready strategy
   */
  public static Gliner2ClassificationStrategy create(
    LoadContext ctx,
    List<ClassificationLabel> labels
  ) {
    var runtime = new GLiNER4jClassifierRuntime(
      ctx.modelDir(),
      ctx.variant(),
      ctx.runtimeConfig()
    );
    var schemaEncoder = labelSchemaEncoder(labels);
    var inputAssembler = new InputAssembler(ctx.tokenizer(), schemaEncoder);
    return new Gliner2ClassificationStrategy(
      ctx.config(),
      ctx.runtimeConfig(),
      ctx.tokenizer(),
      runtime,
      inputAssembler
    );
  }

  // ---- ClassificationStrategy ---------------------------------------------

  @Override
  public List<ClassificationResult> classify(String text, float threshold) {
    return classifyMerged(text, inputAssembler, threshold);
  }

  @Override
  public List<ClassificationResult> classify(
    String text,
    List<ClassificationLabel> labels,
    float threshold
  ) {
    return classifyMerged(text, overrideAssembler(labels), threshold);
  }

  @Override
  public List<List<ClassificationResult>> classifyBatch(
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

    var results = new ArrayList<List<ClassificationResult>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(List.of());
    }

    // Short-circuit: all texts are empty
    if (nonEmptyCount == 0) {
      double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
      telemetry.record(durationMs, batchSize, 0);
      return results;
    }

    var inputs = preproc.inputs();
    var batchIndices = preproc.batchIndices();

    // 2. Merged graph: encoder + in-graph label gather + classifier head in one batched
    // session run. Label marker positions are shared across the batch (identical schema
    // prefix); index 0 is [P], the rest are the [L] labels.
    var logits = runtime.runClassifierFullBatch(
      preproc.batchInputIds(),
      preproc.batchAttentionMask(),
      preproc.maxSeqLen(),
      labelPositions(inputs[batchIndices[0]]),
      nonEmptyCount
    );
    for (int s = 0; s < nonEmptyCount; s++) {
      int origIdx = batchIndices[s];
      results.set(
        origIdx,
        thresholdLabelLogits(logits[s], inputs[origIdx], threshold)
      );
    }

    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    long totalLabels = results.stream().mapToLong(List::size).sum();
    telemetry.record(durationMs, batchSize, totalLabels);
    return results;
  }

  // ---- AbstractNLP hooks ---------------------------------------------------

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
    throw new UnsupportedOperationException(
      "Gliner2ClassificationStrategy scores through the merged classifier_full graph; the split hidden-states path is gone"
    );
  }

  // ---- classifier-specific helpers ----------------------------------------

  /**
   * Single-text classification through the merged graph as a batch of one — the graph
   * embeds the encoder, so the load-time and override assemblers take the exact same path.
   */
  private List<ClassificationResult> classifyMerged(
    String text,
    InputAssembler assembler,
    float threshold
  ) {
    long startNanos = System.nanoTime();
    if (text == null || text.isBlank()) {
      telemetry.record(0.0, 1, 0);
      return List.of();
    }
    var textEncoder = new TextEncoder(text);
    if (textEncoder.getTextLen() == 0) {
      telemetry.record(0.0, 1, 0);
      return List.of();
    }

    var input = assembler.assemble(textEncoder);
    var logits = runtime.runClassifierFullBatch(
      new long[][] { input.inputIds() },
      new long[][] { input.attentionMask() },
      input.inputIds().length,
      labelPositions(input),
      1
    );
    var result = thresholdLabelLogits(logits[0], input, threshold);
    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    telemetry.record(durationMs, 1, result.size());
    return result;
  }

  /** The [L] label-marker positions of {@code input} (drops index 0, which holds [P]). */
  private static long[] labelPositions(PreprocessedInput input) {
    var schemaPositions = input.schemaTokenPositions();
    var labelPositions = new long[schemaPositions.length - 1];
    for (int i = 0; i < labelPositions.length; i++) {
      labelPositions[i] = schemaPositions[i + 1];
    }
    return labelPositions;
  }

  /**
   * Applies sigmoid + threshold to raw per-label logits and sorts by confidence descending.
   */
  private List<ClassificationResult> thresholdLabelLogits(
    float[] logits,
    PreprocessedInput input,
    float threshold
  ) {
    var results = new ArrayList<ClassificationResult>();
    var labelNames = input.fieldNames();
    for (int i = 0; i < logits.length && i < labelNames.size(); i++) {
      float score = LinAlg.sigmoid(logits[i]);
      if (score >= threshold) {
        results.add(new ClassificationResult(labelNames.get(i), score));
      }
    }
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
}
