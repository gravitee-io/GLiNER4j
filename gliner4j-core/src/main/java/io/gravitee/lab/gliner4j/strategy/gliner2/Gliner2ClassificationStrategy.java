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
import io.gravitee.lab.gliner4j.runtime.GLiNER4jClassifierRuntime;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.strategy.ClassificationStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import lombok.extern.slf4j.Slf4j;

/**
 * GLiNER2 (fastino) text-classification strategy: encoder → label embeddings at {@code [L]} markers
 * → classifier-head MLP → sigmoid + threshold.
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
    runtime.initEncoderBuffers(inputAssembler.getSchemaPrefixIds());
    return new Gliner2ClassificationStrategy(
      ctx.config(),
      ctx.tokenizer(),
      runtime,
      inputAssembler
    );
  }

  // ---- ClassificationStrategy ---------------------------------------------

  @Override
  public List<ClassificationResult> classify(String text, float threshold) {
    return doExtractOnce(text, threshold);
  }

  @Override
  public List<ClassificationResult> classify(
    String text,
    List<ClassificationLabel> labels,
    float threshold
  ) {
    return doExtractOverride(text, labels, threshold);
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
