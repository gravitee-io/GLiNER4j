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

import io.gravitee.lab.gliner4j.processor.AssemblerCache;
import io.gravitee.lab.gliner4j.processor.InputAssembler;
import io.gravitee.lab.gliner4j.processor.PreprocessedInput;
import io.gravitee.lab.gliner4j.processor.SchemaEncoder;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.util.List;
import lombok.extern.slf4j.Slf4j;

/**
 * Shared scaffolding for single-schema task facades (NER, classification, …).
 *
 * <p>Every task in this family follows the same pipeline: tokenize the text, assemble it with
 * a {@link SchemaEncoder} that describes the task, run the encoder, then turn the resulting
 * hidden states into the task-specific result. The encoder, the input assembler, the telemetry,
 * and the override-path machinery are all identical across tasks, so they live here. Subclasses
 * only supply the bits that genuinely differ — the schema-encoder factory and the per-task
 * decode step.
 *
 * @param <D>  the definition type (e.g. {@code EntityDefinition}, {@code ClassificationLabel})
 * @param <R>  the per-text result type (e.g. {@code Map<String, List<EntitySpan>>},
 *             {@code List<ClassificationResult>})
 * @param <RT> the concrete runtime; exposed to subclasses for task-head calls
 */
@Slf4j
public abstract class AbstractNLP<D, R, RT extends BaseRuntime>
  implements AutoCloseable {

  protected final GLiNER4jConfig config;
  protected final RuntimeConfig runtimeConfig;
  protected final DjlTokenizerWrapper tokenizer;
  protected final RT runtime;
  protected final InputAssembler inputAssembler;
  protected final GLiNER4jTelemetry telemetry;
  private final AssemblerCache<List<D>, InputAssembler> overrideAssemblers;

  protected AbstractNLP(
    GLiNER4jConfig config,
    RuntimeConfig runtimeConfig,
    DjlTokenizerWrapper tokenizer,
    RT runtime,
    InputAssembler inputAssembler,
    GLiNER4jTelemetry telemetry
  ) {
    this.config = config;
    this.runtimeConfig = runtimeConfig;
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.inputAssembler = inputAssembler;
    this.telemetry = telemetry;
    this.overrideAssemblers = new AssemblerCache<>(
      runtimeConfig.effectiveOverrideCacheSize(),
      defs -> new InputAssembler(tokenizer, buildSchemaEncoder(defs))
    );
  }

  // ---- subclass hooks ------------------------------------------------------

  /**
   * Builds the task-specific {@link SchemaEncoder} for the given definitions. Used by the
   * per-call override path to swap the prompt schema for a single request.
   */
  protected abstract SchemaEncoder buildSchemaEncoder(List<D> definitions);

  /**
   * Turns encoder hidden states into the task-specific result.
   *
   * @param hiddenStates the raw encoder output, shape {@code [1][seqLen][hiddenSize]}
   * @param input        the assembled input for this text
   * @param text         the original text (decoders re-slice character offsets out of it)
   * @param threshold    minimum confidence for a span/label to be kept
   */
  protected abstract R decodeFromHiddenStates(
    float[][][] hiddenStates,
    PreprocessedInput input,
    String text,
    float threshold
  );

  /** The empty result returned for null/blank/empty-after-tokenization texts. */
  protected abstract R emptyResult();

  /**
   * The cached per-call override assembler for {@code defs}. Strategies that route override
   * requests through their own merged-graph paths (instead of {@link #doExtractOverride})
   * use this to swap the prompt schema for a single request.
   */
  protected final InputAssembler overrideAssembler(List<D> defs) {
    return overrideAssemblers.get(defs);
  }

  /** Number of items in {@code result}, recorded into telemetry. */
  protected abstract long resultSize(R result);

  // ---- shared single-text and override paths -------------------------------

  /**
   * Runs the load-time schema against {@code text}. Uses the runtime's prefix-cached encoder.
   */
  protected final R doExtractOnce(String text, float threshold) {
    long startNanos = System.nanoTime();
    if (text == null || text.isBlank()) {
      telemetry.record(0.0, 1, 0);
      return emptyResult();
    }

    var textEncoder = new TextEncoder(text);
    if (textEncoder.getTextLen() == 0) {
      telemetry.record(0.0, 1, 0);
      return emptyResult();
    }

    var input = inputAssembler.assemble(textEncoder);
    var hiddenStates = runtime.runEncoder(
      input.inputIds(),
      input.attentionMask()
    );
    var result = decodeFromHiddenStates(hiddenStates, input, text, threshold);
    recordTelemetry(startNanos, result);
    return result;
  }

  /**
   * Runs a per-call schema against {@code text} via a fresh assembler. Uses {@code runEncoderFull}
   * since the prefix-cache is keyed on the load-time schema.
   */
  protected final R doExtractOverride(
    String text,
    List<D> overrideDefinitions,
    float threshold
  ) {
    long startNanos = System.nanoTime();
    if (text == null || text.isBlank()) {
      telemetry.record(0.0, 1, 0);
      return emptyResult();
    }

    var overrideAssembler = overrideAssemblers.get(overrideDefinitions);

    var textEncoder = new TextEncoder(text);
    if (textEncoder.getTextLen() == 0) {
      telemetry.record(0.0, 1, 0);
      return emptyResult();
    }

    var input = overrideAssembler.assemble(textEncoder);
    var hiddenStates = runtime.runEncoderFull(
      input.inputIds(),
      input.attentionMask()
    );
    var result = decodeFromHiddenStates(hiddenStates, input, text, threshold);
    recordTelemetry(startNanos, result);
    return result;
  }

  private void recordTelemetry(long startNanos, R result) {
    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    telemetry.record(durationMs, 1, resultSize(result));
  }

  // ---- lifecycle -----------------------------------------------------------

  /** Subclass hook invoked at the start of {@link #close()}. */
  protected void onClose() {}

  @Override
  public final void close() {
    onClose();
    runtime.close();
    tokenizer.close();
    log.info("{} closed", getClass().getSimpleName());
  }
}
