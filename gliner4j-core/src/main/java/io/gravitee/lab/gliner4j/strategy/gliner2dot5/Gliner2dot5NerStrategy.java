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
package io.gravitee.lab.gliner4j.strategy.gliner2dot5;

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.postprocess.BoundaryDecoder;
import io.gravitee.lab.gliner4j.processor.AssemblerCache;
import io.gravitee.lab.gliner4j.processor.InputAssembler;
import io.gravitee.lab.gliner4j.processor.PreprocessedInput;
import io.gravitee.lab.gliner4j.processor.SchemaEncoder;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.runtime.Gliner2dot5NerRuntime;
import io.gravitee.lab.gliner4j.runtime.MicroBatcher;
import io.gravitee.lab.gliner4j.runtime.OnnxGliner2dot5NerRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.tokenizer.TokenMapping;
import io.gravitee.lab.gliner4j.utils.GlinerNerSupport;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * GLiNER2.5 (fastino boundary architecture) NER strategy on the merged {@code ner_full.onnx}
 * graph. The prompt is the GLiNER2 one ({@code ( [P] entities ( [E] a [E] b ) ) [SEP_TEXT] …})
 * with the fastino processor's text conventions (lower-cased words, trailing period); each
 * {@code [E]} marker is one boundary query. The graph returns pooled pair logits over a fixed
 * candidate pool, decoded by {@link BoundaryDecoder}.
 *
 * <p>Selected by {@link io.gravitee.lab.gliner4j.arch.gliner2dot5.Gliner2dot5Architecture}.
 */
@Slf4j
public final class Gliner2dot5NerStrategy implements NerStrategy {

  private final RuntimeConfig runtimeConfig;
  private final DjlTokenizerWrapper tokenizer;
  private final Gliner2dot5NerRuntime runtime;
  private final InputAssembler inputAssembler;
  private final AssemblerCache<
    List<EntityDefinition>,
    InputAssembler
  > overrideAssemblers;
  private final BoundaryDecoder decoder;
  private final boolean lowercaseWords;
  private final boolean appendPeriod;
  private final GLiNER4jTelemetry telemetry = new GLiNER4jTelemetry("extract");
  private final MicroBatcher<Map<String, List<EntitySpan>>> microBatcher;

  private Gliner2dot5NerStrategy(
    GLiNER4jConfig config,
    RuntimeConfig runtimeConfig,
    DjlTokenizerWrapper tokenizer,
    Gliner2dot5NerRuntime runtime,
    List<EntityDefinition> entities
  ) {
    this.runtimeConfig = runtimeConfig;
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.inputAssembler = new InputAssembler(
      tokenizer,
      entitySchemaEncoder(entities)
    );
    this.overrideAssemblers = new AssemblerCache<>(
      runtimeConfig.effectiveOverrideCacheSize(),
      defs -> new InputAssembler(tokenizer, entitySchemaEncoder(defs))
    );
    this.decoder = new BoundaryDecoder(
      (float) config.archDouble("pair_temperature", 1.0),
      config.archBoolean("enable_abstention", true),
      (float) config.archDouble("abstention_threshold", 0.5)
    );
    this.lowercaseWords = config.archBoolean("lowercase_words", true);
    this.appendPeriod = config.archBoolean("append_period", true);
    this.microBatcher = runtimeConfig.isMicroBatchingEnabled()
      ? new MicroBatcher<>(
        runtimeConfig.getMicroBatchMaxSize(),
        runtimeConfig.getMicroBatchMaxWaitMicros(),
        this::extractBatch
      )
      : null;
  }

  /**
   * Builds the GLiNER2.5 NER strategy from the family-agnostic load context.
   *
   * @param ctx the load context (model dir, variant, runtime config, parsed config, tokenizer)
   * @param entities the load-time entity schema
   * @return a ready strategy
   */
  public static Gliner2dot5NerStrategy create(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    return create(
      ctx,
      entities,
      new OnnxGliner2dot5NerRuntime(
        ctx.modelDir(),
        ctx.variant(),
        ctx.runtimeConfig()
      )
    );
  }

  /** Same, over an already-built runtime (e.g. the ggml one from {@code gliner4j-llamacpp}). */
  public static Gliner2dot5NerStrategy create(
    LoadContext ctx,
    List<EntityDefinition> entities,
    Gliner2dot5NerRuntime runtime
  ) {
    return new Gliner2dot5NerStrategy(
      ctx.config(),
      ctx.runtimeConfig(),
      ctx.tokenizer(),
      runtime,
      entities
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
    return extractBatchWith(List.of(text), threshold, inputAssembler).get(0);
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
      overrideAssemblers.get(entities)
    ).get(0);
  }

  @Override
  public List<Map<String, List<EntitySpan>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    return extractBatchWith(texts, threshold, inputAssembler);
  }

  @Override
  public void close() {
    if (microBatcher != null) {
      microBatcher.close();
    }
    runtime.close();
    tokenizer.close();
    log.info("Gliner2dot5NerStrategy closed");
  }

  // ---- core ----------------------------------------------------------------

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
    var results = new ArrayList<Map<String, List<EntitySpan>>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(Map.of());
    }

    // 1. Preprocess: fastino word conventions, shared schema prefix.
    var inputs = new PreprocessedInput[batchSize];
    var slots = new ArrayList<Integer>();
    for (int i = 0; i < batchSize; i++) {
      var text = texts.get(i);
      if (text == null || text.isBlank()) continue;
      var enc = new TextEncoder(text, lowercaseWords, appendPeriod);
      if (enc.getTextLen() == 0) continue;
      inputs[i] = assembler.assemble(enc);
      slots.add(i);
    }
    if (slots.isEmpty() || inputs[slots.get(0)].numFields() == 0) {
      telemetry.record(
        (System.nanoTime() - startNanos) / 1_000_000.0,
        batchSize,
        0
      );
      return results;
    }

    // 2. Length-sorted sub-batches (less padding), one merged-graph run each.
    slots.sort(Comparator.comparingInt(i -> inputs[i].inputIds().length));
    var configuredSubBatch = runtimeConfig.getMaxSubBatchSize(); // null = unbounded
    int subBatch = configuredSubBatch == null
      ? slots.size()
      : Math.max(1, configuredSubBatch);
    var schemaPositions = inputs[slots.get(0)].schemaTokenPositions();
    var queryPositions = new long[schemaPositions.length - 1];
    for (int i = 0; i < queryPositions.length; i++) {
      queryPositions[i] = schemaPositions[i + 1];
    }
    long totalEntities = 0;
    for (int from = 0; from < slots.size(); from += subBatch) {
      var chunk = slots.subList(from, Math.min(slots.size(), from + subBatch));
      int bs = chunk.size();
      var ids = new long[bs][];
      var mask = new long[bs][];
      int maxSeqLen = 0;
      int maxTextLen = 0;
      for (int j = 0; j < bs; j++) {
        var in = inputs[chunk.get(j)];
        ids[j] = in.inputIds();
        mask[j] = in.attentionMask();
        maxSeqLen = Math.max(maxSeqLen, ids[j].length);
        maxTextLen = Math.max(maxTextLen, in.textLen());
      }
      var wordPositions = new long[bs * maxTextLen];
      Arrays.fill(wordPositions, -1L);
      for (int j = 0; j < bs; j++) {
        var positions = firstSubwordPositions(inputs[chunk.get(j)]);
        for (int w = 0; w < positions.length; w++) {
          wordPositions[j * maxTextLen + w] = positions[w];
        }
      }
      var scoring = runtime.run(
        ids,
        mask,
        maxSeqLen,
        wordPositions,
        bs,
        maxTextLen,
        queryPositions
      );

      // 3. Decode each row.
      for (int j = 0; j < bs; j++) {
        int origIdx = chunk.get(j);
        var in = inputs[origIdx];
        var spans = decoder.decode(
          scoring,
          j,
          in.fieldNames(),
          in.wordStartChars(),
          in.wordEndChars(),
          texts.get(origIdx),
          in.textLen(),
          threshold
        );
        totalEntities += spans.size();
        results.set(origIdx, GlinerNerSupport.groupByType(spans));
      }
    }

    telemetry.record(
      (System.nanoTime() - startNanos) / 1_000_000.0,
      batchSize,
      totalEntities
    );
    return results;
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

  /** Per-word first-subword positions ({@code -1} for words without a TEXT mapping). */
  private static int[] firstSubwordPositions(PreprocessedInput input) {
    var positions = new int[input.textLen()];
    Arrays.fill(positions, -1);
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
