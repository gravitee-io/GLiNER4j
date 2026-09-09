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
import io.gravitee.lab.gliner4j.postprocess.BoundaryRelationDecoder;
import io.gravitee.lab.gliner4j.processor.AssemblerCache;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInput;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInputAssembler;
import io.gravitee.lab.gliner4j.processor.SchemaUnit;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.runtime.Gliner2dot5RelationRuntime;
import io.gravitee.lab.gliner4j.runtime.MicroBatcher;
import io.gravitee.lab.gliner4j.runtime.OnnxGliner2dot5RelationRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import io.gravitee.lab.gliner4j.strategy.RelationStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * GLiNER2.5 relation extraction on the merged {@code relation_full.onnx} graph. Every relation
 * becomes one prompt unit {@code ( [P] name[: description] ( [R] head [R] tail ) )}, units are
 * joined by {@code [SEP_STRUCT]}, and the two {@code [R]} markers of each unit are the head and
 * tail queries the graph pairs and scores. Only {@code head}/{@code tail} relations are supported,
 * which is also all the fastino schema API exposes.
 */
@Slf4j
public final class Gliner2dot5RelationStrategy implements RelationStrategy {

  private static final String HEAD = "head";
  private static final String TAIL = "tail";

  private final RuntimeConfig runtimeConfig;
  private final DjlTokenizerWrapper tokenizer;
  private final Gliner2dot5RelationRuntime runtime;
  private final List<RelationDefinition> relations;
  private final MultiSchemaInputAssembler inputAssembler;
  private final AssemblerCache<
    List<RelationDefinition>,
    MultiSchemaInputAssembler
  > overrideAssemblers;
  private final BoundaryRelationDecoder decoder;
  private final boolean lowercaseWords;
  private final boolean appendPeriod;
  private final GLiNER4jTelemetry telemetry = new GLiNER4jTelemetry(
    "extract_relations"
  );
  private final MicroBatcher<Map<String, List<RelationInstance>>> microBatcher;

  private Gliner2dot5RelationStrategy(
    GLiNER4jConfig config,
    RuntimeConfig runtimeConfig,
    DjlTokenizerWrapper tokenizer,
    Gliner2dot5RelationRuntime runtime,
    List<RelationDefinition> relations
  ) {
    this.runtimeConfig = runtimeConfig;
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.relations = relations;
    this.inputAssembler = assemblerFor(tokenizer, relations);
    this.overrideAssemblers = new AssemblerCache<>(
      runtimeConfig.effectiveOverrideCacheSize(),
      defs -> assemblerFor(tokenizer, defs)
    );
    this.decoder = new BoundaryRelationDecoder(
      (float) config.archDouble("relation_temperature", 1.0)
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

  public static Gliner2dot5RelationStrategy create(
    LoadContext ctx,
    List<RelationDefinition> relations
  ) {
    return create(
      ctx,
      relations,
      new OnnxGliner2dot5RelationRuntime(
        ctx.modelDir(),
        ctx.variant(),
        ctx.runtimeConfig()
      )
    );
  }

  /** Same, over an already-built runtime (e.g. the ggml one from {@code gliner4j-llamacpp}). */
  public static Gliner2dot5RelationStrategy create(
    LoadContext ctx,
    List<RelationDefinition> relations,
    Gliner2dot5RelationRuntime runtime
  ) {
    return new Gliner2dot5RelationStrategy(
      ctx.config(),
      ctx.runtimeConfig(),
      ctx.tokenizer(),
      runtime,
      relations
    );
  }

  // ---- RelationStrategy ----------------------------------------------------

  @Override
  public Map<String, List<RelationInstance>> extract(
    String text,
    float threshold
  ) {
    if (microBatcher != null) {
      return microBatcher.extract(text, threshold);
    }
    return extractBatchWith(
      List.of(text == null ? "" : text),
      threshold,
      relations,
      inputAssembler
    ).get(0);
  }

  @Override
  public Map<String, List<RelationInstance>> extract(
    String text,
    List<RelationDefinition> overrideRelations,
    float threshold
  ) {
    return extractBatchWith(
      List.of(text == null ? "" : text),
      threshold,
      overrideRelations,
      overrideAssemblers.get(overrideRelations)
    ).get(0);
  }

  @Override
  public List<Map<String, List<RelationInstance>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    return extractBatchWith(texts, threshold, relations, inputAssembler);
  }

  @Override
  public void close() {
    if (microBatcher != null) {
      microBatcher.close();
    }
    runtime.close();
    tokenizer.close();
    log.info("Gliner2dot5RelationStrategy closed");
  }

  // ---- core ----------------------------------------------------------------

  private List<Map<String, List<RelationInstance>>> extractBatchWith(
    List<String> texts,
    float threshold,
    List<RelationDefinition> activeRelations,
    MultiSchemaInputAssembler assembler
  ) {
    if (texts == null) {
      return List.of();
    }
    long startNanos = System.nanoTime();
    int batchSize = texts.size();
    var relationNames = activeRelations
      .stream()
      .map(RelationDefinition::name)
      .toList();
    var results = new ArrayList<Map<String, List<RelationInstance>>>(batchSize);
    for (int i = 0; i < batchSize; i++) {
      results.add(emptyResult(relationNames));
    }

    var inputs = new MultiSchemaInput[batchSize];
    var slots = new ArrayList<Integer>();
    for (int i = 0; i < batchSize; i++) {
      var text = texts.get(i);
      if (text == null || text.isBlank()) continue;
      var enc = new TextEncoder(text, lowercaseWords, appendPeriod);
      if (enc.getTextLen() == 0) continue;
      inputs[i] = assembler.assemble(enc);
      slots.add(i);
    }
    if (slots.isEmpty()) {
      telemetry.record(
        (System.nanoTime() - startNanos) / 1_000_000.0,
        batchSize,
        0
      );
      return results;
    }

    // Head/tail marker positions per unit, shared across the batch (identical schema prefix).
    var layouts = inputs[slots.get(0)].unitLayouts();
    var queryPositions = new long[layouts.size() * 2];
    for (int u = 0; u < layouts.size(); u++) {
      var markers = layouts.get(u).childMarkerPositions();
      queryPositions[2 * u] = markers[0];
      queryPositions[2 * u + 1] = markers[1];
    }

    slots.sort(Comparator.comparingInt(i -> inputs[i].inputIds().length));
    var configuredSubBatch = runtimeConfig.getMaxSubBatchSize(); // null = unbounded
    int subBatch = configuredSubBatch == null
      ? slots.size()
      : Math.max(1, configuredSubBatch);
    long total = 0;
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
        var positions = inputs[chunk.get(j)].wordFirstSubwordPos();
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
      for (int j = 0; j < bs; j++) {
        int origIdx = chunk.get(j);
        var in = inputs[origIdx];
        var decoded = decoder.decode(
          scoring,
          j,
          relationNames,
          HEAD,
          TAIL,
          in.wordStartChars(),
          in.wordEndChars(),
          texts.get(origIdx),
          in.textLen(),
          threshold
        );
        total += decoded.values().stream().mapToLong(List::size).sum();
        results.set(origIdx, decoded);
      }
    }

    telemetry.record(
      (System.nanoTime() - startNanos) / 1_000_000.0,
      batchSize,
      total
    );
    return results;
  }

  private static Map<String, List<RelationInstance>> emptyResult(
    List<String> relationNames
  ) {
    var out = new LinkedHashMap<String, List<RelationInstance>>();
    for (var name : relationNames) {
      out.put(name, List.of());
    }
    return out;
  }

  /**
   * One {@code [R] head [R] tail} unit per relation; a description rides on the parent label as
   * {@code name: description}, which is how the fastino processor renders relation prompts.
   */
  private static MultiSchemaInputAssembler assemblerFor(
    DjlTokenizerWrapper tokenizer,
    List<RelationDefinition> relations
  ) {
    var units = new ArrayList<SchemaUnit>(relations.size());
    for (var relation : relations) {
      if (!relation.fields().equals(List.of(HEAD, TAIL))) {
        throw new IllegalArgumentException(
          "GLiNER2.5 relations are head/tail only; relation '" +
            relation.name() +
            "' declares fields " +
            relation.fields()
        );
      }
      var desc = relation.description();
      var label = desc == null || desc.isBlank()
        ? relation.name()
        : relation.name() + ": " + desc;
      units.add(
        new SchemaUnit(
          SchemaUnit.Kind.RELATION,
          label,
          "[R]",
          List.of(HEAD, TAIL),
          List.of("", "")
        )
      );
    }
    return new MultiSchemaInputAssembler(tokenizer, units);
  }
}
