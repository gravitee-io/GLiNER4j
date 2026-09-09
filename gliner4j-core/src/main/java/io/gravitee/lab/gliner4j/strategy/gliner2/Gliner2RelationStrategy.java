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

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.extractor.Extractor;
import io.gravitee.lab.gliner4j.postprocess.RelationDecoder;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInput;
import io.gravitee.lab.gliner4j.processor.MultiSchemaInputAssembler;
import io.gravitee.lab.gliner4j.processor.SchemaUnit;
import io.gravitee.lab.gliner4j.processor.UnitLayout;
import io.gravitee.lab.gliner4j.runtime.Gliner2SpanRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import io.gravitee.lab.gliner4j.strategy.RelationStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * GLiNER2 relation extraction on the merged {@code ner_full.onnx} graph: one relation unit
 * ({@code ( [P] name ( [R] head [R] tail ) )}) per definition, each scored through the count-aware
 * span head and decoded by {@link RelationDecoder}. Formerly the body of
 * {@link io.gravitee.lab.gliner4j.extractor.RelationExtractor}; that facade now dispatches by
 * family and delegates here for {@code gliner2} bundles.
 */
@Slf4j
public final class Gliner2RelationStrategy
  extends Extractor<RelationDefinition, RelationInstance>
  implements RelationStrategy {

  private final RelationDecoder relationDecoder;

  private Gliner2RelationStrategy(
    GLiNER4jConfig config,
    RuntimeConfig runtimeConfig,
    List<RelationDefinition> relations,
    DjlTokenizerWrapper tokenizer,
    Gliner2SpanRuntime runtime,
    MultiSchemaInputAssembler inputAssembler
  ) {
    super(
      config,
      runtimeConfig,
      relations,
      tokenizer,
      runtime,
      inputAssembler,
      new GLiNER4jTelemetry("extract_relations")
    );
    this.relationDecoder = new RelationDecoder();
  }

  /**
   * Builds the GLiNER2 relation strategy from the family-agnostic load context.
   *
   * @param ctx the load context (model dir, variant, runtime config, parsed config, tokenizer)
   * @param relations the load-time relation schema (non-empty)
   * @return a ready strategy
   */
  public static Gliner2RelationStrategy create(
    LoadContext ctx,
    List<RelationDefinition> relations
  ) {
    return create(
      ctx,
      relations,
      new io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime(
        ctx.modelDir(),
        ctx.variant(),
        ctx.runtimeConfig()
      )
    );
  }

  /** Same, over an already-built span runtime (e.g. the ggml one from {@code gliner4j-llamacpp}). */
  public static Gliner2RelationStrategy create(
    LoadContext ctx,
    List<RelationDefinition> relations,
    Gliner2SpanRuntime runtime
  ) {
    return new Gliner2RelationStrategy(
      ctx.config(),
      ctx.runtimeConfig(),
      relations,
      ctx.tokenizer(),
      runtime,
      assemblerFor(ctx.tokenizer(), relations)
    );
  }

  // ---- RelationStrategy ----------------------------------------------------

  @Override
  public Map<String, List<RelationInstance>> extract(
    String text,
    float threshold
  ) {
    return doExtractOnce(text, threshold);
  }

  @Override
  public Map<String, List<RelationInstance>> extract(
    String text,
    List<RelationDefinition> overrideRelations,
    float threshold
  ) {
    if (overrideRelations.isEmpty()) {
      throw new IllegalArgumentException(
        "Override relations list must not be empty"
      );
    }
    return doExtractOverride(text, overrideRelations, threshold);
  }

  @Override
  public List<Map<String, List<RelationInstance>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    return doExtractBatch(texts, threshold);
  }

  // ---- AbstractMultiUnitFacade hooks --------------------------------------

  @Override
  protected MultiSchemaInputAssembler buildAssembler(
    List<RelationDefinition> relations
  ) {
    return assemblerFor(tokenizer, relations);
  }

  @Override
  protected List<RelationInstance> decodeUnit(
    RelationDefinition relation,
    UnitLayout layout,
    float[][] countLogits,
    float[][][][] spanScores,
    MultiSchemaInput input,
    String text,
    int textLen,
    float threshold
  ) {
    return relationDecoder.decode(
      layout.unit().parentLabel(),
      layout.unit().childNames(),
      countLogits,
      spanScores,
      input.wordStartChars(),
      input.wordEndChars(),
      text,
      textLen,
      threshold
    );
  }

  // ---- helpers -------------------------------------------------------------

  private static MultiSchemaInputAssembler assemblerFor(
    DjlTokenizerWrapper tokenizer,
    List<RelationDefinition> relations
  ) {
    var units = relations.stream().map(SchemaUnit::forRelation).toList();
    return new MultiSchemaInputAssembler(tokenizer, units);
  }
}
