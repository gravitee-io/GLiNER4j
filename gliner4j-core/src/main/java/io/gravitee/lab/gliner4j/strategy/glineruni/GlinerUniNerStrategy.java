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
package io.gravitee.lab.gliner4j.strategy.glineruni;

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.postprocess.SpanDecoder;
import io.gravitee.lab.gliner4j.runtime.GlinerUniNerRuntime;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.utils.GlinerNerSupport;
import io.gravitee.lab.gliner4j.utils.GlinerPrompt;
import io.gravitee.lab.gliner4j.utils.WhitespaceWordSplitter;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Original-GLiNER uni-encoder span NER strategy (span_mode=markerV0, e.g. gliner-pii-base).
 *
 * <p>Builds the GLiNER prompt {@code <<ENT>> t1 <<ENT>> t2 ... <<SEP>> w1 w2 ...}, a {@code
 * words_mask} marking each text word's first subtoken, and full {@code span_idx}/{@code span_mask}
 * up to {@code max_width}; runs the monolithic {@code model.onnx}; then sigmoids the
 * {@code [words][width][class]} logits and reuses the shared {@link SpanDecoder} (threshold +
 * char mapping + greedy non-overlap).
 *
 * <p>Selected by {@link io.gravitee.lab.gliner4j.arch.glineruni.GlinerUniArchitecture}.
 */
@Slf4j
public final class GlinerUniNerStrategy implements NerStrategy {

  private final GLiNER4jConfig config;
  private final DjlTokenizerWrapper tokenizer;
  private final GlinerUniNerRuntime runtime;
  private final List<EntityDefinition> entities;
  private final SpanDecoder spanDecoder = new SpanDecoder();
  private final GLiNER4jTelemetry telemetry = new GLiNER4jTelemetry("extract");

  private final int maxWidth;
  private final String entToken;
  private final String sepToken;
  private final long clsId;
  private final long sepId;

  private GlinerUniNerStrategy(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    GlinerUniNerRuntime runtime,
    List<EntityDefinition> entities
  ) {
    this.config = config;
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.entities = entities;
    this.maxWidth = config.getMaxWidth();
    this.entToken = config.archString("ent_token", "<<ENT>>");
    this.sepToken = config.archString("sep_token", "<<SEP>>");
    // [CLS] … [SEP] ids straight from the tokenizer's post-processor (deberta: 1 / 2).
    // TODO(gliner-x): this strategy also serves the mT5 multilingual model (gliner-x), where two
    // approximations apply: (1) we split on whitespace, but gliner-x was trained with stanza
    // (language-aware) splitting — fine for space-separated languages, but CJK is unsupported and
    // punctuation boundaries may drift; (2) mT5 has no [CLS] token, so encodeWithSpecialTokens("")
    // yields its eos/pad and we prepend it at position 0 — a markerV0 model tolerates this (it
    // pools per-word + per-<<ENT>> reps, not position 0), but it's not the exact training prompt.
    // Revisit with a real word splitter (stanza port / ICU) and mT5-correct special-token handling.
    long[] specials = tokenizer.encodeWithSpecialTokens("");
    this.clsId = specials.length > 0 ? specials[0] : 1L;
    this.sepId = specials.length > 1 ? specials[specials.length - 1] : 2L;
  }

  public static GlinerUniNerStrategy create(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    var runtime = new GlinerUniNerRuntime(
      ctx.modelDir(),
      ctx.variant(),
      ctx.runtimeConfig()
    );
    return new GlinerUniNerStrategy(
      ctx.config(),
      ctx.tokenizer(),
      runtime,
      entities
    );
  }

  // ---- NerStrategy ---------------------------------------------------------

  @Override
  public Map<String, List<EntitySpan>> extract(String text, float threshold) {
    return extractWith(text, entities, threshold);
  }

  @Override
  public Map<String, List<EntitySpan>> extract(
    String text,
    List<EntityDefinition> overrideEntities,
    float threshold
  ) {
    return extractWith(text, overrideEntities, threshold);
  }

  @Override
  public List<Map<String, List<EntitySpan>>> extractBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null) {
      return List.of();
    }
    var out = new ArrayList<Map<String, List<EntitySpan>>>(texts.size());
    for (var text : texts) {
      out.add(extractWith(text, entities, threshold));
    }
    return out;
  }

  // ---- core ----------------------------------------------------------------

  private Map<String, List<EntitySpan>> extractWith(
    String text,
    List<EntityDefinition> labels,
    float threshold
  ) {
    long startNanos = System.nanoTime();
    if (text == null || text.isBlank() || labels.isEmpty()) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    var words = WhitespaceWordSplitter.split(text);
    int textLen = words.size();
    if (textLen == 0) {
      telemetry.record(0.0, 1, 0);
      return Map.of();
    }

    // 1. Build input_ids + words_mask: [CLS] <<ENT>> t1 ... <<SEP>> w1 ... [SEP]
    var idsList = new ArrayList<Long>();
    var wordsMaskList = new ArrayList<Long>();
    idsList.add(clsId);
    wordsMaskList.add(0L);

    for (var label : labels) {
      GlinerPrompt.appendToken(tokenizer, entToken, idsList, wordsMaskList, 0L);
      GlinerPrompt.appendToken(
        tokenizer,
        label.name(),
        idsList,
        wordsMaskList,
        0L
      );
    }
    GlinerPrompt.appendToken(tokenizer, sepToken, idsList, wordsMaskList, 0L);

    for (int w = 0; w < textLen; w++) {
      GlinerPrompt.appendWord(
        tokenizer,
        words.get(w).text(),
        idsList,
        wordsMaskList,
        w + 1L
      );
    }

    idsList.add(sepId);
    wordsMaskList.add(0L);

    long[] inputIds = idsList.stream().mapToLong(Long::longValue).toArray();
    long[] wordsMask = wordsMaskList
      .stream()
      .mapToLong(Long::longValue)
      .toArray();

    // 2. span_idx / span_mask grid up to maxWidth → run the monolithic span model.
    var grid = GlinerNerSupport.buildSpanGrid(textLen, maxWidth);
    var logits = runtime.run(
      inputIds,
      wordsMask,
      textLen,
      grid.spanIdx(),
      grid.spanMask()
    );

    // 3. Decode [words][width][class] logits → flat spans, then group by type.
    var spans = GlinerNerSupport.decodeMarkerV0(
      spanDecoder,
      logits,
      words,
      labels,
      text,
      maxWidth,
      threshold
    );

    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    telemetry.record(durationMs, 1, spans.size());
    return GlinerNerSupport.groupByType(spans);
  }

  @Override
  public void close() {
    runtime.close();
    tokenizer.close();
    log.info("GlinerUniNerStrategy closed");
  }
}
