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
import io.gravitee.lab.gliner4j.postprocess.TokenSpanDecoder;
import io.gravitee.lab.gliner4j.runtime.GlinerUniTokenNerRuntime;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.utils.WhitespaceWordSplitter;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;

/**
 * Original-GLiNER uni-encoder <em>token-level</em> NER strategy (span_mode=token_level, e.g.
 * gliner-multitask-large).
 *
 * <p>Same prompt + {@code words_mask} as the markerV0 uni-encoder strategy
 * ({@code <<ENT>> t1 <<ENT>> t2 ... <<SEP>> words}), but the model takes only the text inputs (no
 * {@code span_idx}) and returns BIO-style {@code [words][class][3]} logits, decoded by
 * {@link TokenSpanDecoder}. Selected by {@link io.gravitee.lab.gliner4j.arch.glineruni.GlinerUniArchitecture}
 * when {@code architecture_config.span_mode == "token_level"}.
 */
@Slf4j
public final class GlinerUniTokenNerStrategy implements NerStrategy {

  private final DjlTokenizerWrapper tokenizer;
  private final GlinerUniTokenNerRuntime runtime;
  private final List<EntityDefinition> entities;
  private final TokenSpanDecoder decoder = new TokenSpanDecoder();
  private final GLiNER4jTelemetry telemetry = new GLiNER4jTelemetry("extract");

  private final String entToken;
  private final String sepToken;
  private final long clsId;
  private final long sepId;

  private GlinerUniTokenNerStrategy(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    GlinerUniTokenNerRuntime runtime,
    List<EntityDefinition> entities
  ) {
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.entities = entities;
    this.entToken = config.archString("ent_token", "<<ENT>>");
    this.sepToken = config.archString("sep_token", "<<SEP>>");
    long[] specials = tokenizer.encodeWithSpecialTokens("");
    this.clsId = specials.length > 0 ? specials[0] : 1L;
    this.sepId = specials.length > 1 ? specials[specials.length - 1] : 2L;
  }

  public static GlinerUniTokenNerStrategy create(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    var runtime = new GlinerUniTokenNerRuntime(
      ctx.modelDir(),
      ctx.variant(),
      ctx.runtimeConfig()
    );
    return new GlinerUniTokenNerStrategy(
      ctx.config(),
      ctx.tokenizer(),
      runtime,
      entities
    );
  }

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

    // [CLS] <<ENT>> t1 <<ENT>> t2 ... <<SEP>> w1 ... [SEP]
    var idsList = new ArrayList<Long>();
    var wordsMaskList = new ArrayList<Long>();
    idsList.add(clsId);
    wordsMaskList.add(0L);
    for (var label : labels) {
      appendTokens(entToken, idsList, wordsMaskList, 0L);
      appendTokens(label.name(), idsList, wordsMaskList, 0L);
    }
    appendTokens(sepToken, idsList, wordsMaskList, 0L);
    for (int w = 0; w < textLen; w++) {
      appendWord(words.get(w).text(), idsList, wordsMaskList, w + 1L);
    }
    idsList.add(sepId);
    wordsMaskList.add(0L);

    long[] inputIds = idsList.stream().mapToLong(Long::longValue).toArray();
    long[] wordsMask = wordsMaskList
      .stream()
      .mapToLong(Long::longValue)
      .toArray();

    var logits = runtime.run(inputIds, wordsMask, textLen); // [words][class][3]

    var labelNames = labels.stream().map(EntityDefinition::name).toList();
    int[] wordStart = new int[textLen];
    int[] wordEnd = new int[textLen];
    for (int i = 0; i < textLen; i++) {
      wordStart[i] = words.get(i).start();
      wordEnd[i] = words.get(i).end();
    }

    var spans = decoder.decode(
      logits,
      labelNames,
      wordStart,
      wordEnd,
      text,
      textLen,
      threshold
    );

    var grouped = spans
      .stream()
      .collect(
        Collectors.groupingBy(
          EntitySpan::type,
          LinkedHashMap::new,
          Collectors.toList()
        )
      );

    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    telemetry.record(durationMs, 1, spans.size());
    return grouped;
  }

  private void appendTokens(
    String token,
    List<Long> ids,
    List<Long> wordsMask,
    long maskValue
  ) {
    for (long id : tokenizer.tokenizeWithIds(token).ids()) {
      ids.add(id);
      wordsMask.add(maskValue);
    }
  }

  private void appendWord(
    String word,
    List<Long> ids,
    List<Long> wordsMask,
    long wordIndex1Based
  ) {
    long[] sub = tokenizer.tokenizeWithIds(word).ids();
    for (int i = 0; i < sub.length; i++) {
      ids.add(sub[i]);
      wordsMask.add(i == 0 ? wordIndex1Based : 0L);
    }
  }

  @Override
  public void close() {
    runtime.close();
    tokenizer.close();
    log.info("GlinerUniTokenNerStrategy closed");
  }
}
