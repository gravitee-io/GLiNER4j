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
package io.gravitee.lab.gliner4j.strategy.glinerbi;

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.postprocess.SpanDecoder;
import io.gravitee.lab.gliner4j.runtime.GlinerBiNerRuntime;
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
 * Original-GLiNER bi-encoder span NER strategy (e.g. gliner-bi-small: deberta text encoder + MiniLM
 * label encoder, span_mode=markerV0).
 *
 * <p>Differs from the uni-encoder strategy in two ways: the text prompt carries only {@code <<ENT>>}
 * markers (one per label, <em>no</em> type words — labels are encoded separately), and labels are
 * tokenized with a <em>second</em> tokenizer (the label encoder's) into {@code labels_input_ids}.
 * Everything else is shared: {@code words_mask}, {@code span_idx}, sigmoid → {@link SpanDecoder}.
 * The i-th {@code <<ENT>>} marker aligns with the i-th label embedding (and logit class i).
 *
 * <p>Selected by {@link io.gravitee.lab.gliner4j.arch.glinerbi.GlinerBiArchitecture}.
 */
@Slf4j
public final class GlinerBiNerStrategy implements NerStrategy {

  private final GLiNER4jConfig config;
  private final DjlTokenizerWrapper tokenizer;
  private final DjlTokenizerWrapper labelTokenizer;
  private final GlinerBiNerRuntime runtime;
  private final List<EntityDefinition> entities;
  private final SpanDecoder spanDecoder = new SpanDecoder();
  private final GLiNER4jTelemetry telemetry = new GLiNER4jTelemetry("extract");

  private final int maxWidth;
  private final String entToken;
  private final String sepToken;
  private final long clsId;
  private final long sepId;

  private GlinerBiNerStrategy(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    DjlTokenizerWrapper labelTokenizer,
    GlinerBiNerRuntime runtime,
    List<EntityDefinition> entities
  ) {
    this.config = config;
    this.tokenizer = tokenizer;
    this.labelTokenizer = labelTokenizer;
    this.runtime = runtime;
    this.entities = entities;
    this.maxWidth = config.getMaxWidth();
    this.entToken = config.archString("ent_token", "<<ENT>>");
    this.sepToken = config.archString("sep_token", "<<SEP>>");
    long[] specials = tokenizer.encodeWithSpecialTokens("");
    this.clsId = specials.length > 0 ? specials[0] : 1L;
    this.sepId = specials.length > 1 ? specials[specials.length - 1] : 2L;
  }

  public static GlinerBiNerStrategy create(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    var runtime = new GlinerBiNerRuntime(
      ctx.modelDir(),
      ctx.variant(),
      ctx.runtimeConfig()
    );
    // Label encoder uses its own tokenizer, exported under labels_tokenizer/.
    var labelTokenizer = new DjlTokenizerWrapper(
      ctx.modelDir().resolve("labels_tokenizer")
    );
    return new GlinerBiNerStrategy(
      ctx.config(),
      ctx.tokenizer(),
      labelTokenizer,
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

    // 1. Text input: [CLS] <<ENT>> x N <<SEP>> w1 ... [SEP]  (marker-only prompt, no type words)
    var idsList = new ArrayList<Long>();
    var wordsMaskList = new ArrayList<Long>();
    idsList.add(clsId);
    wordsMaskList.add(0L);
    for (int n = 0; n < labels.size(); n++) {
      GlinerPrompt.appendToken(tokenizer, entToken, idsList, wordsMaskList, 0L);
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

    // 2. span_idx / span_mask grid up to maxWidth.
    var grid = GlinerNerSupport.buildSpanGrid(textLen, maxWidth);

    // 3. Label tokens via the label encoder's tokenizer: [CLS] label [SEP], padded to longest.
    var labelEnc = new long[labels.size()][];
    int maxLabelLen = 0;
    for (int i = 0; i < labels.size(); i++) {
      labelEnc[i] = labelTokenizer.encodeWithSpecialTokens(
        labels.get(i).name()
      );
      maxLabelLen = Math.max(maxLabelLen, labelEnc[i].length);
    }
    var labelsInputIds = new long[labels.size()][maxLabelLen];
    var labelsAttentionMask = new long[labels.size()][maxLabelLen];
    for (int i = 0; i < labels.size(); i++) {
      for (int j = 0; j < labelEnc[i].length; j++) {
        labelsInputIds[i][j] = labelEnc[i][j];
        labelsAttentionMask[i][j] = 1L;
      }
      // remaining positions stay 0 (pad id) with attention 0
    }

    // 4. Run fused bi-encoder graph → logits [words][width][class]
    var logits = runtime.run(
      inputIds,
      wordsMask,
      textLen,
      grid.spanIdx(),
      grid.spanMask(),
      labelsInputIds,
      labelsAttentionMask
    );

    // 5. Decode [words][width][class] logits → flat spans, then group by type.
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
    labelTokenizer.close();
    log.info("GlinerBiNerStrategy closed");
  }
}
