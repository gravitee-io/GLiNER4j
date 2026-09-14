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
package io.gravitee.lab.gliner4j.llamacpp;

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.postprocess.SpanDecoder;
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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Original-GLiNER uni-encoder NER on the llama.cpp/ggml engine. Same prompt
 * ({@code [CLS] <<ENT>> t1 … <<SEP>> w1 … [SEP]}), words mask and decode as the ONNX
 * {@code GlinerUniNerStrategy}; the forward pass is {@link GgmlGlinerUniModel}.
 */
public final class LlamaGlinerUniNerStrategy implements NerStrategy {

  private static final Logger log = LoggerFactory.getLogger(
    LlamaGlinerUniNerStrategy.class
  );

  private final DjlTokenizerWrapper tokenizer;
  private final GgmlGlinerUniModel model;
  private final List<EntityDefinition> entities;
  private final SpanDecoder spanDecoder = new SpanDecoder();
  private final io.gravitee.lab.gliner4j.postprocess.TokenSpanDecoder tokenDecoder =
    new io.gravitee.lab.gliner4j.postprocess.TokenSpanDecoder();
  /** Bi-encoder only: label encoder + its tokenizer; null for uni-encoders. */
  private final LlamaEncoderBackbone labelEncoder;
  private final DjlTokenizerWrapper labelTokenizer;
  private final java.util.concurrent.ConcurrentHashMap<
    String,
    float[]
  > labelCache = new java.util.concurrent.ConcurrentHashMap<>();
  private final GLiNER4jTelemetry telemetry = new GLiNER4jTelemetry("extract");
  private final int maxWidth;
  private final long classTokenIndex;
  private final String entToken;
  private final String sepToken;
  private final long clsId;
  private final long sepId;

  LlamaGlinerUniNerStrategy(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    GgmlGlinerUniModel model,
    List<EntityDefinition> entities,
    LlamaEncoderBackbone labelEncoder,
    DjlTokenizerWrapper labelTokenizer
  ) {
    this.tokenizer = tokenizer;
    this.model = model;
    this.entities = entities;
    this.labelEncoder = labelEncoder;
    this.labelTokenizer = labelTokenizer;
    this.maxWidth = config.getMaxWidth();
    this.classTokenIndex = config.archLong("class_token_index", -1);
    this.entToken = config.archString("ent_token", "<<ENT>>");
    this.sepToken = config.archString("sep_token", "<<SEP>>");
    long[] specials = tokenizer.encodeWithSpecialTokens("");
    this.clsId = specials.length > 0 ? specials[0] : 1L;
    this.sepId = specials.length > 1 ? specials[specials.length - 1] : 2L;
    if (classTokenIndex < 0 && labelEncoder == null) {
      throw new IllegalStateException(
        "gliner-uni bundle is missing architecture_config.class_token_index"
      );
    }
  }

  public static LlamaGlinerUniNerStrategy create(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    var config = ctx.config();
    var spanMode = config.archString("span_mode", "markerV0");
    if (!"markerV0".equals(spanMode) && !"token_level".equals(spanMode)) {
      throw new UnsupportedOperationException(
        "gliner on llama.cpp supports span_mode=markerV0 or token_level (got " +
          spanMode +
          ")"
      );
    }
    var rc = ctx.runtimeConfig();
    int gpuLayers = LlamaGliclassClassificationStrategy.gpuLayers(rc);
    int threads = LlamaBackbone.threads(rc);
    var model = new GgmlGlinerUniModel(
      GgmlWeights.resolveModelGguf(ctx.modelDir(), ctx.variant()),
      gpuLayers > 0,
      threads
    );
    LlamaEncoderBackbone labelEncoder = null;
    DjlTokenizerWrapper labelTokenizer = null;
    var labelsGguf = config.archString("labels_gguf", null);
    if (labelsGguf != null) {
      labelEncoder = new LlamaEncoderBackbone(
        ctx.modelDir().resolve("gguf").resolve(labelsGguf),
        gpuLayers,
        config.archInt("labels_n_ubatch", 512),
        config.archInt(
          "n_seq_max",
          LlamaGliclassClassificationStrategy.DEFAULT_N_SEQ_MAX
        ),
        threads
      );
      labelTokenizer = new DjlTokenizerWrapper(
        ctx.modelDir().resolve("labels_tokenizer")
      );
    }
    log.info(
      "GLiNER {} llama.cpp strategy loaded (span_mode={}, hidden={})",
      labelsGguf != null ? "bi-encoder" : "uni-encoder",
      spanMode,
      model.modelHidden()
    );
    return new LlamaGlinerUniNerStrategy(
      config,
      ctx.tokenizer(),
      model,
      entities,
      labelEncoder,
      labelTokenizer
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
    var idsList = new ArrayList<Long>();
    var maskList = new ArrayList<Long>();
    idsList.add(clsId);
    maskList.add(0L);
    if (labelEncoder == null) {
      // uni-encoder: [CLS] <<ENT>> t1 <<ENT>> t2 … <<SEP>> words [SEP]
      for (var label : labels) {
        GlinerPrompt.appendToken(tokenizer, entToken, idsList, maskList, 0L);
        GlinerPrompt.appendToken(
          tokenizer,
          label.name(),
          idsList,
          maskList,
          0L
        );
      }
      GlinerPrompt.appendToken(tokenizer, sepToken, idsList, maskList, 0L);
    }
    // bi-encoder (gliner's BaseBiEncoderProcessor.tokenize_inputs): the text goes in alone,
    // [CLS] words [SEP]; the labels are encoded by the label encoder.
    for (int wi = 0; wi < textLen; wi++) {
      GlinerPrompt.appendWord(
        tokenizer,
        words.get(wi).text(),
        idsList,
        maskList,
        wi + 1L
      );
    }
    idsList.add(sepId);
    maskList.add(0L);

    long[] ids = idsList.stream().mapToLong(Long::longValue).toArray();
    var wordPositions = new int[textLen];
    var promptPositions = new ArrayList<Integer>(labels.size());
    for (int i = 0; i < ids.length; i++) {
      long m = maskList.get(i);
      if (m > 0) {
        wordPositions[(int) m - 1] = i;
      }
      if (ids[i] == classTokenIndex) {
        promptPositions.add(i);
      }
    }
    if (labelEncoder == null && promptPositions.size() != labels.size()) {
      log.warn(
        "<<ENT>> marker count ({}) differs from label count ({}); check label names for the marker text",
        promptPositions.size(),
        labels.size()
      );
    }
    int c = Math.min(promptPositions.size(), labels.size());
    var prompts = new int[c];
    for (int k = 0; k < c; k++) prompts[k] = promptPositions.get(k);
    var activeLabels = labels.subList(0, c);

    List<EntitySpan> spans;
    if (model.isTokenLevel()) {
      var logits = model.scoreTokens(ids, wordPositions, prompts);
      var offsets = GlinerNerSupport.charOffsets(words);
      spans = tokenDecoder.decode(
        logits,
        activeLabels.stream().map(EntityDefinition::name).toList(),
        offsets.starts(),
        offsets.ends(),
        text,
        textLen,
        threshold
      );
    } else if (labelEncoder != null) {
      var embeddings = new float[labels.size()][];
      for (int k = 0; k < labels.size(); k++) embeddings[k] = labelEmbedding(
        labels.get(k).name()
      );
      var logits = model.score(ids, wordPositions, null, embeddings, maxWidth);
      spans = GlinerNerSupport.decodeMarkerV0(
        spanDecoder,
        logits,
        words,
        labels,
        text,
        maxWidth,
        threshold
      );
    } else {
      var logits = model.score(ids, wordPositions, prompts, maxWidth);
      spans = GlinerNerSupport.decodeMarkerV0(
        spanDecoder,
        logits,
        words,
        activeLabels,
        text,
        maxWidth,
        threshold
      );
    }
    telemetry.record(
      (System.nanoTime() - startNanos) / 1_000_000.0,
      1,
      spans.size()
    );
    return GlinerNerSupport.groupByType(spans);
  }

  /** Bi-encoder label representation: mean of the label encoder's token states over {@code [CLS] label [SEP]}. */
  private float[] labelEmbedding(String name) {
    return labelCache.computeIfAbsent(name, n -> {
      var rows = labelEncoder.encode(labelTokenizer.encodeWithSpecialTokens(n));
      var out = new float[rows[0].length];
      for (var row : rows)
        for (int d = 0; d < out.length; d++) out[d] += row[d];
      for (int d = 0; d < out.length; d++) out[d] /= rows.length;
      return out;
    });
  }

  @Override
  public void close() {
    model.close();
    if (labelEncoder != null) labelEncoder.close();
    if (labelTokenizer != null) labelTokenizer.close();
    tokenizer.close();
  }
}
