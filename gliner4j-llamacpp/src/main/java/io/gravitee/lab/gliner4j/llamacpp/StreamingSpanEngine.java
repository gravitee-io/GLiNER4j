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
import io.gravitee.lab.gliner4j.processor.AssemblerCache;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The streaming-span inference engine: tokenizer + llama.cpp backbone + ggml span scorer, plus
 * the pool of llama.cpp sequences that {@link StreamingSpanSession}s hold.
 *
 * <p>Prompt layout is gliner's {@code label1 <<LABEL>> label2 <<LABEL>> … <<SEP>> word1 word2 …}
 * (each label and each word tokenized on its own, no BOS/EOS). Word states are the first
 * sub-token's final hidden state; label vectors come from the DeBERTa labels encoder over the
 * prompt rows and are cached for the session.
 */
public final class StreamingSpanEngine implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(
    StreamingSpanEngine.class
  );

  /** Tokenized prompt for one label list. */
  record Prompt(List<String> labels, long[] ids, int[] labelPositions) {}

  /** Tokenized chunk: ids of every word plus the index of each word's first sub-token. */
  record Words(
    List<String> words,
    int[] starts,
    int[] ends,
    long[] ids,
    int[] firstSubtoken
  ) {}

  private final DjlTokenizerWrapper tokenizer;
  private final LlamaBackbone backbone;
  private final GgmlSpanScorer scorer;
  private final long classTokenId;
  private final long sepTokenId;
  private final int maxWidth;
  private final int rightContextWidth;
  private final AssemblerCache<List<String>, Prompt> prompts;
  private final SequencePool sequences;

  private StreamingSpanEngine(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    LlamaBackbone backbone,
    GgmlSpanScorer scorer,
    int cacheSize
  ) {
    this.tokenizer = tokenizer;
    this.backbone = backbone;
    this.scorer = scorer;
    this.classTokenId = config.archLong("class_token_index", -1);
    this.sepTokenId = config.archLong("sep_token_index", -1);
    this.maxWidth = config.archInt("max_width", config.getMaxWidth());
    this.rightContextWidth = config.archInt("right_context_width", maxWidth);
    if (classTokenId < 0 || sepTokenId < 0) {
      throw new IllegalStateException(
        "gliner4j_config.json must carry class_token_index / sep_token_index"
      );
    }
    this.prompts = new AssemblerCache<>(cacheSize, this::buildPrompt);
    this.sequences = new SequencePool(
      backbone.nSeqMax(),
      DecoderKvEngine.SEQUENCE_WAIT_MILLIS
    );
  }

  public static StreamingSpanEngine load(LoadContext ctx) {
    var config = ctx.config();
    var rc = ctx.runtimeConfig();
    var gguf = ctx
      .modelDir()
      .resolve("gguf")
      .resolve(config.archString("backbone_gguf", "backbone-q8_0.gguf"));
    int nCtx = config.archInt("n_ctx", DecoderKvEngine.DEFAULT_N_CTX);
    int nBatch = config.archInt("n_batch", DecoderKvEngine.DEFAULT_N_BATCH);
    int nSeqMax = config.archInt(
      "n_seq_max",
      DecoderKvEngine.DEFAULT_N_SEQ_MAX
    );
    int threads = LlamaBackbone.threads(rc);
    int gpuLayers = switch (rc.getExecutionProvider()) {
      case CPU, OPENVINO -> 0;
      case AUTO, CUDA -> 99;
    };
    var backbone = new LlamaBackbone(
      gguf,
      gpuLayers,
      nCtx,
      nBatch,
      nSeqMax,
      threads
    );
    var scorer = new GgmlSpanScorer(
      ctx.modelDir().resolve("gguf").resolve("scorer.gguf"),
      gpuLayers > 0,
      threads,
      (float) config.archDouble("layer_norm_eps", 1e-7)
    );
    log.info(
      "GLiNER streaming-span engine loaded (backbone={})",
      gguf.getFileName()
    );
    return new StreamingSpanEngine(
      config,
      ctx.tokenizer(),
      backbone,
      scorer,
      rc.effectiveOverrideCacheSize()
    );
  }

  int maxWidth() {
    return maxWidth;
  }

  int rightContextWidth() {
    return rightContextWidth;
  }

  int nCtx() {
    return backbone.nCtx();
  }

  Prompt prompt(List<String> labels) {
    return prompts.get(List.copyOf(labels));
  }

  private Prompt buildPrompt(List<String> labels) {
    var ids = new ArrayList<Long>();
    var positions = new int[labels.size()];
    for (int i = 0; i < labels.size(); i++) {
      for (long id : tokenizer.tokenizeWithIds(labels.get(i)).ids())
        ids.add(id);
      positions[i] = ids.size();
      ids.add(classTokenId);
    }
    ids.add(sepTokenId);
    return new Prompt(
      labels,
      ids.stream().mapToLong(Long::longValue).toArray(),
      positions
    );
  }

  /** Splits a chunk into words (gliner's whitespace splitter) and tokenizes each word on its own. */
  Words words(String chunk) {
    var enc = new io.gravitee.lab.gliner4j.processor.TextEncoder(chunk);
    var words = enc.getWords();
    var ids = new ArrayList<Long>();
    var first = new int[words.size()];
    tokenizer.prefetchTokens(words);
    for (int i = 0; i < words.size(); i++) {
      first[i] = ids.size();
      for (long id : tokenizer.tokenizeWithIds(words.get(i)).ids()) ids.add(id);
    }
    return new Words(
      words,
      enc.getWordStartChars(),
      enc.getWordEndChars(),
      ids.stream().mapToLong(Long::longValue).toArray(),
      first
    );
  }

  int acquireSequence() {
    return sequences.acquire();
  }

  void releaseSequence(int seq) {
    backbone.clearSequence(seq);
    sequences.release(seq);
  }

  /** Decodes {@code ids} at positions {@code startPos…} into {@code seq} and returns every token's state. */
  List<float[]> decode(int seq, long[] ids, int startPos) {
    return backbone.decode(seq, ids, startPos, 0);
  }

  float[][] labelEmbeddings(List<float[]> promptRows, Prompt prompt) {
    return scorer.labelEmbeddings(promptRows, prompt.labelPositions());
  }

  float[][] spanLogits(
    List<float[]> window,
    int[] starts,
    int[] ends,
    int latest,
    float[][] labels
  ) {
    return scorer.spanLogits(window, starts, ends, latest, labels);
  }

  @Override
  public void close() {
    scorer.close();
    backbone.close();
    tokenizer.close();
  }
}
