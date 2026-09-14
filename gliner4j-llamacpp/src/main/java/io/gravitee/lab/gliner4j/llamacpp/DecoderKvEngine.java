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
import io.gravitee.lab.gliner4j.runtime.ExecutionProvider;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Files;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The decoder-kv inference engine: tokenizer + llama.cpp backbone + ONNX scorer, plus the pool of
 * llama.cpp sequence slots that stateless calls borrow and {@link DecoderKvSession}s hold.
 *
 * <p>Sequence layout is fastino's: {@code text<<SEP>>l1<<LABEL>>…<<SEP>>}. The text goes into the
 * KV cache; the label section is decoded on top of it with outputs enabled, scored, and removed
 * again (sessions) or the whole sequence is cleared (stateless).
 */
public final class DecoderKvEngine implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(
    DecoderKvEngine.class
  );

  /** Default context window for a fresh context (tokens). Override with {@code n_ctx}. */
  public static final int DEFAULT_N_CTX = 4096;
  public static final int DEFAULT_N_BATCH = 512;
  public static final int DEFAULT_N_SEQ_MAX = 8;
  /** How long a caller waits for a free sequence slot before failing. */
  public static final long SEQUENCE_WAIT_MILLIS = 30_000;

  private final DjlTokenizerWrapper tokenizer;
  private final LlamaBackbone backbone;
  private final LabelScorer scorer;
  private final String labelToken;
  private final String sepToken;
  private final long classTokenId;
  private final long sepTokenId;
  private final AssemblerCache<List<String>, LabelSection> sections;
  private final SequencePool sequences;

  private DecoderKvEngine(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    LlamaBackbone backbone,
    LabelScorer scorer,
    RuntimeConfig runtimeConfig
  ) {
    this.tokenizer = tokenizer;
    this.backbone = backbone;
    this.scorer = scorer;
    this.labelToken = config.archString("label_token", "<<LABEL>>");
    this.sepToken = config.archString("sep_token", "<<SEP>>");
    this.classTokenId = config.archLong("class_token_index", -1);
    this.sepTokenId = config.archLong("sep_token_index", -1);
    if (classTokenId < 0 || sepTokenId < 0) {
      throw new IllegalStateException(
        "gliner4j_config.json must carry architecture_config.class_token_index / sep_token_index"
      );
    }
    this.sections = new AssemblerCache<>(
      runtimeConfig.effectiveOverrideCacheSize(),
      labels ->
        LabelSection.build(
          tokenizer,
          labels,
          labelToken,
          sepToken,
          classTokenId,
          sepTokenId
        )
    );
    this.sequences = new SequencePool(backbone.nSeqMax(), SEQUENCE_WAIT_MILLIS);
  }

  /** Loads the engine for a {@code gliclass-decoder-kv} bundle. */
  public static DecoderKvEngine load(LoadContext ctx) {
    var config = ctx.config();
    var rc = ctx.runtimeConfig();
    var gguf = ctx
      .modelDir()
      .resolve("gguf")
      .resolve(config.archString("backbone_gguf", "backbone-q8_0.gguf"));
    int nCtx = config.archInt("n_ctx", DEFAULT_N_CTX);
    int nBatch = config.archInt("n_batch", DEFAULT_N_BATCH);
    int nSeqMax = config.archInt("n_seq_max", DEFAULT_N_SEQ_MAX);
    int threads = LlamaBackbone.threads(rc);
    int gpuLayers = switch (rc.getExecutionProvider()) {
      case CPU, OPENVINO -> 0;
      case AUTO, CUDA -> 99; // llama.cpp picks Metal / CUDA / Vulkan from the loaded backends
    };
    var backbone = new LlamaBackbone(
      gguf,
      gpuLayers,
      nCtx,
      nBatch,
      nSeqMax,
      threads
    );
    var scorerGguf = ctx.modelDir().resolve("gguf").resolve("scorer.gguf");
    var scorerBackend = config.archString("scorer_backend", "ggml");
    LabelScorer scorer;
    if ("ggml".equals(scorerBackend) && Files.exists(scorerGguf)) {
      // Same ggml backend family as the backbone (Metal/CUDA/CPU): one GPU runtime per process.
      scorer = new GgmlScorer(
        scorerGguf,
        gpuLayers > 0,
        threads,
        config.archBoolean("normalize_features", false),
        (float) config.archDouble("logit_scale", 1.0),
        (float) config.archDouble("layer_norm_eps", 1e-7)
      );
    } else {
      // ONNX Runtime fallback (older bundles / scorer_backend=onnx), pinned to the CPU provider
      // so it never competes with llama.cpp for the GPU.
      scorer = new DecoderKvScorer(ctx.modelDir(), ctx.variant(), rc);
    }
    log.info(
      "GLiClass decoder-kv engine loaded (backbone={}, scorer={}, ep={})",
      gguf.getFileName(),
      scorer.getClass().getSimpleName(),
      ExecutionProvider.resolve(rc.getExecutionProvider())
    );
    return new DecoderKvEngine(config, ctx.tokenizer(), backbone, scorer, rc);
  }

  public long[] encodeText(String text) {
    return tokenizer.encodeWithSpecialTokens(text);
  }

  public LabelSection section(List<String> labels) {
    return sections.get(List.copyOf(labels));
  }

  public int nCtx() {
    return backbone.nCtx();
  }

  // ---- stateless ------------------------------------------------------------

  /** Scores {@code textIds} against {@code section} in a borrowed sequence. Raw logits. */
  public float[] score(long[] textIds, LabelSection section) {
    int seq = acquireSequence();
    try {
      var ids = new long[textIds.length + section.ids().length];
      System.arraycopy(textIds, 0, ids, 0, textIds.length);
      System.arraycopy(
        section.ids(),
        0,
        ids,
        textIds.length,
        section.ids().length
      );
      // outputs from the first body token (the one after the opening <<SEP>>)
      var rows = backbone.decode(seq, ids, 0, textIds.length + 1);
      return scorer.score(List.of(rows), section)[0];
    } finally {
      backbone.clearSequence(seq);
      releaseSequence(seq);
    }
  }

  // ---- sessions -------------------------------------------------------------

  int acquireSequence() {
    return sequences.acquire();
  }

  void releaseSequence(int seq) {
    backbone.clearSequence(seq);
    sequences.release(seq);
  }

  /** Extends sequence {@code seq}'s cache with {@code ids} at positions {@code startPos…}. */
  void append(int seq, long[] ids, int startPos) {
    backbone.decode(seq, ids, startPos, ids.length);
  }

  /** Scores the cached text of {@code seq} (length {@code textLen}) against {@code section}, leaving the cache as it was. */
  float[] scoreCached(int seq, int textLen, LabelSection section) {
    try {
      var rows = backbone.decode(seq, section.ids(), textLen, 1);
      return scorer.score(List.of(rows), section)[0];
    } finally {
      backbone.removeFrom(seq, textLen);
    }
  }

  void clearSequence(int seq) {
    backbone.clearSequence(seq);
  }

  @Override
  public void close() {
    scorer.close();
    backbone.close();
    tokenizer.close();
  }
}
