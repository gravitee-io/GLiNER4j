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
package io.gravitee.lab.gliner4j.runtime;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Runtime for original-GLiNER uni-encoder span models (e.g. gliner-pii-base, deberta-v3 +
 * span_mode=markerV0). Unlike the GLiNER2 family these ship as a single monolithic {@code
 * model.onnx} (encoder + span head fused), so this is a standalone {@link ArchitectureRuntime}
 * rather than a {@link BaseRuntime} subclass — it reuses the shared {@link OrtSessions} helpers.
 *
 * <p>Graph contract:
 * <pre>
 *   in : input_ids, attention_mask, words_mask, text_lengths (int64); span_idx (int64); span_mask (bool)
 *   out: logits [batch, words, max_width, num_classes]
 * </pre>
 */
@Slf4j
public final class GlinerUniNerRuntime implements ArchitectureRuntime {

  private final OrtEnvironment env;
  private final OrtSession session;

  public GlinerUniNerRuntime(
    Path modelDir,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    try {
      this.env = OrtEnvironment.getEnvironment();
      int numCpus = Runtime.getRuntime().availableProcessors();
      int intra = OrtSessions.getOrDefault(
        runtimeConfig.getEncoderIntraOpThreads(),
        numCpus
      );
      int inter = OrtSessions.getOrDefault(
        runtimeConfig.getEncoderInterOpThreads(),
        Math.max(2, numCpus / 2)
      );

      var variantDir = modelDir.resolve(variant);
      var provider = runtimeConfig.getExecutionProvider() == null
        ? ExecutionProvider.CPU
        : runtimeConfig.getExecutionProvider();
      Path cacheDir = null;
      if (
        runtimeConfig.isOptimizedModelCacheEnabled() &&
        provider.supportsOptimizedModelCache()
      ) {
        cacheDir = modelDir.resolve(
          variant + "_optimized_" + provider.cacheTag()
        );
        try {
          Files.createDirectories(cacheDir);
        } catch (IOException e) {
          log.warn(
            "Could not create optimized-model cache dir {} — caching disabled for this load",
            cacheDir,
            e
          );
          cacheDir = null;
        }
      }

      log.info("Loading model.onnx from {}...", variantDir);
      try (
        var opts = OrtSessions.createSessionOptions(
          intra,
          inter,
          OrtSession.SessionOptions.ExecutionMode.PARALLEL,
          runtimeConfig,
          cacheDir,
          "model.onnx"
        )
      ) {
        this.session = env.createSession(
          variantDir.resolve("model.onnx").toString(),
          opts
        );
      }
      log.info("GLiNER uni-encoder model loaded successfully");
    } catch (OrtException e) {
      throw new RuntimeException(
        "Failed to load GLiNER uni-encoder model.onnx from " +
          modelDir +
          "/" +
          variant,
        e
      );
    }
  }

  /**
   * Runs the monolithic span model for a single text.
   *
   * @param inputIds token IDs [seqLen]
   * @param wordsMask word mask [seqLen] (1-indexed first-subtoken of each text word, else 0)
   * @param textLen number of text words
   * @param spanIdx span (start, end) word-index pairs [numSpans][2]
   * @param spanMask validity per span [numSpans]
   * @return logits [textLen][maxWidth][numClasses]
   */
  public float[][][] run(
    long[] inputIds,
    long[] wordsMask,
    int textLen,
    long[][] spanIdx,
    boolean[] spanMask
  ) {
    int seqLen = inputIds.length;
    int numSpans = spanIdx.length;

    var attentionMask = new long[seqLen];
    java.util.Arrays.fill(attentionMask, 1L);

    try {
      try (
        var idsT = OnnxTensor.createTensor(env, new long[][] { inputIds });
        var maskT = OnnxTensor.createTensor(
          env,
          new long[][] { attentionMask }
        );
        var wordsT = OnnxTensor.createTensor(env, new long[][] { wordsMask });
        var lenT = OnnxTensor.createTensor(env, new long[][] { { textLen } });
        var spanIdxT = OnnxTensor.createTensor(env, new long[][][] { spanIdx });
        var spanMaskT = OnnxTensor.createTensor(
          env,
          new boolean[][] { spanMask }
        );
        var result = session.run(
          Map.of(
            "input_ids",
            idsT,
            "attention_mask",
            maskT,
            "words_mask",
            wordsT,
            "text_lengths",
            lenT,
            "span_idx",
            spanIdxT,
            "span_mask",
            spanMaskT
          )
        )
      ) {
        var logits = (float[][][][]) result.get(0).getValue(); // [1][L][K][C]
        return logits[0];
      }
    } catch (OrtException e) {
      throw new RuntimeException(
        "GLiNER uni-encoder inference failed (" +
          numSpans +
          " spans, " +
          textLen +
          " words)",
        e
      );
    }
  }

  @Override
  public void close() {
    try {
      session.close();
      log.info("GLiNER uni-encoder session closed");
    } catch (Exception e) {
      log.warn("Error closing GLiNER uni-encoder session", e);
    }
  }
}
