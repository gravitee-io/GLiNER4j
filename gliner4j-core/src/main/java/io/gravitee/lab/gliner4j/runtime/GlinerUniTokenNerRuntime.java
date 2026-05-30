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
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Runtime for original-GLiNER <em>token-level</em> uni-encoder NER models (e.g.
 * gliner-multitask-large, span_mode=token_level). The monolithic {@code model.onnx} takes only the
 * text inputs (no span_idx/span_mask) and returns BIO-style token logits.
 *
 * <p>Standalone {@link ArchitectureRuntime} (reuses {@link OrtSessions}). Graph contract:
 * <pre>
 *   in : input_ids, attention_mask, words_mask, text_lengths (int64)
 *   out: logits [batch, words, num_classes, 3]  (3 = start / end / inside)
 * </pre>
 */
@Slf4j
public final class GlinerUniTokenNerRuntime implements ArchitectureRuntime {

  private final OrtEnvironment env;
  private final OrtSession session;

  public GlinerUniTokenNerRuntime(
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
        cacheDir.toFile().mkdirs();
      }

      log.info("Loading token-level model.onnx from {}...", variantDir);
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
      log.info("GLiNER token-level model loaded successfully");
    } catch (OrtException e) {
      throw new RuntimeException(
        "Failed to load GLiNER token-level model.onnx from " +
          modelDir +
          "/" +
          variant,
        e
      );
    }
  }

  /**
   * Runs the token-level model for a single text.
   *
   * @param inputIds token IDs [seqLen]
   * @param wordsMask word mask [seqLen]
   * @param textLen number of text words
   * @return logits [words][numClasses][3]
   */
  public float[][][] run(long[] inputIds, long[] wordsMask, int textLen) {
    int seqLen = inputIds.length;
    var attentionMask = new long[seqLen];
    Arrays.fill(attentionMask, 1L);

    try {
      try (
        var idsT = OnnxTensor.createTensor(env, new long[][] { inputIds });
        var maskT = OnnxTensor.createTensor(
          env,
          new long[][] { attentionMask }
        );
        var wordsT = OnnxTensor.createTensor(env, new long[][] { wordsMask });
        var lenT = OnnxTensor.createTensor(env, new long[][] { { textLen } });
        var result = session.run(
          Map.of(
            "input_ids",
            idsT,
            "attention_mask",
            maskT,
            "words_mask",
            wordsT,
            "text_lengths",
            lenT
          )
        )
      ) {
        var logits = (float[][][][]) result.get(0).getValue(); // [1][words][class][3]
        return logits[0];
      }
    } catch (OrtException e) {
      throw new RuntimeException(
        "GLiNER token-level inference failed (" + textLen + " words)",
        e
      );
    }
  }

  @Override
  public void close() {
    try {
      session.close();
      log.info("GLiNER token-level session closed");
    } catch (Exception e) {
      log.warn("Error closing GLiNER token-level session", e);
    }
  }
}
