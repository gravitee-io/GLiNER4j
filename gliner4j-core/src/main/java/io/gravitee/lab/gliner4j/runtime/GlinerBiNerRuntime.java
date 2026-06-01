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
import java.util.Arrays;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Runtime for original-GLiNER bi-encoder span models (e.g. gliner-bi-small: deberta-v3 text encoder
 * + MiniLM label encoder, span_mode=markerV0). A single fused {@code model.onnx} runs both encoders:
 * it takes the text inputs plus tokenized labels and returns span logits.
 *
 * <p>Standalone {@link ArchitectureRuntime} (reuses {@link OrtSessions}). Graph contract:
 * <pre>
 *   in : input_ids, attention_mask, words_mask, text_lengths (int64); span_idx (int64); span_mask (bool);
 *        labels_input_ids, labels_attention_mask (int64, [num_labels, label_seq_len])
 *   out: logits [batch, words, max_width, num_classes]
 * </pre>
 *
 * <p>Labels are passed as token IDs and re-encoded each call — the precomputed label-embedding
 * cache (the bi-encoder's headline fast-path) is a later optimization.
 */
@Slf4j
public final class GlinerBiNerRuntime implements ArchitectureRuntime {

  private final OrtEnvironment env;
  private final OrtSession session;

  public GlinerBiNerRuntime(
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

      log.info("Loading bi-encoder model.onnx from {}...", variantDir);
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
      log.info("GLiNER bi-encoder model loaded successfully");
    } catch (OrtException e) {
      throw new RuntimeException(
        "Failed to load GLiNER bi-encoder model.onnx from " +
          modelDir +
          "/" +
          variant,
        e
      );
    }
  }

  /**
   * Runs the fused bi-encoder span model for a single text.
   *
   * @param inputIds text token IDs [seqLen]
   * @param wordsMask word mask [seqLen]
   * @param textLen number of text words
   * @param spanIdx span (start, end) pairs [numSpans][2]
   * @param spanMask span validity [numSpans]
   * @param labelsInputIds tokenized labels [numLabels][labelSeqLen]
   * @param labelsAttentionMask label attention mask [numLabels][labelSeqLen]
   * @return logits [textLen][maxWidth][numClasses]
   */
  public float[][][] run(
    long[] inputIds,
    long[] wordsMask,
    int textLen,
    long[][] spanIdx,
    boolean[] spanMask,
    long[][] labelsInputIds,
    long[][] labelsAttentionMask
  ) {
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
        var spanIdxT = OnnxTensor.createTensor(env, new long[][][] { spanIdx });
        var spanMaskT = OnnxTensor.createTensor(
          env,
          new boolean[][] { spanMask }
        );
        var labelIdsT = OnnxTensor.createTensor(env, labelsInputIds);
        var labelMaskT = OnnxTensor.createTensor(env, labelsAttentionMask);
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
            spanMaskT,
            "labels_input_ids",
            labelIdsT,
            "labels_attention_mask",
            labelMaskT
          )
        )
      ) {
        var logits = (float[][][][]) result.get(0).getValue(); // [1][L][K][C]
        return logits[0];
      }
    } catch (OrtException e) {
      throw new RuntimeException(
        "GLiNER bi-encoder inference failed (" +
          labelsInputIds.length +
          " labels, " +
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
      log.info("GLiNER bi-encoder session closed");
    } catch (Exception e) {
      log.warn("Error closing GLiNER bi-encoder session", e);
    }
  }
}
