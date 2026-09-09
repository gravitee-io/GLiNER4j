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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import io.gravitee.lab.gliner4j.runtime.ExecutionProvider;
import io.gravitee.lab.gliner4j.runtime.OrtSessions;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The GLiClass {@code DecoderKVScorer} on ONNX Runtime ({@code onnx/scorer.onnx}): a 2-layer
 * bidirectional DeBERTa encoder over the label-section hidden states, text/label projectors and
 * the pair MLP. Inputs: {@code label_hidden [B, Ls, H]}, {@code label_positions [L]},
 * {@code sep_position [1]}; output {@code logits [B, L]}.
 */
public final class DecoderKvScorer implements LabelScorer {

  private static final Logger log = LoggerFactory.getLogger(
    DecoderKvScorer.class
  );

  private final OrtEnvironment env;
  private final OrtSession session;

  public DecoderKvScorer(
    Path modelDir,
    String variant,
    RuntimeConfig requested
  ) {
    // CPU provider only: the GPU belongs to llama.cpp, and two CUDA runtimes in one process clash.
    var runtimeConfig = requested.getExecutionProvider() ==
      ExecutionProvider.CPU
      ? requested
      : RuntimeConfig.builder()
        .executionProvider(ExecutionProvider.CPU)
        .optimizationLevel(requested.getOptimizationLevel())
        .optimizedModelCacheEnabled(requested.isOptimizedModelCacheEnabled())
        .scoringIntraOpThreads(requested.getScoringIntraOpThreads())
        .scoringInterOpThreads(requested.getScoringInterOpThreads())
        .build();
    var variantDir = modelDir.resolve(variant);
    var path = variantDir.resolve("scorer.onnx");
    if (!Files.exists(path)) {
      throw new IllegalStateException(
        "scorer.onnx not found in " +
          variantDir +
          " — export the bundle with `uv run scripts/export_gliclass_decoder_kv.py ...`"
      );
    }
    try {
      this.env = OrtEnvironment.getEnvironment();
      int numCpus = Runtime.getRuntime().availableProcessors();
      int intra = OrtSessions.getOrDefault(
        runtimeConfig.getScoringIntraOpThreads(),
        Math.max(1, numCpus / 2)
      );
      int inter = OrtSessions.getOrDefault(
        runtimeConfig.getScoringInterOpThreads(),
        1
      );
      var provider = ExecutionProvider.resolve(
        runtimeConfig.getExecutionProvider()
      );
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
            "Could not create optimized-model cache dir {} — caching disabled",
            cacheDir,
            e
          );
          cacheDir = null;
        }
      }
      try (
        var opts = OrtSessions.createSessionOptions(
          intra,
          inter,
          OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL,
          runtimeConfig,
          cacheDir,
          "scorer.onnx"
        )
      ) {
        this.session = env.createSession(path.toString(), opts);
      }
      log.info("Loaded scorer.onnx from {}", variantDir);
    } catch (OrtException e) {
      throw new RuntimeException(
        "Failed to load scorer.onnx from " + variantDir,
        e
      );
    }
  }

  /**
   * Scores one or more texts against one label section.
   *
   * @param labelHidden per text, the {@code bodyLength} hidden rows of the label section
   * @return {@code [texts][labels]} raw logits
   */
  @Override
  public float[][] score(
    List<List<float[]>> labelHidden,
    LabelSection section
  ) {
    int b = labelHidden.size();
    int ls = section.bodyLength();
    int h = labelHidden.get(0).get(0).length;
    var buf = ByteBuffer.allocateDirect(b * ls * h * Float.BYTES)
      .order(ByteOrder.nativeOrder())
      .asFloatBuffer();
    for (var rows : labelHidden) {
      if (rows.size() != ls) {
        throw new IllegalArgumentException(
          "expected " + ls + " label-section rows, got " + rows.size()
        );
      }
      for (var row : rows) {
        buf.put(row);
      }
    }
    buf.rewind();
    var posBuf = directLongs(section.labelPositions());
    var sepBuf = directLongs(new long[] { section.sepPosition() });
    try (
      var hiddenT = OnnxTensor.createTensor(env, buf, new long[] { b, ls, h });
      var posT = OnnxTensor.createTensor(
        env,
        posBuf,
        new long[] { section.labelPositions().length }
      );
      var sepT = OnnxTensor.createTensor(env, sepBuf, new long[] { 1 });
      var result = session.run(
        Map.of(
          "label_hidden",
          hiddenT,
          "label_positions",
          posT,
          "sep_position",
          sepT
        )
      )
    ) {
      var logitsT = (OnnxTensor) result.get(0);
      FloatBuffer flat = logitsT.getFloatBuffer();
      int l = section.labelPositions().length;
      var logits = new float[b][l];
      for (int i = 0; i < b; i++) {
        flat.get(logits[i]);
      }
      return logits;
    } catch (OrtException e) {
      throw new RuntimeException("scorer.onnx inference failed", e);
    }
  }

  private static LongBuffer directLongs(long[] values) {
    var buf = ByteBuffer.allocateDirect(Math.max(1, values.length) * Long.BYTES)
      .order(ByteOrder.nativeOrder())
      .asLongBuffer();
    buf.put(values).rewind();
    return buf;
  }

  @Override
  public void close() {
    try {
      session.close();
    } catch (OrtException e) {
      log.warn("Error closing scorer session", e);
    }
  }
}
