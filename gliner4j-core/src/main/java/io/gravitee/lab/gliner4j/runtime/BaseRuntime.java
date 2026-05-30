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
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Base runtime managing the shared encoder ONNX session and its buffer logic.
 * Subclasses add task-specific heads (span_rep + scoring_head for NER, classifier_head for classification).
 */
@Slf4j
public abstract sealed class BaseRuntime
  implements ArchitectureRuntime
  permits GLiNER4jClassifierRuntime, GLiNER4jNERRuntime, GliclassRuntime {

  /** Default ONNX variant folder name (base FP32). */
  public static final String DEFAULT_VARIANT = "onnx";

  protected final OrtEnvironment env;
  protected final OrtSession encoderSession;

  // Pre-allocated encoder buffers (single-thread assumption — not thread-safe)
  private long[] schemaPrefixIds;
  private int schemaPrefixLen;
  private LongBuffer encoderIdsBuf;
  private LongBuffer encoderMaskBuf;
  private int encoderBufCapacity;

  // Pre-allocated batch encoder buffers (single-thread assumption)
  private LongBuffer batchEncoderIdsBuf;
  private LongBuffer batchEncoderMaskBuf;
  private int batchEncoderBufCapacity;

  protected BaseRuntime(
    Path modelDir,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    try {
      this.env = OrtEnvironment.getEnvironment();
      int numCpus = Runtime.getRuntime().availableProcessors();

      int encoderIntra = runtimeConfig.getEncoderIntraOpThreads() != null
        ? runtimeConfig.getEncoderIntraOpThreads()
        : numCpus;
      int encoderInter = runtimeConfig.getEncoderInterOpThreads() != null
        ? runtimeConfig.getEncoderInterOpThreads()
        : Math.max(2, numCpus / 2);

      var variantDir = modelDir.resolve(variant);
      var provider = runtimeConfig.getExecutionProvider() == null
        ? ExecutionProvider.CPU
        : runtimeConfig.getExecutionProvider();
      // Each backend gets its own optimized-model cache dir (the optimized graph is provider-specific).
      // Providers that emit compiled nodes (CUDA/CoreML/OpenVINO) cannot serialize their graph, so the
      // cache is skipped for them rather than failing the load with an ORT serialization error.
      Path cacheDir = null;
      if (runtimeConfig.isOptimizedModelCacheEnabled()) {
        if (provider.supportsOptimizedModelCache()) {
          cacheDir = modelDir.resolve(
            variant + "_optimized_" + provider.cacheTag()
          );
          cacheDir.toFile().mkdirs();
        } else {
          log.info(
            "Optimized-model cache disabled for execution provider {} (it emits non-serializable compiled nodes)",
            provider
          );
        }
      }

      log.info(
        "Loading ONNX models from {} (cache: {})",
        variantDir,
        cacheDir != null ? cacheDir : "disabled"
      );

      log.info("Loading encoder.onnx...");
      try (
        var opts = createSessionOptions(
          encoderIntra,
          encoderInter,
          OrtSession.SessionOptions.ExecutionMode.PARALLEL,
          runtimeConfig,
          cacheDir,
          "encoder.onnx"
        )
      ) {
        this.encoderSession = env.createSession(
          variantDir.resolve("encoder.onnx").toString(),
          opts
        );
      }

      loadTaskHeads(variantDir, runtimeConfig, cacheDir);

      log.info("All ONNX sessions loaded successfully");
    } catch (OrtException e) {
      throw new RuntimeException(
        "Failed to load ONNX models from " + modelDir + "/" + variant,
        e
      );
    }
  }

  /**
   * Loads task-specific ONNX model heads. Called during construction after the encoder is loaded.
   *
   * @param variantDir the resolved variant directory
   * @param runtimeConfig resource control configuration
   * @param cacheDir optimized model cache directory (may be null)
   */
  protected abstract void loadTaskHeads(
    Path variantDir,
    RuntimeConfig runtimeConfig,
    Path cacheDir
  ) throws OrtException;

  /**
   * Closes task-specific ONNX sessions. Called by {@link #close()}.
   */
  protected abstract void closeTaskHeads() throws OrtException;

  /**
   * Initializes reusable encoder buffers with the schema prefix that is constant across requests.
   *
   * @param schemaPrefix the concatenated schema + separator token IDs
   */
  public void initEncoderBuffers(long[] schemaPrefix) {
    this.schemaPrefixIds = schemaPrefix;
    this.schemaPrefixLen = schemaPrefix.length;
    this.encoderBufCapacity = 0;
  }

  /**
   * Runs the encoder model with cached schema prefix.
   *
   * @param inputIds token IDs [seqLen]
   * @param attentionMask attention mask [seqLen]
   * @return last hidden state [1][seqLen][hiddenSize]
   */
  public float[][][] runEncoder(long[] inputIds, long[] attentionMask) {
    try {
      int seqLen = inputIds.length;
      LongBuffer idsBuf;
      LongBuffer maskBuf;

      if (schemaPrefixIds != null) {
        if (seqLen > encoderBufCapacity) {
          encoderBufCapacity = seqLen + 64;
          encoderIdsBuf = allocateDirectLongBuffer(encoderBufCapacity);
          encoderMaskBuf = allocateDirectLongBuffer(encoderBufCapacity);
          encoderIdsBuf.put(schemaPrefixIds).rewind();
          var ones = new long[encoderBufCapacity];
          Arrays.fill(ones, 1L);
          encoderMaskBuf.put(ones).rewind();
        }
        encoderIdsBuf.limit(encoderBufCapacity).position(schemaPrefixLen);
        encoderIdsBuf.put(inputIds, schemaPrefixLen, seqLen - schemaPrefixLen);
        encoderIdsBuf.rewind().limit(seqLen);
        encoderMaskBuf.rewind().limit(seqLen);

        idsBuf = encoderIdsBuf;
        maskBuf = encoderMaskBuf;
      } else {
        idsBuf = allocateDirectLongBuffer(inputIds);
        maskBuf = allocateDirectLongBuffer(attentionMask);
      }

      var shape = new long[] { 1, seqLen };
      var idsTensor = OnnxTensor.createTensor(env, idsBuf, shape);
      var maskTensor = OnnxTensor.createTensor(env, maskBuf, shape);

      try (
        var result = encoderSession.run(
          Map.of("input_ids", idsTensor, "attention_mask", maskTensor)
        )
      ) {
        return (float[][][]) result.get(0).getValue();
      } finally {
        idsTensor.close();
        maskTensor.close();
      }
    } catch (OrtException e) {
      throw new RuntimeException("Encoder inference failed", e);
    }
  }

  /**
   * Runs the encoder model without buffer caching — used for per-call schema overrides.
   *
   * @param inputIds token IDs [seqLen]
   * @param attentionMask attention mask [seqLen]
   * @return last hidden state [1][seqLen][hiddenSize]
   */
  public float[][][] runEncoderFull(long[] inputIds, long[] attentionMask) {
    try {
      int seqLen = inputIds.length;
      var idsBuf = allocateDirectLongBuffer(inputIds);
      var maskBuf = allocateDirectLongBuffer(attentionMask);

      var shape = new long[] { 1, seqLen };
      var idsTensor = OnnxTensor.createTensor(env, idsBuf, shape);
      var maskTensor = OnnxTensor.createTensor(env, maskBuf, shape);

      try (
        var result = encoderSession.run(
          Map.of("input_ids", idsTensor, "attention_mask", maskTensor)
        )
      ) {
        return (float[][][]) result.get(0).getValue();
      } finally {
        idsTensor.close();
        maskTensor.close();
      }
    } catch (OrtException e) {
      throw new RuntimeException("Encoder inference failed", e);
    }
  }

  /**
   * Runs the encoder model in batch mode.
   *
   * @param inputIds token IDs per text [batchSize][varying seqLen]
   * @param attentionMask attention masks per text [batchSize][varying seqLen]
   * @param maxSeqLen the padded sequence length
   * @return last hidden states [batchSize][maxSeqLen][hiddenSize]
   */
  public float[][][] runEncoderBatch(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen
  ) {
    try {
      int batchSize = inputIds.length;
      int totalElements = batchSize * maxSeqLen;

      if (totalElements > batchEncoderBufCapacity) {
        batchEncoderBufCapacity = totalElements + 256;
        batchEncoderIdsBuf = allocateDirectLongBuffer(batchEncoderBufCapacity);
        batchEncoderMaskBuf = allocateDirectLongBuffer(batchEncoderBufCapacity);
      }
      batchEncoderIdsBuf.clear().limit(totalElements);
      batchEncoderMaskBuf.clear().limit(totalElements);

      var zeroPad = new long[maxSeqLen];
      for (int b = 0; b < batchSize; b++) {
        int seqLen = inputIds[b].length;
        int padLen = maxSeqLen - seqLen;
        batchEncoderIdsBuf.put(inputIds[b]);
        if (padLen > 0) batchEncoderIdsBuf.put(zeroPad, 0, padLen);
        batchEncoderMaskBuf.put(attentionMask[b]);
        if (padLen > 0) batchEncoderMaskBuf.put(zeroPad, 0, padLen);
      }
      batchEncoderIdsBuf.rewind();
      batchEncoderMaskBuf.rewind();

      var shape = new long[] { batchSize, maxSeqLen };
      var idsTensor = OnnxTensor.createTensor(env, batchEncoderIdsBuf, shape);
      var maskTensor = OnnxTensor.createTensor(env, batchEncoderMaskBuf, shape);

      try (
        var result = encoderSession.run(
          Map.of("input_ids", idsTensor, "attention_mask", maskTensor)
        )
      ) {
        return (float[][][]) result.get(0).getValue();
      } finally {
        idsTensor.close();
        maskTensor.close();
      }
    } catch (OrtException e) {
      throw new RuntimeException("Batched encoder inference failed", e);
    }
  }

  @Override
  public void close() {
    try {
      encoderSession.close();
      closeTaskHeads();
      log.info("All ONNX sessions closed");
    } catch (OrtException e) {
      log.warn("Error closing ONNX sessions", e);
    }
  }

  // Session/EP/buffer helpers live in OrtSessions so non-BaseRuntime runtimes can reuse them;
  // these thin delegators keep BaseRuntime subclasses' call sites unchanged.
  protected static OrtSession.SessionOptions createSessionOptions(
    int intraOpThreads,
    int interOpThreads,
    OrtSession.SessionOptions.ExecutionMode executionMode,
    RuntimeConfig config,
    Path cacheDir,
    String modelFileName
  ) throws OrtException {
    return OrtSessions.createSessionOptions(
      intraOpThreads,
      interOpThreads,
      executionMode,
      config,
      cacheDir,
      modelFileName
    );
  }

  protected static LongBuffer allocateDirectLongBuffer(long[] data) {
    return OrtSessions.allocateDirectLongBuffer(data);
  }

  protected static LongBuffer allocateDirectLongBuffer(int capacity) {
    return OrtSessions.allocateDirectLongBuffer(capacity);
  }

  protected static int getOrDefault(Integer value, int defaultValue) {
    return OrtSessions.getOrDefault(value, defaultValue);
  }
}
