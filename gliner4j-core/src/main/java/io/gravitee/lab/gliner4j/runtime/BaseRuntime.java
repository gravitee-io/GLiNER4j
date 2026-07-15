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
import java.nio.LongBuffer;
import java.nio.file.Files;
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
  // The standalone encoder is optional: artifacts that ship a self-contained merged graph
  // (ner_full.onnx / classifier_full.onnx, which embed their own encoder) can omit encoder.onnx,
  // in which case this stays null and the split encoder paths (runEncoder*) are unavailable.
  protected final OrtSession encoderSession;

  // Pooled encoder buffers — concurrent callers each acquire their own holder, sequential
  // callers keep reusing the same one. Encoder holders carry the schema prefix + all-ones
  // mask; batch holders are refilled per call.
  private volatile long[] schemaPrefixIds;
  private final DirectBufferPool encoderBuffers = new DirectBufferPool(2);
  private final DirectBufferPool batchEncoderBuffers = new DirectBufferPool(2);

  // Encoder output binding: name + element type, resolved at load time so batch outputs can
  // be pinned to pooled direct buffers (no native→heap materialization).
  private String encoderOutputName;
  private ai.onnxruntime.OnnxJavaType encoderOutputType;
  private final DirectByteBufferPool encoderOutBuffers =
    new DirectByteBufferPool();

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
      var provider = ExecutionProvider.resolve(
        runtimeConfig.getExecutionProvider()
      );
      // Each backend gets its own optimized-model cache dir (the optimized graph is provider-specific).
      // Providers that emit compiled nodes (CUDA/OpenVINO) cannot serialize their graph, so the
      // cache is skipped for them rather than failing the load with an ORT serialization error.
      Path cacheDir = null;
      if (runtimeConfig.isOptimizedModelCacheEnabled()) {
        if (provider.supportsOptimizedModelCache()) {
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

      if (Files.exists(variantDir.resolve("encoder.onnx"))) {
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
        this.encoderOutputName = encoderSession
          .getOutputNames()
          .iterator()
          .next();
        this.encoderOutputType = encoderSession
              .getOutputInfo()
              .get(encoderOutputName)
              .getInfo() instanceof
            ai.onnxruntime.TensorInfo encoderOutInfo
          ? encoderOutInfo.type
          : ai.onnxruntime.OnnxJavaType.FLOAT;
      } else {
        log.info(
          "encoder.onnx absent — relying on a self-contained merged graph (ner_full/classifier_full)"
        );
        this.encoderSession = null;
        this.encoderOutputName = null;
        this.encoderOutputType = ai.onnxruntime.OnnxJavaType.FLOAT;
      }

      loadTaskHeads(variantDir, runtimeConfig, cacheDir);

      scheduleProfilingFlush(runtimeConfig);

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
   * The sessions whose profiling traces the timed flush should write. Subclasses with task-head
   * sessions add them on top of the encoder.
   */
  protected java.util.List<OrtSession> profilableSessions() {
    return encoderSession == null
      ? java.util.List.of()
      : java.util.List.of(encoderSession);
  }

  /**
   * When profiling is enabled with {@link RuntimeConfig#getProfilingSeconds()}, starts a daemon
   * timer that flushes each session's profiling trace to disk via
   * {@link OrtSession#endProfiling()} after the configured delay — so traces can be collected
   * from a running server instead of relying on a graceful shutdown (which ephemeral hosts may
   * not survive). Each session's trace covers load → flush; ORT records nothing afterwards.
   */
  private void scheduleProfilingFlush(RuntimeConfig runtimeConfig) {
    if (
      runtimeConfig.getProfilingDir() == null ||
      runtimeConfig.getProfilingSeconds() == null
    ) {
      return;
    }
    int delaySeconds = runtimeConfig.getProfilingSeconds();
    var flusher = new Thread(
      () -> {
        try {
          Thread.sleep(delaySeconds * 1000L);
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          return;
        }
        for (OrtSession session : profilableSessions()) {
          try {
            log.info("ORT profiling trace flushed: {}", session.endProfiling());
          } catch (OrtException e) {
            log.warn("Failed to flush ORT profiling trace: {}", e.getMessage());
          }
        }
      },
      "gliner4j-ort-profiling-flush"
    );
    flusher.setDaemon(true);
    flusher.start();
    log.info(
      "ORT profiling traces will be flushed to {} in {}s",
      runtimeConfig.getProfilingDir(),
      delaySeconds
    );
  }

  /**
   * Initializes reusable encoder buffers with the schema prefix that is constant across requests.
   *
   * @param schemaPrefix the concatenated schema + separator token IDs
   */
  public void initEncoderBuffers(long[] schemaPrefix) {
    this.schemaPrefixIds = schemaPrefix;
    // Pooled holders carry the previous prefix — drop them so they are rebuilt on demand.
    encoderBuffers.clear();
  }

  /**
   * Runs the encoder model with cached schema prefix.
   *
   * @param inputIds token IDs [seqLen]
   * @param attentionMask attention mask [seqLen]
   * @return last hidden state [1][seqLen][hiddenSize]
   */
  public float[][][] runEncoder(long[] inputIds, long[] attentionMask) {
    var prefix = schemaPrefixIds;
    if (prefix == null) {
      return runEncoderFull(inputIds, attentionMask);
    }
    var holder = encoderBuffers.acquire();
    try {
      int seqLen = inputIds.length;
      int prefixLen = prefix.length;

      if (seqLen > holder.capacity()) {
        holder.reallocate(seqLen + 64);
        var idsInit = holder.buf(0);
        idsInit.put(prefix).rewind();
        var ones = new long[holder.capacity()];
        Arrays.fill(ones, 1L);
        holder.buf(1).put(ones).rewind();
      }
      var idsBuf = holder.buf(0);
      var maskBuf = holder.buf(1);
      idsBuf.limit(holder.capacity()).position(prefixLen);
      idsBuf.put(inputIds, prefixLen, seqLen - prefixLen);
      idsBuf.rewind().limit(seqLen);
      maskBuf.rewind().limit(seqLen);

      var shape = new long[] { 1, seqLen };
      var idsTensor = OnnxTensor.createTensor(env, idsBuf, shape);
      var maskTensor = OnnxTensor.createTensor(env, maskBuf, shape);

      try (
        var result = encoderSession.run(
          Map.of("input_ids", idsTensor, "attention_mask", maskTensor)
        )
      ) {
        return toNested3d(FloatTensor.of((OnnxTensor) result.get(0)));
      } finally {
        idsTensor.close();
        maskTensor.close();
      }
    } catch (OrtException e) {
      throw new RuntimeException("Encoder inference failed", e);
    } finally {
      encoderBuffers.release(holder);
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
        return toNested3d(FloatTensor.of((OnnxTensor) result.get(0)));
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
    return toNested3d(runEncoderBatchFlat(inputIds, attentionMask, maxSeqLen));
  }

  /**
   * Runs the encoder model in batch mode, returning the hidden states as a flat tensor
   * (one bulk copy out of the ONNX result instead of nested-array materialization).
   *
   * @param inputIds token IDs per text [batchSize][varying seqLen]
   * @param attentionMask attention masks per text [batchSize][varying seqLen]
   * @param maxSeqLen the padded sequence length
   * @return last hidden states, shape [batchSize][maxSeqLen][hiddenSize]
   */
  public FloatTensor runEncoderBatchFlat(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen
  ) {
    var holder = batchEncoderBuffers.acquire();
    try {
      int batchSize = inputIds.length;
      int totalElements = batchSize * maxSeqLen;

      if (totalElements > holder.capacity()) {
        holder.reallocate(totalElements + 256);
      }
      var idsBuf = holder.buf(0);
      var maskBuf = holder.buf(1);
      idsBuf.clear().limit(totalElements);
      maskBuf.clear().limit(totalElements);

      var zeroPad = new long[maxSeqLen];
      for (int b = 0; b < batchSize; b++) {
        int seqLen = inputIds[b].length;
        int padLen = maxSeqLen - seqLen;
        idsBuf.put(inputIds[b]);
        if (padLen > 0) idsBuf.put(zeroPad, 0, padLen);
        maskBuf.put(attentionMask[b]);
        if (padLen > 0) maskBuf.put(zeroPad, 0, padLen);
      }
      idsBuf.rewind();
      maskBuf.rewind();

      var shape = new long[] { batchSize, maxSeqLen };
      var idsTensor = OnnxTensor.createTensor(env, idsBuf, shape);
      var maskTensor = OnnxTensor.createTensor(env, maskBuf, shape);

      try (
        var result = encoderSession.run(
          Map.of("input_ids", idsTensor, "attention_mask", maskTensor)
        )
      ) {
        return FloatTensor.of((OnnxTensor) result.get(0));
      } finally {
        idsTensor.close();
        maskTensor.close();
      }
    } catch (OrtException e) {
      throw new RuntimeException("Batched encoder inference failed", e);
    } finally {
      batchEncoderBuffers.release(holder);
    }
  }

  /**
   * Runs {@code session} with the named output pinned to a pooled direct buffer: ORT writes
   * the result straight into the lease's buffer. Input tensors are managed by the caller.
   * Pinned outputs are fully overwritten, so the pooled buffer needs no zeroing; the caller
   * must {@link PinnedTensorLease#close()} the lease once the tensor is no longer needed.
   */
  protected PinnedTensorLease runPinned(
    OrtSession session,
    Map<String, ? extends ai.onnxruntime.OnnxTensorLike> inputs,
    String outputName,
    ai.onnxruntime.OnnxJavaType outputType,
    long[] outputShape,
    DirectByteBufferPool pool
  ) throws OrtException {
    long numel = 1;
    for (long d : outputShape) {
      numel *= d;
    }
    int elemBytes = outputType == ai.onnxruntime.OnnxJavaType.FLOAT16 ? 2 : 4;
    var holder = pool.acquire();
    boolean handedOff = false;
    try {
      holder.ensureCapacity(
        (int) (numel * elemBytes),
        (int) ((numel * elemBytes) / 4)
      );
      var outBytes = holder.buf();
      outBytes.clear().limit((int) (numel * elemBytes));

      OnnxTensor outTensor = null;
      try {
        outTensor = outputType == ai.onnxruntime.OnnxJavaType.FLOAT16
          ? OnnxTensor.createTensor(
            env,
            outBytes.asShortBuffer(),
            outputShape,
            ai.onnxruntime.OnnxJavaType.FLOAT16
          )
          : OnnxTensor.createTensor(env, outBytes.asFloatBuffer(), outputShape);

        // Result.close() does not close pinned outputs — the lease owns the tensor.
        try (var result = session.run(inputs, Map.of(outputName, outTensor))) {
          var lease = new PinnedTensorLease(
            outTensor,
            outputShape,
            outputType,
            outBytes,
            pool,
            holder
          );
          handedOff = true;
          return lease;
        }
      } catch (OrtException | RuntimeException e) {
        if (outTensor != null) {
          outTensor.close();
        }
        throw e;
      }
    } finally {
      if (!handedOff) {
        pool.release(holder);
      }
    }
  }

  /**
   * Runs the encoder model in batch mode with the hidden states pinned to a pooled direct
   * buffer (no native→heap materialization). The caller must close the returned lease once
   * the hidden states have been consumed.
   *
   * @param inputIds token IDs per text [batchSize][varying seqLen]
   * @param attentionMask attention masks per text [batchSize][varying seqLen]
   * @param maxSeqLen the padded sequence length
   * @param hiddenSize encoder hidden size
   * @return a lease over the hidden states, shape [batchSize][maxSeqLen][hiddenSize]
   */
  public PinnedTensorLease runEncoderBatchPinned(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    int hiddenSize
  ) {
    var holder = batchEncoderBuffers.acquire();
    try {
      int batchSize = inputIds.length;
      int totalElements = batchSize * maxSeqLen;

      if (totalElements > holder.capacity()) {
        holder.reallocate(totalElements + 256);
      }
      var idsBuf = holder.buf(0);
      var maskBuf = holder.buf(1);
      idsBuf.clear().limit(totalElements);
      maskBuf.clear().limit(totalElements);

      var zeroPad = new long[maxSeqLen];
      for (int b = 0; b < batchSize; b++) {
        int seqLen = inputIds[b].length;
        int padLen = maxSeqLen - seqLen;
        idsBuf.put(inputIds[b]);
        if (padLen > 0) idsBuf.put(zeroPad, 0, padLen);
        maskBuf.put(attentionMask[b]);
        if (padLen > 0) maskBuf.put(zeroPad, 0, padLen);
      }
      idsBuf.rewind();
      maskBuf.rewind();

      var shape = new long[] { batchSize, maxSeqLen };
      try (
        var idsTensor = OnnxTensor.createTensor(env, idsBuf, shape);
        var maskTensor = OnnxTensor.createTensor(env, maskBuf, shape)
      ) {
        return runPinned(
          encoderSession,
          Map.of("input_ids", idsTensor, "attention_mask", maskTensor),
          encoderOutputName,
          encoderOutputType,
          new long[] { batchSize, maxSeqLen, hiddenSize },
          encoderOutBuffers
        );
      }
    } catch (OrtException e) {
      throw new RuntimeException("Pinned batched encoder inference failed", e);
    } finally {
      batchEncoderBuffers.release(holder);
    }
  }

  /** Materializes a rank-3 flat tensor as nested arrays via bulk row copies (no reflection). */
  protected static float[][][] toNested3d(FloatTensor tensor) {
    int d0 = tensor.dim(0);
    int d1 = tensor.dim(1);
    int d2 = tensor.dim(2);
    var out = new float[d0][d1][d2];
    long offset = 0;
    for (int i = 0; i < d0; i++) {
      for (int j = 0; j < d1; j++) {
        tensor.copyTo(offset, out[i][j], 0, d2);
        offset += d2;
      }
    }
    return out;
  }

  @Override
  public void close() {
    try {
      if (encoderSession != null) {
        encoderSession.close();
      }
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
