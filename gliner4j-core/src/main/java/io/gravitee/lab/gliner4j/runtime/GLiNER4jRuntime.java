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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Manages the three ONNX Runtime sessions for GLiNER2 inference:
 * encoder, span_rep, and scoring_head.
 */
@Slf4j
public class GLiNER4jRuntime implements AutoCloseable {

  private final OrtEnvironment env;
  private final OrtSession encoderSession;
  private final OrtSession spanRepSession;
  private final OrtSession scoringHeadSession;

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

  // Pre-allocated span index buffer (single-thread assumption)
  private LongBuffer spanIdxBuf;
  private int spanIdxBufCapacity;

  /** Default ONNX variant folder name (base FP32). */
  public static final String DEFAULT_VARIANT = "onnx";

  /**
   * Creates ONNX runtime sessions using the default "onnx" variant and auto-detected resources.
   *
   * @param modelDir root model directory containing variant subfolders
   */
  public GLiNER4jRuntime(Path modelDir) {
    this(modelDir, DEFAULT_VARIANT, RuntimeConfig.defaults());
  }

  /**
   * Creates ONNX runtime sessions for a specific model variant with auto-detected resources.
   *
   * @param modelDir root model directory containing variant subfolders and shared config
   * @param variant  variant folder name (e.g. "onnx", "onnx_fp16", "onnx_quantized")
   */
  public GLiNER4jRuntime(Path modelDir, String variant) {
    this(modelDir, variant, RuntimeConfig.defaults());
  }

  /**
   * Creates ONNX runtime sessions for a specific model variant with explicit resource control.
   *
   * <p>ONNX files are loaded from {@code modelDir/variant/} and runtime-optimized
   * graphs are cached to {@code modelDir/variant_optimized/} (unless disabled via config).
   *
   * @param modelDir      root model directory containing variant subfolders and shared config
   * @param variant       variant folder name (e.g. "onnx", "onnx_fp16", "onnx_quantized")
   * @param runtimeConfig resource control configuration for ORT sessions
   */
  public GLiNER4jRuntime(
    Path modelDir,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    try {
      this.env = OrtEnvironment.getEnvironment();
      int numCpus = Runtime.getRuntime().availableProcessors();

      // Resolve thread counts: explicit config wins, else auto-calculate
      int encoderIntra = runtimeConfig.getEncoderIntraOpThreads() != null
        ? runtimeConfig.getEncoderIntraOpThreads()
        : numCpus;
      int encoderInter = runtimeConfig.getEncoderInterOpThreads() != null
        ? runtimeConfig.getEncoderInterOpThreads()
        : Math.max(2, numCpus / 2);
      int scoringIntra = runtimeConfig.getScoringIntraOpThreads() != null
        ? runtimeConfig.getScoringIntraOpThreads()
        : Math.max(2, numCpus / 4);
      int scoringInter = runtimeConfig.getScoringInterOpThreads() != null
        ? runtimeConfig.getScoringInterOpThreads()
        : 1;

      var variantDir = modelDir.resolve(variant);
      Path cacheDir = null;
      if (runtimeConfig.isOptimizedModelCacheEnabled()) {
        cacheDir = modelDir.resolve(variant + "_optimized");
        cacheDir.toFile().mkdirs();
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
          runtimeConfig.getOptimizationLevel(),
          cacheDir,
          "encoder.onnx"
        )
      ) {
        this.encoderSession =
          env.createSession(
            variantDir.resolve("encoder.onnx").toString(),
            opts
          );
      }

      log.info("Loading span_rep.onnx...");
      try (
        var opts = createSessionOptions(
          encoderIntra,
          encoderInter,
          OrtSession.SessionOptions.ExecutionMode.PARALLEL,
          runtimeConfig.getOptimizationLevel(),
          cacheDir,
          "span_rep.onnx"
        )
      ) {
        this.spanRepSession =
          env.createSession(
            variantDir.resolve("span_rep.onnx").toString(),
            opts
          );
      }

      // Scoring head uses SEQUENTIAL mode with fewer threads per call —
      // concurrency comes from virtual threads during batch processing
      log.info("Loading scoring_head.onnx...");
      try (
        var opts = createSessionOptions(
          scoringIntra,
          scoringInter,
          OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL,
          runtimeConfig.getOptimizationLevel(),
          cacheDir,
          "scoring_head.onnx"
        )
      ) {
        this.scoringHeadSession =
          env.createSession(
            variantDir.resolve("scoring_head.onnx").toString(),
            opts
          );
      }

      log.info("All ONNX sessions loaded successfully");
    } catch (OrtException e) {
      throw new RuntimeException(
        "Failed to load ONNX models from " + modelDir + "/" + variant,
        e
      );
    }
  }

  private static OrtSession.SessionOptions createSessionOptions(
    int intraOpThreads,
    int interOpThreads,
    OrtSession.SessionOptions.ExecutionMode executionMode,
    OrtSession.SessionOptions.OptLevel optLevel,
    Path cacheDir,
    String modelFileName
  ) throws OrtException {
    var opts = new OrtSession.SessionOptions();
    opts.setIntraOpNumThreads(intraOpThreads);
    opts.setInterOpNumThreads(interOpThreads);
    opts.setExecutionMode(executionMode);
    opts.setOptimizationLevel(optLevel);
    if (cacheDir != null) {
      opts.setOptimizedModelFilePath(
        cacheDir.resolve(modelFileName).toString()
      );
    }
    return opts;
  }

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
   * Runs the encoder model.
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
        // Reuse pre-allocated direct buffers, growing on demand
        if (seqLen > encoderBufCapacity) {
          encoderBufCapacity = seqLen + 64; // slight over-allocate
          encoderIdsBuf = allocateDirectLongBuffer(encoderBufCapacity);
          encoderMaskBuf = allocateDirectLongBuffer(encoderBufCapacity);
          // Pre-fill schema prefix into ids buffer
          encoderIdsBuf.put(schemaPrefixIds).rewind();
          // Pre-fill mask with 1s for entire capacity (bulk put)
          var ones = new long[encoderBufCapacity];
          Arrays.fill(ones, 1L);
          encoderMaskBuf.put(ones).rewind();
        }
        // Write text suffix after the cached schema prefix
        encoderIdsBuf.limit(encoderBufCapacity).position(schemaPrefixLen);
        encoderIdsBuf.put(inputIds, schemaPrefixLen, seqLen - schemaPrefixLen);
        encoderIdsBuf.rewind().limit(seqLen);

        // Mask is pre-filled with 1s, just set the limit
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
   * Runs the span representation model.
   *
   * @param textEmbs text embeddings [1][textLen][hiddenSize]
   * @param spanIdx span index pairs [1][numSpans][2]
   * @return span representations [1][textLen][maxWidth][hiddenSize]
   */
  public float[][][][] runSpanRep(float[][][] textEmbs, long[][][] spanIdx) {
    try {
      var embTensor = OnnxTensor.createTensor(env, textEmbs);
      var idxTensor = OnnxTensor.createTensor(env, spanIdx);

      try (
        var result = spanRepSession.run(
          Map.of("token_embeddings", embTensor, "span_idx", idxTensor)
        )
      ) {
        return (float[][][][]) result.get(0).getValue();
      } finally {
        embTensor.close();
        idxTensor.close();
      }
    } catch (OrtException e) {
      throw new RuntimeException("SpanRep inference failed", e);
    }
  }

  /**
   * Runs the span representation model using a flat span index buffer.
   *
   * @param textEmbs text embeddings [1][textLen][hiddenSize]
   * @param spanIdxFlat flat span indices [numSpans*2] laid out as [start0,end0,start1,end1,...]
   * @param numSpans number of spans
   * @return span representations [1][textLen][maxWidth][hiddenSize]
   */
  public float[][][][] runSpanRepFlat(
    float[][][] textEmbs,
    long[] spanIdxFlat,
    int numSpans
  ) {
    try {
      int totalElements = numSpans * 2;
      if (totalElements > spanIdxBufCapacity) {
        spanIdxBufCapacity = totalElements + 128;
        spanIdxBuf = allocateDirectLongBuffer(spanIdxBufCapacity);
      }
      spanIdxBuf.clear().limit(totalElements);
      spanIdxBuf.put(spanIdxFlat, 0, totalElements).rewind();

      var embTensor = OnnxTensor.createTensor(env, textEmbs);
      var idxTensor = OnnxTensor.createTensor(
        env,
        spanIdxBuf,
        new long[] { 1, numSpans, 2 }
      );

      try (
        var result = spanRepSession.run(
          Map.of("token_embeddings", embTensor, "span_idx", idxTensor)
        )
      ) {
        return (float[][][][]) result.get(0).getValue();
      } finally {
        embTensor.close();
        idxTensor.close();
      }
    } catch (OrtException e) {
      throw new RuntimeException("SpanRep inference failed", e);
    }
  }

  /**
   * Runs the span representation model in batch mode.
   *
   * @param textEmbs padded text embeddings [batchSize][maxTextLen][hiddenSize]
   * @param spanIdxFlat flat span indices [batchSize*maxNumSpans*2]
   * @param batchSize number of texts in the batch
   * @param maxNumSpans max number of spans per text (maxTextLen * maxWidth)
   * @return span representations [batchSize][maxTextLen][maxWidth][hiddenSize]
   */
  public float[][][][] runSpanRepBatch(
    float[][][] textEmbs,
    long[] spanIdxFlat,
    int batchSize,
    int maxNumSpans
  ) {
    try {
      int totalElements = batchSize * maxNumSpans * 2;
      if (totalElements > spanIdxBufCapacity) {
        spanIdxBufCapacity = totalElements + 256;
        spanIdxBuf = allocateDirectLongBuffer(spanIdxBufCapacity);
      }
      spanIdxBuf.clear().limit(totalElements);
      spanIdxBuf.put(spanIdxFlat, 0, totalElements).rewind();

      var embTensor = OnnxTensor.createTensor(env, textEmbs);
      var idxTensor = OnnxTensor.createTensor(
        env,
        spanIdxBuf,
        new long[] { batchSize, maxNumSpans, 2 }
      );

      try (
        var result = spanRepSession.run(
          Map.of("token_embeddings", embTensor, "span_idx", idxTensor)
        )
      ) {
        return (float[][][][]) result.get(0).getValue();
      } finally {
        embTensor.close();
        idxTensor.close();
      }
    } catch (OrtException e) {
      throw new RuntimeException("Batched SpanRep inference failed", e);
    }
  }

  /**
   * Runs the scoring head model.
   *
   * @param spanRep span representations [textLen][maxWidth][hiddenSize]
   * @param schemaEmbP the [P] token embedding [hiddenSize]
   * @param schemaEmbFields field embeddings [numFields][hiddenSize]
   * @param count predicted count (scalar)
   * @return scoring result with count logits and span scores
   */
  public ScoringResult runScoringHead(
    float[][][] spanRep,
    float[] schemaEmbP,
    float[][] schemaEmbFields,
    long count
  ) {
    try {
      var spanRepTensor = OnnxTensor.createTensor(env, spanRep);
      var pTensor = OnnxTensor.createTensor(env, schemaEmbP);
      var fieldsTensor = OnnxTensor.createTensor(env, schemaEmbFields);
      var countTensor = OnnxTensor.createTensor(
        env,
        allocateDirectLongBuffer(new long[] { count }),
        new long[] {}
      );

      try (
        var result = scoringHeadSession.run(
          Map.of(
            "span_rep",
            spanRepTensor,
            "schema_emb_p",
            pTensor,
            "schema_emb_fields",
            fieldsTensor,
            "count",
            countTensor
          )
        )
      ) {
        var countLogits = (float[][]) result
          .get("count_logits")
          .get()
          .getValue();
        var spanScores = (float[][][][]) result
          .get("span_scores")
          .get()
          .getValue();
        return new ScoringResult(countLogits, spanScores);
      } finally {
        spanRepTensor.close();
        pTensor.close();
        fieldsTensor.close();
        countTensor.close();
      }
    } catch (OrtException e) {
      throw new RuntimeException("ScoringHead inference failed", e);
    }
  }

  /**
   * Runs the encoder model in batch mode — all inputs are padded to maxSeqLen and
   * processed in a single ONNX call with shape [batchSize, maxSeqLen].
   *
   * @param inputIds token IDs per text [batchSize][varying seqLen]
   * @param attentionMask attention masks per text [batchSize][varying seqLen]
   * @param maxSeqLen the padded sequence length (longest in the batch)
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

      // Reuse pre-allocated direct buffers, growing on demand
      if (totalElements > batchEncoderBufCapacity) {
        batchEncoderBufCapacity = totalElements + 256;
        batchEncoderIdsBuf = allocateDirectLongBuffer(batchEncoderBufCapacity);
        batchEncoderMaskBuf = allocateDirectLongBuffer(batchEncoderBufCapacity);
      }
      batchEncoderIdsBuf.clear().limit(totalElements);
      batchEncoderMaskBuf.clear().limit(totalElements);

      var zeroPad = new long[maxSeqLen]; // default-initialized to 0
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

  /**
   * Runs the encoder model without buffer caching — used for per-call entity overrides
   * where the schema prefix differs from the one cached at load time.
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

  private static LongBuffer allocateDirectLongBuffer(long[] data) {
    return ByteBuffer
      .allocateDirect(data.length * Long.BYTES)
      .order(ByteOrder.nativeOrder())
      .asLongBuffer()
      .put(data)
      .rewind();
  }

  private static LongBuffer allocateDirectLongBuffer(int capacity) {
    return ByteBuffer
      .allocateDirect(capacity * Long.BYTES)
      .order(ByteOrder.nativeOrder())
      .asLongBuffer();
  }

  @Override
  public void close() {
    try {
      encoderSession.close();
      spanRepSession.close();
      scoringHeadSession.close();
      log.info("All ONNX sessions closed");
    } catch (OrtException e) {
      log.warn("Error closing ONNX sessions", e);
    }
  }
}
