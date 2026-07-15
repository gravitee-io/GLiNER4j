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
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * NER runtime managing span_rep and scoring_head ONNX sessions
 * on top of the shared encoder from {@link BaseRuntime}.
 */
@Slf4j
public non-sealed class GLiNER4jNERRuntime extends BaseRuntime {

  private OrtSession spanRepSession;
  private OrtSession scoringHeadSession;
  // Merged span_rep + scoring_head graph (span_scoring.onnx): one session run per bucket, the
  // large span-representation intermediate never leaves the GPU. Present only in artifacts
  // composed offline via onnx.compose; null otherwise (split-session path).
  private OrtSession spanScoringSession;
  // Full merged graph (ner_full.onnx): encoder + word/schema gather + span_scoring in ONE
  // session — only int64 ids/positions go up, count_logits/span_scores come down. Present
  // only in artifacts composed offline; null otherwise.
  private OrtSession nerFullSession;
  // True when scoring_head.onnx was exported with a leading batch axis on span_rep
  // (rank-4 input). Older artifacts (rank-3) keep the per-text path.
  private boolean scoringHeadBatched;
  // span_rep output binding: name + element type (FLOAT for fp32 exports, FLOAT16 for fp16),
  // resolved at load time so outputs can be pinned to pooled direct buffers.
  private String spanRepOutputName;
  private ai.onnxruntime.OnnxJavaType spanRepOutputType;

  // Pooled span index buffers — safe for concurrent span_rep calls
  private final DirectBufferPool spanIdxBuffers = new DirectBufferPool(1);
  // Pooled direct buffers backing pinned span_rep output tensors
  private final DirectByteBufferPool spanRepOutBuffers =
    new DirectByteBufferPool();
  // Pooled ids/mask/word_positions buffers for the full merged graph
  private final DirectBufferPool nerFullInputBuffers = new DirectBufferPool(3);
  // Pooled buffer for the flattened schema_emb_fields input — createTensor on a float[][]
  // marshals through reflection (one boxed Float per element on every scoring call).
  private final DirectByteBufferPool schemaFieldsBuffers =
    new DirectByteBufferPool();

  public GLiNER4jNERRuntime(Path modelDir) {
    this(modelDir, DEFAULT_VARIANT, RuntimeConfig.defaults());
  }

  public GLiNER4jNERRuntime(Path modelDir, String variant) {
    this(modelDir, variant, RuntimeConfig.defaults());
  }

  public GLiNER4jNERRuntime(
    Path modelDir,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    super(modelDir, variant, runtimeConfig);
  }

  @Override
  protected void loadTaskHeads(
    Path variantDir,
    RuntimeConfig runtimeConfig,
    Path cacheDir
  ) throws OrtException {
    int numCpus = Runtime.getRuntime().availableProcessors();

    int encoderIntra = getOrDefault(
      runtimeConfig.getEncoderIntraOpThreads(),
      numCpus
    );
    int encoderInter = getOrDefault(
      runtimeConfig.getEncoderInterOpThreads(),
      Math.max(2, numCpus / 2)
    );
    int scoringIntra = getOrDefault(
      runtimeConfig.getScoringIntraOpThreads(),
      Math.max(2, numCpus / 4)
    );
    int scoringInter = getOrDefault(
      runtimeConfig.getScoringInterOpThreads(),
      1
    );

    // The split heads (span_rep + scoring_head) are only needed when the artifact does not ship a
    // merged graph — a self-contained ner_full.onnx deployment omits them. Load them only if
    // present; the per-text/extractor split paths require them and fail clearly if absent.
    if (java.nio.file.Files.exists(variantDir.resolve("span_rep.onnx"))) {
      log.info("Loading span_rep.onnx...");
      try (
        var opts = createSessionOptions(
          encoderIntra,
          encoderInter,
          OrtSession.SessionOptions.ExecutionMode.PARALLEL,
          runtimeConfig,
          cacheDir,
          "span_rep.onnx"
        )
      ) {
        this.spanRepSession = env.createSession(
          variantDir.resolve("span_rep.onnx").toString(),
          opts
        );
      }
      this.spanRepOutputName = spanRepSession
        .getOutputNames()
        .iterator()
        .next();
      this.spanRepOutputType = spanRepSession
            .getOutputInfo()
            .get(spanRepOutputName)
            .getInfo() instanceof
          TensorInfo outInfo
        ? outInfo.type
        : ai.onnxruntime.OnnxJavaType.FLOAT;
    }

    if (java.nio.file.Files.exists(variantDir.resolve("scoring_head.onnx"))) {
      log.info("Loading scoring_head.onnx...");
      try (
        var opts = createSessionOptions(
          scoringIntra,
          scoringInter,
          OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL,
          runtimeConfig,
          cacheDir,
          "scoring_head.onnx"
        )
      ) {
        this.scoringHeadSession = env.createSession(
          variantDir.resolve("scoring_head.onnx").toString(),
          opts
        );
      }
      var spanRepInput = scoringHeadSession.getInputInfo().get("span_rep");
      this.scoringHeadBatched =
        spanRepInput != null &&
        spanRepInput.getInfo() instanceof TensorInfo tensorInfo &&
        tensorInfo.getShape().length == 4;
      if (scoringHeadBatched) {
        log.info("scoring_head.onnx supports batched span_rep");
      }
    }

    var mergedPath = variantDir.resolve("span_scoring.onnx");
    if (java.nio.file.Files.exists(mergedPath)) {
      log.info("Loading span_scoring.onnx (merged span_rep + scoring_head)...");
      try (
        var opts = createSessionOptions(
          encoderIntra,
          encoderInter,
          OrtSession.SessionOptions.ExecutionMode.PARALLEL,
          runtimeConfig,
          cacheDir,
          "span_scoring.onnx"
        )
      ) {
        this.spanScoringSession = env.createSession(
          mergedPath.toString(),
          opts
        );
      }
    }

    var fullPath = variantDir.resolve("ner_full.onnx");
    if (java.nio.file.Files.exists(fullPath)) {
      log.info(
        "Loading ner_full.onnx (full merged graph: encoder + gather + span_scoring)..."
      );
      try (
        var opts = createSessionOptions(
          encoderIntra,
          encoderInter,
          OrtSession.SessionOptions.ExecutionMode.PARALLEL,
          runtimeConfig,
          cacheDir,
          "ner_full.onnx"
        )
      ) {
        this.nerFullSession = env.createSession(fullPath.toString(), opts);
      }
    }
  }

  /**
   * Whether the artifact ships a merged {@code span_scoring.onnx} graph, enabling
   * {@link #runSpanScoringBatchFlat} (span_rep never leaves the GPU).
   */
  public boolean isSpanScoringMerged() {
    return spanScoringSession != null;
  }

  /**
   * Whether the artifact ships the full merged graph ({@code ner_full.onnx}) — one session
   * run per bucket, encoder hidden states never leave the device.
   */
  public boolean isNerFullGraph() {
    return nerFullSession != null;
  }

  /**
   * Whether scoring_head.onnx accepts a leading batch axis on span_rep, enabling
   * {@link #runScoringHeadBatch}.
   */
  public boolean isScoringHeadBatched() {
    return scoringHeadBatched;
  }

  @Override
  protected void closeTaskHeads() throws OrtException {
    if (spanRepSession != null) {
      spanRepSession.close();
    }
    if (scoringHeadSession != null) {
      scoringHeadSession.close();
    }
    if (spanScoringSession != null) {
      spanScoringSession.close();
    }
    if (nerFullSession != null) {
      nerFullSession.close();
    }
  }

  @Override
  protected java.util.List<OrtSession> profilableSessions() {
    var sessions = new java.util.ArrayList<OrtSession>();
    if (encoderSession != null) {
      sessions.add(encoderSession);
    }
    if (spanRepSession != null) {
      sessions.add(spanRepSession);
    }
    if (scoringHeadSession != null) {
      sessions.add(scoringHeadSession);
    }
    if (spanScoringSession != null) {
      sessions.add(spanScoringSession);
    }
    if (nerFullSession != null) {
      sessions.add(nerFullSession);
    }
    return sessions;
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
      try (
        var embTensor = OnnxTensor.createTensor(env, textEmbs);
        var idxTensor = OnnxTensor.createTensor(env, spanIdx);
        var result = spanRepSession.run(
          Map.of("token_embeddings", embTensor, "span_idx", idxTensor)
        )
      ) {
        return (float[][][][]) result.get(0).getValue();
      }
    } catch (OrtException e) {
      throw new RuntimeException("SpanRep inference failed", e);
    }
  }

  /**
   * Runs the span representation model using a flat span index buffer.
   *
   * @param textEmbs text embeddings [1][textLen][hiddenSize]
   * @param spanIdxFlat flat span indices [numSpans*2]
   * @param numSpans number of spans
   * @return span representations [1][textLen][maxWidth][hiddenSize]
   */
  public float[][][][] runSpanRepFlat(
    float[][][] textEmbs,
    long[] spanIdxFlat,
    int numSpans
  ) {
    var holder = spanIdxBuffers.acquire();
    try {
      int totalElements = numSpans * 2;
      if (totalElements > holder.capacity()) {
        holder.reallocate(totalElements + 128);
      }
      var spanIdxBuf = holder.buf(0);
      spanIdxBuf.clear().limit(totalElements);
      spanIdxBuf.put(spanIdxFlat, 0, totalElements).rewind();

      try (
        var embTensor = OnnxTensor.createTensor(env, textEmbs);
        var idxTensor = OnnxTensor.createTensor(
          env,
          spanIdxBuf,
          new long[] { 1, numSpans, 2 }
        );
        var result = spanRepSession.run(
          Map.of("token_embeddings", embTensor, "span_idx", idxTensor)
        )
      ) {
        return (float[][][][]) result.get(0).getValue();
      }
    } catch (OrtException e) {
      throw new RuntimeException("SpanRep inference failed", e);
    } finally {
      spanIdxBuffers.release(holder);
    }
  }

  /**
   * Runs the span representation model in batch mode.
   *
   * @param textEmbs padded text embeddings [batchSize][maxTextLen][hiddenSize]
   * @param spanIdxFlat flat span indices [batchSize*maxNumSpans*2]
   * @param batchSize number of texts in the batch
   * @param maxNumSpans max number of spans per text
   * @return span representations [batchSize][maxTextLen][maxWidth][hiddenSize]
   */
  public float[][][][] runSpanRepBatch(
    float[][][] textEmbs,
    long[] spanIdxFlat,
    int batchSize,
    int maxNumSpans
  ) {
    var holder = spanIdxBuffers.acquire();
    try {
      int totalElements = batchSize * maxNumSpans * 2;
      if (totalElements > holder.capacity()) {
        holder.reallocate(totalElements + 256);
      }
      var spanIdxBuf = holder.buf(0);
      spanIdxBuf.clear().limit(totalElements);
      spanIdxBuf.put(spanIdxFlat, 0, totalElements).rewind();

      try (
        var embTensor = OnnxTensor.createTensor(env, textEmbs);
        var idxTensor = OnnxTensor.createTensor(
          env,
          spanIdxBuf,
          new long[] { batchSize, maxNumSpans, 2 }
        );
        var result = spanRepSession.run(
          Map.of("token_embeddings", embTensor, "span_idx", idxTensor)
        )
      ) {
        return (float[][][][]) result.get(0).getValue();
      }
    } catch (OrtException e) {
      throw new RuntimeException("Batched SpanRep inference failed", e);
    } finally {
      spanIdxBuffers.release(holder);
    }
  }

  /**
   * Runs the span representation model in batch mode with a flat text-embedding buffer,
   * returning the span representations as a flat tensor. Both directions avoid nested-array
   * marshalling: the input buffer is wrapped zero-copy, the output is one bulk copy.
   *
   * @param textEmbs padded text embeddings, row-major [batchSize][maxTextLen][hiddenSize],
   *                 in a direct buffer positioned at 0 with limit == batchSize*maxTextLen*hiddenSize
   * @param batchSize number of texts in the batch
   * @param maxTextLen the padded text length
   * @param hiddenSize encoder hidden size
   * @param spanIdxFlat flat span indices [batchSize*maxNumSpans*2]
   * @param maxNumSpans max number of spans per text
   * @return span representations, shape [batchSize][maxTextLen][maxWidth][hiddenSize]
   */
  public FloatTensor runSpanRepBatchFlat(
    java.nio.FloatBuffer textEmbs,
    int batchSize,
    int maxTextLen,
    int hiddenSize,
    long[] spanIdxFlat,
    int maxNumSpans
  ) {
    var holder = spanIdxBuffers.acquire();
    try {
      int totalElements = batchSize * maxNumSpans * 2;
      if (totalElements > holder.capacity()) {
        holder.reallocate(totalElements + 256);
      }
      var spanIdxBuf = holder.buf(0);
      spanIdxBuf.clear().limit(totalElements);
      spanIdxBuf.put(spanIdxFlat, 0, totalElements).rewind();

      try (
        var embTensor = OnnxTensor.createTensor(
          env,
          textEmbs,
          new long[] { batchSize, maxTextLen, hiddenSize }
        );
        var idxTensor = OnnxTensor.createTensor(
          env,
          spanIdxBuf,
          new long[] { batchSize, maxNumSpans, 2 }
        );
        var result = spanRepSession.run(
          Map.of("token_embeddings", embTensor, "span_idx", idxTensor)
        )
      ) {
        return FloatTensor.of((OnnxTensor) result.get(0));
      }
    } catch (OrtException e) {
      throw new RuntimeException("Batched SpanRep inference failed", e);
    } finally {
      spanIdxBuffers.release(holder);
    }
  }

  /**
   * Runs the span representation model in batch mode with the output <em>pinned</em> to a pooled
   * direct buffer: ORT writes the span representations straight into the leased buffer (no
   * native→heap materialization), and the returned lease's tensor can be fed to the scoring head
   * zero-copy. The caller must {@link PinnedTensorLease#close()} the lease once the tensor is no
   * longer needed.
   *
   * @param textEmbs padded text embeddings, row-major [batchSize][maxTextLen][hiddenSize],
   *                 in a direct buffer positioned at 0 with limit == batchSize*maxTextLen*hiddenSize
   * @param batchSize number of texts in the batch
   * @param maxTextLen the padded text length
   * @param hiddenSize encoder hidden size
   * @param maxWidth max span width
   * @param spanIdxFlat flat span indices [batchSize*maxTextLen*maxWidth*2]
   * @return a lease over the span representations, shape [batchSize][maxTextLen][maxWidth][hiddenSize]
   */
  public PinnedTensorLease runSpanRepBatchPinned(
    java.nio.FloatBuffer textEmbs,
    int batchSize,
    int maxTextLen,
    int hiddenSize,
    int maxWidth,
    long[] spanIdxFlat
  ) {
    int maxNumSpans = maxTextLen * maxWidth;
    var idxHolder = spanIdxBuffers.acquire();
    try {
      int totalIdxElements = batchSize * maxNumSpans * 2;
      if (totalIdxElements > idxHolder.capacity()) {
        idxHolder.reallocate(totalIdxElements + 256);
      }
      var spanIdxBuf = idxHolder.buf(0);
      spanIdxBuf.clear().limit(totalIdxElements);
      spanIdxBuf.put(spanIdxFlat, 0, totalIdxElements).rewind();

      try (
        var embTensor = OnnxTensor.createTensor(
          env,
          textEmbs,
          new long[] { batchSize, maxTextLen, hiddenSize }
        );
        var idxTensor = OnnxTensor.createTensor(
          env,
          spanIdxBuf,
          new long[] { batchSize, maxNumSpans, 2 }
        )
      ) {
        return runPinned(
          spanRepSession,
          Map.of("token_embeddings", embTensor, "span_idx", idxTensor),
          spanRepOutputName,
          spanRepOutputType,
          new long[] { batchSize, maxTextLen, maxWidth, hiddenSize },
          spanRepOutBuffers
        );
      }
    } catch (OrtException e) {
      throw new RuntimeException("Pinned SpanRep inference failed", e);
    } finally {
      spanIdxBuffers.release(idxHolder);
    }
  }

  /**
   * Runs the scoring head over several texts in one call with the span representations passed
   * as a pinned tensor lease — the lease's tensor is used directly as the session input
   * (zero-copy), and the span scores come back pinned to a pooled direct buffer once the
   * output's count dimension has been learned from the first run. The span_rep lease is not
   * closed by this method; the returned result must be closed after decoding.
   */
  public FlatBatchScoringResult runScoringHeadBatchFlat(
    PinnedTensorLease spanRep,
    float[] schemaEmbP,
    float[][] schemaEmbFields,
    long count
  ) {
    if (!scoringHeadBatched) {
      throw new IllegalStateException(
        "scoring_head.onnx does not support batched span_rep (re-export with a batch axis)"
      );
    }
    var fieldsHolder = schemaFieldsBuffers.acquire();
    try {
      int numFields = schemaEmbFields.length;
      int fieldHidden = schemaEmbFields[0].length;
      int fieldsBytes = numFields * fieldHidden * 4;
      fieldsHolder.ensureCapacity(fieldsBytes, fieldsBytes / 4);
      var fieldsBuf = fieldsHolder.buf();
      fieldsBuf.clear().limit(fieldsBytes);
      var fieldsFloats = fieldsBuf.asFloatBuffer();
      for (int i = 0; i < numFields; i++) {
        fieldsFloats.put(i * fieldHidden, schemaEmbFields[i], 0, fieldHidden);
      }

      try (
        var pTensor = OnnxTensor.createTensor(env, schemaEmbP);
        var fieldsTensor = OnnxTensor.createTensor(
          env,
          fieldsFloats,
          new long[] { numFields, fieldHidden }
        );
        var countTensor = OnnxTensor.createTensor(
          env,
          allocateDirectLongBuffer(new long[] { count }),
          new long[] {}
        )
      ) {
        var inputs = Map.of(
          "span_rep",
          (ai.onnxruntime.OnnxTensorLike) spanRep.tensor(),
          "schema_emb_p",
          pTensor,
          "schema_emb_fields",
          fieldsTensor,
          "count",
          countTensor
        );

        return runScoring(scoringHeadSession, inputs);
      }
    } catch (OrtException e) {
      throw new RuntimeException("Batched ScoringHead inference failed", e);
    } finally {
      schemaFieldsBuffers.release(fieldsHolder);
    }
  }

  /**
   * Runs the merged {@code span_scoring.onnx} graph over one bucket: span_rep + scoring_head in
   * a single session run, the span-representation intermediate staying on the device. Mirrors
   * {@link #runScoringHeadBatchFlat} semantics; the returned result must be closed after
   * decoding.
   *
   * @param textEmbs padded text embeddings, row-major [batchSize][maxTextLen][hiddenSize]
   * @param batchSize number of texts in the bucket
   * @param maxTextLen the padded text length
   * @param hiddenSize encoder hidden size
   * @param maxWidth max span width
   * @param spanIdxFlat flat span indices [batchSize*maxTextLen*maxWidth*2]
   * @param schemaEmbP the [P] token embedding [hiddenSize]
   * @param schemaEmbFields field embeddings [numFields][hiddenSize]
   * @param count predicted count (scalar)
   */
  public FlatBatchScoringResult runSpanScoringBatchFlat(
    java.nio.FloatBuffer textEmbs,
    int batchSize,
    int maxTextLen,
    int hiddenSize,
    int maxWidth,
    long[] spanIdxFlat,
    float[] schemaEmbP,
    float[][] schemaEmbFields,
    long count
  ) {
    if (spanScoringSession == null) {
      throw new IllegalStateException(
        "span_scoring.onnx is not part of this artifact"
      );
    }
    int maxNumSpans = maxTextLen * maxWidth;
    var idxHolder = spanIdxBuffers.acquire();
    var fieldsHolder = schemaFieldsBuffers.acquire();
    try {
      int totalIdxElements = batchSize * maxNumSpans * 2;
      if (totalIdxElements > idxHolder.capacity()) {
        idxHolder.reallocate(totalIdxElements + 256);
      }
      var spanIdxBuf = idxHolder.buf(0);
      spanIdxBuf.clear().limit(totalIdxElements);
      spanIdxBuf.put(spanIdxFlat, 0, totalIdxElements).rewind();

      int numFields = schemaEmbFields.length;
      int fieldHidden = schemaEmbFields[0].length;
      int fieldsBytes = numFields * fieldHidden * 4;
      fieldsHolder.ensureCapacity(fieldsBytes, fieldsBytes / 4);
      var fieldsBuf = fieldsHolder.buf();
      fieldsBuf.clear().limit(fieldsBytes);
      var fieldsFloats = fieldsBuf.asFloatBuffer();
      for (int i = 0; i < numFields; i++) {
        fieldsFloats.put(i * fieldHidden, schemaEmbFields[i], 0, fieldHidden);
      }

      try (
        var embTensor = OnnxTensor.createTensor(
          env,
          textEmbs,
          new long[] { batchSize, maxTextLen, hiddenSize }
        );
        var idxTensor = OnnxTensor.createTensor(
          env,
          spanIdxBuf,
          new long[] { batchSize, maxNumSpans, 2 }
        );
        var pTensor = OnnxTensor.createTensor(env, schemaEmbP);
        var fieldsTensor = OnnxTensor.createTensor(
          env,
          fieldsFloats,
          new long[] { numFields, fieldHidden }
        );
        var countTensor = OnnxTensor.createTensor(
          env,
          allocateDirectLongBuffer(new long[] { count }),
          new long[] {}
        )
      ) {
        var inputs = Map.of(
          "token_embeddings",
          (ai.onnxruntime.OnnxTensorLike) embTensor,
          "span_idx",
          idxTensor,
          "schema_emb_p",
          pTensor,
          "schema_emb_fields",
          fieldsTensor,
          "count",
          countTensor
        );

        return runScoring(spanScoringSession, inputs);
      }
    } catch (OrtException e) {
      throw new RuntimeException("Merged SpanScoring inference failed", e);
    } finally {
      spanIdxBuffers.release(idxHolder);
      schemaFieldsBuffers.release(fieldsHolder);
    }
  }

  /**
   * Runs the full merged graph ({@code ner_full.onnx}) over one bucket: encoder,
   * word/schema gathers and span scoring in a single session run. Only int64 inputs cross
   * to the device; the hidden states and span representations never leave it. The returned
   * result must be closed after decoding.
   *
   * @param inputIds packed per-row token IDs (rows may be shorter than {@code maxSeqLen})
   * @param attentionMask packed per-row attention masks
   * @param maxSeqLen padded sequence length
   * @param wordPositionsFlat row-major [batchSize][maxTextLen] first-subword positions, -1 pad
   * @param batchSize number of texts in the bucket
   * @param maxTextLen padded word count
   * @param pPosition position of the [P] token (may be -1 — zeroed in-graph)
   * @param fieldPositions positions of the per-field marker tokens
   * @param spanIdxFlat flat span indices [batchSize*maxTextLen*maxWidth*2]
   * @param maxWidth max span width
   * @param count predicted count (scalar)
   */
  public FlatBatchScoringResult runNerFullBatch(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] wordPositionsFlat,
    int batchSize,
    int maxTextLen,
    long pPosition,
    long[] fieldPositions,
    long[] spanIdxFlat,
    int maxWidth,
    long count
  ) {
    if (nerFullSession == null) {
      throw new IllegalStateException(
        "ner_full.onnx is not part of this artifact"
      );
    }
    int numFields = fieldPositions.length;
    int maxNumSpans = maxTextLen * maxWidth;
    var inHolder = nerFullInputBuffers.acquire();
    var idxHolder = spanIdxBuffers.acquire();
    try {
      int tokElements = batchSize * maxSeqLen;
      int wpElements = batchSize * maxTextLen;
      int needed = Math.max(tokElements, wpElements);
      if (needed > inHolder.capacity()) {
        inHolder.reallocate(needed + 256);
      }
      var idsBuf = inHolder.buf(0);
      var maskBuf = inHolder.buf(1);
      var wpBuf = inHolder.buf(2);
      idsBuf.clear().limit(tokElements);
      maskBuf.clear().limit(tokElements);
      wpBuf.clear().limit(wpElements);
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
      wpBuf.put(wordPositionsFlat, 0, wpElements).rewind();

      int totalIdxElements = batchSize * maxNumSpans * 2;
      if (totalIdxElements > idxHolder.capacity()) {
        idxHolder.reallocate(totalIdxElements + 256);
      }
      var spanIdxBuf = idxHolder.buf(0);
      spanIdxBuf.clear().limit(totalIdxElements);
      spanIdxBuf.put(spanIdxFlat, 0, totalIdxElements).rewind();

      var tokShape = new long[] { batchSize, maxSeqLen };
      try (
        var idsTensor = OnnxTensor.createTensor(env, idsBuf, tokShape);
        var maskTensor = OnnxTensor.createTensor(env, maskBuf, tokShape);
        var wpTensor = OnnxTensor.createTensor(
          env,
          wpBuf,
          new long[] { batchSize, maxTextLen }
        );
        var pTensor = OnnxTensor.createTensor(
          env,
          allocateDirectLongBuffer(new long[] { pPosition }),
          new long[] { 1 }
        );
        var fieldsTensor = OnnxTensor.createTensor(
          env,
          allocateDirectLongBuffer(fieldPositions),
          new long[] { numFields }
        );
        var idxTensor = OnnxTensor.createTensor(
          env,
          spanIdxBuf,
          new long[] { batchSize, maxNumSpans, 2 }
        );
        var countTensor = OnnxTensor.createTensor(
          env,
          allocateDirectLongBuffer(new long[] { count }),
          new long[] {}
        )
      ) {
        var inputs = Map.of(
          "input_ids",
          (ai.onnxruntime.OnnxTensorLike) idsTensor,
          "attention_mask",
          maskTensor,
          "word_positions",
          wpTensor,
          "p_position",
          pTensor,
          "field_positions",
          fieldsTensor,
          "span_idx",
          idxTensor,
          "count",
          countTensor
        );
        return runScoring(nerFullSession, inputs);
      }
    } catch (OrtException e) {
      throw new RuntimeException("Full-graph NER inference failed", e);
    } finally {
      nerFullInputBuffers.release(inHolder);
      spanIdxBuffers.release(idxHolder);
    }
  }

  /**
   * Shared tail of the batched scoring paths: runs {@code session} and returns count logits
   * plus span scores as heap-copied flat tensors.
   *
   * <p>Outputs are NOT pinned to pre-sized buffers. Pinning a pre-declared output shape is
   * unsafe here: {@code span_scores}' count dimension is symbolic (sliced to the {@code count}
   * input at run time), and binding a pre-sized output for it races under concurrent runs on
   * the shared session (observed as sporadic {@code Tensor size mismatch} failures). Since NER
   * requests a single count-instance ({@code SCORING_COUNT_INSTANCES}), {@code span_scores} is
   * small, so a plain run plus one bulk {@link FloatTensor#of} copy is cheap and race-free.
   */
  private FlatBatchScoringResult runScoring(
    OrtSession session,
    Map<String, ai.onnxruntime.OnnxTensorLike> inputs
  ) throws OrtException {
    try (var result = session.run(inputs)) {
      var countLogits = result.get("count_logits");
      var spanScores = result.get("span_scores");
      if (countLogits.isPresent() && spanScores.isPresent()) {
        return new FlatBatchScoringResult(
          toNested2d(FloatTensor.of((OnnxTensor) countLogits.get())),
          FloatTensor.of((OnnxTensor) spanScores.get()),
          () -> {}
        );
      }
      throw new OrtException(
        "ScoringHead inference failed: missing output tensors"
      );
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
    if (scoringHeadBatched) {
      // Batched graph: run with batch=1 and unwrap.
      var batch = runScoringHeadBatch(
        new float[][][][] { spanRep },
        schemaEmbP,
        schemaEmbFields,
        count
      );
      return new ScoringResult(batch.countLogits(), batch.spanScores()[0]);
    }
    try {
      try (
        var spanRepTensor = OnnxTensor.createTensor(env, spanRep);
        var pTensor = OnnxTensor.createTensor(env, schemaEmbP);
        var fieldsTensor = OnnxTensor.createTensor(env, schemaEmbFields);
        var countTensor = OnnxTensor.createTensor(
          env,
          allocateDirectLongBuffer(new long[] { count }),
          new long[] {}
        );
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
        var countLogits = result.get("count_logits");
        var spanScores = result.get("span_scores");
        if (countLogits.isPresent() && spanScores.isPresent()) {
          return new ScoringResult(
            (float[][]) countLogits.get().getValue(),
            (float[][][][]) spanScores.get().getValue()
          );
        }
        throw new OrtException(
          "ScoringHead inference failed: missing output tensors"
        );
      }
    } catch (OrtException e) {
      throw new RuntimeException("ScoringHead inference failed", e);
    }
  }

  /** Materializes a small rank-2 flat tensor as nested arrays via bulk row copies. */
  private static float[][] toNested2d(FloatTensor tensor) {
    int d0 = tensor.dim(0);
    int d1 = tensor.dim(1);
    var out = new float[d0][d1];
    for (int i = 0; i < d0; i++) {
      tensor.copyTo((long) i * d1, out[i], 0, d1);
    }
    return out;
  }

  public BatchScoringResult runScoringHeadBatch(
    float[][][][] spanRep,
    float[] schemaEmbP,
    float[][] schemaEmbFields,
    long count
  ) {
    if (!scoringHeadBatched) {
      throw new IllegalStateException(
        "scoring_head.onnx does not support batched span_rep (re-export with a batch axis)"
      );
    }
    try {
      try (
        var spanRepTensor = OnnxTensor.createTensor(env, spanRep);
        var pTensor = OnnxTensor.createTensor(env, schemaEmbP);
        var fieldsTensor = OnnxTensor.createTensor(env, schemaEmbFields);
        var countTensor = OnnxTensor.createTensor(
          env,
          allocateDirectLongBuffer(new long[] { count }),
          new long[] {}
        );
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
        var countLogits = result.get("count_logits");
        var spanScores = result.get("span_scores");
        if (countLogits.isPresent() && spanScores.isPresent()) {
          return new BatchScoringResult(
            (float[][]) countLogits.get().getValue(),
            (float[][][][][]) spanScores.get().getValue()
          );
        }
        throw new OrtException(
          "ScoringHead inference failed: missing output tensors"
        );
      }
    } catch (OrtException e) {
      throw new RuntimeException("Batched ScoringHead inference failed", e);
    }
  }
}
