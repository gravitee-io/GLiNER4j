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
import java.nio.file.Path;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * NER runtime for the merged GLiNER2 graph ({@code ner_full.onnx}): encoder, word/schema
 * gathers and span scoring in a single session run. The split graphs
 * ({@code encoder.onnx} / {@code span_rep.onnx} / {@code scoring_head.onnx}) are no longer
 * loaded — every NER, relation and structure path scores through the merged graph.
 */
@Slf4j
public non-sealed class GLiNER4jNERRuntime extends BaseRuntime {

  private OrtSession nerFullSession;

  // Pooled span index buffers — safe for concurrent runNerFullBatch calls
  private final DirectBufferPool spanIdxBuffers = new DirectBufferPool(1);
  // Pooled ids/mask/word_positions buffers for the full merged graph
  private final DirectBufferPool nerFullInputBuffers = new DirectBufferPool(3);

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
    var fullPath = variantDir.resolve("ner_full.onnx");
    if (!java.nio.file.Files.exists(fullPath)) {
      throw new IllegalStateException(
        "ner_full.onnx not found in " +
          variantDir +
          " — this runtime requires the merged graph. Re-export the bundle with " +
          "`uv run scripts/export_onnx.py gliner2 ...` (or the matching task in Taskfile.yml)."
      );
    }

    int numCpus = Runtime.getRuntime().availableProcessors();
    int encoderIntra = getOrDefault(
      runtimeConfig.getEncoderIntraOpThreads(),
      numCpus
    );
    int encoderInter = getOrDefault(
      runtimeConfig.getEncoderInterOpThreads(),
      Math.max(2, numCpus / 2)
    );

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

  @Override
  protected void closeTaskHeads() throws OrtException {
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
    if (nerFullSession != null) {
      sessions.add(nerFullSession);
    }
    return sessions;
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
   * Shared tail of the merged scoring path: runs {@code session} and returns count logits
   * plus span scores as heap-copied flat tensors.
   *
   * <p>Outputs are NOT pinned to pre-sized buffers. Pinning a pre-declared output shape is
   * unsafe here: {@code span_scores}' count dimension is symbolic (sliced to the {@code count}
   * input at run time), and binding a pre-sized output for it races under concurrent runs on
   * the shared session (observed as sporadic {@code Tensor size mismatch} failures). A plain
   * run plus one bulk {@link FloatTensor#of} copy is cheap and race-free.
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
}
