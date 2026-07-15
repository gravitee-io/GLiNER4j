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
 * Classification runtime managing the classifier_head ONNX session
 * on top of the shared encoder from {@link BaseRuntime}.
 */
@Slf4j
public non-sealed class GLiNER4jClassifierRuntime extends BaseRuntime {

  private OrtSession classifierHeadSession;
  // Full merged graph (classifier_full.onnx): encoder + in-graph label gather + classifier_head
  // in one session. Present only in artifacts composed offline; null otherwise.
  private OrtSession classifierFullSession;
  private final DirectBufferPool classifierInputBuffers = new DirectBufferPool(
    2
  );

  public GLiNER4jClassifierRuntime(
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

    int scoringIntra = getOrDefault(
      runtimeConfig.getScoringIntraOpThreads(),
      Math.max(2, numCpus / 4)
    );
    int scoringInter = getOrDefault(
      runtimeConfig.getScoringInterOpThreads(),
      1
    );

    // Split classifier_head is only needed by the non-merged (per-text) path; a self-contained
    // classifier_full.onnx deployment omits it.
    if (
      java.nio.file.Files.exists(variantDir.resolve("classifier_head.onnx"))
    ) {
      log.info("Loading classifier_head.onnx...");
      try (
        var opts = createSessionOptions(
          scoringIntra,
          scoringInter,
          OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL,
          runtimeConfig,
          cacheDir,
          "classifier_head.onnx"
        )
      ) {
        this.classifierHeadSession = env.createSession(
          variantDir.resolve("classifier_head.onnx").toString(),
          opts
        );
      }
    }

    var fullPath = variantDir.resolve("classifier_full.onnx");
    if (java.nio.file.Files.exists(fullPath)) {
      log.info(
        "Loading classifier_full.onnx (merged graph: encoder + label gather + classifier_head)..."
      );
      int encoderIntra = getOrDefault(
        runtimeConfig.getEncoderIntraOpThreads(),
        numCpus
      );
      int encoderInter = getOrDefault(
        runtimeConfig.getEncoderInterOpThreads(),
        Math.max(2, numCpus / 2)
      );
      try (
        var opts = createSessionOptions(
          encoderIntra,
          encoderInter,
          OrtSession.SessionOptions.ExecutionMode.PARALLEL,
          runtimeConfig,
          cacheDir,
          "classifier_full.onnx"
        )
      ) {
        this.classifierFullSession = env.createSession(
          fullPath.toString(),
          opts
        );
      }
    }
  }

  /**
   * Whether the artifact ships the merged classifier graph ({@code classifier_full.onnx}) — one
   * session run per batch, no standalone encoder/head round trip.
   */
  public boolean isClassifierFullGraph() {
    return classifierFullSession != null;
  }

  @Override
  protected void closeTaskHeads() throws OrtException {
    if (classifierHeadSession != null) {
      classifierHeadSession.close();
    }
    if (classifierFullSession != null) {
      classifierFullSession.close();
    }
  }

  @Override
  protected java.util.List<OrtSession> profilableSessions() {
    var sessions = new java.util.ArrayList<OrtSession>();
    if (encoderSession != null) {
      sessions.add(encoderSession);
    }
    if (classifierHeadSession != null) {
      sessions.add(classifierHeadSession);
    }
    if (classifierFullSession != null) {
      sessions.add(classifierFullSession);
    }
    return sessions;
  }

  /**
   * Runs the classifier head MLP on label embeddings (split path).
   *
   * @param labelEmbeddings label embeddings [numLabels][hiddenSize]
   * @return logits [numLabels][1] (raw scores before activation)
   */
  public float[][] runClassifierHead(float[][] labelEmbeddings) {
    if (classifierHeadSession == null) {
      throw new IllegalStateException(
        "classifier_head.onnx is not part of this artifact (merged classifier_full only)"
      );
    }
    try {
      try (
        var embTensor = OnnxTensor.createTensor(env, labelEmbeddings);
        var result = classifierHeadSession.run(
          Map.of("label_embeddings", embTensor)
        )
      ) {
        return (float[][]) result.get(0).getValue();
      }
    } catch (OrtException e) {
      throw new RuntimeException("Classifier head inference failed", e);
    }
  }

  /**
   * Runs the full merged classifier graph over a padded batch: encoder, in-graph label gather and
   * classifier head in one session. Label positions are identical across the batch (the schema
   * prefix is shared), so a single {@code label_positions} vector is gathered from every row.
   *
   * @param inputIds packed per-row token IDs
   * @param attentionMask packed per-row attention masks
   * @param maxSeqLen padded sequence length
   * @param labelPositions the [L] label-marker positions (shared across the batch)
   * @param batchSize number of texts
   * @return per-text logits {@code [batchSize][numLabels]} (raw scores before sigmoid)
   */
  public float[][] runClassifierFullBatch(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] labelPositions,
    int batchSize
  ) {
    if (classifierFullSession == null) {
      throw new IllegalStateException(
        "classifier_full.onnx is not part of this artifact"
      );
    }
    int numLabels = labelPositions.length;
    var holder = classifierInputBuffers.acquire();
    try {
      int tokElements = batchSize * maxSeqLen;
      if (tokElements > holder.capacity()) {
        holder.reallocate(tokElements + 256);
      }
      var idsBuf = holder.buf(0);
      var maskBuf = holder.buf(1);
      idsBuf.clear().limit(tokElements);
      maskBuf.clear().limit(tokElements);
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

      var tokShape = new long[] { batchSize, maxSeqLen };
      try (
        var idsTensor = OnnxTensor.createTensor(env, idsBuf, tokShape);
        var maskTensor = OnnxTensor.createTensor(env, maskBuf, tokShape);
        var posTensor = OnnxTensor.createTensor(
          env,
          allocateDirectLongBuffer(labelPositions),
          new long[] { numLabels }
        );
        var result = classifierFullSession.run(
          Map.of(
            "input_ids",
            idsTensor,
            "attention_mask",
            maskTensor,
            "label_positions",
            posTensor
          )
        )
      ) {
        // logits come back flat as [batchSize*numLabels][1]; reshape to [batchSize][numLabels]
        var flat = FloatTensor.of((OnnxTensor) result.get(0));
        var logits = new float[batchSize][numLabels];
        for (int b = 0; b < batchSize; b++) {
          for (int l = 0; l < numLabels; l++) {
            logits[b][l] = flat.get((long) (b * numLabels + l));
          }
        }
        return logits;
      }
    } catch (OrtException e) {
      throw new RuntimeException("Merged classifier inference failed", e);
    } finally {
      classifierInputBuffers.release(holder);
    }
  }
}
