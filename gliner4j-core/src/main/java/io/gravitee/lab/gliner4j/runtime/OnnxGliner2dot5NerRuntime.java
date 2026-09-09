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
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * NER runtime for the GLiNER2.5 merged graph ({@code ner_full.onnx}): encoder, word/query
 * gathers, boundary encoding, start/end/inside marginals, the shared candidate pool and the
 * pooled pair scorer in a single session run. Only int64 inputs cross to the device; the
 * outputs are the {@code [batch, queries, pool]} pair logits, the {@code [batch, pool, 2]}
 * half-open word-boundary candidates and the {@code [batch, queries]} abstention logits.
 */
@Slf4j
public final class OnnxGliner2dot5NerRuntime implements Gliner2dot5NerRuntime {

  private final OrtEnvironment env;
  private final OrtSession session;

  public OnnxGliner2dot5NerRuntime(
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
      var fullPath = variantDir.resolve("ner_full.onnx");
      if (!Files.exists(fullPath)) {
        throw new IllegalStateException(
          "ner_full.onnx not found in " +
            variantDir +
            " — export the bundle with `uv run scripts/export_onnx.py gliner2dot5 ...` " +
            "(or the matching task in Taskfile.yml)."
        );
      }
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
            "Could not create optimized-model cache dir {} — caching disabled for this load",
            cacheDir,
            e
          );
          cacheDir = null;
        }
      }

      log.info(
        "Loading ner_full.onnx from {} (GLiNER2.5 merged graph: encoder + boundary head + shared pool)...",
        variantDir
      );
      try (
        var opts = OrtSessions.createSessionOptions(
          intra,
          inter,
          OrtSession.SessionOptions.ExecutionMode.PARALLEL,
          runtimeConfig,
          cacheDir,
          "ner_full.onnx"
        )
      ) {
        this.session = env.createSession(fullPath.toString(), opts);
      }
      log.info("GLiNER2.5 ner_full.onnx loaded successfully");
    } catch (OrtException e) {
      throw new RuntimeException(
        "Failed to load GLiNER2.5 ner_full.onnx from " +
          modelDir +
          "/" +
          variant,
        e
      );
    }
  }

  /**
   * Runs the merged graph over one padded sub-batch.
   *
   * @param inputIds per-row token IDs (rows may be shorter than {@code maxSeqLen}; zero-padded)
   * @param attentionMask per-row attention masks (zero-padded)
   * @param maxSeqLen padded sequence length
   * @param wordPositionsFlat row-major {@code [batchSize][maxTextLen]} first-subword positions,
   *                          {@code -1} for padding words
   * @param batchSize number of rows
   * @param maxTextLen padded word count
   * @param queryPositions first-subword positions of the {@code [E]} markers (shared across rows)
   * @return the heap-materialized scoring tensors
   */
  @Override
  public Scoring run(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] wordPositionsFlat,
    int batchSize,
    int maxTextLen,
    long[] queryPositions
  ) {
    int tokElements = batchSize * maxSeqLen;
    var idsBuf = directLongs(tokElements);
    var maskBuf = directLongs(tokElements);
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
    var wpBuf = directLongs(batchSize * maxTextLen);
    wpBuf.put(wordPositionsFlat, 0, batchSize * maxTextLen).rewind();
    var qBuf = directLongs(queryPositions.length);
    qBuf.put(queryPositions).rewind();

    var tokShape = new long[] { batchSize, maxSeqLen };
    try (
      var idsTensor = OnnxTensor.createTensor(env, idsBuf, tokShape);
      var maskTensor = OnnxTensor.createTensor(env, maskBuf, tokShape);
      var wpTensor = OnnxTensor.createTensor(
        env,
        wpBuf,
        new long[] { batchSize, maxTextLen }
      );
      var qTensor = OnnxTensor.createTensor(
        env,
        qBuf,
        new long[] { queryPositions.length }
      );
      var result = session.run(
        Map.of(
          "input_ids",
          idsTensor,
          "attention_mask",
          maskTensor,
          "word_positions",
          wpTensor,
          "query_positions",
          qTensor
        )
      )
    ) {
      var pair = (OnnxTensor) result.get("pair_logits").orElseThrow();
      var cands = (OnnxTensor) result.get("candidates").orElseThrow();
      var nulls = (OnnxTensor) result.get("null_logits").orElseThrow();
      var pairShape = pair.getInfo().getShape();
      int numQueries = (int) pairShape[1];
      int poolSize = (int) pairShape[2];
      var pairLogits = new float[batchSize * numQueries * poolSize];
      pair.getFloatBuffer().get(pairLogits);
      var candidates = new long[batchSize * poolSize * 2];
      cands.getLongBuffer().get(candidates);
      var nullLogits = new float[batchSize * numQueries];
      nulls.getFloatBuffer().get(nullLogits);
      return new Scoring(
        pairLogits,
        candidates,
        nullLogits,
        batchSize,
        numQueries,
        poolSize
      );
    } catch (OrtException e) {
      throw new RuntimeException("GLiNER2.5 merged-graph inference failed", e);
    }
  }

  private static LongBuffer directLongs(int elements) {
    return ByteBuffer.allocateDirect(Math.max(elements, 1) * Long.BYTES)
      .order(ByteOrder.nativeOrder())
      .asLongBuffer();
  }

  @Override
  public void close() {
    try {
      session.close();
      log.info("GLiNER2.5 ner_full session closed");
    } catch (Exception e) {
      log.warn("Error closing GLiNER2.5 ner_full session", e);
    }
  }
}
