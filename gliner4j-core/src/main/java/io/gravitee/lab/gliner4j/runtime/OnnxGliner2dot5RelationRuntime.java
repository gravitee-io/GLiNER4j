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
 * Relation runtime for the GLiNER2.5 merged graph ({@code relation_full.onnx}): the NER
 * pipeline over the {@code [R] head} / {@code [R] tail} queries, typed capped pair generation
 * and the sparse relation scorer in a single session run. Outputs the {@code [batch,
 * relations, pairs]} relation logits and the {@code [batch, relations, pairs, 4]} half-open
 * word boundaries {@code (head_start, head_end, tail_start, tail_end)} of each proposed pair.
 */
@Slf4j
public final class OnnxGliner2dot5RelationRuntime
  implements Gliner2dot5RelationRuntime {

  private final OrtEnvironment env;
  private final OrtSession session;

  public OnnxGliner2dot5RelationRuntime(
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
      var fullPath = variantDir.resolve("relation_full.onnx");
      if (!Files.exists(fullPath)) {
        throw new IllegalStateException(
          "relation_full.onnx not found in " +
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
        "Loading relation_full.onnx from {} (GLiNER2.5 merged relation graph)...",
        variantDir
      );
      try (
        var opts = OrtSessions.createSessionOptions(
          intra,
          inter,
          OrtSession.SessionOptions.ExecutionMode.PARALLEL,
          runtimeConfig,
          cacheDir,
          "relation_full.onnx"
        )
      ) {
        this.session = env.createSession(fullPath.toString(), opts);
      }
      log.info("GLiNER2.5 relation_full.onnx loaded successfully");
    } catch (OrtException e) {
      throw new RuntimeException(
        "Failed to load GLiNER2.5 relation_full.onnx from " +
          modelDir +
          "/" +
          variant,
        e
      );
    }
  }

  /**
   * Runs the merged relation graph over one padded sub-batch.
   *
   * @param queryPositions first-subword positions of the {@code [R]} markers, head then tail per
   *                       relation (length {@code 2 × relations}), shared across rows
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
      var logitsT = (OnnxTensor) result.get("relation_logits").orElseThrow();
      var pairsT = (OnnxTensor) result.get("relation_pairs").orElseThrow();
      var shape = logitsT.getInfo().getShape();
      int numRelations = (int) shape[1];
      int pairCap = (int) shape[2];
      var logits = new float[batchSize * numRelations * pairCap];
      logitsT.getFloatBuffer().get(logits);
      var pairs = new long[batchSize * numRelations * pairCap * 4];
      pairsT.getLongBuffer().get(pairs);
      return new Scoring(logits, pairs, batchSize, numRelations, pairCap);
    } catch (OrtException e) {
      throw new RuntimeException(
        "GLiNER2.5 relation-graph inference failed",
        e
      );
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
      log.info("GLiNER2.5 relation_full session closed");
    } catch (Exception e) {
      log.warn("Error closing GLiNER2.5 relation_full session", e);
    }
  }
}
