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

  // Pre-allocated span index buffer (single-thread assumption)
  private LongBuffer spanIdxBuf;
  private int spanIdxBufCapacity;

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
  }

  @Override
  protected void closeTaskHeads() throws OrtException {
    spanRepSession.close();
    scoringHeadSession.close();
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
    try {
      int totalElements = numSpans * 2;
      if (totalElements > spanIdxBufCapacity) {
        spanIdxBufCapacity = totalElements + 128;
        spanIdxBuf = allocateDirectLongBuffer(spanIdxBufCapacity);
      }
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
    try {
      int totalElements = batchSize * maxNumSpans * 2;
      if (totalElements > spanIdxBufCapacity) {
        spanIdxBufCapacity = totalElements + 256;
        spanIdxBuf = allocateDirectLongBuffer(spanIdxBufCapacity);
      }
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
}
