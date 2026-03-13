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

  /**
   * Creates ONNX runtime sessions for all three model files.
   *
   * @param modelDir path to directory containing encoder.onnx, span_rep.onnx, scoring_head.onnx
   */
  public GLiNER4jRuntime(Path modelDir) {
    try {
      this.env = OrtEnvironment.getEnvironment();
      var opts = new OrtSession.SessionOptions();
      opts.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors());

      log.info("Loading encoder.onnx...");
      this.encoderSession =
        env.createSession(modelDir.resolve("encoder.onnx").toString(), opts);

      log.info("Loading span_rep.onnx...");
      this.spanRepSession =
        env.createSession(modelDir.resolve("span_rep.onnx").toString(), opts);

      log.info("Loading scoring_head.onnx...");
      this.scoringHeadSession =
        env.createSession(
          modelDir.resolve("scoring_head.onnx").toString(),
          opts
        );

      log.info("All ONNX sessions loaded successfully");
    } catch (OrtException e) {
      throw new RuntimeException(
        "Failed to load ONNX models from " + modelDir,
        e
      );
    }
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
      var idsTensor = OnnxTensor.createTensor(
        env,
        LongBuffer.wrap(inputIds),
        new long[] { 1, inputIds.length }
      );
      var maskTensor = OnnxTensor.createTensor(
        env,
        LongBuffer.wrap(attentionMask),
        new long[] { 1, attentionMask.length }
      );

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
        LongBuffer.wrap(new long[] { count }),
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
