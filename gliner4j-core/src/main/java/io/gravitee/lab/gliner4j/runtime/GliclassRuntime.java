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
 * GLiClass runtime: the shared single encoder from {@link BaseRuntime} plus the GLiClass
 * {@code score_head.onnx} (both feature projectors + the dot-product scorer).
 *
 * <p>GLiClass is uni-encoder, so it reuses {@link BaseRuntime}'s encoder session and buffers
 * exactly like the GLiNER2 runtimes; only the task head differs.
 */
@Slf4j
public non-sealed class GliclassRuntime extends BaseRuntime {

  private OrtSession scoreHeadSession;

  public GliclassRuntime(
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

    log.info("Loading score_head.onnx...");
    try (
      var opts = createSessionOptions(
        scoringIntra,
        scoringInter,
        OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL,
        runtimeConfig,
        cacheDir,
        "score_head.onnx"
      )
    ) {
      this.scoreHeadSession = env.createSession(
        variantDir.resolve("score_head.onnx").toString(),
        opts
      );
    }
  }

  @Override
  protected void closeTaskHeads() throws OrtException {
    scoreHeadSession.close();
  }

  /**
   * Runs the GLiClass scoring head: projects the pooled text vector and the per-label class
   * vectors and computes per-label logits via the dot-product scorer.
   *
   * @param textEmb the pooled text (CLS) embedding, shape {@code [hidden]}
   * @param classEmbs the per-label class embeddings, shape {@code [numLabels][hidden]}
   * @return raw per-label logits, shape {@code [numLabels]} (sigmoid applied by the caller)
   */
  public float[] runScoreHead(float[] textEmb, float[][] classEmbs) {
    int numLabels = classEmbs.length;
    var text2d = new float[][] { textEmb }; // [1, hidden]
    var class3d = new float[][][] { classEmbs }; // [1, numLabels, hidden]
    try {
      try (
        var textTensor = OnnxTensor.createTensor(env, text2d);
        var classTensor = OnnxTensor.createTensor(env, class3d);
        var result = scoreHeadSession.run(
          Map.of("text_emb", textTensor, "class_embs", classTensor)
        )
      ) {
        var logits = (float[][]) result.get(0).getValue(); // [1, numLabels]
        return logits[0];
      }
    } catch (OrtException e) {
      throw new RuntimeException(
        "GLiClass score head inference failed (" + numLabels + " labels)",
        e
      );
    }
  }
}
