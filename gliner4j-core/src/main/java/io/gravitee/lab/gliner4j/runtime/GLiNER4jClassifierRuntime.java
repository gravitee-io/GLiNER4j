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
public class GLiNER4jClassifierRuntime extends BaseRuntime {

  private OrtSession classifierHeadSession;

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

    int scoringIntra = runtimeConfig.getScoringIntraOpThreads() != null
      ? runtimeConfig.getScoringIntraOpThreads()
      : Math.max(2, numCpus / 4);
    int scoringInter = runtimeConfig.getScoringInterOpThreads() != null
      ? runtimeConfig.getScoringInterOpThreads()
      : 1;

    log.info("Loading classifier_head.onnx...");
    try (
      var opts = createSessionOptions(
        scoringIntra,
        scoringInter,
        OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL,
        runtimeConfig.getOptimizationLevel(),
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

  @Override
  protected void closeTaskHeads() throws OrtException {
    classifierHeadSession.close();
  }

  /**
   * Runs the classifier head MLP on label embeddings.
   *
   * @param labelEmbeddings label embeddings [numLabels][hiddenSize]
   * @return logits [numLabels][1] (raw scores before activation)
   */
  public float[][] runClassifierHead(float[][] labelEmbeddings) {
    try {
      var embTensor = OnnxTensor.createTensor(env, labelEmbeddings);

      try (
        var result = classifierHeadSession.run(
          Map.of("label_embeddings", embTensor)
        )
      ) {
        return (float[][]) result.get(0).getValue();
      } finally {
        embTensor.close();
      }
    } catch (OrtException e) {
      throw new RuntimeException("Classifier head inference failed", e);
    }
  }
}
