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

/**
 * The GLiNER2 classifier forward pass (merged {@code classifier_full} graph: encoder → {@code [L]}
 * rows → classifier MLP). Implemented by the ONNX Runtime {@link GLiNER4jClassifierRuntime} and by
 * the ggml runtime in {@code gliner4j-llamacpp}.
 */
public interface Gliner2ClassifierRuntime extends ArchitectureRuntime {
  /** Raw (pre-sigmoid) logits {@code [batchSize][labelPositions.length]}. */
  float[][] runClassifierFullBatch(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] labelPositions,
    int batchSize
  );
}
