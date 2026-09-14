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
 * The GLiNER2.5 (boundary architecture) relation forward pass as the strategies consume it:
 * the merged relation graph (the NER pipeline over the [R] head/tail queries → typed capped pair generation → sparse relation scorer).
 * Implemented by the ONNX Runtime {@link OnnxGliner2dot5RelationRuntime} and by the ggml runtime in
 * {@code gliner4j-llamacpp}.
 */
public interface Gliner2dot5RelationRuntime extends ArchitectureRuntime {
  /** One merged-graph run, materialized on the heap. Invalid pairs carry the -1e4 mask logit. */
  public record Scoring(
    float[] relationLogits,
    long[] relationPairs,
    int batchSize,
    int numRelations,
    int pairCap
  ) {
    public float logit(int row, int relation, int pair) {
      return relationLogits[(row * numRelations + relation) * pairCap + pair];
    }

    /** {@code component}: 0 = head start, 1 = head end, 2 = tail start, 3 = tail end. */
    public int boundary(int row, int relation, int pair, int component) {
      return (int) relationPairs[((row * numRelations + relation) * pairCap +
          pair) *
        4 +
      component];
    }
  }

  Scoring run(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] wordPositionsFlat,
    int batchSize,
    int maxTextLen,
    long[] queryPositions
  );
}
