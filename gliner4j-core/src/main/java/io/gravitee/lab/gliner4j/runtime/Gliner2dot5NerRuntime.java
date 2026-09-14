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
 * The GLiNER2.5 (boundary architecture) ner forward pass as the strategies consume it:
 * the merged NER graph (encoder → word/query gathers → boundary encoding → start/end/inside marginals → shared candidate pool → pooled pair scorer).
 * Implemented by the ONNX Runtime {@link OnnxGliner2dot5NerRuntime} and by the ggml runtime in
 * {@code gliner4j-llamacpp}.
 */
public interface Gliner2dot5NerRuntime extends ArchitectureRuntime {
  /**
   * One merged-graph run, materialized on the heap (small: {@code batch × queries × pool}).
   *
   * @param pairLogits flat {@code [batch][numQueries][poolSize]} pair logits; invalid pool
   *                   slots carry the head's {@code -1e4} mask sentinel (sigmoid ⇒ 0)
   * @param candidates flat {@code [batch][poolSize][2]} half-open {@code [start, end)} word
   *                   boundaries per pool slot
   * @param nullLogits flat {@code [batch][numQueries]} abstention logits
   */
  public record Scoring(
    float[] pairLogits,
    long[] candidates,
    float[] nullLogits,
    int batchSize,
    int numQueries,
    int poolSize
  ) {
    public float pairLogit(int row, int query, int slot) {
      return pairLogits[(row * numQueries + query) * poolSize + slot];
    }

    public int candidateStart(int row, int slot) {
      return (int) candidates[(row * poolSize + slot) * 2];
    }

    public int candidateEnd(int row, int slot) {
      return (int) candidates[(row * poolSize + slot) * 2 + 1];
    }

    public float nullLogit(int row, int query) {
      return nullLogits[row * numQueries + query];
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
