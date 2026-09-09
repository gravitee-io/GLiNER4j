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
 * The GLiNER2 span-model forward pass as the strategies, extractors and the unified facade consume
 * it: the merged {@code ner_full} graph (encoder → span representations → count head → scores).
 * Implemented by the ONNX Runtime {@link GLiNER4jNERRuntime} and by the ggml runtime in
 * {@code gliner4j-llamacpp}; everything above this interface is engine-agnostic.
 */
public interface Gliner2SpanRuntime extends ArchitectureRuntime {
  /**
   * Scores one schema unit for a (length-bucketed) batch of inputs.
   *
   * @param inputIds per-row token ids (unpadded; the implementation pads to {@code maxSeqLen})
   * @param attentionMask per-row attention masks
   * @param maxSeqLen longest row
   * @param wordPositionsFlat {@code [batchSize × maxTextLen]} first-subtoken position of each text word, {@code -1} = padding
   * @param batchSize number of rows
   * @param maxTextLen longest text (in words)
   * @param pPosition position of the unit's {@code [P]} marker (shared prefix; {@code -1} = none)
   * @param fieldPositions positions of the unit's field markers
   * @param spanIdxFlat {@code [batchSize × maxTextLen·maxWidth × 2]} span grid ({@code SpanIndexCache} layout)
   * @param maxWidth span width
   * @param count count instances to score (1 for entities, {@code max_count} for relations / structures)
   * @return {@code countLogits[1][max_count]} of row 0 and sigmoid span scores
   *         {@code [batch][count][fields][maxTextLen][maxWidth]}
   */
  FlatBatchScoringResult runNerFullBatch(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] wordPositionsFlat,
    int batchSize,
    int maxTextLen,
    long pPosition,
    long[] fieldPositions,
    long[] spanIdxFlat,
    int maxWidth,
    long count
  );
}
