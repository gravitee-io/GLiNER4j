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
 * The original-GLiNER bi-encoder span forward pass as
 * {@link io.gravitee.lab.gliner4j.strategy.glinerbi.GlinerBiNerStrategy} consumes it: text
 * encoder and label encoder fused in one graph, returning span logits. Implemented by the ONNX
 * Runtime {@link GlinerBiNerRuntime}; the interface is the seam that lets the strategy run over a
 * fake runtime in unit tests.
 */
public interface GlinerBiSpanRuntime extends ArchitectureRuntime {
  /**
   * Runs the fused bi-encoder span model for a single text.
   *
   * @param inputIds text token IDs [seqLen], unpadded (the attention mask is all ones)
   * @param wordsMask word mask [seqLen]
   * @param textLen number of text words
   * @param spanIdx span (start, end) pairs [numSpans][2]
   * @param spanMask span validity [numSpans]
   * @param labelsInputIds tokenized labels [numLabels][labelSeqLen], padded to the longest label
   * @param labelsAttentionMask label attention mask [numLabels][labelSeqLen], 0 on padding
   * @return logits [textLen][maxWidth][numClasses]
   */
  float[][][] run(
    long[] inputIds,
    long[] wordsMask,
    int textLen,
    long[][] spanIdx,
    boolean[] spanMask,
    long[][] labelsInputIds,
    long[][] labelsAttentionMask
  );
}
