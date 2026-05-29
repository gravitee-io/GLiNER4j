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
package io.gravitee.lab.gliner4j.processor;

import java.util.List;

/**
 * Common preprocess step for batched extraction: tokenize each input text, assemble it through
 * a shared {@link InputAssembler}, repack non-empty slots into contiguous arrays for the encoder,
 * and report the batch-wide max sequence length.
 *
 * <p>Null/blank/empty-after-tokenization texts are left as {@code null} in {@link Result#inputs()}
 * and excluded from the contiguous batch. Callers can use {@link Result#batchIndices()} to map
 * a batch slot back to the original text position when writing per-text results.
 */
public final class BatchPreprocessor {

  private BatchPreprocessor() {}

  /**
   * Per-batch preprocessing output.
   *
   * @param inputs            per-original-index preprocessed inputs; {@code null} when the source
   *                          text was null/blank/empty after tokenization
   * @param batchIndices      maps a contiguous batch slot {@code s ∈ [0, nonEmptyCount)} back to
   *                          the original text index in {@code inputs}
   * @param batchInputIds     packed token IDs, one row per non-empty slot, in batch-slot order
   * @param batchAttentionMask packed attention masks, same ordering as {@code batchInputIds}
   * @param nonEmptyCount     number of non-null slots (i.e. {@code batchIndices.length})
   * @param maxSeqLen         largest sequence length across the non-empty inputs
   */
  public record Result(
    PreprocessedInput[] inputs,
    int[] batchIndices,
    long[][] batchInputIds,
    long[][] batchAttentionMask,
    int nonEmptyCount,
    int maxSeqLen
  ) {}

  /**
   * Preprocesses every text in {@code texts} with the provided {@code assembler}.
   *
   * @param texts     input texts (may contain null, blank, or whitespace-only entries)
   * @param assembler the {@link InputAssembler} to render each text through
   * @return the assembled per-text inputs plus the packed batch arrays for the encoder
   */
  public static Result preprocess(
    List<String> texts,
    InputAssembler assembler
  ) {
    int batchSize = texts.size();
    var inputs = new PreprocessedInput[batchSize];
    int maxSeqLen = 0;
    int nonEmptyCount = 0;
    for (int i = 0; i < batchSize; i++) {
      var text = texts.get(i);
      if (text == null || text.isBlank()) continue;
      var enc = new TextEncoder(text);
      if (enc.getTextLen() == 0) continue;
      inputs[i] = assembler.assemble(enc);
      maxSeqLen = Math.max(maxSeqLen, inputs[i].inputIds().length);
      nonEmptyCount++;
    }

    var batchIndices = new int[nonEmptyCount];
    var batchInputIds = new long[nonEmptyCount][];
    var batchAttentionMask = new long[nonEmptyCount][];
    int slot = 0;
    for (int i = 0; i < batchSize; i++) {
      if (inputs[i] != null) {
        batchIndices[slot] = i;
        batchInputIds[slot] = inputs[i].inputIds();
        batchAttentionMask[slot] = inputs[i].attentionMask();
        slot++;
      }
    }

    return new Result(
      inputs,
      batchIndices,
      batchInputIds,
      batchAttentionMask,
      nonEmptyCount,
      maxSeqLen
    );
  }

  /**
   * Multi-schema variant of {@link Result} — same packing semantics but the per-text
   * inputs are {@link MultiSchemaInput}s (with precomputed unit layouts).
   */
  public record MultiResult(
    MultiSchemaInput[] inputs,
    int[] batchIndices,
    long[][] batchInputIds,
    long[][] batchAttentionMask,
    int nonEmptyCount,
    int maxSeqLen
  ) {}

  /**
   * Multi-schema variant of {@link #preprocess(List, InputAssembler)}.
   *
   * @param texts     input texts (may contain null, blank, or whitespace-only entries)
   * @param assembler the {@link MultiSchemaInputAssembler} to render each text through
   * @return the assembled per-text multi-schema inputs plus the packed batch arrays
   */
  public static MultiResult preprocess(
    List<String> texts,
    MultiSchemaInputAssembler assembler
  ) {
    int batchSize = texts.size();
    var inputs = new MultiSchemaInput[batchSize];
    int maxSeqLen = 0;
    int nonEmptyCount = 0;
    for (int i = 0; i < batchSize; i++) {
      var text = texts.get(i);
      if (text == null || text.isBlank()) continue;
      var enc = new TextEncoder(text);
      if (enc.getTextLen() == 0) continue;
      inputs[i] = assembler.assemble(enc);
      maxSeqLen = Math.max(maxSeqLen, inputs[i].inputIds().length);
      nonEmptyCount++;
    }

    var batchIndices = new int[nonEmptyCount];
    var batchInputIds = new long[nonEmptyCount][];
    var batchAttentionMask = new long[nonEmptyCount][];
    int slot = 0;
    for (int i = 0; i < batchSize; i++) {
      if (inputs[i] != null) {
        batchIndices[slot] = i;
        batchInputIds[slot] = inputs[i].inputIds();
        batchAttentionMask[slot] = inputs[i].attentionMask();
        slot++;
      }
    }

    return new MultiResult(
      inputs,
      batchIndices,
      batchInputIds,
      batchAttentionMask,
      nonEmptyCount,
      maxSeqLen
    );
  }
}
