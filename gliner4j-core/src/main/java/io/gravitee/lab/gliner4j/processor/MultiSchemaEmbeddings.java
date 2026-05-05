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

/**
 * Extracts per-unit and text embeddings from encoder hidden states for the multi-unit prompt path.
 *
 * <p>Schema unit positions are pre-resolved in {@link UnitLayout}, so extraction is a direct
 * index into the hidden-state matrix — no scanning of token IDs is required.
 */
public final class MultiSchemaEmbeddings {

  private MultiSchemaEmbeddings() {}

  /**
   * One unit's parent ({@code [P]}) and per-field child-marker embeddings.
   *
   * @param schemaEmbP the {@code [P]} hidden state
   * @param schemaEmbFields the child-marker hidden states, one per child
   */
  public record UnitEmbeddings(float[] schemaEmbP, float[][] schemaEmbFields) {}

  /**
   * Reads one unit's parent and child-marker embeddings from encoder hidden states.
   *
   * @param hiddenState encoder output for a single batch slot, shape {@code [seqLen][hiddenSize]}
   * @param layout the unit's resolved positions
   * @return the unit's parent and child-marker embeddings
   */
  public static UnitEmbeddings extractUnit(
    float[][] hiddenState,
    UnitLayout layout
  ) {
    var pEmb = hiddenState[layout.parentTokenPos()];
    var positions = layout.childMarkerPositions();
    var fieldEmbs = new float[positions.length][];
    for (int i = 0; i < positions.length; i++) {
      fieldEmbs[i] = hiddenState[positions[i]];
    }
    return new UnitEmbeddings(pEmb, fieldEmbs);
  }

  /**
   * Reads each text word's first-subword embedding into the target matrix.
   *
   * @param hiddenState encoder output for a single batch slot, shape {@code [seqLen][hiddenSize]}
   * @param wordFirstSubwordPos absolute position of each word's first subword in the input
   * @param target output matrix, shape {@code [textLen][hiddenSize]} (rows are overwritten)
   */
  public static void extractText(
    float[][] hiddenState,
    int[] wordFirstSubwordPos,
    float[][] target
  ) {
    for (int w = 0; w < target.length; w++) {
      target[w] = hiddenState[wordFirstSubwordPos[w]];
    }
  }
}
