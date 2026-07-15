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
package io.gravitee.lab.gliner4j.postprocess;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.lab.gliner4j.runtime.FloatTensor;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;

/**
 * Differential test: the flat {@link SpanDecoder#decode(FloatTensor, int, List, int[], int[],
 * String, int, float)} must produce exactly the spans of the nested-array decode for every
 * batch row, including padded rows (textLen shorter than the tensor's padded text length).
 */
class SpanDecoderFlatTest {

  private static final int BATCH = 3;
  private static final int COUNT = 2; // count instances; only instance 0 is decoded
  private static final int FIELDS = 4;
  private static final int PADDED_TEXT_LEN = 12;
  private static final int MAX_WIDTH = 5;
  private static final float THRESHOLD = 0.6f;

  private final SpanDecoder decoder = new SpanDecoder();

  @Test
  void flat_decode_matches_nested_decode_for_every_batch_row() {
    var random = new Random(42);

    // Synthetic batched scores [batch][count][fields][paddedTextLen][maxWidth], values in [0,1)
    // with plenty above and below the threshold, plus exact-threshold edge values.
    var nested = new float[BATCH][COUNT][FIELDS][PADDED_TEXT_LEN][MAX_WIDTH];
    for (int b = 0; b < BATCH; b++) {
      for (int c = 0; c < COUNT; c++) {
        for (int f = 0; f < FIELDS; f++) {
          for (int t = 0; t < PADDED_TEXT_LEN; t++) {
            for (int w = 0; w < MAX_WIDTH; w++) {
              nested[b][c][f][t][w] = random.nextFloat();
            }
          }
        }
      }
    }
    nested[0][0][1][2][1] = THRESHOLD; // >= comparison boundary
    nested[1][0][0][0][0] = Math.nextDown(THRESHOLD);

    var flat = flatten(nested);
    var fieldNames = List.of("person", "email", "phone", "iban");

    // Word char offsets for a synthetic text of PADDED_TEXT_LEN words ("w0 w1 w2 ...").
    var text = new StringBuilder();
    var wordStart = new int[PADDED_TEXT_LEN];
    var wordEnd = new int[PADDED_TEXT_LEN];
    for (int i = 0; i < PADDED_TEXT_LEN; i++) {
      wordStart[i] = text.length();
      text.append('w').append(i);
      wordEnd[i] = text.length();
      text.append(' ');
    }

    // Row 0: full length; row 1: shorter than padded (exercises the textLen guard);
    // row 2: single word.
    int[] textLens = { PADDED_TEXT_LEN, 7, 1 };

    for (int row = 0; row < BATCH; row++) {
      var expected = decoder.decode(
        nested[row],
        fieldNames,
        wordStart,
        wordEnd,
        text.toString(),
        textLens[row],
        THRESHOLD
      );
      var actual = decoder.decode(
        flat,
        row,
        fieldNames,
        wordStart,
        wordEnd,
        text.toString(),
        textLens[row],
        THRESHOLD
      );
      assertThat(actual)
        .as("batch row %d (textLen=%d)", row, textLens[row])
        .containsExactlyElementsOf(expected);
    }
  }

  private static FloatTensor flatten(float[][][][][] nested) {
    int total = BATCH * COUNT * FIELDS * PADDED_TEXT_LEN * MAX_WIDTH;
    var buf = ByteBuffer.allocateDirect(total * Float.BYTES)
      .order(ByteOrder.nativeOrder())
      .asFloatBuffer();
    for (var byCount : nested) {
      for (var byField : byCount) {
        for (var byText : byField) {
          for (var byWidth : byText) {
            buf.put(byWidth);
          }
        }
      }
    }
    buf.rewind();
    return new FloatTensor(
      buf,
      new long[] { BATCH, COUNT, FIELDS, PADDED_TEXT_LEN, MAX_WIDTH }
    );
  }
}
