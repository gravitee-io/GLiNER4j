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

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Test;

class FloatTensorTest {

  /** [2][3][4] row-major tensor with element value = 100*i + 10*j + k. */
  private static FloatTensor tensor234() {
    var buf = ByteBuffer.allocateDirect(24 * Float.BYTES)
      .order(ByteOrder.nativeOrder())
      .asFloatBuffer();
    for (int i = 0; i < 2; i++) {
      for (int j = 0; j < 3; j++) {
        for (int k = 0; k < 4; k++) {
          buf.put(100f * i + 10f * j + k);
        }
      }
    }
    buf.rewind();
    return new FloatTensor(buf, new long[] { 2, 3, 4 });
  }

  @Test
  void dims_and_strides_follow_row_major_layout() {
    var t = tensor234();
    assertThat(t.dim(0)).isEqualTo(2);
    assertThat(t.dim(1)).isEqualTo(3);
    assertThat(t.dim(2)).isEqualTo(4);
    assertThat(t.stride(0)).isEqualTo(12);
    assertThat(t.stride(1)).isEqualTo(4);
    assertThat(t.stride(2)).isEqualTo(1);
  }

  @Test
  void get_reads_by_absolute_flat_index() {
    var t = tensor234();
    // [1][2][3] -> 1*12 + 2*4 + 3 = 23
    assertThat(t.get(1 * t.stride(0) + 2 * t.stride(1) + 3)).isEqualTo(123f);
    assertThat(t.get(0)).isEqualTo(0f);
  }

  @Test
  void copyTo_array_copies_a_row() {
    var t = tensor234();
    var row = new float[4];
    t.copyTo(1 * t.stride(0) + 1 * t.stride(1), row, 0, 4);
    assertThat(row).containsExactly(110f, 111f, 112f, 113f);
  }

  @Test
  void copyTo_buffer_copies_at_absolute_positions_without_moving_cursors() {
    var t = tensor234();
    var dst = ByteBuffer.allocateDirect(8 * Float.BYTES)
      .order(ByteOrder.nativeOrder())
      .asFloatBuffer();
    t.copyTo(4, dst, 2, 4); // row [0][1] into dst[2..6)
    assertThat(dst.get(2)).isEqualTo(10f);
    assertThat(dst.get(5)).isEqualTo(13f);
    assertThat(dst.position()).isZero();
    assertThat(t.data().position()).isZero();
  }
}
