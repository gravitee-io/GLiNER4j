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

import java.nio.FloatBuffer;

/**
 * Read-only view over a flat, row-major float tensor — implemented by heap-copied
 * {@link FloatTensor}s and by {@link PinnedTensorLease}s over pooled direct buffers
 * (which may hold FLOAT16 elements, converted on read).
 */
public interface FloatTensorView {
  long[] shape();

  /** Size of dimension {@code i}. */
  int dim(int i);

  /** Number of elements in one step of dimension {@code i} (product of trailing dims). */
  long stride(int i);

  /** Absolute element read (converted to float when the backing type is FLOAT16). */
  float get(long index);

  /** Number of elements actually backing this view (the readable buffer capacity). */
  long capacity();

  /** Copies {@code length} elements starting at {@code srcOffset} into {@code dst[dstOffset..]}. */
  void copyRowAsFloats(long srcOffset, float[] dst, int dstOffset, int length);

  /** Copies {@code length} elements starting at {@code srcOffset} into {@code dst} at {@code dstOffset}. */
  void copyRowAsFloats(
    long srcOffset,
    FloatBuffer dst,
    int dstOffset,
    int length
  );
}
