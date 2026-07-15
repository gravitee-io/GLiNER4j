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

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;

/**
 * A flat, row-major float tensor: one contiguous {@link java.nio.FloatBuffer} plus its shape.
 *
 * <p>Replaces nested {@code float[][]...} arrays on the hot inference path. Reading an ONNX
 * output through {@link #of(OnnxTensor)} is a single bulk copy ({@code getFloatBuffer}),
 * versus {@code getValue()}'s reflective element-by-element materialization
 * ({@code OrtUtil.fillArrayFromBuffer} + {@code Array.get} boxing); feeding a
 * {@code FloatTensor}'s buffer back into {@code OnnxTensor.createTensor(env, buffer, shape)}
 * is zero-copy for direct buffers, versus the reflective shape walk
 * ({@code TensorInfo.extractShape} + {@code fillBufferFromArray}) of nested arrays.
 *
 * <p>The buffer's position/limit are not part of the contract: accessors use absolute
 * indexing, and {@link #data()} returns the buffer rewound. Not thread-safe for writes;
 * concurrent absolute reads are safe.
 *
 * @param data  the tensor elements, row-major, starting at index 0
 * @param shape the tensor dimensions
 */
public record FloatTensor(java.nio.FloatBuffer data, long[] shape) implements
  FloatTensorView {
  /** Reads an ONNX tensor into a flat {@code FloatTensor} with one bulk copy. */
  public static FloatTensor of(OnnxTensor tensor) throws OrtException {
    var info = tensor.getInfo();
    if (info.type == ai.onnxruntime.OnnxJavaType.FLOAT) {
      // Direct-buffer copy: keeps the data off-heap, so feeding it back into
      // createTensor is zero-copy (a heap buffer from getFloatBuffer() would force
      // ORT to re-copy it into a fresh zero-filled direct buffer per call).
      var buf = tensor
        .getByteBuffer()
        .order(java.nio.ByteOrder.nativeOrder())
        .asFloatBuffer();
      return new FloatTensor(buf, info.getShape());
    }
    // Non-FLOAT element types (e.g. FLOAT16 outputs): getFloatBuffer() converts.
    return new FloatTensor(tensor.getFloatBuffer(), info.getShape());
  }

  /** Size of dimension {@code i}. */
  @Override
  public int dim(int i) {
    return (int) shape[i];
  }

  /** Number of elements in one step of dimension {@code i} (product of trailing dims). */
  public long stride(int i) {
    long s = 1;
    for (int d = i + 1; d < shape.length; d++) {
      s *= shape[d];
    }
    return s;
  }

  /** Absolute element read. */
  public float get(long index) {
    return data.get((int) index);
  }

  /** Number of float elements backing this tensor. */
  @Override
  public long capacity() {
    return data.capacity();
  }

  /** Copies {@code length} elements starting at {@code srcOffset} into {@code dst[dstOffset..]}. */
  public void copyTo(long srcOffset, float[] dst, int dstOffset, int length) {
    data.get((int) srcOffset, dst, dstOffset, length);
  }

  /** Copies {@code length} elements starting at {@code srcOffset} into {@code dst} at {@code dstOffset}. */
  public void copyTo(
    long srcOffset,
    java.nio.FloatBuffer dst,
    int dstOffset,
    int length
  ) {
    dst.put(dstOffset, data, (int) srcOffset, length);
  }

  @Override
  public void copyRowAsFloats(
    long srcOffset,
    float[] dst,
    int dstOffset,
    int length
  ) {
    data.get((int) srcOffset, dst, dstOffset, length);
  }

  @Override
  public void copyRowAsFloats(
    long srcOffset,
    java.nio.FloatBuffer dst,
    int dstOffset,
    int length
  ) {
    dst.put(dstOffset, data, (int) srcOffset, length);
  }

  /** The backing buffer, rewound — suitable for {@code OnnxTensor.createTensor(env, buf, shape)}. */
  public java.nio.FloatBuffer rewound() {
    return data.rewind();
  }
}
