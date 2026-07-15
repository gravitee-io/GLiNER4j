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

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;

/**
 * An ONNX output tensor pinned to a pooled direct buffer.
 *
 * <p>ORT writes the output straight into the leased buffer (no native→heap copy), and the
 * wrapped {@link #tensor()} can be fed back as another session's input zero-copy. The element
 * type mirrors the model variant (FLOAT for fp32 exports, FLOAT16 for fp16); reads convert on
 * the fly where a float view is needed.
 *
 * <p>{@link #close()} closes the ORT tensor wrapper (the pooled buffer itself is returned to
 * the pool, not freed) and is idempotent. The lease must stay open while the tensor is used as
 * a session input or read from.
 */
public final class PinnedTensorLease implements FloatTensorView, AutoCloseable {

  private final OnnxTensor tensor;
  private final long[] shape;
  private final OnnxJavaType type;
  private final ByteBuffer data;
  private final DirectByteBufferPool pool;
  private final DirectByteBufferPool.Holder holder;
  private boolean closed;

  PinnedTensorLease(
    OnnxTensor tensor,
    long[] shape,
    OnnxJavaType type,
    ByteBuffer data,
    DirectByteBufferPool pool,
    DirectByteBufferPool.Holder holder
  ) {
    this.tensor = tensor;
    this.shape = shape;
    this.type = type;
    this.data = data;
    this.pool = pool;
    this.holder = holder;
  }

  /** The pinned ORT tensor — usable directly as a session input while the lease is open. */
  public OnnxTensor tensor() {
    return tensor;
  }

  @Override
  public long[] shape() {
    return shape;
  }

  @Override
  public int dim(int i) {
    return (int) shape[i];
  }

  @Override
  public long stride(int i) {
    long s = 1;
    for (int d = i + 1; d < shape.length; d++) {
      s *= shape[d];
    }
    return s;
  }

  @Override
  public float get(long index) {
    if (type == OnnxJavaType.FLOAT) {
      return data.asFloatBuffer().get((int) index);
    }
    return Float.float16ToFloat(data.asShortBuffer().get((int) index));
  }

  @Override
  public long capacity() {
    return type == OnnxJavaType.FLOAT
      ? data.asFloatBuffer().capacity()
      : data.asShortBuffer().capacity();
  }

  @Override
  public void copyRowAsFloats(
    long srcOffset,
    float[] dst,
    int dstOffset,
    int length
  ) {
    if (type == OnnxJavaType.FLOAT) {
      data.asFloatBuffer().get((int) srcOffset, dst, dstOffset, length);
      return;
    }
    var shorts = data.asShortBuffer();
    for (int i = 0; i < length; i++) {
      dst[dstOffset + i] = Float.float16ToFloat(
        shorts.get((int) srcOffset + i)
      );
    }
  }

  @Override
  public void copyRowAsFloats(
    long srcOffset,
    FloatBuffer dst,
    int dstOffset,
    int length
  ) {
    if (type == OnnxJavaType.FLOAT) {
      dst.put(dstOffset, data.asFloatBuffer(), (int) srcOffset, length);
      return;
    }
    var shorts = data.asShortBuffer();
    for (int i = 0; i < length; i++) {
      dst.put(
        dstOffset + i,
        Float.float16ToFloat(shorts.get((int) srcOffset + i))
      );
    }
  }

  @Override
  public void close() {
    if (closed) {
      return;
    }
    closed = true;
    tensor.close();
    pool.release(holder);
  }
}
