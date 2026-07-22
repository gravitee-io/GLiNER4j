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

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Pool of reusable direct {@link FloatBuffer}s for tensor inputs, mirroring
 * {@link DirectBufferPool} for floats.
 *
 * <p>Reusing buffers avoids the per-allocation zero-fill of
 * {@code ByteBuffer.allocateDirect} ({@code Bits.setMemory}), which is significant for
 * multi-megabyte embedding tensors. Reused buffers carry stale content — callers must
 * overwrite (or explicitly zero) every region the consumer reads, including padding.
 */
public final class FloatBufferPool {

  private static final int MAX_POOLED = 32;

  /** One reusable direct float buffer. */
  public static final class Holder {

    private FloatBuffer buf;
    private int capacity;

    private Holder() {
      this.capacity = 0;
    }

    public FloatBuffer buf() {
      return buf;
    }

    public int capacity() {
      return capacity;
    }

    /** Replaces the buffer with a fresh (zeroed) allocation of {@code newCapacity} floats. */
    public void reallocate(int newCapacity) {
      this.buf = ByteBuffer.allocateDirect(newCapacity * Float.BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer();
      this.capacity = newCapacity;
    }

    /** Ensures capacity, reallocating with headroom when too small. */
    public void ensureCapacity(int required, int headroom) {
      if (required > capacity) {
        reallocate(required + headroom);
      }
    }
  }

  private final ConcurrentLinkedQueue<Holder> pool =
    new ConcurrentLinkedQueue<>();
  // ConcurrentLinkedQueue.size() is O(n); track it separately for the O(1) cap check.
  // The check-then-offer is racy, so the cap is approximate — it only bounds how much
  // direct memory a burst can pin.
  private final AtomicInteger pooled = new AtomicInteger();

  public Holder acquire() {
    var holder = pool.poll();
    if (holder != null) {
      pooled.decrementAndGet();
      return holder;
    }
    return new Holder();
  }

  public void release(Holder holder) {
    if (pooled.get() < MAX_POOLED) {
      pool.offer(holder);
      pooled.incrementAndGet();
    }
  }
}
