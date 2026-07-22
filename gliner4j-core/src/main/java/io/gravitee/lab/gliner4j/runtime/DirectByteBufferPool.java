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
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Pool of reusable direct {@link ByteBuffer}s for pinned output tensors — byte-based so the
 * same pool serves fp32 and fp16 element types via {@code asFloatBuffer()}/{@code
 * asShortBuffer()} views.
 *
 * <p>Reused buffers carry stale content; pinned outputs are fully overwritten by the
 * session run, so no zeroing is needed.
 */
public final class DirectByteBufferPool {

  private static final int MAX_POOLED = 32;

  /** One reusable direct byte buffer. */
  public static final class Holder {

    private ByteBuffer buf;
    private int capacity;

    private Holder() {
      this.capacity = 0;
    }

    public ByteBuffer buf() {
      return buf;
    }

    /** Ensures capacity in bytes, reallocating with headroom when too small. */
    public void ensureCapacity(int requiredBytes, int headroomBytes) {
      if (requiredBytes > capacity) {
        this.buf = ByteBuffer.allocateDirect(
          requiredBytes + headroomBytes
        ).order(ByteOrder.nativeOrder());
        this.capacity = requiredBytes + headroomBytes;
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
