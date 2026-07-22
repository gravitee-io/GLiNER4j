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

import java.nio.LongBuffer;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Lock-free pool of reusable direct {@link LongBuffer} sets, so concurrent inference calls on
 * one runtime each get their own input buffers instead of racing on a single shared one.
 *
 * <p>A sequential caller keeps hitting the same holder (no allocation regression); concurrent
 * callers grow the pool on demand. Holders released while the pool is at capacity are dropped
 * to the GC so bursts don't pin direct memory forever.
 */
final class DirectBufferPool {

  /** One reusable set of equally sized direct buffers. */
  static final class Holder {

    private LongBuffer[] bufs;
    private int capacity;

    private Holder(int bufferCount) {
      this.bufs = new LongBuffer[bufferCount];
      this.capacity = 0;
    }

    LongBuffer buf(int i) {
      return bufs[i];
    }

    int capacity() {
      return capacity;
    }

    /**
     * Replaces every buffer with a fresh direct buffer of {@code newCapacity} longs.
     * The caller re-fills any constant content (schema prefix, all-ones mask) afterwards.
     */
    void reallocate(int newCapacity) {
      for (int i = 0; i < bufs.length; i++) {
        bufs[i] = OrtSessions.allocateDirectLongBuffer(newCapacity);
      }
      this.capacity = newCapacity;
    }
  }

  private static final int MAX_POOLED = 32;

  private final ConcurrentLinkedQueue<Holder> pool =
    new ConcurrentLinkedQueue<>();
  // ConcurrentLinkedQueue.size() is O(n); track it separately for the O(1) cap check.
  // The check-then-offer is racy, so the cap is approximate — that's fine, it only
  // bounds how much direct memory a burst can pin.
  private final AtomicInteger pooled = new AtomicInteger();
  private final int bufferCount;

  DirectBufferPool(int bufferCount) {
    this.bufferCount = bufferCount;
  }

  Holder acquire() {
    var holder = pool.poll();
    if (holder != null) {
      pooled.decrementAndGet();
      return holder;
    }
    return new Holder(bufferCount);
  }

  void release(Holder holder) {
    if (pooled.get() < MAX_POOLED) {
      pool.offer(holder);
      pooled.incrementAndGet();
    }
  }

  /** Drops all pooled holders (e.g. when constant buffer content becomes stale). */
  void clear() {
    pool.clear();
    pooled.set(0);
  }
}
