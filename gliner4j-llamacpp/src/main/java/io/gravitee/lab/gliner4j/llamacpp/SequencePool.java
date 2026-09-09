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
package io.gravitee.lab.gliner4j.llamacpp;

import java.util.BitSet;
import java.util.concurrent.TimeUnit;

/**
 * Pool of llama.cpp sequence ids ({@code 0 … nSeqMax-1}). Stateless calls borrow one for the
 * duration of a decode, sessions hold one until closed. Acquisition blocks (bounded) when every
 * slot is busy, so a burst of callers queues behind the batched decoder instead of failing.
 */
final class SequencePool {

  private final BitSet inUse;
  private final int size;
  private final long timeoutMillis;

  SequencePool(int size, long timeoutMillis) {
    this.size = size;
    this.inUse = new BitSet(size);
    this.timeoutMillis = timeoutMillis;
  }

  int size() {
    return size;
  }

  synchronized int acquire() {
    long deadline =
      System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(timeoutMillis);
    while (true) {
      int seq = inUse.nextClearBit(0);
      if (seq < size) {
        inUse.set(seq);
        return seq;
      }
      long remaining = deadline - System.nanoTime();
      if (remaining <= 0) {
        throw new IllegalStateException(
          "all " +
            size +
            " llama.cpp sequences stayed busy for " +
            timeoutMillis +
            " ms — close sessions or raise n_seq_max"
        );
      }
      try {
        TimeUnit.NANOSECONDS.timedWait(this, remaining);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException(
          "interrupted while waiting for a llama.cpp sequence",
          e
        );
      }
    }
  }

  synchronized void release(int seq) {
    inUse.clear(seq);
    notifyAll();
  }
}
