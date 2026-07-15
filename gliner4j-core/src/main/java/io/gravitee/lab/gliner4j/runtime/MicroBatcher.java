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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;

/**
 * Coalesces concurrent single-text extract() calls into batched extractBatch() calls.
 *
 * <p>Callers block on a future while a single drainer (virtual thread) collects requests for
 * at most {@code maxWaitMicros} (or until {@code maxBatchSize} accumulate) and funnels them
 * into one batch call, amortizing the encoder run across concurrent callers. Requests with
 * different thresholds within one drain are grouped into one batch call per threshold.
 *
 * <p>Opt-in via {@link RuntimeConfig#isMicroBatchingEnabled()}: it adds up to
 * {@code maxWaitMicros} latency per call in exchange for concurrent throughput.
 *
 * @param <R> the per-text result type
 */
public final class MicroBatcher<R> implements AutoCloseable {

  /** Runs one batch: texts (parallel to callers) at a given threshold. */
  @FunctionalInterface
  public interface BatchFunction<R> {
    List<R> apply(List<String> texts, float threshold);
  }

  private record Request<R>(
    String text,
    float threshold,
    CompletableFuture<R> future
  ) {}

  private final LinkedBlockingQueue<Request<R>> queue =
    new LinkedBlockingQueue<>();
  private final BatchFunction<R> batchFn;
  private final int maxBatchSize;
  private final long maxWaitNanos;
  private final Thread drainer;
  private volatile boolean closed;

  public MicroBatcher(
    int maxBatchSize,
    long maxWaitMicros,
    BatchFunction<R> batchFn
  ) {
    this.batchFn = batchFn;
    this.maxBatchSize = Math.max(1, maxBatchSize);
    this.maxWaitNanos = TimeUnit.MICROSECONDS.toNanos(
      Math.max(0, maxWaitMicros)
    );
    this.drainer = Thread.ofVirtual()
      .name("gliner4j-micro-batcher")
      .start(this::drainLoop);
  }

  /**
   * Submits one text and blocks until its batch completes.
   *
   * @param text      the input text (null/blank allowed — handled by the batch path)
   * @param threshold the extraction threshold
   * @return this text's result
   */
  public R extract(String text, float threshold) {
    if (closed) {
      throw new IllegalStateException("MicroBatcher is closed");
    }
    var request = new Request<R>(text, threshold, new CompletableFuture<>());
    queue.add(request);
    try {
      return request.future().get();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(
        "Interrupted while waiting for micro-batch result",
        e
      );
    } catch (ExecutionException e) {
      if (e.getCause() instanceof RuntimeException re) {
        throw re;
      }
      throw new IllegalStateException(e.getCause());
    }
  }

  private void drainLoop() {
    var pending = new ArrayList<Request<R>>(maxBatchSize);
    while (!closed || !queue.isEmpty()) {
      try {
        var first = queue.poll(50, TimeUnit.MILLISECONDS);
        if (first == null) {
          continue;
        }
        pending.add(first);
        long deadline = System.nanoTime() + maxWaitNanos;
        while (pending.size() < maxBatchSize) {
          long remaining = deadline - System.nanoTime();
          if (remaining <= 0) {
            break;
          }
          var next = queue.poll(remaining, TimeUnit.NANOSECONDS);
          if (next == null) {
            break;
          }
          pending.add(next);
        }
        runBatches(pending);
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        break;
      } finally {
        pending.clear();
      }
    }
    failPending();
  }

  private void runBatches(List<Request<R>> requests) {
    // Group by threshold, preserving arrival order within each group (usually one group).
    var groups = new LinkedHashMap<Float, List<Request<R>>>();
    for (var request : requests) {
      groups
        .computeIfAbsent(request.threshold(), t -> new ArrayList<>())
        .add(request);
    }
    for (var group : groups.entrySet()) {
      var batch = group.getValue();
      var texts = new ArrayList<String>(batch.size());
      for (var request : batch) {
        texts.add(request.text());
      }
      try {
        var results = batchFn.apply(texts, group.getKey());
        for (int i = 0; i < batch.size(); i++) {
          batch.get(i).future().complete(results.get(i));
        }
      } catch (Throwable t) {
        for (var request : batch) {
          request.future().completeExceptionally(t);
        }
      }
    }
  }

  private void failPending() {
    Request<R> leftover;
    while ((leftover = queue.poll()) != null) {
      leftover
        .future()
        .completeExceptionally(
          new IllegalStateException("MicroBatcher closed")
        );
    }
  }

  @Override
  public void close() {
    closed = true;
    drainer.interrupt();
    try {
      drainer.join();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    failPending();
  }
}
