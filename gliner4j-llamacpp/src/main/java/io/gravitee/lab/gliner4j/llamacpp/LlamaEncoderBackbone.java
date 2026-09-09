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

import io.gravitee.llama.cpp.AttentionType;
import io.gravitee.llama.cpp.LlamaBatch;
import io.gravitee.llama.cpp.LlamaContext;
import io.gravitee.llama.cpp.LlamaContextParams;
import io.gravitee.llama.cpp.LlamaModel;
import io.gravitee.llama.cpp.LlamaModelParams;
import io.gravitee.llama.cpp.PoolingType;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A bidirectional encoder (ModernBERT, BERT, XLM-R, …) on llama.cpp: {@code llama_encode} with
 * {@code pooling=NONE} returns the final hidden state of every token, the same tensor as HF
 * {@code last_hidden_state}.
 *
 * <p>Unlike the causal {@link LlamaBackbone} there is no KV cache and no positions to continue:
 * every request is a whole sequence encoded from position 0. llama.cpp runs an encode as a single
 * micro-batch, so the packed batch must fit {@code n_ubatch} tokens; the dispatcher thread packs
 * pending requests (one sequence id each, up to {@code n_seq_max}) into one {@code llama_encode}
 * and completes each caller with its own rows. Parallel callers therefore batch for free.
 */
public final class LlamaEncoderBackbone implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(
    LlamaEncoderBackbone.class
  );

  private record Request(long[] ids, CompletableFuture<float[][]> result) {}

  private final Arena arena;
  private final LlamaModel model;
  private final LlamaContext context;
  private final LlamaBatch batch;
  private final int nUbatch;
  private final int nSeqMax;
  private final int hiddenSize;
  private final ReentrantLock contextLock = new ReentrantLock();
  private final LinkedBlockingQueue<Request> queue =
    new LinkedBlockingQueue<>();
  private final Thread dispatcher;
  private final AtomicLong requestsEncoded = new AtomicLong();
  private final AtomicLong batchesEncoded = new AtomicLong();
  private volatile boolean closed;

  public LlamaEncoderBackbone(
    Path gguf,
    int nGpuLayers,
    int nUbatch,
    int nSeqMax,
    int nThreads
  ) {
    this.arena = Arena.ofShared();
    LlamaBackbone.initBackend();
    log.info(
      "Loading encoder {} (gpuLayers={}, nUbatch={}, nSeqMax={}, threads={})",
      gguf,
      nGpuLayers,
      nUbatch,
      nSeqMax,
      nThreads
    );
    this.model = new LlamaModel(
      arena,
      gguf,
      new LlamaModelParams(arena).nGpuLayers(nGpuLayers).useMmap(true)
    );
    this.context = new LlamaContext(
      arena,
      model,
      new LlamaContextParams(arena)
        .nCtx(nUbatch)
        .nBatch(nUbatch)
        .nUBatch(nUbatch)
        .nSeqMax(nSeqMax)
        .nThreads(nThreads)
        .nThreadsBatch(nThreads)
        .embeddings(true)
        .poolingType(PoolingType.NONE)
        .attentionType(AttentionType.NON_CAUSAL)
    );
    this.nUbatch = context.nUBatch();
    this.nSeqMax = context.nSeqMax();
    this.hiddenSize = model.nEmbdOut();
    this.batch = new LlamaBatch(arena, this.nUbatch, 0, 1);
    this.dispatcher = new Thread(this::dispatchLoop, "gliner4j-llama-encode");
    this.dispatcher.setDaemon(true);
    this.dispatcher.start();
    log.info(
      "Encoder ready: hidden={}, nUbatch={}, nSeqMax={}",
      hiddenSize,
      this.nUbatch,
      this.nSeqMax
    );
  }

  public int nUbatch() {
    return nUbatch;
  }

  public int nSeqMax() {
    return nSeqMax;
  }

  public int hiddenSize() {
    return hiddenSize;
  }

  /** Requests completed so far (diagnostics). */
  public long requestsEncoded() {
    return requestsEncoded.get();
  }

  /** {@code llama_encode} calls so far; fewer than {@link #requestsEncoded} means coalescing happened. */
  public long batchesEncoded() {
    return batchesEncoded.get();
  }

  /**
   * Encodes one sequence and returns the hidden state of every token, in order. Blocks until the
   * dispatcher has run the request (possibly packed with other callers' sequences).
   */
  public float[][] encode(long[] ids) {
    if (closed) {
      throw new IllegalStateException("encoder is closed");
    }
    if (ids.length == 0) {
      return new float[0][];
    }
    if (ids.length > nUbatch) {
      throw new IllegalStateException(
        "sequence of " +
          ids.length +
          " tokens exceeds n_ubatch=" +
          nUbatch +
          " — llama.cpp encodes a sequence in one micro-batch; raise architecture_config.n_ubatch " +
          "or shorten the input"
      );
    }
    var request = new Request(ids, new CompletableFuture<>());
    queue.add(request);
    try {
      return request.result().get();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(
        "interrupted while waiting for llama_encode",
        e
      );
    } catch (ExecutionException e) {
      var cause = e.getCause();
      throw cause instanceof RuntimeException re
        ? re
        : new IllegalStateException("llama_encode failed", cause);
    }
  }

  /** Encodes several sequences; callers on other threads may still be packed into the same encodes. */
  public List<float[][]> encodeAll(List<long[]> sequences) {
    var futures = new ArrayList<CompletableFuture<float[][]>>(sequences.size());
    for (var ids : sequences) {
      if (ids.length > nUbatch) {
        throw new IllegalStateException(
          "sequence of " + ids.length + " tokens exceeds n_ubatch=" + nUbatch
        );
      }
      var request = new Request(ids, new CompletableFuture<>());
      futures.add(request.result());
      queue.add(request);
    }
    var out = new ArrayList<float[][]>(sequences.size());
    for (var f : futures) {
      try {
        out.add(f.get());
      } catch (InterruptedException e) {
        Thread.currentThread().interrupt();
        throw new IllegalStateException(
          "interrupted while waiting for llama_encode",
          e
        );
      } catch (ExecutionException e) {
        var cause = e.getCause();
        throw cause instanceof RuntimeException re
          ? re
          : new IllegalStateException("llama_encode failed", cause);
      }
    }
    return out;
  }

  // ---- dispatcher ------------------------------------------------------------

  private void dispatchLoop() {
    var backlog = new ArrayDeque<Request>();
    while (!closed) {
      if (backlog.isEmpty()) {
        try {
          backlog.add(queue.take());
        } catch (InterruptedException e) {
          break;
        }
      }
      queue.drainTo(backlog);
      // Oldest first; stop at the first request that does not fit so ordering is preserved.
      var picked = new ArrayList<Request>();
      int tokens = 0;
      while (!backlog.isEmpty() && picked.size() < nSeqMax) {
        var r = backlog.peekFirst();
        if (tokens + r.ids().length > nUbatch) {
          break;
        }
        picked.add(backlog.pollFirst());
        tokens += r.ids().length;
      }
      runBatch(picked);
    }
    var abort = new IllegalStateException("encoder closed");
    backlog.forEach(r -> r.result().completeExceptionally(abort));
    queue.forEach(r -> r.result().completeExceptionally(abort));
  }

  private void runBatch(List<Request> requests) {
    contextLock.lock();
    try {
      batch.clear();
      var offsets = new int[requests.size()];
      int cursor = 0;
      for (int k = 0; k < requests.size(); k++) {
        var r = requests.get(k);
        offsets[k] = cursor;
        var seqIds = List.of(k);
        for (int i = 0; i < r.ids().length; i++) {
          batch.add((int) r.ids()[i], i, seqIds, true);
        }
        cursor += r.ids().length;
      }
      int rc = batch.encode(context);
      batchesEncoded.incrementAndGet();
      if (rc != 0) {
        var failure = new IllegalStateException(
          "llama_encode failed with code " + rc
        );
        requests.forEach(r -> r.result().completeExceptionally(failure));
        return;
      }
      for (int k = 0; k < requests.size(); k++) {
        var r = requests.get(k);
        var rows = new float[r.ids().length][];
        for (int i = 0; i < rows.length; i++) {
          rows[i] = context.getEmbeddingsIth(offsets[k] + i);
        }
        requestsEncoded.incrementAndGet();
        r.result().complete(rows);
      }
    } catch (RuntimeException e) {
      requests.forEach(r -> r.result().completeExceptionally(e));
    } finally {
      contextLock.unlock();
    }
  }

  @Override
  public void close() {
    closed = true;
    dispatcher.interrupt();
    try {
      dispatcher.join(5_000);
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
    }
    contextLock.lock();
    try {
      batch.free();
      context.free();
      model.free();
    } finally {
      contextLock.unlock();
    }
    arena.close();
    log.info(
      "Encoder closed ({} requests in {} encodes)",
      requestsEncoded.get(),
      batchesEncoded.get()
    );
  }
}
