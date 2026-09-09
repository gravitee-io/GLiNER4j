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
import io.gravitee.llama.cpp.LlamaLogLevel;
import io.gravitee.llama.cpp.LlamaLogger;
import io.gravitee.llama.cpp.LlamaModel;
import io.gravitee.llama.cpp.LlamaModelParams;
import io.gravitee.llama.cpp.LlamaRuntime;
import io.gravitee.llama.cpp.PoolingType;
import io.gravitee.llama.cpp.nativelib.LlamaLibLoader;
import java.lang.foreign.Arena;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The causal backbone on llama.cpp: one GGUF model, one context in embeddings mode with
 * {@code pooling=NONE}, so every decoded token yields its final-norm hidden state (llama.cpp's
 * {@code result_norm}, the same tensor as HF {@code last_hidden_state}).
 *
 * <p><b>Concurrency.</b> A llama.cpp context is single-threaded, but one {@code llama_decode}
 * can carry tokens of many sequences. Callers therefore never touch the context directly:
 * {@link #decode} enqueues a request and a dispatcher thread packs whatever is pending — one
 * request per sequence, up to {@code n_batch} tokens — into a single decode and completes each
 * caller with its own rows. Under load requests coalesce naturally (arrivals during one decode
 * form the next batch); a lone caller pays no extra latency. Memory operations
 * ({@link #removeFrom}, {@link #clearSequence}) take the same context lock.
 */
public final class LlamaBackbone implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(
    LlamaBackbone.class
  );
  private static final Object BACKEND_LOCK = new Object();
  private static boolean backendInitialized;
  // Keeps the native log callback (and its upcall stub) alive for the JVM's lifetime.
  private static Arena backendArena;
  private static LlamaLogger nativeLogger;

  private record Request(
    int seqId,
    long[] ids,
    int startPos,
    int outputFrom,
    CompletableFuture<List<float[]>> result
  ) {}

  private final Arena arena;
  private final LlamaModel model;
  private final LlamaContext context;
  private final LlamaBatch batch;
  private final int nCtx;
  private final int nBatch;
  private final int nSeqMax;
  private final int hiddenSize;
  private final ReentrantLock contextLock = new ReentrantLock();
  private final LinkedBlockingQueue<Request> queue =
    new LinkedBlockingQueue<>();
  private final Thread dispatcher;
  private final AtomicLong requestsDecoded = new AtomicLong();
  private final AtomicLong batchesDecoded = new AtomicLong();
  private volatile boolean closed;

  public LlamaBackbone(
    Path gguf,
    int nGpuLayers,
    int nCtx,
    int nBatch,
    int nSeqMax,
    int nThreads
  ) {
    this.arena = Arena.ofShared();
    initBackend();
    log.info(
      "Loading backbone {} (gpuLayers={}, nCtx={}, nBatch={}, nSeqMax={}, threads={})",
      gguf,
      nGpuLayers,
      nCtx,
      nBatch,
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
        .nCtx(nCtx)
        .nBatch(nBatch)
        .nUBatch(nBatch)
        .nSeqMax(nSeqMax)
        .nThreads(nThreads)
        .nThreadsBatch(nThreads)
        .embeddings(true)
        .poolingType(PoolingType.NONE)
        .attentionType(AttentionType.CAUSAL)
    );
    this.nCtx = context.nCtx();
    this.nBatch = context.nBatch();
    this.nSeqMax = context.nSeqMax();
    this.hiddenSize = model.nEmbdOut();
    this.batch = new LlamaBatch(arena, this.nBatch, 0, 1);
    this.dispatcher = new Thread(this::dispatchLoop, "gliner4j-llama-decode");
    this.dispatcher.setDaemon(true);
    this.dispatcher.start();
    log.info(
      "Backbone ready: hidden={}, nCtx={}, nBatch={}, nSeqMax={}",
      hiddenSize,
      this.nCtx,
      this.nBatch,
      this.nSeqMax
    );
  }

  static void initBackend() {
    synchronized (BACKEND_LOCK) {
      if (!backendInitialized) {
        var libPath = LlamaLibLoader.load();
        backendArena = Arena.ofShared();
        // llama.cpp is chatty on stderr at INFO (every tensor at model load); keep warnings and
        // errors and route them through SLF4J. Continuation fragments (progress dots, blank
        // lines) arrive at the CONT level, above WARN; drop anything without letters.
        nativeLogger = new LlamaLogger(backendArena);
        nativeLogger.setLogging(LlamaLogLevel.WARN, message -> {
          var text = message.strip();
          if (text.chars().anyMatch(Character::isLetter)) {
            log.warn("llama.cpp: {}", text);
          }
        });
        LlamaRuntime.llama_backend_init();
        LlamaRuntime.ggml_backend_load_all_from_path(backendArena, libPath);
        backendInitialized = true;
      }
    }
  }

  /**
   * ggml threads for one native call when {@code RuntimeConfig} does not pin them. ggml threads
   * spin at barriers, so running one per logical CPU (efficiency cores, SMT siblings) makes every
   * decode several times slower and wildly jittery — on Apple silicon, dropping from one thread per
   * logical CPU to the performance cores alone cut both single calls and batches sharply. Default:
   * the performance cores on macOS, half the logical CPUs elsewhere; override with the
   * {@code gliner4j.llamacpp.threads} system property or {@code encoderIntraOpThreads}.
   */
  public static int defaultThreads() {
    var prop = Integer.getInteger("gliner4j.llamacpp.threads");
    if (prop != null && prop > 0) {
      return prop;
    }
    int logical = Runtime.getRuntime().availableProcessors();
    if (System.getProperty("os.name", "").toLowerCase().contains("mac")) {
      try {
        var proc = new ProcessBuilder(
          "sysctl",
          "-n",
          "hw.perflevel0.physicalcpu"
        )
          .redirectErrorStream(true)
          .start();
        var out = new String(proc.getInputStream().readAllBytes()).strip();
        if (proc.waitFor() == 0 && !out.isEmpty()) {
          return Math.max(1, Math.min(logical, Integer.parseInt(out)));
        }
      } catch (Exception ignored) {
        // fall through to the generic heuristic
      }
    }
    return Math.max(1, logical / 2);
  }

  /** Threads for a llama.cpp/ggml engine: the pinned {@code encoderIntraOpThreads} or {@link #defaultThreads()}. */
  public static int threads(io.gravitee.lab.gliner4j.runtime.RuntimeConfig rc) {
    return rc.getEncoderIntraOpThreads() != null
      ? rc.getEncoderIntraOpThreads()
      : defaultThreads();
  }

  public int nCtx() {
    return nCtx;
  }

  public int nBatch() {
    return nBatch;
  }

  public int nSeqMax() {
    return nSeqMax;
  }

  public int hiddenSize() {
    return hiddenSize;
  }

  /** Decode requests completed so far (diagnostics). */
  public long requestsDecoded() {
    return requestsDecoded.get();
  }

  /** {@code llama_decode} calls issued so far; fewer than {@link #requestsDecoded} means coalescing happened. */
  public long batchesDecoded() {
    return batchesDecoded.get();
  }

  /**
   * Decodes {@code ids} into sequence {@code seqId} at positions {@code startPos…} and returns the
   * hidden state of every token whose index is {@code >= outputFrom}, in order. Blocks until the
   * dispatcher has run the request (possibly packed with other sequences' requests).
   */
  public List<float[]> decode(
    int seqId,
    long[] ids,
    int startPos,
    int outputFrom
  ) {
    if (closed) {
      throw new IllegalStateException("backbone is closed");
    }
    if (startPos + ids.length > nCtx) {
      throw new IllegalStateException(
        "sequence would exceed the context window: " +
          (startPos + ids.length) +
          " > nCtx=" +
          nCtx +
          " — raise nCtx or start a new session"
      );
    }
    var request = new Request(
      seqId,
      ids,
      startPos,
      outputFrom,
      new CompletableFuture<>()
    );
    queue.add(request);
    try {
      return request.result().get();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(
        "interrupted while waiting for llama_decode",
        e
      );
    } catch (ExecutionException e) {
      var cause = e.getCause();
      throw cause instanceof RuntimeException re
        ? re
        : new IllegalStateException("llama_decode failed", cause);
    }
  }

  // ---- dispatcher ------------------------------------------------------------

  private void dispatchLoop() {
    // Requests that did not fit in the previous round, in arrival order; served before new arrivals.
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
      // Pack one request per sequence, at most nBatch tokens, oldest first. Everything else waits
      // for the next round in its original order.
      var picked = new ArrayList<Request>();
      var seqs = new HashSet<Integer>();
      int tokens = 0;
      var rest = new ArrayDeque<Request>();
      for (var r : backlog) {
        boolean fits =
          !seqs.contains(r.seqId()) && tokens + r.ids().length <= nBatch;
        if (fits || (picked.isEmpty() && r.ids().length > nBatch)) {
          picked.add(r);
          seqs.add(r.seqId());
          tokens += r.ids().length;
          if (r.ids().length > nBatch) {
            // An oversized request runs alone, chunked.
            break;
          }
        } else {
          rest.add(r);
        }
      }
      if (picked.size() == 1 && picked.get(0).ids().length > nBatch) {
        // Requests skipped before the oversized one keep their place ahead of later ones.
        var it = backlog.iterator();
        rest.clear();
        boolean seen = false;
        while (it.hasNext()) {
          var r = it.next();
          if (r == picked.get(0)) {
            seen = true;
            continue;
          }
          rest.add(r);
        }
      }
      backlog = rest;
      runBatch(picked);
    }
    var abort = new IllegalStateException("backbone closed");
    backlog.forEach(r -> r.result().completeExceptionally(abort));
    queue.forEach(r -> r.result().completeExceptionally(abort));
  }

  private void runBatch(List<Request> requests) {
    contextLock.lock();
    try {
      if (requests.size() == 1 && requests.get(0).ids().length > nBatch) {
        // A single oversized request: decode it in nBatch-sized chunks, alone.
        var r = requests.get(0);
        r.result().complete(decodeChunked(r));
        return;
      }
      batch.clear();
      var offsets = new int[requests.size()];
      int cursor = 0;
      for (int k = 0; k < requests.size(); k++) {
        var r = requests.get(k);
        offsets[k] = cursor;
        var seqIds = List.of(r.seqId());
        for (int i = 0; i < r.ids().length; i++) {
          // In embeddings mode llama.cpp computes an output for every token anyway; flag them
          // all and read back only the requested ones.
          batch.add((int) r.ids()[i], r.startPos() + i, seqIds, true);
        }
        cursor += r.ids().length;
      }
      int rc = context.decode(batch);
      batchesDecoded.incrementAndGet();
      if (rc != 0) {
        var failure = new IllegalStateException(
          "llama_decode failed with code " + rc
        );
        requests.forEach(r -> r.result().completeExceptionally(failure));
        return;
      }
      for (int k = 0; k < requests.size(); k++) {
        var r = requests.get(k);
        var rows = new ArrayList<float[]>(
          Math.max(0, r.ids().length - r.outputFrom())
        );
        for (int i = r.outputFrom(); i < r.ids().length; i++) {
          rows.add(context.getEmbeddingsIth(offsets[k] + i));
        }
        requestsDecoded.incrementAndGet();
        r.result().complete(rows);
      }
    } catch (RuntimeException e) {
      requests.forEach(r -> r.result().completeExceptionally(e));
    } finally {
      contextLock.unlock();
    }
  }

  private List<float[]> decodeChunked(Request r) {
    var seqIds = List.of(r.seqId());
    var out = new ArrayList<float[]>();
    for (int from = 0; from < r.ids().length; from += nBatch) {
      int to = Math.min(r.ids().length, from + nBatch);
      batch.clear();
      for (int i = from; i < to; i++) {
        batch.add((int) r.ids()[i], r.startPos() + i, seqIds, true);
      }
      int rc = context.decode(batch);
      batchesDecoded.incrementAndGet();
      if (rc != 0) {
        throw new IllegalStateException("llama_decode failed with code " + rc);
      }
      for (int i = Math.max(from, r.outputFrom()); i < to; i++) {
        out.add(context.getEmbeddingsIth(i - from));
      }
    }
    requestsDecoded.incrementAndGet();
    return out;
  }

  // ---- memory --------------------------------------------------------------

  /** Drops every cached token of {@code seqId} at position {@code >= pos}. */
  public void removeFrom(int seqId, int pos) {
    contextLock.lock();
    try {
      context.getMemory().seqRm(seqId, pos, -1);
    } finally {
      contextLock.unlock();
    }
  }

  /** Drops every cached token of {@code seqId}. */
  public void clearSequence(int seqId) {
    contextLock.lock();
    try {
      context.getMemory().seqRm(seqId, -1, -1);
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
      "Backbone closed ({} requests in {} decodes)",
      requestsDecoded.get(),
      batchesDecoded.get()
    );
  }
}
