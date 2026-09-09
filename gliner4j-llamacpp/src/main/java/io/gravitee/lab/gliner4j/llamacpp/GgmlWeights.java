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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A ggml backend (GPU device when requested and available, else CPU) holding the tensors of one
 * GGUF file, plus a graph allocator reused across compute calls. Shared by the ggml graph heads.
 */
final class GgmlWeights implements AutoCloseable {

  private static final Logger log = LoggerFactory.getLogger(GgmlWeights.class);
  static final long GRAPH_SIZE = 2048;
  static final int GRAPH_TENSORS = 1024;

  final Arena arena = Arena.ofShared();
  final MemorySegment backend;
  final MemorySegment galloc;
  /**
   * When the DEBERTA plugin is loaded and the weights live on CUDA: a {@code ggml_backend_sched}
   * over [DEBERTA, CUDA, CPU] runs the graphs (the fused attention node goes to the plugin, zero
   * copies since it shares CUDA's buffer type); otherwise NULL and the gallocr path is used.
   */
  private final MemorySegment sched;
  private final MemorySegment debertaBackend;
  private final MemorySegment cpuBackend;
  /** {@code ggml_custom_op_t} marker of the fused attention op, or NULL. */
  private final MemorySegment debertaAttnFn;
  private final boolean debertaAttnQkvF32;
  private final MemorySegment lstmFn;
  /** Metal rejects mixed-type binary ops (f16 + f32); CUDA and the CPU backend accept them. */
  final boolean metal;
  /** Weights and graphs live on a GPU backend (CUDA / Metal / Vulkan …). */
  final boolean gpu;
  final int typeF32;
  final int typeI32;
  private final MemorySegment weightsCtx;
  private final MemorySegment weightsBuffer;
  private final Map<String, MemorySegment> tensors = new HashMap<>();
  private final Map<String, Object> metadata;

  GgmlWeights(Path gguf, boolean useGpu, int cpuThreads, String what) {
    LlamaBackbone.initBackend(); // loads libggml + backends; no-op when a backbone already did
    this.typeF32 = Ggml.typeF32();
    this.typeI32 = Ggml.typeI32();
    MemorySegment chosen = null;
    String device = "CPU";
    if (useGpu) {
      var dev = Ggml.gpuDevice();
      if (dev.address() != 0) {
        chosen = Ggml.deviceInit(dev);
        device = Ggml.deviceName(dev);
        log.info("{} on {}", what, device);
      }
    }
    this.metal = device.startsWith("Metal");
    this.gpu = !"CPU".equals(device);
    if (chosen == null || chosen.address() == 0) {
      chosen = Ggml.cpuInit();
      Ggml.cpuSetThreads(chosen, Math.max(1, cpuThreads));
      log.info("{} on CPU ({} threads)", what, cpuThreads);
    }
    this.backend = chosen;

    var opened = Ggml.ggufOpen(arena, gguf.toString());
    var ggufCtx = opened[0];
    var hostCtx = opened[1];
    this.metadata = Ggml.ggufMetadata(ggufCtx);
    var host = new ArrayList<MemorySegment>();
    for (
      var t = Ggml.firstTensor(hostCtx);
      t.address() != 0;
      t = Ggml.nextTensor(hostCtx, t)
    ) {
      host.add(t);
    }
    this.weightsCtx = Ggml.init(
      arena,
      Ggml.tensorOverhead() * (host.size() + 8),
      true
    );
    var names = new ArrayList<String>(host.size());
    for (var t : host) {
      var name = Ggml.name(t);
      var copy = Ggml.dupTensor(weightsCtx, t);
      Ggml.setName(copy, arena, name);
      tensors.put(name, copy);
      names.add(name);
    }
    this.weightsBuffer = Ggml.allocCtxTensors(weightsCtx, backend);
    if (weightsBuffer.address() == 0) {
      throw new IllegalStateException(
        "failed to allocate " + what + " weights on the ggml backend"
      );
    }
    for (int i = 0; i < host.size(); i++) {
      Ggml.tensorSet(
        tensors.get(names.get(i)),
        Ggml.data(host.get(i)),
        Ggml.nbytes(host.get(i))
      );
    }
    Ggml.ggufFree(ggufCtx);
    Ggml.free(hostCtx);
    this.galloc = Ggml.gallocrNew(backend);

    MemorySegment schedPtr = MemorySegment.NULL;
    MemorySegment deberta = MemorySegment.NULL;
    MemorySegment cpu = MemorySegment.NULL;
    MemorySegment attnFn = MemorySegment.NULL;
    boolean attnQkvF32 = false;
    MemorySegment lstmFn = MemorySegment.NULL;
    var plugin = LlamaBackbone.debertaPlugin();
    if (plugin.address() != 0 && device.startsWith("CUDA")) {
      attnFn = Ggml.regGetProcAddress(plugin, "gliner4j_deberta_attn_fn");
      attnQkvF32 =
        Ggml.regGetProcAddress(plugin, "gliner4j_deberta_qkv_f32").address() !=
        0;
      lstmFn = Ggml.regGetProcAddress(plugin, "gliner4j_lstm_fn");
      deberta = Ggml.deviceInit(Ggml.regDevGet(plugin, 0));
      if (attnFn.address() != 0 && deberta.address() != 0) {
        cpu = Ggml.cpuInit();
        Ggml.cpuSetThreads(cpu, Math.max(1, cpuThreads));
        // A patched ggml orders consecutive splits on different backends with stream-side event
        // waits (see native/ggml-deberta/patches); then the scheduler is created with events
        // (parallel=true) and the plugin drops its host-side synchronisation.
        boolean events =
          Ggml.hasSymbol("ggml_backend_sched_cross_backend_events") &&
          Boolean.parseBoolean(
            System.getProperty("gliner4j.ggml.eventOrdering", "true")
          );
        schedPtr = Ggml.schedNew(
          arena,
          new MemorySegment[] { deberta, backend, cpu },
          SCHED_GRAPH_SIZE,
          events
        );
        var setHostSync = Ggml.regGetProcAddress(
          plugin,
          "gliner4j_deberta_set_host_sync"
        );
        if (setHostSync.address() != 0) {
          Ggml.callBoolFn(setHostSync, !events);
        }
        log.info(
          "{}: fused DeBERTa attention via the DEBERTA plugin ({} ordering)",
          what,
          events ? "event" : "host-sync"
        );
        var report = Ggml.regGetProcAddress(
          plugin,
          "gliner4j_deberta_kernel_report"
        );
        if (report.address() != 0 && !KERNEL_REPORTED.getAndSet(true)) {
          // resolves the kernel for this device (autotune on first use) and says which one
          log.info("DEBERTA plugin: {}", Ggml.callStringFn(report));
        }
      } else {
        log.warn(
          "DEBERTA plugin present but unusable (no CUDA device?), using the composed graph"
        );
        attnFn = MemorySegment.NULL;
        lstmFn = MemorySegment.NULL;
        if (deberta.address() != 0) Ggml.backendFree(deberta);
        deberta = MemorySegment.NULL;
      }
    }
    this.sched = schedPtr;
    this.debertaBackend = deberta;
    this.cpuBackend = cpu;
    this.debertaAttnFn = attnFn;
    this.debertaAttnQkvF32 =
      attnFn.address() != 0 &&
      attnQkvF32 &&
      Boolean.parseBoolean(System.getProperty(RAW_QKV_PROPERTY, "false"));
    this.lstmFn = lstmFn;
    log.info("Loaded {} ({} tensors)", gguf.getFileName(), tensors.size());
  }

  /**
   * Max nodes + leafs per graph handed to the scheduler (fixed at creation; exceeding it aborts in
   * {@code ggml_backend_sched_alloc_graph}). The encoder is ≈ 96·layers, the GLiNER2 heads a few
   * dozen, but the uni-encoder's word LSTM adds ≈ 40 nodes per word — a 512-token text needs
   * ~20k. The hash set and copy tables this sizes cost a few megabytes.
   */
  private static final long SCHED_GRAPH_SIZE = 65536;
  /**
   * {@code true} feeds the f32 q / k / v projections straight into the fused attention op (it
   * converts while staging) instead of casting them to f16 first. Off by default: the kernel
   * re-reads K / V once per key tile, so f32 operands double that traffic and cost more than the
   * three single-pass casts they remove. Kept for A/B on other cards.
   */
  static final String RAW_QKV_PROPERTY = "gliner4j.ggml.rawQkv";
  private static final java.util.concurrent.atomic.AtomicBoolean KERNEL_REPORTED =
    new java.util.concurrent.atomic.AtomicBoolean();

  /** Whether graphs may use the fused disentangled-attention op ({@link #debertaAttnFn()}). */
  boolean fusedDebertaAttention() {
    return debertaAttnFn.address() != 0;
  }

  /**
   * Whether the fused op takes the f32 q / k / v projections directly (converted while staging),
   * so the graph skips three cast ops per layer. Older plugins want f16 operands.
   */
  boolean debertaAttnQkvF32() {
    return debertaAttnQkvF32;
  }

  MemorySegment debertaAttnFn() {
    return debertaAttnFn;
  }

  /** Whether graphs may use the plugin's fused bidirectional LSTM ({@link #lstmFn()}); older plugins lack it. */
  boolean fusedLstm() {
    return lstmFn.address() != 0;
  }

  MemorySegment lstmFn() {
    return lstmFn;
  }

  MemorySegment get(String name) {
    var t = tensors.get(name);
    if (t == null) {
      throw new IllegalStateException("GGUF is missing tensor " + name);
    }
    return t;
  }

  boolean has(String name) {
    return tensors.containsKey(name);
  }

  /**
   * The single-file model GGUF for a bundle: {@code variant} names a quantisation
   * ({@code q8_0}, {@code q4_0}, … → {@code gguf/model-<variant>.gguf}) or a file name; anything
   * else (including {@code f16} and the ONNX default variant) is the f16 {@code gguf/model.gguf}.
   */
  static java.nio.file.Path resolveModelGguf(
    java.nio.file.Path modelDir,
    String variant
  ) {
    var gguf = modelDir.resolve("gguf");
    if (variant != null && !variant.isBlank()) {
      for (var candidate : java.util.List.of(
        variant,
        "model-" + variant + ".gguf"
      )) {
        var p = gguf.resolve(candidate);
        if (java.nio.file.Files.isRegularFile(p)) {
          return p;
        }
      }
    }
    return gguf.resolve("model.gguf");
  }

  /** GGUF metadata value for {@code key}, or {@code def} if absent. */
  int metaInt(String key, int def) {
    var v = metadata.get(key);
    return v instanceof Number n ? n.intValue() : def;
  }

  float metaFloat(String key, float def) {
    var v = metadata.get(key);
    return v instanceof Number n ? n.floatValue() : def;
  }

  String metaString(String key, String def) {
    var v = metadata.get(key);
    return v != null ? v.toString() : def;
  }

  /** A fresh no-alloc graph context sized for one head graph; free it after compute. */
  MemorySegment newGraphContext(Arena call) {
    return newGraphContext(call, GRAPH_SIZE);
  }

  /** A graph context for up to {@code nodes} graph nodes (and about as many tensors). */
  MemorySegment newGraphContext(Arena call, long nodes) {
    return Ggml.init(
      call,
      Ggml.tensorOverhead() * (nodes + 64) + Ggml.graphOverhead(nodes),
      true
    );
  }

  /** {@code y = W x + b} for a torch {@code Linear} saved as {@code prefix.weight/.bias}. */
  MemorySegment linear(MemorySegment ctx, MemorySegment x, String prefix) {
    return Ggml.add(
      ctx,
      Ggml.mulMat(ctx, get(prefix + ".weight"), x),
      get(prefix + ".bias")
    );
  }

  /** gliner's {@code create_projection_layer}: Linear → ReLU → (Dropout) → Linear, indices 0 and 3. */
  MemorySegment projection(MemorySegment ctx, MemorySegment x, String prefix) {
    return linear(
      ctx,
      Ggml.relu(ctx, linear(ctx, x, prefix + ".0")),
      prefix + ".3"
    );
  }

  /** Allocates the graph ending in {@code output} and runs it on the backend. */
  void compute(MemorySegment ctx, MemorySegment graph, MemorySegment output) {
    Ggml.setOutput(output);
    Ggml.buildForwardExpand(graph, output);
    alloc(graph);
  }

  /** Allocates a graph whose outputs were already expanded into it. */
  void alloc(MemorySegment graph) {
    if (sched.address() != 0) {
      Ggml.schedReset(sched);
      if (!Ggml.schedAllocGraph(sched, graph)) {
        throw new IllegalStateException(
          "ggml_backend_sched_alloc_graph failed"
        );
      }
      return;
    }
    if (!Ggml.gallocrAllocGraph(galloc, graph)) {
      throw new IllegalStateException("ggml_gallocr_alloc_graph failed");
    }
  }

  void run(MemorySegment graph) {
    int rc = sched.address() != 0
      ? Ggml.schedGraphCompute(sched, graph)
      : Ggml.graphCompute(backend, graph);
    if (rc != 0) {
      throw new IllegalStateException(
        "ggml graph compute failed with status " + rc
      );
    }
  }

  private final java.util.List<Runnable> onClose = new java.util.ArrayList<>();

  /** Registers a callback run by {@link #close()} before the backend goes away (buffers owned by graph helpers). */
  synchronized void onClose(Runnable r) {
    onClose.add(r);
  }

  @Override
  public synchronized void close() {
    for (var r : onClose) r.run();
    onClose.clear();
    if (sched.address() != 0) {
      Ggml.schedFree(sched);
      Ggml.backendFree(debertaBackend);
      Ggml.backendFree(cpuBackend);
    }
    Ggml.gallocrFree(galloc);
    Ggml.bufferFree(weightsBuffer);
    Ggml.free(weightsCtx);
    Ggml.backendFree(backend);
    arena.close();
  }
}
