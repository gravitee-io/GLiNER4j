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
    if (useGpu) {
      var dev = Ggml.gpuDevice();
      if (dev.address() != 0) {
        chosen = Ggml.deviceInit(dev);
        log.info("{} on {}", what, Ggml.deviceName(dev));
      }
    }
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
    log.info("Loaded {} ({} tensors)", gguf.getFileName(), tensors.size());
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
    if (!Ggml.gallocrAllocGraph(galloc, graph)) {
      throw new IllegalStateException("ggml_gallocr_alloc_graph failed");
    }
  }

  void run(MemorySegment graph) {
    int rc = Ggml.graphCompute(backend, graph);
    if (rc != 0) {
      throw new IllegalStateException(
        "ggml_backend_graph_compute failed with status " + rc
      );
    }
  }

  @Override
  public synchronized void close() {
    Ggml.gallocrFree(galloc);
    Ggml.bufferFree(weightsBuffer);
    Ggml.free(weightsCtx);
    Ggml.backendFree(backend);
    arena.close();
  }
}
