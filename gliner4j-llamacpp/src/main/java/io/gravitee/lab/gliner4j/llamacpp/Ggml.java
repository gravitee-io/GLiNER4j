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

import io.gravitee.llama.cpp.LlamaRuntime;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;

/**
 * Thin, typed wrappers over the ggml / ggml-backend / gguf C API that llamaj.cpp's jextract
 * bindings already expose (dispatched through {@link LlamaRuntime#invoke} so the platform
 * package is resolved at runtime). Only what {@link GgmlScorer} needs.
 */
final class Ggml {

  private static final Class<?> S = MemorySegment.class;
  private static final Class<?>[] S1 = { S };
  private static final Class<?>[] S2 = { S, S };
  private static final Class<?>[] S3 = { S, S, S };

  private Ggml() {}

  static int typeF32() {
    return LlamaRuntime.llama_h("GGML_TYPE_F32", new Class[0]);
  }

  static int typeI32() {
    return LlamaRuntime.llama_h("GGML_TYPE_I32", new Class[0]);
  }

  static int typeI64() {
    return LlamaRuntime.llama_h("GGML_TYPE_I64", new Class[0]);
  }

  static int deviceTypeGpu() {
    return LlamaRuntime.llama_h("GGML_BACKEND_DEVICE_TYPE_GPU", new Class[0]);
  }

  // ---- contexts ------------------------------------------------------------

  static MemorySegment init(Arena arena, long memSize, boolean noAlloc) {
    MemorySegment params = LlamaRuntime.invoke(
      "ggml_init_params",
      "allocate",
      new Class[] { SegmentAllocator.class },
      arena
    );
    LlamaRuntime.invoke(
      "ggml_init_params",
      "mem_size",
      new Class[] { S, long.class },
      params,
      memSize
    );
    LlamaRuntime.invoke(
      "ggml_init_params",
      "mem_buffer",
      S2,
      params,
      MemorySegment.NULL
    );
    LlamaRuntime.invoke(
      "ggml_init_params",
      "no_alloc",
      new Class[] { S, boolean.class },
      params,
      noAlloc
    );
    MemorySegment ctx = LlamaRuntime.llama_h("ggml_init", S1, params);
    if (ctx.address() == 0) {
      throw new IllegalStateException(
        "ggml_init failed (mem_size=" + memSize + ")"
      );
    }
    return ctx;
  }

  static void free(MemorySegment ctx) {
    LlamaRuntime.llama_h("ggml_free", S1, ctx);
  }

  static long tensorOverhead() {
    return LlamaRuntime.llama_h("ggml_tensor_overhead", new Class[0]);
  }

  static long graphOverhead(long size) {
    return LlamaRuntime.llama_h(
      "ggml_graph_overhead_custom",
      new Class[] { long.class, boolean.class },
      size,
      false
    );
  }

  // ---- tensors -------------------------------------------------------------

  static MemorySegment newTensor1d(MemorySegment ctx, int type, long ne0) {
    return LlamaRuntime.llama_h(
      "ggml_new_tensor_1d",
      new Class[] { S, int.class, long.class },
      ctx,
      type,
      ne0
    );
  }

  static MemorySegment newTensor2d(
    MemorySegment ctx,
    int type,
    long ne0,
    long ne1
  ) {
    return LlamaRuntime.llama_h(
      "ggml_new_tensor_2d",
      new Class[] { S, int.class, long.class, long.class },
      ctx,
      type,
      ne0,
      ne1
    );
  }

  static MemorySegment dupTensor(MemorySegment ctx, MemorySegment src) {
    return LlamaRuntime.llama_h("ggml_dup_tensor", S2, ctx, src);
  }

  static MemorySegment getTensor(MemorySegment ctx, Arena arena, String name) {
    return LlamaRuntime.llama_h(
      "ggml_get_tensor",
      S2,
      ctx,
      arena.allocateFrom(name)
    );
  }

  static MemorySegment firstTensor(MemorySegment ctx) {
    return LlamaRuntime.llama_h("ggml_get_first_tensor", S1, ctx);
  }

  static MemorySegment nextTensor(MemorySegment ctx, MemorySegment t) {
    return LlamaRuntime.llama_h("ggml_get_next_tensor", S2, ctx, t);
  }

  static void setName(MemorySegment t, Arena arena, String name) {
    LlamaRuntime.llama_h("ggml_set_name", S2, t, arena.allocateFrom(name));
  }

  static String name(MemorySegment t) {
    MemorySegment s = LlamaRuntime.llama_h("ggml_get_name", S1, t);
    return s.reinterpret(Long.MAX_VALUE).getString(0);
  }

  static long nbytes(MemorySegment t) {
    return LlamaRuntime.llama_h("ggml_nbytes", S1, t);
  }

  static long ne(MemorySegment t, int dim) {
    long size = LlamaRuntime.invoke("ggml_tensor", "sizeof", new Class[0]);
    return LlamaRuntime.invoke(
      "ggml_tensor",
      "ne",
      new Class[] { S, long.class },
      t.reinterpret(size),
      (long) dim
    );
  }

  static long nb(MemorySegment t, int dim) {
    long size = LlamaRuntime.invoke("ggml_tensor", "sizeof", new Class[0]);
    return LlamaRuntime.invoke(
      "ggml_tensor",
      "nb",
      new Class[] { S, long.class },
      t.reinterpret(size),
      (long) dim
    );
  }

  static MemorySegment data(MemorySegment t) {
    long size = LlamaRuntime.invoke("ggml_tensor", "sizeof", new Class[0]);
    return LlamaRuntime.invoke("ggml_tensor", "data", S1, t.reinterpret(size));
  }

  static void setInput(MemorySegment t) {
    LlamaRuntime.llama_h("ggml_set_input", S1, t);
  }

  static void setOutput(MemorySegment t) {
    LlamaRuntime.llama_h("ggml_set_output", S1, t);
  }

  // ---- ops -----------------------------------------------------------------

  static MemorySegment mulMat(
    MemorySegment ctx,
    MemorySegment a,
    MemorySegment b
  ) {
    return LlamaRuntime.llama_h("ggml_mul_mat", S3, ctx, a, b);
  }

  static MemorySegment add(
    MemorySegment ctx,
    MemorySegment a,
    MemorySegment b
  ) {
    return LlamaRuntime.llama_h("ggml_add", S3, ctx, a, b);
  }

  static MemorySegment mul(
    MemorySegment ctx,
    MemorySegment a,
    MemorySegment b
  ) {
    return LlamaRuntime.llama_h("ggml_mul", S3, ctx, a, b);
  }

  static MemorySegment scale(MemorySegment ctx, MemorySegment a, float s) {
    return LlamaRuntime.llama_h(
      "ggml_scale",
      new Class[] { S, S, float.class },
      ctx,
      a,
      s
    );
  }

  static MemorySegment softMax(MemorySegment ctx, MemorySegment a) {
    return LlamaRuntime.llama_h(
      "ggml_soft_max_ext",
      new Class[] { S, S, S, float.class, float.class },
      ctx,
      a,
      MemorySegment.NULL,
      1.0f,
      0.0f
    );
  }

  static MemorySegment norm(MemorySegment ctx, MemorySegment a, float eps) {
    return LlamaRuntime.llama_h(
      "ggml_norm",
      new Class[] { S, S, float.class },
      ctx,
      a,
      eps
    );
  }

  static MemorySegment l2Norm(MemorySegment ctx, MemorySegment a, float eps) {
    return LlamaRuntime.llama_h(
      "ggml_l2_norm",
      new Class[] { S, S, float.class },
      ctx,
      a,
      eps
    );
  }

  static MemorySegment geluErf(MemorySegment ctx, MemorySegment a) {
    return LlamaRuntime.llama_h("ggml_gelu_erf", S2, ctx, a);
  }

  static MemorySegment relu(MemorySegment ctx, MemorySegment a) {
    return LlamaRuntime.llama_h("ggml_relu", S2, ctx, a);
  }

  static MemorySegment sub(
    MemorySegment ctx,
    MemorySegment a,
    MemorySegment b
  ) {
    return LlamaRuntime.llama_h("ggml_sub", S3, ctx, a, b);
  }

  static MemorySegment reshape4d(
    MemorySegment ctx,
    MemorySegment a,
    long ne0,
    long ne1,
    long ne2,
    long ne3
  ) {
    return LlamaRuntime.llama_h(
      "ggml_reshape_4d",
      new Class[] { S, S, long.class, long.class, long.class, long.class },
      ctx,
      a,
      ne0,
      ne1,
      ne2,
      ne3
    );
  }

  static MemorySegment silu(MemorySegment ctx, MemorySegment a) {
    return LlamaRuntime.llama_h("ggml_silu", S2, ctx, a);
  }

  /** Sums along ne0: {@code (ne0, ne1, …) → (1, ne1, …)}. */
  static MemorySegment sumRows(MemorySegment ctx, MemorySegment a) {
    return LlamaRuntime.llama_h("ggml_sum_rows", S2, ctx, a);
  }

  /** Softmax over ne0 of {@code a·scale + mask} ({@code mask} may be {@code MemorySegment.NULL}; it broadcasts over ne2/ne3). */
  static MemorySegment softMaxExt(
    MemorySegment ctx,
    MemorySegment a,
    MemorySegment mask,
    float scale
  ) {
    return LlamaRuntime.llama_h(
      "ggml_soft_max_ext",
      new Class[] { S, S, S, float.class, float.class },
      ctx,
      a,
      mask,
      scale,
      0.0f
    );
  }

  static MemorySegment newTensor3d(
    MemorySegment ctx,
    int type,
    long ne0,
    long ne1,
    long ne2
  ) {
    return LlamaRuntime.llama_h(
      "ggml_new_tensor_3d",
      new Class[] { S, int.class, long.class, long.class, long.class },
      ctx,
      type,
      ne0,
      ne1,
      ne2
    );
  }

  static MemorySegment sigmoid(MemorySegment ctx, MemorySegment a) {
    return LlamaRuntime.llama_h("ggml_sigmoid", S2, ctx, a);
  }

  static MemorySegment tanh(MemorySegment ctx, MemorySegment a) {
    return LlamaRuntime.llama_h("ggml_tanh", S2, ctx, a);
  }

  /** {@code a} with the rows named by {@code idx} (I64) replaced by the rows of {@code b}; returns a view of {@code a}. */
  static MemorySegment setRows(
    MemorySegment ctx,
    MemorySegment a,
    MemorySegment b,
    MemorySegment idx
  ) {
    return LlamaRuntime.llama_h(
      "ggml_set_rows",
      new Class[] { S, S, S, S },
      ctx,
      a,
      b,
      idx
    );
  }

  static MemorySegment view1d(
    MemorySegment ctx,
    MemorySegment a,
    long ne0,
    long offset
  ) {
    return LlamaRuntime.llama_h(
      "ggml_view_1d",
      new Class[] { S, S, long.class, long.class },
      ctx,
      a,
      ne0,
      offset
    );
  }

  static MemorySegment getRows(
    MemorySegment ctx,
    MemorySegment a,
    MemorySegment idx
  ) {
    return LlamaRuntime.llama_h("ggml_get_rows", S3, ctx, a, idx);
  }

  static MemorySegment cont(MemorySegment ctx, MemorySegment a) {
    return LlamaRuntime.llama_h("ggml_cont", S2, ctx, a);
  }

  static MemorySegment permute(
    MemorySegment ctx,
    MemorySegment a,
    int a0,
    int a1,
    int a2,
    int a3
  ) {
    return LlamaRuntime.llama_h(
      "ggml_permute",
      new Class[] { S, S, int.class, int.class, int.class, int.class },
      ctx,
      a,
      a0,
      a1,
      a2,
      a3
    );
  }

  static MemorySegment reshape2d(
    MemorySegment ctx,
    MemorySegment a,
    long ne0,
    long ne1
  ) {
    return LlamaRuntime.llama_h(
      "ggml_reshape_2d",
      new Class[] { S, S, long.class, long.class },
      ctx,
      a,
      ne0,
      ne1
    );
  }

  static MemorySegment reshape3d(
    MemorySegment ctx,
    MemorySegment a,
    long ne0,
    long ne1,
    long ne2
  ) {
    return LlamaRuntime.llama_h(
      "ggml_reshape_3d",
      new Class[] { S, S, long.class, long.class, long.class },
      ctx,
      a,
      ne0,
      ne1,
      ne2
    );
  }

  static MemorySegment view2d(
    MemorySegment ctx,
    MemorySegment a,
    long ne0,
    long ne1,
    long nb1,
    long offset
  ) {
    return LlamaRuntime.llama_h(
      "ggml_view_2d",
      new Class[] { S, S, long.class, long.class, long.class, long.class },
      ctx,
      a,
      ne0,
      ne1,
      nb1,
      offset
    );
  }

  static MemorySegment concat(
    MemorySegment ctx,
    MemorySegment a,
    MemorySegment b,
    int dim
  ) {
    return LlamaRuntime.llama_h(
      "ggml_concat",
      new Class[] { S, S, S, int.class },
      ctx,
      a,
      b,
      dim
    );
  }

  static MemorySegment repeat(
    MemorySegment ctx,
    MemorySegment a,
    MemorySegment b
  ) {
    return LlamaRuntime.llama_h("ggml_repeat", S3, ctx, a, b);
  }

  // ---- graphs & backends ---------------------------------------------------

  static MemorySegment newGraph(MemorySegment ctx, long size) {
    return LlamaRuntime.llama_h(
      "ggml_new_graph_custom",
      new Class[] { S, long.class, boolean.class },
      ctx,
      size,
      false
    );
  }

  static void buildForwardExpand(MemorySegment graph, MemorySegment t) {
    LlamaRuntime.llama_h("ggml_build_forward_expand", S2, graph, t);
  }

  static MemorySegment gpuDevice() {
    return LlamaRuntime.llama_h(
      "ggml_backend_dev_by_type",
      new Class[] { int.class },
      deviceTypeGpu()
    );
  }

  static String deviceName(MemorySegment dev) {
    MemorySegment s = LlamaRuntime.llama_h("ggml_backend_dev_name", S1, dev);
    return s.reinterpret(Long.MAX_VALUE).getString(0);
  }

  static MemorySegment deviceInit(MemorySegment dev) {
    return LlamaRuntime.llama_h(
      "ggml_backend_dev_init",
      S2,
      dev,
      MemorySegment.NULL
    );
  }

  static MemorySegment cpuInit() {
    return LlamaRuntime.llama_h("ggml_backend_cpu_init", new Class[0]);
  }

  static void cpuSetThreads(MemorySegment backend, int n) {
    LlamaRuntime.llama_h(
      "ggml_backend_cpu_set_n_threads",
      new Class[] { S, int.class },
      backend,
      n
    );
  }

  static void backendFree(MemorySegment backend) {
    LlamaRuntime.llama_h("ggml_backend_free", S1, backend);
  }

  static MemorySegment allocCtxTensors(
    MemorySegment ctx,
    MemorySegment backend
  ) {
    return LlamaRuntime.llama_h(
      "ggml_backend_alloc_ctx_tensors",
      S2,
      ctx,
      backend
    );
  }

  static void bufferFree(MemorySegment buffer) {
    LlamaRuntime.llama_h("ggml_backend_buffer_free", S1, buffer);
  }

  static MemorySegment gallocrNew(MemorySegment backend) {
    MemorySegment buft = LlamaRuntime.llama_h(
      "ggml_backend_get_default_buffer_type",
      S1,
      backend
    );
    return LlamaRuntime.llama_h("ggml_gallocr_new", S1, buft);
  }

  static boolean gallocrAllocGraph(MemorySegment galloc, MemorySegment graph) {
    return LlamaRuntime.llama_h("ggml_gallocr_alloc_graph", S2, galloc, graph);
  }

  static void gallocrFree(MemorySegment galloc) {
    LlamaRuntime.llama_h("ggml_gallocr_free", S1, galloc);
  }

  static int graphCompute(MemorySegment backend, MemorySegment graph) {
    return LlamaRuntime.llama_h(
      "ggml_backend_graph_compute",
      S2,
      backend,
      graph
    );
  }

  static void tensorSet(MemorySegment t, MemorySegment data, long size) {
    LlamaRuntime.llama_h(
      "ggml_backend_tensor_set",
      new Class[] { S, S, long.class, long.class },
      t,
      data,
      0L,
      size
    );
  }

  static void tensorGet(MemorySegment t, MemorySegment data, long size) {
    LlamaRuntime.llama_h(
      "ggml_backend_tensor_get",
      new Class[] { S, S, long.class, long.class },
      t,
      data,
      0L,
      size
    );
  }

  static void setFloats(MemorySegment t, Arena arena, float[] values) {
    var seg = arena.allocate((long) values.length * Float.BYTES);
    MemorySegment.copy(
      values,
      0,
      seg,
      ValueLayout.JAVA_FLOAT,
      0,
      values.length
    );
    tensorSet(t, seg, seg.byteSize());
  }

  static void setLongs(MemorySegment t, Arena arena, long[] values) {
    var seg = arena.allocateFrom(ValueLayout.JAVA_LONG, values);
    tensorSet(t, seg, (long) values.length * Long.BYTES);
  }

  static void setInts(MemorySegment t, Arena arena, int[] values) {
    var seg = arena.allocate((long) values.length * Integer.BYTES);
    MemorySegment.copy(values, 0, seg, ValueLayout.JAVA_INT, 0, values.length);
    tensorSet(t, seg, seg.byteSize());
  }

  static float[] getFloats(MemorySegment t, Arena arena, int count) {
    var seg = arena.allocate((long) count * Float.BYTES);
    tensorGet(t, seg, seg.byteSize());
    return seg.toArray(ValueLayout.JAVA_FLOAT);
  }

  // ---- gguf ----------------------------------------------------------------

  /** Loads a GGUF with its tensor data into a fresh CPU ggml context: returns {gguf, ctx}. */
  static MemorySegment[] ggufOpen(Arena arena, String path) {
    MemorySegment params = LlamaRuntime.invoke(
      "gguf_init_params",
      "allocate",
      new Class[] { SegmentAllocator.class },
      arena
    );
    var ctxSlot = arena.allocate(ValueLayout.ADDRESS);
    LlamaRuntime.invoke(
      "gguf_init_params",
      "no_alloc",
      new Class[] { S, boolean.class },
      params,
      false
    );
    LlamaRuntime.invoke("gguf_init_params", "ctx", S2, params, ctxSlot);
    MemorySegment gguf = LlamaRuntime.llama_h(
      "gguf_init_from_file",
      S2,
      arena.allocateFrom(path),
      params
    );
    if (gguf.address() == 0) {
      throw new IllegalStateException("gguf_init_from_file failed for " + path);
    }
    return new MemorySegment[] { gguf, ctxSlot.get(ValueLayout.ADDRESS, 0) };
  }

  static void ggufFree(MemorySegment gguf) {
    LlamaRuntime.llama_h("gguf_free", S1, gguf);
  }

  /** Scalar GGUF metadata (uint32 / float32 / bool / string) as a map; other value types are skipped. */
  static java.util.Map<String, Object> ggufMetadata(MemorySegment gguf) {
    var out = new java.util.LinkedHashMap<String, Object>();
    long n = LlamaRuntime.llama_h("gguf_get_n_kv", S1, gguf);
    var sig = new Class[] { S, long.class };
    for (long i = 0; i < n; i++) {
      MemorySegment keySeg = LlamaRuntime.llama_h("gguf_get_key", sig, gguf, i);
      var key = keySeg.reinterpret(Long.MAX_VALUE).getString(0);
      int type = LlamaRuntime.llama_h("gguf_get_kv_type", sig, gguf, i);
      switch (type) {
        case 4 -> out.put(
          key,
          (int) LlamaRuntime.llama_h("gguf_get_val_u32", sig, gguf, i)
        );
        case 6 -> out.put(
          key,
          (float) LlamaRuntime.llama_h("gguf_get_val_f32", sig, gguf, i)
        );
        case 7 -> out.put(
          key,
          (boolean) LlamaRuntime.llama_h("gguf_get_val_bool", sig, gguf, i)
        );
        case 8 -> {
          MemorySegment v = LlamaRuntime.llama_h(
            "gguf_get_val_str",
            sig,
            gguf,
            i
          );
          out.put(key, v.reinterpret(Long.MAX_VALUE).getString(0));
        }
        default -> {}
      }
    }
    return out;
  }
}
