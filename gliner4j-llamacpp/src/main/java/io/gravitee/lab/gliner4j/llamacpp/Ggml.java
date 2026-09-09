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

  static int typeF16() {
    return LlamaRuntime.llama_h("GGML_TYPE_F16", new Class[0]);
  }

  static int typeQ8_0() {
    return LlamaRuntime.llama_h("GGML_TYPE_Q8_0", new Class[0]);
  }

  static int precF32() {
    return LlamaRuntime.llama_h("GGML_PREC_F32", new Class[0]);
  }

  private static final java.util.concurrent.ConcurrentHashMap<
    String,
    java.lang.invoke.MethodHandle
  > RAW = new java.util.concurrent.ConcurrentHashMap<>();

  /**
   * A downcall to a ggml symbol the jextract bindings do not cover, resolved from the
   * {@code System.load}ed libggml / libggml-base (see {@code LlamaLibLoader}).
   */
  private static java.lang.invoke.MethodHandle raw(
    String symbol,
    java.lang.foreign.FunctionDescriptor fd
  ) {
    return RAW.computeIfAbsent(symbol, s ->
      java.lang.foreign.Linker.nativeLinker().downcallHandle(
        java.lang.foreign.SymbolLookup.loaderLookup()
          .find(s)
          .orElseThrow(() ->
            new IllegalStateException(s + " not found in loaded ggml libraries")
          ),
        fd
      )
    );
  }

  /** {@code ggml_cast(ctx, a, type)} — a typed copy (works on non-contiguous views). */
  static MemorySegment cast(MemorySegment ctx, MemorySegment a, int type) {
    try {
      return (MemorySegment) raw(
        "ggml_cast",
        java.lang.foreign.FunctionDescriptor.of(
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_INT
        )
      ).invokeExact(ctx, a, type);
    } catch (RuntimeException e) {
      throw e;
    } catch (Throwable t) {
      throw new IllegalStateException("ggml_cast failed", t);
    }
  }

  /**
   * Fused attention {@code softmax(scale·qkᵀ + mask)·v}. {@code q (d, n_q, heads, ne3)},
   * {@code k / v (d, n_kv, heads_kv, ne3)} (v NOT transposed), {@code mask (n_kv, n_q, ne32, ne33)}
   * F16 contiguous, result {@code (d, heads, n_q, ne3)} (permuted). On CUDA the mask must not
   * carry a head dimension ({@code ne32 == 1}).
   */
  static MemorySegment flashAttnExt(
    MemorySegment ctx,
    MemorySegment q,
    MemorySegment k,
    MemorySegment v,
    MemorySegment mask,
    float scale
  ) {
    return LlamaRuntime.llama_h(
      "ggml_flash_attn_ext",
      new Class[] { S, S, S, S, S, float.class, float.class, float.class },
      ctx,
      q,
      k,
      v,
      mask,
      scale,
      0.0f,
      0.0f
    );
  }

  /**
   * {@code ggml_mul_mat_set_prec(t, GGML_PREC_F32)}: with an f16 {@code src0} CUDA then runs the
   * tensor-core GEMM with f32 accumulation and an f32 result, instead of an f16 result plus a
   * separate f16→f32 conversion pass.
   */
  static MemorySegment mulMatPrec(MemorySegment t, int prec) {
    try {
      raw(
        "ggml_mul_mat_set_prec",
        java.lang.foreign.FunctionDescriptor.ofVoid(
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_INT
        )
      ).invokeExact(t, prec);
      return t;
    } catch (RuntimeException e) {
      throw e;
    } catch (Throwable x) {
      throw new IllegalStateException("ggml_mul_mat_set_prec failed", x);
    }
  }

  static void flashAttnExtSetPrec(MemorySegment t, int prec) {
    try {
      raw(
        "ggml_flash_attn_ext_set_prec",
        java.lang.foreign.FunctionDescriptor.ofVoid(
          ValueLayout.ADDRESS,
          ValueLayout.JAVA_INT
        )
      ).invokeExact(t, prec);
    } catch (RuntimeException e) {
      throw e;
    } catch (Throwable x) {
      throw new IllegalStateException("ggml_flash_attn_ext_set_prec failed", x);
    }
  }

  // ---- backend registry, plugins, scheduler ---------------------------------

  /** {@code ggml_backend_load(path)}: loads a backend plugin; NULL when rejected (missing symbol, API version). */
  static MemorySegment backendLoad(String path) {
    try (var a = Arena.ofConfined()) {
      return LlamaRuntime.llama_h(
        "ggml_backend_load",
        S1,
        a.allocateFrom(path)
      );
    }
  }

  static MemorySegment regByName(String name) {
    try (var a = Arena.ofConfined()) {
      return LlamaRuntime.llama_h(
        "ggml_backend_reg_by_name",
        S1,
        a.allocateFrom(name)
      );
    }
  }

  static String regName(MemorySegment reg) {
    MemorySegment s = LlamaRuntime.llama_h("ggml_backend_reg_name", S1, reg);
    return s.reinterpret(Long.MAX_VALUE).getString(0);
  }

  static MemorySegment regDevGet(MemorySegment reg, long index) {
    return LlamaRuntime.llama_h(
      "ggml_backend_reg_dev_get",
      new Class[] { S, long.class },
      reg,
      index
    );
  }

  /** {@code ggml_backend_reg_get_proc_address}: a function pointer exported by a backend plugin, or NULL. */
  static MemorySegment regGetProcAddress(MemorySegment reg, String name) {
    try (var a = Arena.ofConfined()) {
      return (MemorySegment) raw(
        "ggml_backend_reg_get_proc_address",
        java.lang.foreign.FunctionDescriptor.of(
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS
        )
      ).invokeExact(reg, a.allocateFrom(name));
    } catch (RuntimeException e) {
      throw e;
    } catch (Throwable t) {
      throw new IllegalStateException(
        "ggml_backend_reg_get_proc_address failed",
        t
      );
    }
  }

  /** Calls a {@code const char * (*)(void)} exported through {@link #regGetProcAddress}. */
  static String callStringFn(MemorySegment fn) {
    try {
      var h = java.lang.foreign.Linker.nativeLinker().downcallHandle(
        fn,
        java.lang.foreign.FunctionDescriptor.of(ValueLayout.ADDRESS)
      );
      var s = (MemorySegment) h.invokeExact();
      return s.reinterpret(Long.MAX_VALUE).getString(0);
    } catch (RuntimeException e) {
      throw e;
    } catch (Throwable t) {
      throw new IllegalStateException("native string function failed", t);
    }
  }

  /**
   * {@code ggml_custom_4d}: an opaque op computed by whichever backend recognises {@code fun}
   * (see the DEBERTA plugin). {@code args} are copied into the node's sources by ggml.
   */
  static MemorySegment custom4d(
    MemorySegment ctx,
    int type,
    long ne0,
    long ne1,
    long ne2,
    long ne3,
    MemorySegment[] args,
    MemorySegment fun,
    int nTasks,
    MemorySegment userdata
  ) {
    try (var a = Arena.ofConfined()) {
      var arr = a.allocate(ValueLayout.ADDRESS, args.length);
      for (int i = 0; i < args.length; i++) {
        arr.setAtIndex(ValueLayout.ADDRESS, i, args[i]);
      }
      return LlamaRuntime.llama_h(
        "ggml_custom_4d",
        new Class[] {
          S,
          int.class,
          long.class,
          long.class,
          long.class,
          long.class,
          S,
          int.class,
          S,
          int.class,
          S,
        },
        ctx,
        type,
        ne0,
        ne1,
        ne2,
        ne3,
        arr,
        args.length,
        fun,
        nTasks,
        userdata
      );
    }
  }

  /** {@code ggml_backend_sched_new(backends, NULL, n, graphSize, parallel, opOffload)}; the last backend must be the CPU one. */
  /** Whether a symbol is exported by the loaded ggml libraries (feature detection, e.g. patched schedulers). */
  static boolean hasSymbol(String name) {
    return java.lang.foreign.SymbolLookup.loaderLookup().find(name).isPresent();
  }

  /** Calls a {@code void (*)(bool)} exported through {@link #regGetProcAddress}. */
  static void callBoolFn(MemorySegment fn, boolean value) {
    try {
      java.lang.foreign.Linker.nativeLinker()
        .downcallHandle(
          fn,
          java.lang.foreign.FunctionDescriptor.ofVoid(ValueLayout.JAVA_BOOLEAN)
        )
        .invokeExact(value);
    } catch (RuntimeException e) {
      throw e;
    } catch (Throwable t) {
      throw new IllegalStateException("native bool function failed", t);
    }
  }

  static MemorySegment schedNew(
    Arena arena,
    MemorySegment[] backends,
    long graphSize
  ) {
    return schedNew(arena, backends, graphSize, false);
  }

  /** {@code parallel} allocates per-backend events (4 input copies) — needed for stream-side cross-backend ordering. */
  static MemorySegment schedNew(
    Arena arena,
    MemorySegment[] backends,
    long graphSize,
    boolean parallel
  ) {
    var arr = arena.allocate(ValueLayout.ADDRESS, backends.length);
    for (int i = 0; i < backends.length; i++) {
      arr.setAtIndex(ValueLayout.ADDRESS, i, backends[i]);
    }
    return LlamaRuntime.llama_h(
      "ggml_backend_sched_new",
      new Class[] { S, S, int.class, long.class, boolean.class, boolean.class },
      arr,
      MemorySegment.NULL,
      backends.length,
      graphSize,
      parallel,
      false
    );
  }

  static void schedFree(MemorySegment sched) {
    LlamaRuntime.llama_h("ggml_backend_sched_free", S1, sched);
  }

  static void schedReset(MemorySegment sched) {
    LlamaRuntime.llama_h("ggml_backend_sched_reset", S1, sched);
  }

  static boolean schedAllocGraph(MemorySegment sched, MemorySegment graph) {
    return LlamaRuntime.llama_h(
      "ggml_backend_sched_alloc_graph",
      S2,
      sched,
      graph
    );
  }

  static int schedGraphCompute(MemorySegment sched, MemorySegment graph) {
    return LlamaRuntime.llama_h(
      "ggml_backend_sched_graph_compute",
      S2,
      sched,
      graph
    );
  }

  static int schedNumSplits(MemorySegment sched) {
    return LlamaRuntime.llama_h("ggml_backend_sched_get_n_splits", S1, sched);
  }

  /** {@code ggml_view_4d}: strides {@code nb1..nb3} and {@code offset} in bytes; {@code nb0} is the element size. */
  static MemorySegment view4d(
    MemorySegment ctx,
    MemorySegment a,
    long ne0,
    long ne1,
    long ne2,
    long ne3,
    long nb1,
    long nb2,
    long nb3,
    long offset
  ) {
    return LlamaRuntime.llama_h(
      "ggml_view_4d",
      new Class[] {
        S,
        S,
        long.class,
        long.class,
        long.class,
        long.class,
        long.class,
        long.class,
        long.class,
        long.class,
      },
      ctx,
      a,
      ne0,
      ne1,
      ne2,
      ne3,
      nb1,
      nb2,
      nb3,
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

  static int deviceTypeCpu() {
    return LlamaRuntime.llama_h("GGML_BACKEND_DEVICE_TYPE_CPU", new Class[0]);
  }

  /**
   * The CPU backend through the device registry. {@code ggml_backend_cpu_init} lives in the
   * {@code libggml-cpu*} plugin, which is dlopen'ed by {@code ggml_backend_load_all_from_path}
   * (RTLD_LOCAL) and never {@code System.load}ed, so a direct symbol lookup fails on every build
   * with {@code GGML_BACKEND_DL=ON} (the Linux prebuilt natives and any custom CUDA build).
   */
  static MemorySegment cpuDevice() {
    MemorySegment dev = LlamaRuntime.llama_h(
      "ggml_backend_dev_by_type",
      new Class[] { int.class },
      deviceTypeCpu()
    );
    if (dev.address() == 0) {
      throw new IllegalStateException(
        "ggml CPU backend not registered — is libggml-cpu*.so next to libggml.so?"
      );
    }
    return dev;
  }

  static MemorySegment cpuInit() {
    return deviceInit(cpuDevice());
  }

  /**
   * Backend-agnostic {@code ggml_backend_set_n_threads}: the CPU plugin exports it only through
   * {@code ggml_backend_reg_get_proc_address} (libggml-base, which is {@code System.load}ed), and
   * the returned function pointer is called with a plain FFM downcall.
   */
  static void cpuSetThreads(MemorySegment backend, int n) {
    try {
      MemorySegment reg = LlamaRuntime.llama_h(
        "ggml_backend_dev_backend_reg",
        S1,
        cpuDevice()
      );
      var linker = java.lang.foreign.Linker.nativeLinker();
      var getProc = linker.downcallHandle(
        java.lang.foreign.SymbolLookup.loaderLookup()
          .find("ggml_backend_reg_get_proc_address")
          .orElseThrow(() ->
            new IllegalStateException(
              "ggml_backend_reg_get_proc_address not found in loaded libggml-base"
            )
          ),
        java.lang.foreign.FunctionDescriptor.of(
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS,
          ValueLayout.ADDRESS
        )
      );
      try (var a = Arena.ofConfined()) {
        var fn = (MemorySegment) getProc.invokeExact(
          reg,
          a.allocateFrom("ggml_backend_set_n_threads")
        );
        if (fn.address() == 0) {
          return; // backend has no thread control; keep its default
        }
        var setThreads = linker.downcallHandle(
          fn,
          java.lang.foreign.FunctionDescriptor.ofVoid(
            ValueLayout.ADDRESS,
            ValueLayout.JAVA_INT
          )
        );
        setThreads.invokeExact(backend, n);
      }
    } catch (RuntimeException e) {
      throw e;
    } catch (Throwable t) {
      throw new IllegalStateException("ggml_backend_set_n_threads failed", t);
    }
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

  /** {@code ggml_backend_tensor_copy(src, dst)}: same-shape copy, device-to-device when both live on the backend. */
  static void tensorCopy(MemorySegment src, MemorySegment dst) {
    LlamaRuntime.llama_h("ggml_backend_tensor_copy", S2, src, dst);
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
