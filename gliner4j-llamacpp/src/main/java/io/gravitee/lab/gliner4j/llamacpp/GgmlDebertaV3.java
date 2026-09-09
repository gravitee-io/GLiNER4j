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

/**
 * A full DeBERTa-v2/v3 encoder (embeddings + disentangled-attention stack) as a ggml graph over
 * token ids — the backbone of GLiNER2 / GLiNER2.5 / original-GLiNER bundles, which llama.cpp cannot
 * run natively. Weights come from {@code encoder.gguf} written by
 * {@code export_llamacpp.py} ({@code export_deberta_encoder_gguf}), which already folds the
 * rel-embedding LayerNorm and, since {@code share_att_key=true}, the per-layer position keys /
 * queries ({@code layer.N.pos_key} / {@code pos_query}, {@code 2·att_span} rows). Disentangled
 * attention after DeBERTa (He et al.), <a href="https://github.com/microsoft/DeBERTa">microsoft/DeBERTa</a>
 * (MIT) — reimplemented here, no code shared.
 *
 * <p>Differences from the 2-layer scorer encoder in {@link GgmlDeberta}: word embeddings + LayerNorm
 * in front (no absolute positions), log-bucketed relative positions ({@code position_buckets=256}:
 * identity for {@code |i−j| ≤ 128}, logarithmic beyond), and content/position scores all scaled by
 * {@code 1/sqrt(3·d)}.
 *
 * <h2>Disentangled attention without the big gather</h2>
 *
 * <p>HF computes {@code c2p[i][j] = q_i · posKey[bucket(i−j)]} with a {@code gather} over an
 * {@code n×n×heads} index — on ggml that is {@code n·n·heads} one-element {@code get_rows} per
 * layer, which is what dominated CUDA time (96% in a 512-token profile). Instead:
 *
 * <ol>
 *   <li>Expand the {@code 2·att_span}-row position table once per layer to one row per relative
 *       offset {@code d = i−j ∈ [−(n−1), n−1]} with a cheap {@code get_rows} of {@code 2n−1}
 *       rows ({@code posIdx[d + n − 1] = bucket(d)}). The position scores
 *       {@code (2n−1, n)} are then linear in {@code i−j}.</li>
 *   <li>Feed keys / values in reversed sequence order ({@code j' = n−1−j}). With that,
 *       {@code c2p[j'][i]} sits at flat index {@code i·2n + j'} of the position-score matrix: a
 *       strided {@link Ggml#view4d} with row stride {@code 2n}, no gather. Same for {@code p2c}
 *       up to a transpose. Softmax and the {@code probs·V} product are order-invariant as long
 *       as {@code V} (and the padding mask) use the same reversed order.</li>
 *   <li>Run {@code softmax(scale·q·kᵀ + bias)·v} as one {@code ggml_flash_attn_ext} with
 *       {@code bias = scale·(c2p + p2c)} (+ padding) as its F16 mask, instead of a materialised
 *       {@code n×n×heads} score matrix, a separate softmax and two more copies. CUDA's flash
 *       kernels refuse a per-head mask, so heads are folded into the batch axis (one head,
 *       {@code heads·B} sequences) — see {@link #layer}.</li>
 * </ol>
 *
 * <h2>Fused kernel (DEBERTA plugin)</h2>
 *
 * <p>With the out-of-tree {@code libggml-deberta} backend loaded ({@link LlamaBackbone}) and the
 * weights on CUDA, {@link #fusedLayer} replaces the whole position-score / bias / flash-attention
 * section by one {@code ggml_custom_4d} node: {@code softmax(scale·q·kᵀ + q·P[i−j] + k·Qp[i−j])·v}
 * computed inside a shared-memory tile, nothing n² ever materialised. Its inputs are f16 head
 * views of the fused qkv projection and the per-length, padded f16 position tables kept in
 * {@link PosTables} across calls. {@code GgmlWeights} then runs graphs through a
 * {@code ggml_backend_sched} over [DEBERTA, CUDA, CPU]; the plugin shares CUDA's buffer type, so
 * no copies happen at the split boundaries. CPU, Metal and plugin-less runs keep the graph above.
 *
 * <p>On CUDA the cost fell in stages: strided views instead of the n² gather, then flash attention,
 * then f16 heads / bias with per-head tables and a fused qkv projection, then the fused kernel —
 * which lands below ONNX Runtime CUDA fp16 on the same call.
 */
final class GgmlDebertaV3 {

  /** Graph handles the caller feeds after allocation. */
  record Built(
    MemorySegment output,
    MemorySegment ids,
    MemorySegment posIdx,
    MemorySegment revIdx,
    MemorySegment mask,
    MemorySegment lens,
    int n,
    int batch
  ) {}

  private final GgmlWeights w;
  private final String prefix;
  private final int hidden;
  private final int heads;
  private final int headSize;
  private final int layers;
  private final int attSpan;
  private final int buckets;
  private final int maxRel;
  private final float eps;
  /** Position tables stored {@code (head_dim, 2·att_span, heads)} and pre-scaled (export ≥ pos_per_head). */
  private final boolean posPerHead;

  private static final float MASKED = -1e9f;
  /** Padding rows on each side of the position tables the fused kernel reads (mirrors GLINER4J_DEBERTA_POS_PAD). */
  static final int POS_PAD = 96;

  GgmlDebertaV3(GgmlWeights w) {
    this(w, "");
  }

  /** @param prefix tensor-name prefix inside a merged GGUF (e.g. {@code "enc."}); metadata keys are unprefixed */
  GgmlDebertaV3(GgmlWeights w, String prefix) {
    this.w = w;
    this.prefix = prefix;
    this.hidden = w.metaInt(
      "deberta.hidden_size",
      (int) Ggml.ne(w.get(prefix + "embeddings.word_embeddings.weight"), 0)
    );
    this.layers = w.metaInt("deberta.num_layers", -1);
    this.heads = w.metaInt("deberta.num_heads", hidden / 64);
    this.headSize = hidden / heads;
    this.buckets = w.metaInt("deberta.position_buckets", 256);
    this.maxRel = w.metaInt("deberta.max_relative_positions", 512);
    this.attSpan = w.metaInt(
      "deberta.att_span",
      buckets > 0 ? buckets : maxRel
    );
    this.eps = w.metaFloat("deberta.layer_norm_eps", 1e-7f);
    this.posPerHead = w.metaInt("deberta.pos_per_head", 0) == 1;
    if (layers < 0 || !w.has(prefix + "layer.0.pos_key")) {
      throw new IllegalStateException(
        "encoder.gguf is not a gliner4j-deberta-v3 export (missing metadata / layer.0.pos_key)"
      );
    }
  }

  int hidden() {
    return hidden;
  }

  int layers() {
    return layers;
  }

  /** Approximate graph-node count of {@link #build}, for sizing the graph context. */
  int graphNodes() {
    return 96 * layers + 32;
  }

  /** Builds the encoder over {@code n} token ids (fed later through {@link #feed}). */
  Built build(MemorySegment ctx, int n) {
    return build(ctx, n, 1);
  }

  /**
   * Builds the encoder over {@code batch} rows of {@code n} token ids each (shorter rows are
   * padded and their padding keys masked out). The output is {@code (hidden, n·batch)}: row
   * {@code b}'s token {@code i} is column {@code i + n·b}.
   */
  Built build(MemorySegment ctx, int n, int batch) {
    var ids = Ggml.newTensor1d(ctx, w.typeI32, (long) n * batch);
    long win = 2L * n - 1;
    // fused: the padded f16 tables come from the per-length cache, no gather in this graph
    MemorySegment posIdx = null;
    if (!fused()) {
      posIdx = posPerHead
        ? Ggml.newTensor2d(ctx, w.typeI32, win, heads) // get_rows index per head
        : Ggml.newTensor1d(ctx, w.typeI32, win);
    }
    // The fused kernel indexes i − j itself, so the reversed-key gather index is not part of its
    // graph (an input no node reads is never allocated, and feeding it would abort).
    var revIdx = fused()
      ? null
      : Ggml.newTensor1d(ctx, w.typeI32, (long) n * batch);
    Ggml.setInput(ids);
    if (posIdx != null) Ggml.setInput(posIdx);
    if (revIdx != null) Ggml.setInput(revIdx);
    // valid tokens per row for the fused attention kernel (its padding mask); only part of the
    // fused graph — an input no node reads is never allocated, and feeding it would abort
    MemorySegment lens = null;
    if (fused()) {
      lens = Ggml.newTensor1d(ctx, w.typeI32, batch);
      Ggml.setInput(lens);
    }
    MemorySegment mask = null;
    if (batch > 1 && !fused()) {
      mask = Ggml.reshape4d(
        ctx,
        Ggml.newTensor3d(ctx, w.typeF32, n, n, batch),
        n,
        n,
        1,
        batch
      );
      Ggml.setInput(mask);
    }
    var h = layerNorm(
      ctx,
      Ggml.getRows(
        ctx,
        w.get(prefix + "embeddings.word_embeddings.weight"),
        ids
      ),
      prefix + "embeddings.LayerNorm"
    );
    float attnScale = (float) (1.0 / Math.sqrt(headSize * 3.0));
    // The attention bias is assembled in f16 (flash attention wants an F16 mask anyway), so the
    // padding mask is converted once per graph rather than once per layer.
    var mask16 = mask == null ? null : Ggml.cast(ctx, mask, Ggml.typeF16());
    for (int l = 0; l < layers; l++) {
      h = fused()
        ? fusedLayer(ctx, h, l, n, batch, posIdx, lens, attnScale)
        : layer(ctx, h, l, n, batch, posIdx, revIdx, mask16, attnScale);
    }
    return new Built(h, ids, posIdx, revIdx, mask, lens, n, batch);
  }

  /** Feeds token ids and the relative-position / key-order indices once the graph is allocated. */
  void feed(Built b, Arena call, long[] tokenIds) {
    feed(b, call, new long[][] { tokenIds });
  }

  /** Batched feed: rows shorter than {@code n} are padded with id 0 and masked as keys. */
  void feed(Built b, Arena call, long[][] rows) {
    int n = b.n();
    int batch = b.batch();
    if (rows.length != batch) {
      throw new IllegalArgumentException(
        "expected " + batch + " rows, got " + rows.length
      );
    }
    var ids = new int[n * batch];
    var rev = new int[n * batch];
    for (int r = 0; r < batch; r++) {
      for (int i = 0; i < rows[r].length; i++) ids[r * n + i] =
        (int) rows[r][i];
      for (int j = 0; j < n; j++) rev[r * n + j] = r * n + (n - 1 - j);
    }
    Ggml.setInts(b.ids(), call, ids);
    if (b.revIdx() != null) Ggml.setInts(b.revIdx(), call, rev);
    if (b.lens() != null) {
      var lens = new int[batch];
      for (int r = 0; r < batch; r++) lens[r] = rows[r].length;
      Ggml.setInts(b.lens(), call, lens);
    }
    if (b.posIdx() != null) {
      Ggml.setInts(b.posIdx(), call, positionIndex(n, 0));
    }
    if (b.mask() != null) {
      // Keys run reversed (j' = n−1−j): padding keys j ≥ len are j' < n − len.
      var m = new float[n * n * batch];
      for (int r = 0; r < batch; r++) {
        int pad = n - rows[r].length;
        for (int q = 0; q < n; q++) {
          int base = (r * n + q) * n;
          for (int k = 0; k < pad; k++) m[base + k] = MASKED;
        }
      }
      Ggml.setFloats(b.mask(), call, m);
    }
  }

  /**
   * {@code get_rows} index of the per-offset table for length {@code n}: entry {@code d} (with
   * {@code pad} extra clamped rows on each side) is the bucketed table row of offset
   * {@code d − pad − (n − 1)}; replicated per head when the tables are stored per head.
   */
  private int[] positionIndex(int n, int pad) {
    int win = 2 * n - 1 + 2 * pad;
    var pos = new int[posPerHead ? win * heads : win];
    for (int d = 0; d < win; d++) pos[d] = relativeRow(d - pad - (n - 1), 0);
    if (posPerHead) {
      for (int hd = 1; hd < heads; hd++) System.arraycopy(
        pos,
        0,
        pos,
        hd * win,
        win
      );
    }
    return pos;
  }

  /**
   * f16, padded per-offset position tables of every layer for one sequence length, kept on the
   * backend across calls: they depend only on the length and the weights, and gathering plus
   * casting them per layer per call cost more than the fused attention itself. Tables built for
   * a length {@code n_tab} serve every {@code n ≤ n_tab} (the kernel derives the offset origin
   * from the table size), so lengths are bucketed.
   */
  private final class PosTables implements AutoCloseable {

    final Arena arena = Arena.ofShared();
    final MemorySegment ctx;
    final MemorySegment buffer;
    final MemorySegment[] key = new MemorySegment[layers];
    final MemorySegment[] query = new MemorySegment[layers];

    PosTables(int n) {
      long win = 2L * n - 1 + 2L * POS_PAD;
      ctx = Ggml.init(arena, Ggml.tensorOverhead() * (2L * layers + 8), true);
      int f16 = Ggml.typeF16();
      for (int l = 0; l < layers; l++) {
        key[l] = Ggml.newTensor3d(ctx, f16, headSize, win, heads);
        query[l] = Ggml.newTensor3d(ctx, f16, headSize, win, heads);
      }
      buffer = Ggml.allocCtxTensors(ctx, w.backend);
      if (buffer.address() == 0) {
        throw new IllegalStateException(
          "failed to allocate the position-table cache"
        );
      }
      // one graph gathers + casts every layer's tables, then they are copied into place
      try (var call = Arena.ofConfined()) {
        var gctx = w.newGraphContext(call, 8L * layers + 32);
        try {
          var posIdx = posPerHead
            ? Ggml.newTensor2d(gctx, w.typeI32, win, heads)
            : Ggml.newTensor1d(gctx, w.typeI32, win);
          Ggml.setInput(posIdx);
          var outs = new MemorySegment[2 * layers];
          var graph = Ggml.newGraph(gctx, 8L * layers + 32);
          for (int l = 0; l < layers; l++) {
            var p = prefix + "layer." + l + ".";
            outs[2 * l] = Ggml.cast(
              gctx,
              posTable(gctx, p + "pos_key", posIdx, win, 1f),
              f16
            );
            outs[2 * l + 1] = Ggml.cast(
              gctx,
              posTable(gctx, p + "pos_query", posIdx, win, 1f),
              f16
            );
            Ggml.setOutput(outs[2 * l]);
            Ggml.setOutput(outs[2 * l + 1]);
            Ggml.buildForwardExpand(graph, outs[2 * l]);
            Ggml.buildForwardExpand(graph, outs[2 * l + 1]);
          }
          w.alloc(graph);
          Ggml.setInts(posIdx, call, positionIndex(n, POS_PAD));
          w.run(graph);
          for (int l = 0; l < layers; l++) {
            Ggml.tensorCopy(outs[2 * l], key[l]);
            Ggml.tensorCopy(outs[2 * l + 1], query[l]);
          }
        } finally {
          Ggml.free(gctx);
        }
      }
    }

    @Override
    public void close() {
      Ggml.bufferFree(buffer);
      Ggml.free(ctx);
      arena.close();
    }
  }

  /** Cache key: lengths rounded up to this, so a corpus of varying lengths hits a few buckets. */
  private static final int POS_CACHE_BUCKET = 64;
  private static final int POS_CACHE_ENTRIES = 12;
  private final java.util.LinkedHashMap<Integer, PosTables> posCache =
    new java.util.LinkedHashMap<>(8, 0.75f, true) {
      @Override
      protected boolean removeEldestEntry(
        java.util.Map.Entry<Integer, PosTables> eldest
      ) {
        if (size() > POS_CACHE_ENTRIES) {
          eldest.getValue().close();
          return true;
        }
        return false;
      }
    };
  private boolean cacheHooked;

  private PosTables posTables(int n) {
    if (!cacheHooked) {
      w.onClose(() -> {
        posCache.values().forEach(PosTables::close);
        posCache.clear();
      });
      cacheHooked = true;
    }
    int bucket =
      ((n + POS_CACHE_BUCKET - 1) / POS_CACHE_BUCKET) * POS_CACHE_BUCKET;
    return posCache.computeIfAbsent(bucket, PosTables::new); // tables for n_tab >= n serve n
  }

  /**
   * HF {@code make_log_bucket_position}: identity inside {@code ±bucket/2}, log-spaced buckets
   * beyond, signed. Returns the (clamped) row into the {@code 2·att_span} position table.
   */
  int relativeRow(int i, int j) {
    int rel = i - j;
    int bucket = rel;
    if (buckets > 0) {
      int mid = buckets / 2;
      int abs = Math.abs(rel);
      if (abs > mid) {
        double logPos =
          Math.ceil(
            (Math.log((double) abs / mid) /
                Math.log((double) (maxRel - 1) / mid)) *
              (mid - 1)
          ) +
          mid;
        bucket = (int) logPos * Integer.signum(rel);
      }
    }
    return Math.max(0, Math.min(2 * attSpan - 1, bucket + attSpan));
  }

  /**
   * {@code (hidden, n·batch) → (headSize, n, heads, batch)} as a strided view — no copy. Feed it to
   * {@code mul_mat} / {@code ggml_cast} directly (both read permuted operands).
   */
  private MemorySegment headsView(
    MemorySegment ctx,
    MemorySegment t,
    long n,
    long batch
  ) {
    long es = Ggml.nb(t, 0);
    long row = Ggml.nb(t, 1); // t may be a row-strided slice of the fused qkv projection
    return Ggml.view4d(
      ctx,
      t,
      headSize,
      n,
      heads,
      batch,
      row,
      (long) headSize * es,
      row * n,
      0
    );
  }

  /**
   * The fused disentangled-attention op (DEBERTA plugin, CUDA) replaces the whole
   * position-score / bias / flash-attention section of {@link #layer}. It needs the per-head,
   * pre-scaled position tables of newer bundles.
   */
  private boolean fused() {
    return w.fusedDebertaAttention() && posPerHead;
  }

  private MemorySegment fusedLayer(
    MemorySegment ctx,
    MemorySegment h,
    int l,
    int n,
    int batch,
    MemorySegment posIdx,
    MemorySegment lens,
    float attnScale
  ) {
    var p = prefix + "layer." + l + ".";
    long winExp = 2L * n - 1;
    int f16 = Ggml.typeF16();
    // The projections go in as the kernel's f16 operands: newer plugins convert the f32 matmul
    // output while staging (no cast ops), older ones want one cast of the fused projection. Either
    // way sliced into q / k / v views (natural order — the kernel indexes i − j itself).
    boolean rawQkv = w.debertaAttnQkvF32();
    MemorySegment qLin;
    MemorySegment kLin;
    MemorySegment vLin;
    if (w.has(p + "qkv.weight")) {
      var qkvLin = w.linear(ctx, h, p + "qkv"); // (3·hidden, n·B)
      var qkv = rawQkv ? qkvLin : Ggml.cast(ctx, qkvLin, f16);
      long es = Ggml.nb(qkv, 0);
      long nb1 = Ggml.nb(qkv, 1);
      qLin = Ggml.view2d(ctx, qkv, hidden, (long) n * batch, nb1, 0);
      kLin = Ggml.view2d(ctx, qkv, hidden, (long) n * batch, nb1, hidden * es);
      vLin = Ggml.view2d(
        ctx,
        qkv,
        hidden,
        (long) n * batch,
        nb1,
        2L * hidden * es
      );
    } else {
      qLin = w.linear(ctx, h, p + "q");
      kLin = w.linear(ctx, h, p + "k");
      vLin = w.linear(ctx, h, p + "v");
      if (!rawQkv) {
        qLin = Ggml.cast(ctx, qLin, f16);
        kLin = Ggml.cast(ctx, kLin, f16);
        vLin = Ggml.cast(ctx, vLin, f16);
      }
    }
    // position tables over the padded offset window, f16 (hd, 2n−1+2·PAD, heads), from the cache
    var tables = posTables(n);
    var posKey = tables.key[l];
    var posQuery = tables.query[l];
    var attn = Ggml.custom4d(
      ctx,
      w.typeF32,
      headSize,
      heads,
      n,
      batch,
      new MemorySegment[] {
        headsView(ctx, qLin, n, batch),
        headsView(ctx, kLin, n, batch),
        headsView(ctx, vLin, n, batch),
        posKey,
        posQuery,
        lens,
      },
      w.debertaAttnFn(),
      1,
      MemorySegment.NULL
    ); // (hd, heads, n, B) — same permuted layout as ggml_flash_attn_ext
    var merged = Ggml.reshape2d(ctx, attn, hidden, (long) n * batch);
    var attnOut = layerNorm(
      ctx,
      Ggml.add(ctx, w.linear(ctx, merged, p + "attn_out"), h),
      p + "attn_ln"
    );
    var inter = Ggml.geluErf(ctx, w.linear(ctx, attnOut, p + "ffn_up"));
    return layerNorm(
      ctx,
      Ggml.add(ctx, w.linear(ctx, inter, p + "ffn_down"), attnOut),
      p + "ffn_ln"
    );
  }

  private MemorySegment layerNorm(
    MemorySegment ctx,
    MemorySegment t,
    String prefix
  ) {
    return Ggml.add(
      ctx,
      Ggml.mul(ctx, Ggml.norm(ctx, t, eps), w.get(prefix + ".weight")),
      w.get(prefix + ".bias")
    );
  }

  /** {@code (hidden, n·batch) → (headSize, n·batch, heads)} as a strided view (rows merged into ne1). */
  private MemorySegment mergedView(
    MemorySegment ctx,
    MemorySegment t,
    long n,
    long batch
  ) {
    long es = Ggml.nb(t, 0);
    long row = Ggml.nb(t, 1);
    return Ggml.view4d(
      ctx,
      t,
      headSize,
      n * batch,
      heads,
      1,
      row,
      (long) headSize * es,
      row * n * batch,
      0
    );
  }

  /**
   * {@code (hidden, n)} (batch 1) → {@code (headSize, n, 1, heads)}: the "heads folded into the
   * batch axis" layout flash attention gets, as a view (head stride on ne3).
   */
  private MemorySegment foldedView(MemorySegment ctx, MemorySegment t, long n) {
    long es = Ggml.nb(t, 0);
    long row = Ggml.nb(t, 1);
    return Ggml.view4d(
      ctx,
      t,
      headSize,
      n,
      1,
      heads,
      row,
      (long) headSize * es,
      (long) headSize * es,
      0
    );
  }

  private MemorySegment posTable(
    MemorySegment ctx,
    String name,
    MemorySegment posIdx,
    long winExp,
    float attnScale
  ) {
    if (posPerHead) {
      return Ggml.getRows(ctx, w.get(name), posIdx); // (hd, 2n−1, heads)
    }
    return Ggml.cont(
      ctx,
      headsView(
        ctx,
        Ggml.scale(ctx, Ggml.getRows(ctx, w.get(name), posIdx), attnScale),
        winExp,
        1
      )
    );
  }

  /**
   * The {@code n×n} band of a {@code (2n−1, n, heads, B)} position-score matrix whose entry
   * {@code [d, x]} is wanted at {@code [y, x]} with {@code d = x + y}: flat index
   * {@code x·(2n−1) + x + y = x·2n + y}, i.e. a view with {@code ne0 = y}, row stride {@code 2n}.
   */
  private MemorySegment band(
    MemorySegment ctx,
    MemorySegment full,
    int n,
    int batch
  ) {
    // full is (2n−1, n·B, heads): row (i + b·n) of head h starts at ((h·B + b)·n + i)·(2n−1);
    // the band for batch b reads i·2n + j' from there — uniform strides in all four dims.
    long es = Ggml.nb(full, 0);
    long win = 2L * n - 1;
    return Ggml.view4d(
      ctx,
      full,
      n,
      n,
      heads,
      batch,
      2L * n * es,
      win * n * batch * es,
      win * n * es,
      0
    );
  }

  private MemorySegment layer(
    MemorySegment ctx,
    MemorySegment h,
    int l,
    int n,
    int batch,
    MemorySegment posIdx,
    MemorySegment revIdx,
    MemorySegment mask16,
    float attnScale
  ) {
    var p = prefix + "layer." + l + ".";
    long winExp = 2L * n - 1;
    int f16 = Ggml.typeF16();
    long hb = (long) heads * batch;

    MemorySegment qLin; // (hidden, n·B), rows in sequence order
    MemorySegment kLin; // (hidden, n·B), rows in reversed order j' = n−1−j
    MemorySegment vLin;
    if (w.has(p + "qkv.weight")) {
      // One fused projection, sliced by rows; K / V rows are then gathered in reversed order.
      var qkv = w.linear(ctx, h, p + "qkv"); // (3·hidden, n·B)
      long es = Ggml.nb(qkv, 0);
      long nb1 = Ggml.nb(qkv, 1);
      qLin = Ggml.view2d(ctx, qkv, hidden, (long) n * batch, nb1, 0);
      kLin = Ggml.getRows(
        ctx,
        Ggml.view2d(ctx, qkv, hidden, (long) n * batch, nb1, hidden * es),
        revIdx
      );
      vLin = Ggml.getRows(
        ctx,
        Ggml.view2d(ctx, qkv, hidden, (long) n * batch, nb1, 2L * hidden * es),
        revIdx
      );
    } else {
      qLin = w.linear(ctx, h, p + "q");
      var hRev = Ggml.getRows(ctx, h, revIdx);
      kLin = w.linear(ctx, hRev, p + "k");
      vLin = w.linear(ctx, hRev, p + "v");
    }
    var q = headsView(ctx, qLin, n, batch); // (hd, n, heads, B) view
    // For the position products the batch rows are merged into the sequence axis — rows sit
    // consecutively in the projection, so (hd, n·B, heads) is one uniform-stride view — giving
    // heads large GEMMs instead of heads·B small ones (cuBLAS is far better at the former).
    var qM = mergedView(ctx, qLin, n, batch);
    var kM = mergedView(ctx, kLin, n, batch);
    // K / V go to flash attention as f16 with heads folded into the batch axis (see below). At
    // batch 1 that layout is a plain strided view of the f16 projection (contiguous cast, fast
    // path); for B > 1 the cast reads the permuted view and materialises the head split.
    MemorySegment k1;
    MemorySegment v1;
    if (batch == 1) {
      k1 = foldedView(ctx, Ggml.cast(ctx, kLin, f16), n);
      v1 = foldedView(ctx, Ggml.cast(ctx, vLin, f16), n);
    } else {
      // Contiguous f16 cast (fast path) then an f16 permute copy: half the bytes of casting the
      // permuted f32 view element by element.
      k1 = Ggml.reshape4d(
        ctx,
        Ggml.cont(ctx, headsView(ctx, Ggml.cast(ctx, kLin, f16), n, batch)),
        headSize,
        n,
        1,
        hb
      );
      v1 = Ggml.reshape4d(
        ctx,
        Ggml.cont(ctx, headsView(ctx, Ggml.cast(ctx, vLin, f16), n, batch)),
        headSize,
        n,
        1,
        hb
      );
    }

    // Position table expanded to one row per offset d = i−j (+n−1) as (hd, 2n−1, heads, 1), which
    // broadcasts over B. Kept f32: for this K=64 batched product an f32 SGEMM beats every
    // alternative measured on Turing (f16 cuBLAS + f16→f32 pass over the (2n−1)·n·heads result,
    // q8_0 MMQ with its quantisation pass). New bundles store the tables per head and pre-scaled,
    // so the gather lands in the attention layout directly; old ones need the split and the scale.
    var posKey = posTable(ctx, p + "pos_key", posIdx, winExp, attnScale);
    var posQuery = posTable(ctx, p + "pos_query", posIdx, winExp, attnScale);

    // c2p[j'][i] = q_i · posKey[i−j] = c2pFull[d = i + j'][i]  → band(x = i, y = j')
    var c2pFull = Ggml.mulMat(ctx, posKey, qM); // (2n−1, n_q·B, heads)
    var c2p = band(ctx, c2pFull, n, batch); // (n_k', n_q, heads, B) f32 strided view
    // p2c[j'][i] = k_j · posQuery[i−j] = p2cFull[d = i + j'][j'] → band(x = j', y = i), transposed.
    // Cast the band first (row-contiguous read), then transpose the dense f16 block: that shape hits
    // CUDA's tiled transpose copy, whereas transposing the strided band view is an uncoalesced copy.
    var p2cFull = Ggml.mulMat(ctx, posQuery, kM); // (2n−1, n_k'·B, heads)
    var p2c16 = Ggml.cont(
      ctx,
      Ggml.permute(
        ctx,
        Ggml.cast(ctx, band(ctx, p2cFull, n, batch), f16),
        1,
        0,
        2,
        3
      )
    ); // (n_k', n_q, heads, B)
    // Fused attention: softmax(scale·q·kᵀ + bias)·v, bias = scale·(c2p + p2c) [+ padding], all f16.
    // CUDA's flash-attention kernels refuse a per-head mask, so heads are folded into the batch
    // axis: one head, heads·B independent sequences, mask (n_k', n_q, 1, heads·B).
    // CUDA and the CPU backend add an f32 operand into an f16 one directly, so the c2p band needs
    // no cast pass of its own; Metal wants matching types and keeps the extra pass.
    var bias16 = w.metal
      ? Ggml.add(ctx, p2c16, Ggml.cast(ctx, c2p, f16))
      : Ggml.add(ctx, p2c16, c2p);
    if (mask16 != null) {
      bias16 = Ggml.add(ctx, bias16, mask16); // (n, n, 1, B) broadcasts over heads
    }
    var biasF16 = Ggml.reshape4d(ctx, bias16, n, n, 1, hb);
    var q1 = batch == 1
      ? foldedView(ctx, qLin, n)
      : Ggml.reshape4d(ctx, Ggml.cont(ctx, q), headSize, n, 1, hb);
    var attn = Ggml.flashAttnExt(ctx, q1, k1, v1, biasF16, attnScale); // (hd, 1, n_q, heads·B)
    var merged = Ggml.reshape2d(
      ctx,
      Ggml.cont(
        ctx,
        Ggml.permute(
          ctx,
          Ggml.reshape4d(ctx, attn, headSize, n, heads, batch),
          0,
          2,
          1,
          3
        )
      ), // (hd, heads, n, B)
      hidden,
      (long) n * batch
    );
    var attnOut = layerNorm(
      ctx,
      Ggml.add(ctx, w.linear(ctx, merged, p + "attn_out"), h),
      p + "attn_ln"
    );
    var inter = Ggml.geluErf(ctx, w.linear(ctx, attnOut, p + "ffn_up"));
    return layerNorm(
      ctx,
      Ggml.add(ctx, w.linear(ctx, inter, p + "ffn_down"), attnOut),
      p + "ffn_ln"
    );
  }
}
