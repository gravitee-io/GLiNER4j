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
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * A full DeBERTa-v2/v3 encoder (embeddings + disentangled-attention stack) as a ggml graph over
 * token ids — the backbone of GLiNER2 / GLiNER2.5 / original-GLiNER bundles, which llama.cpp cannot
 * run natively. Weights come from {@code encoder.gguf} written by
 * {@code export_llamacpp.py} ({@code export_deberta_encoder_gguf}), which already folds the
 * rel-embedding LayerNorm and, since {@code share_att_key=true}, the per-layer position keys /
 * queries ({@code layer.N.pos_key} / {@code pos_query}, {@code 2·att_span} rows).
 *
 * <p>Differences from the 2-layer scorer encoder in {@link GgmlDeberta}: word embeddings + LayerNorm
 * in front (no absolute positions), log-bucketed relative positions ({@code position_buckets=256}:
 * identity for {@code |i−j| ≤ 128}, logarithmic beyond), and content/position scores all scaled by
 * {@code 1/sqrt(3·d)}. One sequence per graph; padding is never needed.
 */
final class GgmlDebertaV3 {

  /** Graph handles the caller feeds after allocation. */
  record Built(
    MemorySegment output,
    MemorySegment ids,
    MemorySegment c2pIdx,
    MemorySegment p2cIdx,
    MemorySegment mask,
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
  private final Map<Long, int[][]> indexCache = new ConcurrentHashMap<>();
  /** Largest per-(n, batch) gather-index array kept in the cache (ints). */
  private static final int INDEX_CACHE_MAX_INTS = 4 << 20;
  private static final float MASKED = -1e9f;

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
    return 64 * layers + 32;
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
    var c2pIdx = Ggml.newTensor3d(ctx, w.typeI32, (long) n * n, heads, batch);
    var p2cIdx = Ggml.newTensor3d(ctx, w.typeI32, (long) n * n, heads, batch);
    Ggml.setInput(ids);
    Ggml.setInput(c2pIdx);
    Ggml.setInput(p2cIdx);
    MemorySegment mask = null;
    if (batch > 1) {
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
    int win = 2 * attSpan;
    for (int l = 0; l < layers; l++) {
      h = layer(ctx, h, l, n, batch, win, c2pIdx, p2cIdx, mask, attnScale);
    }
    return new Built(h, ids, c2pIdx, p2cIdx, mask, n, batch);
  }

  /** Feeds token ids and the relative-position gather indices once the graph is allocated. */
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
    for (int r = 0; r < batch; r++) {
      for (int i = 0; i < rows[r].length; i++) ids[r * n + i] =
        (int) rows[r][i];
    }
    Ggml.setInts(b.ids(), call, ids);
    long key = ((long) n << 8) | batch;
    int[][] idx = indexCache.get(key);
    if (idx == null) {
      idx = gatherIndices(n, batch);
      if ((long) idx[0].length <= INDEX_CACHE_MAX_INTS) indexCache.putIfAbsent(
        key,
        idx
      );
    }
    Ggml.setInts(b.c2pIdx(), call, idx[0]);
    Ggml.setInts(b.p2cIdx(), call, idx[1]);
    if (b.mask() != null) {
      var m = new float[n * n * batch];
      for (int r = 0; r < batch; r++) {
        int len = rows[r].length;
        for (int q = 0; q < n; q++) {
          int base = (r * n + q) * n;
          for (int k = len; k < n; k++) m[base + k] = MASKED;
        }
      }
      Ggml.setFloats(b.mask(), call, m);
    }
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

  private int[][] gatherIndices(int n, int batch) {
    var base = gatherIndices(n);
    if (batch == 1) return base;
    var c2p = new int[base[0].length * batch];
    var p2c = new int[base[1].length * batch];
    for (int r = 0; r < batch; r++) {
      System.arraycopy(base[0], 0, c2p, r * base[0].length, base[0].length);
      System.arraycopy(base[1], 0, p2c, r * base[1].length, base[1].length);
    }
    return new int[][] { c2p, p2c };
  }

  private int[][] gatherIndices(int n) {
    int win = 2 * attSpan;
    var c2p = new int[n * n * heads];
    var p2c = new int[n * n * heads];
    var row = new int[n * n];
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
        row[i * n + j] = relativeRow(i, j);
      }
    }
    for (int hd = 0; hd < heads; hd++) {
      int base = hd * n * n;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          int r = row[i * n + j];
          // scores layout: ne0 = key j, ne1 = query i. c2p reads row i of (q·posKeyᵀ), p2c row j of (k·posQueryᵀ).
          c2p[base + i * n + j] = i * win + r;
          p2c[base + i * n + j] = j * win + r;
        }
      }
    }
    return new int[][] { c2p, p2c };
  }

  /** {@code (hidden, n·batch) → (headSize, n, heads, batch)}. */
  private MemorySegment splitHeads(
    MemorySegment ctx,
    MemorySegment t,
    long n,
    long batch
  ) {
    return Ggml.cont(
      ctx,
      Ggml.permute(
        ctx,
        Ggml.reshape4d(ctx, t, headSize, heads, n, batch),
        0,
        2,
        1,
        3
      )
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

  private MemorySegment layer(
    MemorySegment ctx,
    MemorySegment h,
    int l,
    int n,
    int batch,
    int win,
    MemorySegment c2pIdx,
    MemorySegment p2cIdx,
    MemorySegment mask,
    float attnScale
  ) {
    var p = prefix + "layer." + l + ".";
    var q = splitHeads(ctx, w.linear(ctx, h, p + "q"), n, batch); // (hd, n, heads, B)
    var k = splitHeads(ctx, w.linear(ctx, h, p + "k"), n, batch);
    var v = splitHeads(ctx, w.linear(ctx, h, p + "v"), n, batch);
    var posKey = splitHeads(ctx, w.get(p + "pos_key"), win, 1); // (hd, win, heads, 1) — broadcasts over B
    var posQuery = splitHeads(ctx, w.get(p + "pos_query"), win, 1);

    var scores = Ggml.mulMat(ctx, k, q); // (n_k, n_q, heads, B)
    var c2pFull = Ggml.mulMat(ctx, posKey, q); // (win, n_q, heads, B)
    var c2p = Ggml.reshape4d(
      ctx,
      Ggml.getRows(
        ctx,
        Ggml.reshape4d(ctx, c2pFull, 1, (long) win * n, heads, batch),
        c2pIdx
      ),
      n,
      n,
      heads,
      batch
    );
    var p2cFull = Ggml.mulMat(ctx, posQuery, k); // (win, n_k, heads, B)
    var p2c = Ggml.reshape4d(
      ctx,
      Ggml.getRows(
        ctx,
        Ggml.reshape4d(ctx, p2cFull, 1, (long) win * n, heads, batch),
        p2cIdx
      ),
      n,
      n,
      heads,
      batch
    );
    var total = Ggml.add(ctx, Ggml.add(ctx, scores, c2p), p2c);
    var probs = Ggml.softMaxExt(
      ctx,
      total,
      mask == null ? MemorySegment.NULL : mask,
      attnScale
    );

    var vT = Ggml.cont(ctx, Ggml.permute(ctx, v, 1, 0, 2, 3)); // (n_k, hd, heads, B)
    var context = Ggml.mulMat(ctx, vT, probs); // (hd, n_q, heads, B)
    var merged = Ggml.reshape2d(
      ctx,
      Ggml.cont(ctx, Ggml.permute(ctx, context, 0, 2, 1, 3)), // (hd, heads, n, B)
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
