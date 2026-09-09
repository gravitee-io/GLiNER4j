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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A DeBERTa-v2 encoder stack (disentangled attention, {@code pos_att_type=[p2c, c2p]}, separate
 * position projections, no buckets, no rel-embedding norm, no conv) as a ggml graph over dense
 * inputs {@code (hidden, n)}. Shared by the router scorer and the streaming-span labels encoder;
 * the two only differ in the weight-name prefix. Disentangled attention after DeBERTa (He et al.),
 * <a href="https://github.com/microsoft/DeBERTa">microsoft/DeBERTa</a> (MIT) — reimplemented here,
 * no code shared.
 *
 * <p><b>Composed graph</b> (CPU, Metal, no plugin): relative attention only needs the {@code 2·n−1}
 * position rows around the centre, so the {@code q·posKeyᵀ} / {@code k·posQueryᵀ} products run on
 * that window and the {@code (i, j) → i−j} gather is one {@code ggml_get_rows} over the flattened
 * scores.
 *
 * <p><b>Fused kernel</b> (DEBERTA plugin, CUDA): the same {@code ggml_custom_4d} op
 * {@link GgmlDebertaV3} uses — {@code softmax(scale·q·kᵀ + q·P[i−j] + k·Qp[i−j])·v} in one
 * shared-memory tile, nothing n² materialised. Unlike the v3 backbone this stack projects its
 * position tables in-graph from the {@code rel} embeddings, so they are built (pre-scaled, f16,
 * padded by {@link GgmlDebertaV3#POS_PAD} clamped rows) per call instead of coming from a cache.
 * Opt-in ({@code -Dgliner4j.ggml.fusedScorer=true}): the kernel keeps its {@code q·kᵀ} tile in
 * f16, which is exact enough for the LayerNorm-bounded backbone but not for the raw decoder
 * hidden states these scorers take (Qwen3 outlier channels push {@code q·k} past what f16
 * resolves — logits drift by 1e-3 at unit scale and collapse at 50×), and the graph is short
 * enough that the speedup is worth little in absolute terms.
 */
final class GgmlDeberta {

  private static final Logger LOG = LoggerFactory.getLogger(GgmlDeberta.class);

  /** {@code -Dgliner4j.ggml.fusedScorer=true} routes these scorers through the fused op (see class doc). */
  static final String FUSED_PROPERTY = "gliner4j.ggml.fusedScorer";

  /**
   * Graph handles the caller feeds after allocation. {@code c2pIdx}/{@code p2cIdx} belong to the
   * composed graph, {@code relIdx}/{@code lens} to the fused one; the other pair is {@code null}
   * (an input no node reads is never allocated by the scheduler).
   */
  record Built(
    MemorySegment output,
    MemorySegment c2pIdx,
    MemorySegment p2cIdx,
    MemorySegment relIdx,
    MemorySegment lens,
    int n
  ) {}

  private final GgmlWeights w;
  private final String layerPrefix;
  private final String relEmbeddings;
  private final int hidden;
  private final int heads;
  private final int headSize;
  private final int layers;
  private final int posSpan;
  private final float eps;

  GgmlDeberta(
    GgmlWeights w,
    String layerPrefix,
    String relEmbeddings,
    float eps
  ) {
    this.w = w;
    this.layerPrefix = layerPrefix;
    this.relEmbeddings = relEmbeddings;
    this.eps = eps;
    this.hidden = (int) Ggml.ne(w.get(relEmbeddings), 0);
    this.headSize = 64; // hidden // 64 heads, as in DecoderKVScorer / gliner's normalize_context_encoder_config
    this.heads = hidden / headSize;
    int n = 0;
    while (w.has(layerPrefix + n + ".attention.self.query_proj.weight")) n++;
    this.layers = n;
    this.posSpan = (int) (Ggml.ne(w.get(relEmbeddings), 1) / 2);
    if (w.fusedDebertaAttention()) {
      LOG.info(
        "{}: {} disentangled attention ({}={})",
        layerPrefix,
        fused() ? "fused" : "composed",
        FUSED_PROPERTY,
        Boolean.getBoolean(FUSED_PROPERTY)
      );
    }
  }

  int hidden() {
    return hidden;
  }

  int layers() {
    return layers;
  }

  int posSpan() {
    return posSpan;
  }

  /** Whether the fused disentangled-attention op (DEBERTA plugin, CUDA) replaces the composed graph. */
  private boolean fused() {
    return w.fusedDebertaAttention() && Boolean.getBoolean(FUSED_PROPERTY);
  }

  /** Builds the stack over {@code x (hidden, n)}; all {@code n} positions attend to each other. */
  Built build(MemorySegment ctx, MemorySegment x, int n) {
    if (n > posSpan) {
      throw new IllegalArgumentException(
        n + " tokens exceed the relative-position span " + posSpan
      );
    }
    float attnScale = (float) (1.0 / Math.sqrt(headSize * 3.0));
    var rel = w.get(relEmbeddings); // ne=(hidden, 2·posSpan)
    var h = x;
    if (fused()) {
      long win = 2L * n - 1 + 2L * GgmlDebertaV3.POS_PAD;
      var relIdx = Ggml.newTensor1d(ctx, w.typeI32, win);
      Ggml.setInput(relIdx);
      var lens = Ggml.newTensor1d(ctx, w.typeI32, 1);
      Ggml.setInput(lens);
      var relWin = Ggml.getRows(ctx, rel, relIdx); // (hidden, win), rows clamped into rel
      for (int l = 0; l < layers; l++) {
        h = fusedLayer(ctx, h, relWin, l, n, win, lens, attnScale);
      }
      return new Built(h, null, null, relIdx, lens, n);
    }
    int win = 2 * n - 1;
    var c2pIdx = Ggml.newTensor2d(ctx, w.typeI32, (long) n * n, heads);
    var p2cIdx = Ggml.newTensor2d(ctx, w.typeI32, (long) n * n, heads);
    Ggml.setInput(c2pIdx);
    Ggml.setInput(p2cIdx);
    long nb1 = Ggml.nb(rel, 1);
    var relWin = Ggml.view2d(
      ctx,
      rel,
      hidden,
      win,
      nb1,
      (posSpan - (n - 1)) * nb1
    );
    for (int l = 0; l < layers; l++) {
      h = layer(ctx, h, relWin, l, n, win, c2pIdx, p2cIdx, attnScale);
    }
    return new Built(h, c2pIdx, p2cIdx, null, null, n);
  }

  /** Feeds the index / length inputs once the graph is allocated. */
  void feed(Built b, Arena call) {
    int n = b.n();
    if (b.relIdx() != null) {
      Ggml.setInts(b.relIdx(), call, relIndex(n));
      Ggml.setInts(b.lens(), call, new int[] { n });
      return;
    }
    int win = 2 * n - 1;
    Ggml.setInts(b.c2pIdx(), call, gatherIndex(n, win, heads, true));
    Ggml.setInts(b.p2cIdx(), call, gatherIndex(n, win, heads, false));
  }

  /**
   * Rows of {@code rel} for the padded offset window the fused kernel reads: row
   * {@code POS_PAD + d + n − 1} holds offset {@code d = i − j}, i.e. {@code rel[posSpan + d]},
   * clamped so the padding rows (never part of a score) stay finite.
   */
  private int[] relIndex(int n) {
    int pad = GgmlDebertaV3.POS_PAD;
    int win = 2 * n - 1 + 2 * pad;
    var idx = new int[win];
    for (int r = 0; r < win; r++) {
      int d = r - pad - (n - 1);
      idx[r] = Math.max(0, Math.min(2 * posSpan - 1, posSpan + d));
    }
    return idx;
  }

  private static int[] gatherIndex(int n, int win, int heads, boolean c2p) {
    var idx = new int[n * n * heads];
    for (int hd = 0; hd < heads; hd++) {
      int base = hd * n * n;
      for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
          int r = i - j + n - 1;
          idx[base + i * n + j] = c2p ? i * win + r : j * win + r;
        }
      }
    }
    return idx;
  }

  /** {@code (hidden, n) → (headSize, n, heads)}, contiguous copy. */
  private MemorySegment splitHeads(MemorySegment ctx, MemorySegment t, long n) {
    return Ggml.cont(
      ctx,
      Ggml.permute(ctx, Ggml.reshape3d(ctx, t, headSize, heads, n), 0, 2, 1, 3)
    );
  }

  /** {@code (hidden, n) → (headSize, n, heads, 1)} as a strided view — no copy. */
  private MemorySegment headsView(MemorySegment ctx, MemorySegment t, long n) {
    long es = Ggml.nb(t, 0);
    long row = Ggml.nb(t, 1);
    return Ggml.view4d(
      ctx,
      t,
      headSize,
      n,
      heads,
      1,
      row,
      (long) headSize * es,
      row * n,
      0
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

  /**
   * One layer on the fused op: f16 head views of q / k / v, the two position projections of the
   * padded window pre-scaled and split per head (contiguous, as the kernel requires), then the
   * shared output projection / LayerNorm / FFN tail.
   */
  private MemorySegment fusedLayer(
    MemorySegment ctx,
    MemorySegment h,
    MemorySegment relWin,
    int l,
    int n,
    long win,
    MemorySegment lens,
    float attnScale
  ) {
    var p = layerPrefix + l;
    int f16 = Ggml.typeF16();
    // newer plugins take the f32 projections and convert while staging (no cast ops)
    boolean rawQkv = w.debertaAttnQkvF32();
    var q = w.linear(ctx, h, p + ".attention.self.query_proj");
    var k = w.linear(ctx, h, p + ".attention.self.key_proj");
    var v = w.linear(ctx, h, p + ".attention.self.value_proj");
    if (!rawQkv) {
      q = Ggml.cast(ctx, q, f16);
      k = Ggml.cast(ctx, k, f16);
      v = Ggml.cast(ctx, v, f16);
    }
    // the kernel scales q·k itself and expects the tables pre-scaled
    var posKey = splitHeads(
      ctx,
      Ggml.cast(
        ctx,
        Ggml.scale(
          ctx,
          w.linear(ctx, relWin, p + ".attention.self.pos_key_proj"),
          attnScale
        ),
        f16
      ),
      win
    );
    var posQuery = splitHeads(
      ctx,
      Ggml.cast(
        ctx,
        Ggml.scale(
          ctx,
          w.linear(ctx, relWin, p + ".attention.self.pos_query_proj"),
          attnScale
        ),
        f16
      ),
      win
    );
    var attn = Ggml.custom4d(
      ctx,
      w.typeF32,
      headSize,
      heads,
      n,
      1,
      new MemorySegment[] {
        headsView(ctx, q, n),
        headsView(ctx, k, n),
        headsView(ctx, v, n),
        posKey,
        posQuery,
        lens,
      },
      w.debertaAttnFn(),
      1,
      MemorySegment.NULL
    ); // (headSize, heads, n, 1)
    return tail(ctx, h, Ggml.reshape2d(ctx, attn, hidden, n), p);
  }

  private MemorySegment layer(
    MemorySegment ctx,
    MemorySegment h,
    MemorySegment relWin,
    int l,
    int n,
    int win,
    MemorySegment c2pIdx,
    MemorySegment p2cIdx,
    float attnScale
  ) {
    var p = layerPrefix + l;
    var q = splitHeads(
      ctx,
      w.linear(ctx, h, p + ".attention.self.query_proj"),
      n
    );
    var k = splitHeads(
      ctx,
      w.linear(ctx, h, p + ".attention.self.key_proj"),
      n
    );
    var v = splitHeads(
      ctx,
      w.linear(ctx, h, p + ".attention.self.value_proj"),
      n
    );
    var posKey = splitHeads(
      ctx,
      w.linear(ctx, relWin, p + ".attention.self.pos_key_proj"),
      win
    );
    var posQuery = splitHeads(
      ctx,
      w.linear(ctx, relWin, p + ".attention.self.pos_query_proj"),
      win
    );

    var scores = Ggml.mulMat(ctx, k, q); // (n_k, n_q, heads)
    var c2pFull = Ggml.mulMat(ctx, posKey, q); // (win, n_q, heads)
    var c2p = Ggml.reshape3d(
      ctx,
      Ggml.getRows(
        ctx,
        Ggml.reshape3d(ctx, c2pFull, 1, (long) win * n, heads),
        c2pIdx
      ),
      n,
      n,
      heads
    );
    var p2cFull = Ggml.mulMat(ctx, posQuery, k); // (win, n_k, heads)
    var p2c = Ggml.reshape3d(
      ctx,
      Ggml.getRows(
        ctx,
        Ggml.reshape3d(ctx, p2cFull, 1, (long) win * n, heads),
        p2cIdx
      ),
      n,
      n,
      heads
    );
    var total = Ggml.scale(
      ctx,
      Ggml.add(ctx, Ggml.add(ctx, scores, c2p), p2c),
      attnScale
    );
    var probs = Ggml.softMax(ctx, total);

    var vT = Ggml.cont(ctx, Ggml.permute(ctx, v, 1, 0, 2, 3)); // (n_k, headSize, heads)
    var context = Ggml.mulMat(ctx, vT, probs); // (headSize, n_q, heads)
    var merged = Ggml.reshape2d(
      ctx,
      Ggml.cont(ctx, Ggml.permute(ctx, context, 0, 2, 1, 3)),
      hidden,
      n
    );
    return tail(ctx, h, merged, p);
  }

  /** Output projection + residual LayerNorm + FFN, shared by both attention branches. */
  private MemorySegment tail(
    MemorySegment ctx,
    MemorySegment h,
    MemorySegment merged,
    String p
  ) {
    var attnOut = layerNorm(
      ctx,
      Ggml.add(ctx, w.linear(ctx, merged, p + ".attention.output.dense"), h),
      p + ".attention.output.LayerNorm"
    );
    var inter = Ggml.geluErf(
      ctx,
      w.linear(ctx, attnOut, p + ".intermediate.dense")
    );
    return layerNorm(
      ctx,
      Ggml.add(ctx, w.linear(ctx, inter, p + ".output.dense"), attnOut),
      p + ".output.LayerNorm"
    );
  }
}
