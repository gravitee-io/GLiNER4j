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
 * A DeBERTa-v2 encoder stack (disentangled attention, {@code pos_att_type=[p2c, c2p]}, separate
 * position projections, no buckets, no rel-embedding norm, no conv) as a ggml graph over dense
 * inputs {@code (hidden, n)}. Shared by the router scorer and the streaming-span labels encoder;
 * the two only differ in the weight-name prefix.
 *
 * <p>Relative attention only needs the {@code 2·n−1} position rows around the centre, so the
 * {@code q·posKeyᵀ} / {@code k·posQueryᵀ} products run on that window and the {@code (i, j) → i−j}
 * gather is one {@code ggml_get_rows} over the flattened scores.
 */
final class GgmlDeberta {

  /** Graph handles the caller feeds after allocation. */
  record Built(
    MemorySegment output,
    MemorySegment c2pIdx,
    MemorySegment p2cIdx,
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

  /** Builds the stack over {@code x (hidden, n)}; all {@code n} positions attend to each other. */
  Built build(MemorySegment ctx, MemorySegment x, int n) {
    if (n > posSpan) {
      throw new IllegalArgumentException(
        n + " tokens exceed the relative-position span " + posSpan
      );
    }
    int win = 2 * n - 1;
    var c2pIdx = Ggml.newTensor2d(ctx, w.typeI32, (long) n * n, heads);
    var p2cIdx = Ggml.newTensor2d(ctx, w.typeI32, (long) n * n, heads);
    Ggml.setInput(c2pIdx);
    Ggml.setInput(p2cIdx);
    var rel = w.get(relEmbeddings); // ne=(hidden, 2·posSpan)
    long nb1 = Ggml.nb(rel, 1);
    var relWin = Ggml.view2d(
      ctx,
      rel,
      hidden,
      win,
      nb1,
      (posSpan - (n - 1)) * nb1
    );
    float attnScale = (float) (1.0 / Math.sqrt(headSize * 3.0));
    var h = x;
    for (int l = 0; l < layers; l++) {
      h = layer(ctx, h, relWin, l, n, win, c2pIdx, p2cIdx, attnScale);
    }
    return new Built(h, c2pIdx, p2cIdx, n);
  }

  /** Feeds the gather indices once the graph is allocated. */
  void feed(Built b, Arena call) {
    int win = 2 * b.n() - 1;
    Ggml.setInts(b.c2pIdx(), call, gatherIndex(b.n(), win, heads, true));
    Ggml.setInts(b.p2cIdx(), call, gatherIndex(b.n(), win, heads, false));
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

  private MemorySegment splitHeads(MemorySegment ctx, MemorySegment t, long n) {
    return Ggml.cont(
      ctx,
      Ggml.permute(ctx, Ggml.reshape3d(ctx, t, headSize, heads, n), 0, 2, 1, 3)
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
