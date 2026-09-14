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
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

/**
 * GLiNER2.5 (fastino "boundary" architecture, shared candidate pool) on ggml. The encoder-bound
 * work runs as graphs; the small, index-heavy candidate selection (top-k boundaries, Cartesian
 * pairing, per-query quota, dedup, final top-k) and the relation pair generation run on the host
 * exactly as the ONNX {@code ner_full} / {@code relation_full} graphs do them.
 *
 * <ul>
 *   <li>Stage A (graph): DeBERTa encoder → text / query rows → {@code BoundaryEncoder} (left/right
 *       projections, 2 windowed pre-norm attention blocks, 1 residual SwiGLU) → start/end/inside
 *       marginals, pool and scorer boundary projections, content values, query projection, FiLM
 *       parameters, abstention logits.</li>
 *   <li>Host: candidate pool selection ({@code DocumentCandidatePool}), inside-evidence prefix,
 *       content prefix means, length features.</li>
 *   <li>Stage B (graph): {@code SharedPoolScorer} candidate assembly + norm, query dot product,
 *       FiLM MLP over {@code (candidate, query)} pairs.</li>
 *   <li>Stage C (graph, relations): {@code SparseRelationScorer} MLP + biaffine content term over
 *       the host-selected head/tail pairs.</li>
 * </ul>
 * Weights: {@code gguf/model.gguf} from {@code export_llamacpp_gliner2.py gliner2dot5}.
 */
public final class GgmlGliner2dot5Model implements AutoCloseable {

  static final float MASK_LOGIT = -1e4f;

  /** NER result for one row: {@code pairLogits[Q][C]}, candidates {@code [C][2]}, {@code nullLogits[Q]}. */
  public record NerResult(
    float[][] pairLogits,
    int[][] candidates,
    boolean[] valid,
    float[] nullLogits
  ) {}

  /** Relation result for one row: {@code logits[R][P]}, pairs {@code [R][P][4]}. */
  public record RelationResult(float[][] logits, int[][][] pairs) {}

  private final GgmlWeights w;
  private final GgmlDebertaV3 encoder;
  private final int hidden;
  private final int dim;
  private final int heads;
  private final int window;
  private final int topK;
  private final int poolSize;
  private final int quota;
  private final int contentDim;
  private final float eps;
  private final int relHeads;
  private final int relTails;
  private final int pairCap;
  private final float argThreshold;
  private final boolean biaffine;
  private final String classifierOut;

  public GgmlGliner2dot5Model(Path gguf, boolean useGpu, int cpuThreads) {
    this.w = new GgmlWeights(gguf, useGpu, cpuThreads, "ggml GLiNER2.5");
    this.encoder = new GgmlDebertaV3(w, "enc.");
    this.hidden = encoder.hidden();
    this.dim = w.metaInt("g25.boundary_dim", 128);
    this.heads = w.metaInt("g25.attention_heads", 4);
    this.window = w.metaInt("g25.attention_window", 128);
    this.topK = w.metaInt("g25.pool_top_k", 32);
    this.poolSize = w.metaInt("g25.pool_size", 192);
    this.quota = Math.min(w.metaInt("g25.min_pool_per_query", 8), topK * topK);
    this.contentDim = w.metaInt("g25.content_dim", 64);
    this.eps = w.metaFloat("g25.layer_norm_eps", 1e-5f);
    this.relHeads = w.metaInt("g25.relation_heads", 32);
    this.relTails = w.metaInt("g25.relation_tails", 32);
    this.pairCap = w.metaInt("g25.relation_pair_cap", 64);
    this.argThreshold = w.metaFloat("g25.relation_argument_threshold", 0.2f);
    this.biaffine = w.has("rel.head_content_projection.weight");
    this.classifierOut = w.has("classifier.3.weight")
      ? "classifier.3"
      : "classifier.2";
  }

  public int poolSize() {
    return poolSize;
  }

  public int pairCap() {
    return pairCap;
  }

  // ---- stage A ----------------------------------------------------------------

  /** Everything the host and the later graphs need from one encoder pass. */
  private record StageA(
    int l,
    int q,
    float[] startLogits, // (N, Q): [i + N*q]
    float[] endLogits,
    float[] insideLogits, // (L, Q)
    float[] poolS, // (D, N): [d + D*i]
    float[] poolE,
    float[] scS,
    float[] scE,
    float[] values, // (cd, L)
    float[] qproj, // (D, Q)
    float[] film, // (2D, Q)
    float[] nullLogits, // (Q)
    float[] text, // (H, L)
    float[] query // (H, Q)
  ) {}

  /** Batched head outputs of a stage-A graph: every tensor carries the rows on its last axis. */
  private record HeadOuts(
    MemorySegment startLogits, // (BN, Q, B)
    MemorySegment endLogits,
    MemorySegment insideLogits, // (L, Q, B)
    MemorySegment poolS, // (D, BN, B)
    MemorySegment poolE,
    MemorySegment scS,
    MemorySegment scE,
    MemorySegment values, // (cd, L, B)
    MemorySegment qproj, // (D, Q, B)
    MemorySegment film, // (2D, Q, B)
    MemorySegment nullL, // (1, Q, B) or null
    MemorySegment query, // (H, Q, B)
    MemorySegment text // (H, L, B)
  ) {}

  /** Max nodes the batched heads add to the stage-A graph on top of the encoder. */
  private static final long HEAD_NODES = 256;

  private StageA stageA(
    long[] ids,
    int[] wordPositions,
    int[] queryPositions,
    boolean withText
  ) {
    return stageABatch(
      new long[][] { ids },
      new int[][] { wordPositions },
      queryPositions,
      withText
    )[0];
  }

  /**
   * Stage A for {@code B} rows in one graph: the rows are padded to the longest and encoded
   * together ({@link GgmlDebertaV3#build(MemorySegment, int, int)}), then the boundary encoder
   * and the heads run once over the padded {@code (·, words, B)} stack — padded word slots hold
   * a real token's state and are masked out of the boundary attention, the {@code [EOS]} state
   * is gathered at each row's own length, and only the first {@code bn_r} columns of a row are
   * read back. Short rows are launch-bound on CUDA, so one graph for the batch is what keeps the
   * GPU busy; a single row is the {@code B = 1} case of the same code.
   */
  private StageA[] stageABatch(
    long[][] idsRows,
    int[][] wordPositionsRows,
    int[] queryPositions,
    boolean withText
  ) {
    int batch = idsRows.length;
    int n = 0;
    int l = 0;
    for (int r = 0; r < batch; r++) {
      n = Math.max(n, idsRows[r].length);
      l = Math.max(l, wordPositionsRows[r].length);
    }
    int bn = l + 1;
    int q = queryPositions.length;
    long nodes = encoder.graphNodes() + HEAD_NODES;
    // host-side gathers: word / query states per row, [BOS] + words and words + [EOS] rows of
    // the per-row (H, L+2) table, and the boundary attention mask (window + padding)
    var wordAll = new int[l * batch];
    var queryAll = new int[q * batch];
    var leftAll = new int[bn * batch];
    var rightAll = new int[bn * batch];
    var mask = new float[bn * bn * batch];
    for (int r = 0; r < batch; r++) {
      var wp = wordPositionsRows[r];
      int lr = wp.length;
      for (int i = 0; i < l; i++) wordAll[r * l + i] =
        (i < lr ? wp[i] : 0) + r * n;
      for (int i = 0; i < q; i++) queryAll[r * q + i] =
        queryPositions[i] + r * n;
      int base = r * (l + 2);
      for (int i = 0; i < bn; i++) {
        leftAll[r * bn + i] = base + i; // 0 = [BOS], then word i-1
        rightAll[r * bn + i] = base + (i < lr ? i + 1 : l + 1); // word i, then [EOS]
      }
      int bnr = lr + 1;
      for (int i = 0; i < bn; i++) for (int j = 0; j < bn; j++) {
        mask[(r * bn + i) * bn + j] = j < bnr && Math.abs(i - j) <= window
          ? 0f
          : MASK_LOGIT; // row = query i, ne0 = key j
      }
    }
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, nodes);
      try {
        var built = encoder.build(ctx, n, batch);
        var h = built.output(); // (H, n·B): column of row r token i = i + n·r
        var wordIdx = Ggml.newTensor1d(ctx, w.typeI32, (long) l * batch);
        var queryIdx = Ggml.newTensor1d(ctx, w.typeI32, (long) q * batch);
        var leftIdx = Ggml.newTensor1d(ctx, w.typeI32, (long) bn * batch);
        var rightIdx = Ggml.newTensor1d(ctx, w.typeI32, (long) bn * batch);
        var attnMask = Ggml.newTensor3d(ctx, w.typeF32, bn, bn, batch);
        for (var t : new MemorySegment[] {
          wordIdx,
          queryIdx,
          leftIdx,
          rightIdx,
          attnMask,
        })
          Ggml.setInput(t);
        var text = Ggml.reshape3d(
          ctx,
          Ggml.getRows(ctx, h, wordIdx),
          hidden,
          l,
          batch
        ); // (H, L, B)
        var query = Ggml.reshape3d(
          ctx,
          Ggml.getRows(ctx, h, queryIdx),
          hidden,
          q,
          batch
        ); // (H, Q, B)
        var o = heads(
          ctx,
          text,
          query,
          leftIdx,
          rightIdx,
          Ggml.reshape4d(ctx, attnMask, bn, bn, 1, batch),
          l,
          q,
          batch
        );
        var graph = Ggml.newGraph(ctx, nodes);
        for (var t : new MemorySegment[] {
          o.startLogits(),
          o.endLogits(),
          o.insideLogits(),
          o.poolS(),
          o.poolE(),
          o.scS(),
          o.scE(),
          o.values(),
          o.qproj(),
          o.film(),
          o.query(),
          o.nullL(),
          withText ? o.text() : null,
        }) {
          if (t == null) continue;
          Ggml.setOutput(t);
          Ggml.buildForwardExpand(graph, t);
        }
        w.alloc(graph);
        encoder.feed(built, call, idsRows);
        Ggml.setInts(wordIdx, call, wordAll);
        Ggml.setInts(queryIdx, call, queryAll);
        Ggml.setInts(leftIdx, call, leftAll);
        Ggml.setInts(rightIdx, call, rightAll);
        Ggml.setFloats(attnMask, call, mask);
        w.run(graph);
        // read the padded stacks once, then cut each row to its own length
        var startAll = Ggml.getFloats(o.startLogits(), call, bn * q * batch);
        var endAll = Ggml.getFloats(o.endLogits(), call, bn * q * batch);
        var insideAll = Ggml.getFloats(o.insideLogits(), call, l * q * batch);
        var poolSAll = Ggml.getFloats(o.poolS(), call, dim * bn * batch);
        var poolEAll = Ggml.getFloats(o.poolE(), call, dim * bn * batch);
        var scSAll = Ggml.getFloats(o.scS(), call, dim * bn * batch);
        var scEAll = Ggml.getFloats(o.scE(), call, dim * bn * batch);
        var valuesAll = Ggml.getFloats(
          o.values(),
          call,
          contentDim * l * batch
        );
        var qprojAll = Ggml.getFloats(o.qproj(), call, dim * q * batch);
        var filmAll = Ggml.getFloats(o.film(), call, 2 * dim * q * batch);
        var queryAllF = Ggml.getFloats(o.query(), call, hidden * q * batch);
        var nullAll = o.nullL() != null
          ? Ggml.getFloats(o.nullL(), call, q * batch)
          : null;
        var textAll = withText
          ? Ggml.getFloats(o.text(), call, hidden * l * batch)
          : null;
        var result = new StageA[batch];
        for (int r = 0; r < batch; r++) {
          int lr = wordPositionsRows[r].length;
          int bnr = lr + 1;
          result[r] = new StageA(
            lr,
            q,
            rows(startAll, bn, q, r, bnr),
            rows(endAll, bn, q, r, bnr),
            rows(insideAll, l, q, r, lr),
            cols(poolSAll, dim, bn, r, bnr),
            cols(poolEAll, dim, bn, r, bnr),
            cols(scSAll, dim, bn, r, bnr),
            cols(scEAll, dim, bn, r, bnr),
            cols(valuesAll, contentDim, l, r, lr),
            cols(qprojAll, dim, q, r, q),
            cols(filmAll, 2 * dim, q, r, q),
            nullAll != null
              ? Arrays.copyOfRange(nullAll, r * q, (r + 1) * q)
              : filled(q, MASK_LOGIT),
            textAll != null ? cols(textAll, hidden, l, r, lr) : null,
            cols(queryAllF, hidden, q, r, q)
          );
        }
        return result;
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /** Row {@code r} of a {@code (X, N, B)} stack, first {@code keep} columns: {@code [x + X·i]}. */
  private static float[] cols(float[] all, int x, int nMax, int r, int keep) {
    var out = new float[x * keep];
    System.arraycopy(all, r * x * nMax, out, 0, x * keep);
    return out;
  }

  /** Row {@code r} of a {@code (N, Q, B)} logit stack, first {@code keep} positions: {@code [i + keep·q]}. */
  private static float[] rows(float[] all, int nMax, int q, int r, int keep) {
    var out = new float[keep * q];
    for (int qq = 0; qq < q; qq++) System.arraycopy(
      all,
      (r * q + qq) * nMax,
      out,
      qq * keep,
      keep
    );
    return out;
  }

  /** {@code (X, N, B)} stack → {@code (hd, N, heads, B)} per-head layout, contiguous. */
  private MemorySegment splitHeadsBatch(
    MemorySegment ctx,
    MemorySegment t,
    int hd,
    long n,
    long batch
  ) {
    return Ggml.cont(
      ctx,
      Ggml.permute(ctx, Ggml.reshape4d(ctx, t, hd, heads, n, batch), 0, 2, 1, 3)
    );
  }

  /** Contiguous copy of columns {@code [off, off + x)} of a {@code (X0, N, B)} stack: {@code (x, N, B)}. */
  private MemorySegment sliceBatch(
    MemorySegment ctx,
    MemorySegment t,
    long x,
    long n,
    long batch,
    long off
  ) {
    long es = Ggml.nb(t, 0);
    var v = Ggml.view4d(
      ctx,
      t,
      x,
      n,
      batch,
      1,
      Ggml.nb(t, 1),
      Ggml.nb(t, 2),
      Ggml.nb(t, 2) * batch,
      off * es
    );
    return Ggml.reshape3d(ctx, Ggml.cont(ctx, v), x, n, batch);
  }

  /** Boundary encoder + boundary / pooling / scorer projections over the padded row stack. */
  private HeadOuts heads(
    MemorySegment ctx,
    MemorySegment text,
    MemorySegment query,
    MemorySegment leftIdx,
    MemorySegment rightIdx,
    MemorySegment attnMask,
    int l,
    int q,
    int batch
  ) {
    int bn = l + 1;
    // BoundaryEncoder: per row the table [BOS] · words · [EOS], gathered into the left / right rows
    var tmpl = Ggml.newTensor3d(ctx, w.typeF32, hidden, 1, batch);
    var bos = Ggml.repeat(
      ctx,
      Ggml.reshape3d(ctx, w.get("benc.bos_state"), hidden, 1, 1),
      tmpl
    );
    var eos = Ggml.repeat(
      ctx,
      Ggml.reshape3d(ctx, w.get("benc.eos_state"), hidden, 1, 1),
      tmpl
    );
    var table = Ggml.reshape2d(
      ctx,
      Ggml.concat(ctx, Ggml.concat(ctx, bos, text, 1), eos, 1),
      hidden,
      (long) (l + 2) * batch
    ); // (H, (L+2)·B)
    var left = Ggml.reshape3d(
      ctx,
      Ggml.getRows(ctx, table, leftIdx),
      hidden,
      bn,
      batch
    );
    var right = Ggml.reshape3d(
      ctx,
      Ggml.getRows(ctx, table, rightIdx),
      hidden,
      bn,
      batch
    );
    var states = layerNorm(
      ctx,
      w.linear(
        ctx,
        Ggml.concat(
          ctx,
          w.linear(ctx, left, "benc.left_projection"),
          w.linear(ctx, right, "benc.right_projection"),
          0
        ),
        "benc.output_projection"
      ),
      "benc.layer_norm"
    ); // (D, BN, B)
    int hd = dim / heads;
    float attnScale = (float) (1.0 / Math.sqrt(hd));
    for (int b = 0; w.has("benc.attention_blocks." + b + ".norm.weight"); b++) {
      var p = "benc.attention_blocks." + b + ".";
      var qkv = w.linear(
        ctx,
        layerNorm(ctx, states, p + "norm"),
        p + "qkv_projection"
      ); // (3D, BN, B)
      var qh = splitHeadsBatch(
        ctx,
        sliceBatch(ctx, qkv, dim, bn, batch, 0),
        hd,
        bn,
        batch
      );
      var kh = splitHeadsBatch(
        ctx,
        sliceBatch(ctx, qkv, dim, bn, batch, dim),
        hd,
        bn,
        batch
      );
      var vh = splitHeadsBatch(
        ctx,
        sliceBatch(ctx, qkv, dim, bn, batch, 2L * dim),
        hd,
        bn,
        batch
      );
      var scores = Ggml.mulMat(ctx, kh, qh); // (BN_k, BN_q, heads, B)
      var probs = Ggml.softMaxExt(ctx, scores, attnMask, attnScale); // mask (BN, BN, 1, B)
      var vT = Ggml.cont(ctx, Ggml.permute(ctx, vh, 1, 0, 2, 3)); // (BN_k, hd, heads, B)
      var ctxv = Ggml.mulMat(ctx, vT, probs); // (hd, BN_q, heads, B)
      var merged = Ggml.reshape3d(
        ctx,
        Ggml.cont(ctx, Ggml.permute(ctx, ctxv, 0, 2, 1, 3)),
        dim,
        bn,
        batch
      );
      states = Ggml.add(
        ctx,
        states,
        w.linear(ctx, merged, p + "output_projection")
      );
    }
    for (
      int b = 0;
      w.has("benc.refinement_blocks." + b + ".norm.weight");
      b++
    ) {
      var p = "benc.refinement_blocks." + b + ".";
      var ip = w.linear(
        ctx,
        layerNorm(ctx, states, p + "norm"),
        p + "input_projection"
      ); // (2F, BN, B)
      int f = (int) (Ggml.ne(ip, 0) / 2);
      var value = sliceBatch(ctx, ip, f, bn, batch, 0);
      var gate = sliceBatch(ctx, ip, f, bn, batch, f);
      var upd = Ggml.mul(ctx, value, Ggml.silu(ctx, gate));
      states = Ggml.add(
        ctx,
        states,
        w.linear(ctx, upd, p + "output_projection")
      );
    }

    float mScale = (float) (1.0 / Math.sqrt(dim));
    var startLogits = Ggml.scale(
      ctx,
      Ggml.mulMat(
        ctx,
        w.linear(ctx, states, "bqh.start_boundary_projection"),
        w.linear(ctx, query, "bqh.start_query_projection")
      ),
      mScale
    ); // (BN, Q, B)
    var endLogits = Ggml.scale(
      ctx,
      Ggml.mulMat(
        ctx,
        w.linear(ctx, states, "bqh.end_boundary_projection"),
        w.linear(ctx, query, "bqh.end_query_projection")
      ),
      mScale
    );
    var insideLogits = Ggml.scale(
      ctx,
      Ggml.mulMat(
        ctx,
        w.linear(ctx, text, "bqh.inside_text_projection"),
        w.linear(ctx, query, "bqh.inside_query_projection")
      ),
      mScale
    ); // (L, Q, B)
    var poolS = w.linear(ctx, states, "pool.start_projection");
    var poolE = w.linear(ctx, states, "pool.end_projection");
    var scS = w.linear(ctx, states, "scorer.start_projection");
    var scE = w.linear(ctx, states, "scorer.end_projection");
    var values = w.linear(ctx, text, "scorer.content_pooler.value_projection"); // (cd, L, B)
    var qproj = w.linear(ctx, query, "scorer.query_projection"); // (D, Q, B)
    var film = w.linear(ctx, qproj, "scorer.film"); // (2D, Q, B)
    var nullL = w.has("null_projection.weight")
      ? w.linear(ctx, query, "null_projection")
      : null; // (1, Q, B)
    return new HeadOuts(
      startLogits,
      endLogits,
      insideLogits,
      poolS,
      poolE,
      scS,
      scE,
      values,
      qproj,
      film,
      nullL,
      query,
      text
    );
  }

  // ---- host: candidate pool ---------------------------------------------------

  private record Pool(int[] s, int[] e, boolean[] valid, float[] compat) {}

  /** Scratch table of {@link #buildPool}: best candidate index per (start, end) pair. */
  private int[] bestByPair;

  /**
   * Indices of the {@code k} best scores, score descending and index ascending on ties (matches
   * ONNX TopK on rows padded with the mask logit); {@code -1} past {@code count}. One primitive
   * sort of packed (inverted sortable score, index) keys — the boxed comparator version of this
   * cost more per call than the GPU pass.
   */
  private static int[] topKIndices(float[] scores, int count, int k) {
    var keys = new long[count];
    for (int i = 0; i < count; i++) {
      int bits = Float.floatToIntBits(scores[i]);
      int sortable = bits < 0 ? bits ^ 0x7fffffff : bits; // monotone in the float value
      keys[i] = ((long) (~sortable) << 32) | (i & 0xffffffffL); // ascending = score desc, index asc
    }
    Arrays.sort(keys);
    var out = new int[k];
    for (int i = 0; i < k; i++) out[i] = i < count ? (int) keys[i] : -1;
    return out;
  }

  private Pool buildPool(StageA a) {
    int bn = a.l() + 1;
    int q = a.q();
    int k = topK;
    var unionStart = new float[bn];
    var unionEnd = new float[bn];
    for (int i = 0; i < bn; i++) {
      float ms = -Float.MAX_VALUE,
        me = -Float.MAX_VALUE;
      for (int qq = 0; qq < q; qq++) {
        ms = Math.max(ms, a.startLogits()[i + bn * qq]);
        me = Math.max(me, a.endLogits()[i + bn * qq]);
      }
      unionStart[i] = ms;
      unionEnd[i] = me;
    }
    var starts = topKIndices(unionStart, bn, k);
    var ends = topKIndices(unionEnd, bn, k);
    int kk = k * k;
    var pairS = new int[kk];
    var pairE = new int[kk];
    var pairValid = new boolean[kk];
    var compat = new float[kk];
    var unionPair = new float[kk];
    for (int x = 0; x < k; x++) {
      for (int y = 0; y < k; y++) {
        int i = x * k + y;
        int s = Math.max(starts[x], 0),
          e = Math.max(ends[y], 0);
        boolean valid = starts[x] >= 0 && ends[y] >= 0 && e > s;
        pairS[i] = s;
        pairE[i] = e;
        pairValid[i] = valid;
        compat[i] =
          dot(a.poolS(), s * dim, a.poolE(), e * dim, dim) /
          (float) Math.sqrt(dim);
        unionPair[i] = compat[i] + unionStart[s] + unionEnd[e];
      }
    }
    // per-query quota band
    int m = q * quota + kk + poolSize;
    var allS = new int[m];
    var allE = new int[m];
    var allScore = new float[m];
    var allValid = new boolean[m];
    var perQuery = new float[kk];
    for (int qq = 0; qq < q; qq++) {
      for (int i = 0; i < kk; i++) {
        perQuery[i] = pairValid[i]
          ? a.startLogits()[pairS[i] + bn * qq] +
          a.endLogits()[pairE[i] + bn * qq] +
          compat[i]
          : MASK_LOGIT;
      }
      var ranked = topKIndices(perQuery, kk, quota);
      for (int r = 0; r < quota; r++) {
        int o = qq * quota + r;
        int i = ranked[r];
        allS[o] = pairS[i];
        allE[o] = pairE[i];
        allValid[o] = pairValid[i];
        allScore[o] = -MASK_LOGIT * 0.5f + (quota - r);
      }
    }
    System.arraycopy(pairS, 0, allS, q * quota, kk);
    System.arraycopy(pairE, 0, allE, q * quota, kk);
    System.arraycopy(pairValid, 0, allValid, q * quota, kk);
    System.arraycopy(unionPair, 0, allScore, q * quota, kk);
    for (int i = 0; i < m; i++) if (!allValid[i]) allScore[i] = MASK_LOGIT;
    // dedup: keep the best-scoring occurrence of each (start, end); ties → lowest index
    var keep = new boolean[m];
    var finalScore = new float[m];
    if ((long) bn * bn <= (1 << 22)) {
      var best = bestByPair;
      if (best == null || best.length < bn * bn) best = bestByPair =
        new int[Math.max(bn * bn, 1 << 12)];
      Arrays.fill(best, 0, bn * bn, -1);
      for (int i = 0; i < m; i++) {
        if (!allValid[i]) continue;
        int key = allS[i] * bn + allE[i];
        int cur = best[key];
        if (cur < 0 || allScore[i] > allScore[cur]) best[key] = i;
      }
      for (int i = 0; i < m; i++) {
        keep[i] = allValid[i] && best[allS[i] * bn + allE[i]] == i;
        finalScore[i] = keep[i] ? allScore[i] : MASK_LOGIT;
      }
    } else {
      var best = new HashMap<Long, Integer>();
      for (int i = 0; i < m; i++) {
        if (!allValid[i]) continue;
        long key = (long) allS[i] * bn + allE[i];
        var cur = best.get(key);
        if (cur == null || allScore[i] > allScore[cur]) best.put(key, i);
      }
      for (int i = 0; i < m; i++) {
        keep[i] = allValid[i] && best.get((long) allS[i] * bn + allE[i]) == i;
        finalScore[i] = keep[i] ? allScore[i] : MASK_LOGIT;
      }
    }
    var top = topKIndices(finalScore, m, poolSize);
    var selS = new int[poolSize];
    var selE = new int[poolSize];
    var selValid = new boolean[poolSize];
    var selCompat = new float[poolSize];
    for (int c = 0; c < poolSize; c++) {
      int i = top[c];
      selValid[c] = i >= 0 && keep[i];
      selS[c] = selValid[c] ? allS[i] : 0;
      selE[c] = selValid[c] ? allE[i] : 0;
      selCompat[c] = selValid[c]
        ? dot(a.poolS(), selS[c] * dim, a.poolE(), selE[c] * dim, dim) /
        (float) Math.sqrt(dim)
        : 0f;
    }
    return new Pool(selS, selE, selValid, selCompat);
  }

  // ---- stage B ------------------------------------------------------------------

  private float[][] scorePool(StageA a, Pool pool) {
    return scorePoolBatch(new StageA[] { a }, new Pool[] { pool })[0];
  }

  /**
   * Candidate scoring for {@code B} rows in one graph: the rows' candidates are laid out along
   * one axis ({@code C·B}, the scorer start / end tables concatenated with per-row offsets) and
   * the per-row query features sit on a batch axis, so the query dot products run as one batched
   * matmul and the FiLM conditioning as one broadcast over {@code (D, Q, C, B)}.
   */
  private float[][][] scorePoolBatch(StageA[] as, Pool[] pools) {
    int batch = as.length;
    int q = as[0].q();
    int c = poolSize;
    int bnTotal = 0;
    for (var a : as) bnTotal += a.l() + 1;
    // host features: length, content means (per row), candidate indices into the stacked tables
    var lenFeat = new float[3 * c * batch];
    var content = new float[contentDim * c * batch];
    var compat = new float[c * batch];
    var validMask = new float[c * batch];
    var sIdxAll = new int[c * batch];
    var eIdxAll = new int[c * batch];
    var scSAll = new float[dim * bnTotal];
    var scEAll = new float[dim * bnTotal];
    var gammaP1 = new float[dim * q * batch];
    var beta = new float[dim * q * batch];
    var qprojAll = new float[dim * q * batch];
    int bnOffset = 0;
    for (int r = 0; r < batch; r++) {
      var a = as[r];
      var pool = pools[r];
      int l = a.l();
      int bn = l + 1;
      System.arraycopy(a.scS(), 0, scSAll, dim * bnOffset, dim * bn);
      System.arraycopy(a.scE(), 0, scEAll, dim * bnOffset, dim * bn);
      var prefix = new double[(l + 1) * contentDim];
      for (int t = 0; t < l; t++) {
        for (int d = 0; d < contentDim; d++) {
          prefix[(t + 1) * contentDim + d] =
            prefix[t * contentDim + d] + a.values()[d + contentDim * t];
        }
      }
      float tl = Math.max(l, 1);
      for (int i = 0; i < c; i++) {
        int o = r * c + i;
        int s = pool.s()[i],
          e = pool.e()[i];
        float len = Math.max(e - s, 1);
        lenFeat[3 * o] = (float) Math.log1p(len);
        lenFeat[3 * o + 1] = len / tl;
        lenFeat[3 * o + 2] = (float) (1.0 / Math.sqrt(len));
        for (int d = 0; d < contentDim; d++) {
          content[d + contentDim * o] = (float) ((prefix[e * contentDim + d] -
              prefix[s * contentDim + d]) /
            len);
        }
        compat[o] = pool.compat()[i];
        validMask[o] = pool.valid()[i] ? 1f : 0f;
        sIdxAll[o] = bnOffset + s;
        eIdxAll[o] = bnOffset + e;
      }
      for (int qq = 0; qq < q; qq++) {
        int o = r * q + qq;
        for (int d = 0; d < dim; d++) {
          gammaP1[d + dim * o] = 1f + a.film()[d + 2 * dim * qq];
          beta[d + dim * o] = a.film()[dim + d + 2 * dim * qq];
          qprojAll[d + dim * o] = a.qproj()[d + dim * qq];
        }
      }
      bnOffset += bn;
    }

    float[] base;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, 96);
      try {
        var scS = in2(ctx, dim, bnTotal);
        var scE = in2(ctx, dim, bnTotal);
        var sIdx = Ggml.newTensor1d(ctx, w.typeI32, (long) c * batch);
        var eIdx = Ggml.newTensor1d(ctx, w.typeI32, (long) c * batch);
        Ggml.setInput(sIdx);
        Ggml.setInput(eIdx);
        var lenT = in2(ctx, 3, (long) c * batch);
        var compatT = in2(ctx, 1, (long) c * batch);
        var contentT = in2(ctx, contentDim, (long) c * batch);
        var maskT = in2(ctx, 1, (long) c * batch);
        var qprojT = in2(ctx, dim, (long) q * batch);
        var gammaT = in2(ctx, dim, (long) q * batch);
        var betaT = in2(ctx, dim, (long) q * batch);

        var cand = Ggml.add(
          ctx,
          Ggml.getRows(ctx, scS, sIdx),
          Ggml.getRows(ctx, scE, eIdx)
        );
        cand = Ggml.add(
          ctx,
          cand,
          w.linear(ctx, lenT, "scorer.length_projection")
        );
        cand = Ggml.add(
          ctx,
          cand,
          w.linear(ctx, compatT, "scorer.prior_projection")
        );
        cand = Ggml.add(
          ctx,
          cand,
          w.linear(
            ctx,
            layerNorm(ctx, contentT, "scorer.content_pooler.layer_norm"),
            "scorer.content_projection"
          )
        );
        cand = Ggml.mul(
          ctx,
          layerNorm(ctx, cand, "scorer.candidate_norm"),
          maskT
        ); // (D, C·B)
        var cand3 = Ggml.reshape3d(ctx, cand, dim, c, batch);
        var qproj3 = Ggml.reshape3d(ctx, qprojT, dim, q, batch);
        var dots = Ggml.scale(
          ctx,
          Ggml.mulMat(ctx, cand3, qproj3),
          (float) (1.0 / Math.sqrt(dim))
        ); // (C, Q, B)
        var tmpl = Ggml.newTensor3d(ctx, w.typeF32, dim, q, (long) c * batch);
        var cand4 = Ggml.reshape4d(
          ctx,
          Ggml.repeat(
            ctx,
            Ggml.reshape3d(ctx, cand, dim, 1, (long) c * batch),
            tmpl
          ),
          dim,
          q,
          c,
          batch
        ); // (D, Q, C, B)
        var cond = Ggml.add(
          ctx,
          Ggml.mul(ctx, cand4, Ggml.reshape4d(ctx, gammaT, dim, q, 1, batch)),
          Ggml.reshape4d(ctx, betaT, dim, q, 1, batch)
        );
        var fo = w.linear(
          ctx,
          Ggml.geluErf(
            ctx,
            w.linear(
              ctx,
              Ggml.reshape2d(ctx, cond, dim, (long) q * c * batch),
              "scorer.film_output.0"
            )
          ),
          "scorer.film_output.3"
        ); // (1, Q·C·B)
        var filmT = Ggml.cont(
          ctx,
          Ggml.permute(ctx, Ggml.reshape3d(ctx, fo, q, c, batch), 1, 0, 2, 3)
        ); // (C, Q, B)
        var total = Ggml.add(ctx, dots, filmT);
        var graph = Ggml.newGraph(ctx, 96);
        w.compute(ctx, graph, total);
        Ggml.setFloats(scS, call, scSAll);
        Ggml.setFloats(scE, call, scEAll);
        Ggml.setInts(sIdx, call, sIdxAll);
        Ggml.setInts(eIdx, call, eIdxAll);
        Ggml.setFloats(lenT, call, lenFeat);
        Ggml.setFloats(compatT, call, compat);
        Ggml.setFloats(contentT, call, content);
        Ggml.setFloats(maskT, call, validMask);
        Ggml.setFloats(qprojT, call, qprojAll);
        Ggml.setFloats(gammaT, call, gammaP1);
        Ggml.setFloats(betaT, call, beta);
        w.run(graph);
        base = Ggml.getFloats(total, call, c * q * batch); // [i + C·qq + C·Q·r]
      } finally {
        Ggml.free(ctx);
      }
    }
    // inside evidence prefix (fp32, mean-centred) per row and query
    var out = new float[batch][q][c];
    for (int r = 0; r < batch; r++) {
      var a = as[r];
      var pool = pools[r];
      int l = a.l();
      int bn = l + 1;
      var insidePrefix = new double[l + 1];
      for (int qq = 0; qq < q; qq++) {
        double sum = 0;
        for (int t = 0; t < l; t++) sum += a.insideLogits()[t + l * qq];
        double mean = l > 0 ? sum / l : 0;
        insidePrefix[0] = 0;
        for (int t = 0; t < l; t++) insidePrefix[t + 1] =
          insidePrefix[t] + (a.insideLogits()[t + l * qq] - mean);
        for (int i = 0; i < c; i++) {
          if (!pool.valid()[i]) {
            out[r][qq][i] = MASK_LOGIT;
            continue;
          }
          int s = pool.s()[i],
            e = pool.e()[i];
          double score =
            base[i + c * qq + c * q * r] +
            a.startLogits()[s + bn * qq] +
            a.endLogits()[e + bn * qq];
          double interval = insidePrefix[e] - insidePrefix[s] + mean * (e - s);
          score += interval / Math.sqrt(Math.max(e - s, 1));
          out[r][qq][i] = (float) score;
        }
      }
    }
    return out;
  }

  /**
   * Longest (padded) row for which {@link #scoreEntitiesBatch} beats per-row
   * {@link #scoreEntities} on CUDA — same rule as {@link GgmlGliner2Model#prefersBatchedScoring}:
   * short rows are launch-bound, long ones already fill the GPU. Never on CPU or Metal.
   */
  public boolean prefersBatchedScoring(int n) {
    return (
      w.gpu &&
      !w.metal &&
      n <= Integer.getInteger("gliner4j.ggml.batchMaxTokens", 384)
    );
  }

  /** NER for one unpadded row. */
  public synchronized NerResult scoreEntities(
    long[] ids,
    int[] wordPositions,
    int[] queryPositions
  ) {
    return scoreEntitiesBatch(
      new long[][] { ids },
      new int[][] { wordPositions },
      queryPositions
    )[0];
  }

  /** NER for {@code B} unpadded rows sharing the same row-relative {@code queryPositions}. */
  public synchronized NerResult[] scoreEntitiesBatch(
    long[][] idsRows,
    int[][] wordPositionsRows,
    int[] queryPositions
  ) {
    var as = stageABatch(idsRows, wordPositionsRows, queryPositions, false);
    var pools = new Pool[as.length];
    for (int r = 0; r < as.length; r++) pools[r] = buildPool(as[r]);
    var logits = scorePoolBatch(as, pools);
    var out = new NerResult[as.length];
    for (int r = 0; r < as.length; r++) {
      var cands = new int[poolSize][2];
      for (int i = 0; i < poolSize; i++) {
        cands[i][0] = pools[r].s()[i];
        cands[i][1] = pools[r].e()[i];
      }
      out[r] = new NerResult(
        logits[r],
        cands,
        pools[r].valid(),
        as[r].nullLogits()
      );
    }
    return out;
  }

  // ---- stage C: relations -------------------------------------------------------

  /** Relations for one unpadded row; {@code queryPositions} = head_0, tail_0, head_1, tail_1, …. */
  public synchronized RelationResult scoreRelations(
    long[] ids,
    int[] wordPositions,
    int[] queryPositions
  ) {
    var a = stageA(ids, wordPositions, queryPositions, true);
    var pool = buildPool(a);
    var pairLogits = scorePool(a, pool);
    int r = queryPositions.length / 2;
    int l = a.l();
    int p = pairCap;
    int kh = relHeads,
      kt = relTails;
    var hs = new int[r * p];
    var he = new int[r * p];
    var ts = new int[r * p];
    var te = new int[r * p];
    var selValid = new boolean[r * p];
    for (int rr = 0; rr < r; rr++) {
      var head = selectArguments(pairLogits[2 * rr], pool, kh);
      var tail = selectArguments(pairLogits[2 * rr + 1], pool, kt);
      var flat = new float[kh * kt];
      var flatValid = new boolean[kh * kt];
      for (int x = 0; x < kh; x++) {
        for (int y = 0; y < kt; y++) {
          int i = x * kt + y;
          boolean same =
            head.s()[x] == tail.s()[y] && head.e()[x] == tail.e()[y];
          flatValid[i] = head.valid()[x] && tail.valid()[y] && !same;
          flat[i] = flatValid[i] ? head.prob()[x] * tail.prob()[y] : -1f;
        }
      }
      var keep = topKIndices(flat, kh * kt, p);
      for (int j = 0; j < p; j++) {
        int i = keep[j];
        int o = rr * p + j;
        int hi = i / kt,
          ti = i % kt;
        selValid[o] = i >= 0 && flatValid[i];
        hs[o] = head.s()[hi];
        he[o] = head.e()[hi];
        ts[o] = tail.s()[ti];
        te[o] = tail.e()[ti];
      }
    }
    // host features
    int rp = r * p;
    var rel = new float[2 * hidden * rp];
    var order = new float[rp];
    var dist = new float[rp];
    var headContent = new float[hidden * rp];
    var tailContent = new float[hidden * rp];
    var heM1 = new int[rp];
    var teM1 = new int[rp];
    var prefix = new double[(l + 1) * hidden];
    for (int t = 0; t < l; t++) for (int d = 0; d < hidden; d++) prefix[(t +
          1) *
        hidden +
      d] =
      prefix[t * hidden + d] + a.text()[d + hidden * t];
    float tl = Math.max(l, 1);
    for (int o = 0; o < rp; o++) {
      int rr = o / p;
      System.arraycopy(a.query(), 2 * rr * hidden, rel, o * 2 * hidden, hidden);
      System.arraycopy(
        a.query(),
        (2 * rr + 1) * hidden,
        rel,
        o * 2 * hidden + hidden,
        hidden
      );
      int delta = ts[o] - hs[o];
      order[o] = Integer.signum(delta);
      dist[o] = Math.abs(delta) / tl;
      heM1[o] = Math.max(he[o] - 1, 0);
      teM1[o] = Math.max(te[o] - 1, 0);
      float hw = Math.max(he[o] - hs[o], 1),
        tw = Math.max(te[o] - ts[o], 1);
      for (int d = 0; d < hidden; d++) {
        headContent[d + hidden * o] = (float) ((prefix[he[o] * hidden + d] -
            prefix[hs[o] * hidden + d]) /
          hw);
        tailContent[d + hidden * o] = (float) ((prefix[te[o] * hidden + d] -
            prefix[ts[o] * hidden + d]) /
          tw);
      }
    }
    float[] scores;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, 96);
      try {
        var text = in2(ctx, hidden, l);
        var hsT = Ggml.newTensor1d(ctx, w.typeI32, rp);
        var heT = Ggml.newTensor1d(ctx, w.typeI32, rp);
        var tsT = Ggml.newTensor1d(ctx, w.typeI32, rp);
        var teT = Ggml.newTensor1d(ctx, w.typeI32, rp);
        for (var t : new MemorySegment[] { hsT, heT, tsT, teT })
          Ggml.setInput(t);
        var relT = in2(ctx, 2 * hidden, rp);
        var orderT = in2(ctx, 1, rp);
        var distT = in2(ctx, 1, rp);
        var feats = Ggml.concat(
          ctx,
          Ggml.getRows(ctx, text, hsT),
          Ggml.getRows(ctx, text, heT),
          0
        );
        feats = Ggml.concat(ctx, feats, Ggml.getRows(ctx, text, tsT), 0);
        feats = Ggml.concat(ctx, feats, Ggml.getRows(ctx, text, teT), 0);
        feats = Ggml.concat(ctx, feats, relT, 0);
        feats = Ggml.concat(ctx, feats, orderT, 0);
        feats = Ggml.concat(ctx, feats, distT, 0); // (4H + 2H + 2, RP)
        var score = w.linear(
          ctx,
          Ggml.geluErf(ctx, w.linear(ctx, feats, "rel.mlp.0")),
          "rel.mlp.3"
        ); // (1, RP)
        MemorySegment hcT = null,
          tcT = null;
        if (biaffine) {
          hcT = in2(ctx, hidden, rp);
          tcT = in2(ctx, hidden, rp);
          var hc = w.linear(ctx, hcT, "rel.head_content_projection");
          var tc = w.linear(ctx, tcT, "rel.tail_content_projection");
          var gate = Ggml.sigmoid(
            ctx,
            w.linear(ctx, relT, "rel.relation_content_gate")
          );
          var bi = Ggml.scale(
            ctx,
            Ggml.sumRows(ctx, Ggml.mul(ctx, Ggml.mul(ctx, hc, gate), tc)),
            (float) (1.0 / Math.sqrt(hidden))
          );
          var lin = w.linear(
            ctx,
            Ggml.concat(ctx, Ggml.concat(ctx, hc, tc, 0), relT, 0),
            "rel.content_linear"
          );
          score = Ggml.add(ctx, Ggml.add(ctx, score, bi), lin);
        }
        var graph = Ggml.newGraph(ctx, 96);
        w.compute(ctx, graph, score);
        Ggml.setFloats(text, call, a.text());
        Ggml.setInts(hsT, call, hs);
        Ggml.setInts(heT, call, heM1);
        Ggml.setInts(tsT, call, ts);
        Ggml.setInts(teT, call, teM1);
        Ggml.setFloats(relT, call, rel);
        Ggml.setFloats(orderT, call, order);
        Ggml.setFloats(distT, call, dist);
        if (biaffine) {
          Ggml.setFloats(hcT, call, headContent);
          Ggml.setFloats(tcT, call, tailContent);
        }
        w.run(graph);
        scores = Ggml.getFloats(score, call, rp);
      } finally {
        Ggml.free(ctx);
      }
    }
    var logits = new float[r][p];
    var pairs = new int[r][p][4];
    for (int o = 0; o < rp; o++) {
      int rr = o / p,
        j = o % p;
      logits[rr][j] = selValid[o] ? scores[o] : MASK_LOGIT;
      if (selValid[o]) {
        pairs[rr][j][0] = hs[o];
        pairs[rr][j][1] = he[o];
        pairs[rr][j][2] = ts[o];
        pairs[rr][j][3] = te[o];
      }
    }
    return new RelationResult(logits, pairs);
  }

  private record Args(float[] prob, int[] s, int[] e, boolean[] valid) {}

  private Args selectArguments(float[] logits, Pool pool, int k) {
    int c = poolSize;
    var prob = new float[c];
    var masked = new float[c];
    var valid = new boolean[c];
    for (int i = 0; i < c; i++) {
      prob[i] = (float) (1.0 / (1.0 + Math.exp(-logits[i])));
      valid[i] = pool.valid()[i] && prob[i] >= argThreshold;
      masked[i] = valid[i] ? prob[i] : -1f;
    }
    var idx = topKIndices(masked, c, k);
    var out = new Args(new float[k], new int[k], new int[k], new boolean[k]);
    for (int j = 0; j < k; j++) {
      int i = Math.max(idx[j], 0);
      out.prob()[j] = prob[i];
      out.s()[j] = pool.s()[i];
      out.e()[j] = pool.e()[i];
      out.valid()[j] = idx[j] >= 0 && valid[i];
    }
    return out;
  }

  // ---- classification -----------------------------------------------------------

  /** Classifier logits for the {@code [L]} marker rows. */
  public synchronized float[] classify(long[] ids, int[] labelPositions) {
    int n = ids.length;
    int l = labelPositions.length;
    if (l == 0) return new float[0];
    long nodes = encoder.graphNodes() + 32;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, nodes);
      try {
        var built = encoder.build(ctx, n);
        var lIdx = Ggml.newTensor1d(ctx, w.typeI32, l);
        Ggml.setInput(lIdx);
        var rows = Ggml.getRows(ctx, built.output(), lIdx);
        var logits = w.linear(
          ctx,
          Ggml.relu(ctx, w.linear(ctx, rows, "classifier.0")),
          classifierOut
        );
        var graph = Ggml.newGraph(ctx, nodes);
        w.compute(ctx, graph, logits);
        encoder.feed(built, call, ids);
        Ggml.setInts(lIdx, call, labelPositions);
        w.run(graph);
        return Ggml.getFloats(logits, call, l);
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  // ---- helpers ------------------------------------------------------------------

  private MemorySegment in2(MemorySegment ctx, long ne0, long ne1) {
    var t = Ggml.newTensor2d(ctx, w.typeF32, ne0, ne1);
    Ggml.setInput(t);
    return t;
  }

  private MemorySegment splitHeads(
    MemorySegment ctx,
    MemorySegment t,
    int hd,
    long n
  ) {
    return Ggml.cont(
      ctx,
      Ggml.permute(ctx, Ggml.reshape3d(ctx, t, hd, heads, n), 0, 2, 1, 3)
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

  private static float dot(float[] a, int ao, float[] b, int bo, int n) {
    float s = 0;
    for (int i = 0; i < n; i++) s += a[ao + i] * b[bo + i];
    return s;
  }

  private static float[] filled(int n, float v) {
    var out = new float[n];
    Arrays.fill(out, v);
    return out;
  }

  @Override
  public synchronized void close() {
    w.close();
  }
}
