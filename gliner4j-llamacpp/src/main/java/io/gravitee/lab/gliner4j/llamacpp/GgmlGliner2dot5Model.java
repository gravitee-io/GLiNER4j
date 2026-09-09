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

  private StageA stageA(
    long[] ids,
    int[] wordPositions,
    int[] queryPositions,
    boolean withText
  ) {
    int n = ids.length;
    int l = wordPositions.length;
    int q = queryPositions.length;
    int bn = l + 1;
    long nodes = encoder.graphNodes() + 160;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, nodes);
      try {
        var built = encoder.build(ctx, n);
        var wordIdx = Ggml.newTensor1d(ctx, w.typeI32, l);
        var queryIdx = Ggml.newTensor1d(ctx, w.typeI32, q);
        Ggml.setInput(wordIdx);
        Ggml.setInput(queryIdx);
        MemorySegment attnMask = null;
        if (bn > window + 1) {
          attnMask = Ggml.newTensor2d(ctx, w.typeF32, bn, bn);
          Ggml.setInput(attnMask);
        }
        var h = built.output();
        var text = Ggml.getRows(ctx, h, wordIdx); // (H, L)
        var query = Ggml.getRows(ctx, h, queryIdx); // (H, Q)

        // BoundaryEncoder
        var bos = Ggml.reshape2d(ctx, w.get("benc.bos_state"), hidden, 1);
        var eos = Ggml.reshape2d(ctx, w.get("benc.eos_state"), hidden, 1);
        var left = Ggml.concat(ctx, bos, text, 1); // (H, N)
        var right = Ggml.concat(ctx, text, eos, 1);
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
        ); // (D, N)
        int hd = dim / heads;
        float attnScale = (float) (1.0 / Math.sqrt(hd));
        for (
          int b = 0;
          w.has("benc.attention_blocks." + b + ".norm.weight");
          b++
        ) {
          var p = "benc.attention_blocks." + b + ".";
          var qkv = w.linear(
            ctx,
            layerNorm(ctx, states, p + "norm"),
            p + "qkv_projection"
          ); // (3D, N)
          long nb1 = Ggml.nb(qkv, 1);
          long dBytes = (long) dim * Float.BYTES;
          var qh = splitHeads(
            ctx,
            Ggml.cont(ctx, Ggml.view2d(ctx, qkv, dim, bn, nb1, 0)),
            hd,
            bn
          );
          var kh = splitHeads(
            ctx,
            Ggml.cont(ctx, Ggml.view2d(ctx, qkv, dim, bn, nb1, dBytes)),
            hd,
            bn
          );
          var vh = splitHeads(
            ctx,
            Ggml.cont(ctx, Ggml.view2d(ctx, qkv, dim, bn, nb1, 2 * dBytes)),
            hd,
            bn
          );
          var scores = Ggml.mulMat(ctx, kh, qh); // (N_k, N_q, heads)
          var probs = Ggml.softMaxExt(
            ctx,
            scores,
            attnMask == null ? MemorySegment.NULL : attnMask,
            attnScale
          );
          var vT = Ggml.cont(ctx, Ggml.permute(ctx, vh, 1, 0, 2, 3)); // (N_k, hd, heads)
          var ctxv = Ggml.mulMat(ctx, vT, probs); // (hd, N_q, heads)
          var merged = Ggml.reshape2d(
            ctx,
            Ggml.cont(ctx, Ggml.permute(ctx, ctxv, 0, 2, 1, 3)),
            dim,
            bn
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
          ); // (2F, N)
          int f = (int) (Ggml.ne(ip, 0) / 2);
          long nb1 = Ggml.nb(ip, 1);
          var value = Ggml.cont(ctx, Ggml.view2d(ctx, ip, f, bn, nb1, 0));
          var gate = Ggml.cont(
            ctx,
            Ggml.view2d(ctx, ip, f, bn, nb1, (long) f * Float.BYTES)
          );
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
        ); // (N, Q)
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
        ); // (L, Q)
        var poolS = w.linear(ctx, states, "pool.start_projection");
        var poolE = w.linear(ctx, states, "pool.end_projection");
        var scS = w.linear(ctx, states, "scorer.start_projection");
        var scE = w.linear(ctx, states, "scorer.end_projection");
        var values = w.linear(
          ctx,
          text,
          "scorer.content_pooler.value_projection"
        ); // (cd, L)
        var qproj = w.linear(ctx, query, "scorer.query_projection"); // (D, Q)
        var film = w.linear(ctx, qproj, "scorer.film"); // (2D, Q)
        var nullL = w.has("null_projection.weight")
          ? w.linear(ctx, query, "null_projection")
          : null;

        var outs = new ArrayList<MemorySegment>(
          List.of(
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
            query
          )
        );
        if (nullL != null) outs.add(nullL);
        if (withText) outs.add(text);
        var graph = Ggml.newGraph(ctx, nodes);
        for (var t : outs) {
          Ggml.setOutput(t);
          Ggml.buildForwardExpand(graph, t);
        }
        w.alloc(graph);
        encoder.feed(built, call, ids);
        Ggml.setInts(wordIdx, call, wordPositions);
        Ggml.setInts(queryIdx, call, queryPositions);
        if (attnMask != null) {
          var m = new float[bn * bn];
          for (int i = 0; i < bn; i++) for (int j = 0; j < bn; j++) {
            m[i * bn + j] = Math.abs(i - j) <= window ? 0f : MASK_LOGIT; // row = query i, ne0 = key j
          }
          Ggml.setFloats(attnMask, call, m);
        }
        w.run(graph);
        float[] nullOut = nullL != null
          ? Ggml.getFloats(nullL, call, q)
          : filled(q, MASK_LOGIT);
        return new StageA(
          l,
          q,
          Ggml.getFloats(startLogits, call, bn * q),
          Ggml.getFloats(endLogits, call, bn * q),
          Ggml.getFloats(insideLogits, call, l * q),
          Ggml.getFloats(poolS, call, dim * bn),
          Ggml.getFloats(poolE, call, dim * bn),
          Ggml.getFloats(scS, call, dim * bn),
          Ggml.getFloats(scE, call, dim * bn),
          Ggml.getFloats(values, call, contentDim * l),
          Ggml.getFloats(qproj, call, dim * q),
          Ggml.getFloats(film, call, 2 * dim * q),
          nullOut,
          withText ? Ggml.getFloats(text, call, hidden * l) : null,
          Ggml.getFloats(query, call, hidden * q)
        );
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  // ---- host: candidate pool ---------------------------------------------------

  private record Pool(int[] s, int[] e, boolean[] valid, float[] compat) {}

  private static int[] topKIndices(float[] scores, int count, int k) {
    // stable: score desc, index asc — matches ONNX TopK on rows padded with the mask logit
    Integer[] idx = new Integer[count];
    for (int i = 0; i < count; i++) idx[i] = i;
    Arrays.sort(idx, (a, b) ->
      scores[a] == scores[b]
        ? Integer.compare(a, b)
        : Float.compare(scores[b], scores[a])
    );
    var out = new int[k];
    for (int i = 0; i < k; i++) out[i] = i < count ? idx[i] : -1;
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
    var best = new HashMap<Long, Integer>();
    for (int i = 0; i < m; i++) {
      if (!allValid[i]) continue;
      long key = (long) allS[i] * bn + allE[i];
      var cur = best.get(key);
      if (cur == null || allScore[i] > allScore[cur]) best.put(key, i);
    }
    var finalScore = new float[m];
    var keep = new boolean[m];
    for (int i = 0; i < m; i++) {
      keep[i] = allValid[i] && best.get((long) allS[i] * bn + allE[i]) == i;
      finalScore[i] = keep[i] ? allScore[i] : MASK_LOGIT;
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
    int l = a.l();
    int bn = l + 1;
    int q = a.q();
    int c = poolSize;
    // host features: length, content means, inside evidence
    var lenFeat = new float[3 * c];
    var content = new float[contentDim * c];
    var prefix = new double[(l + 1) * contentDim];
    for (int t = 0; t < l; t++) {
      for (int d = 0; d < contentDim; d++) {
        prefix[(t + 1) * contentDim + d] =
          prefix[t * contentDim + d] + a.values()[d + contentDim * t];
      }
    }
    float tl = Math.max(l, 1);
    for (int i = 0; i < c; i++) {
      int s = pool.s()[i],
        e = pool.e()[i];
      float len = Math.max(e - s, 1);
      lenFeat[3 * i] = (float) Math.log1p(len);
      lenFeat[3 * i + 1] = len / tl;
      lenFeat[3 * i + 2] = (float) (1.0 / Math.sqrt(len));
      for (int d = 0; d < contentDim; d++) {
        content[d + contentDim * i] = (float) ((prefix[e * contentDim + d] -
            prefix[s * contentDim + d]) /
          len);
      }
    }
    var gammaP1 = new float[dim * q];
    var beta = new float[dim * q];
    for (int qq = 0; qq < q; qq++) {
      for (int d = 0; d < dim; d++) {
        gammaP1[d + dim * qq] = 1f + a.film()[d + 2 * dim * qq];
        beta[d + dim * qq] = a.film()[dim + d + 2 * dim * qq];
      }
    }
    var validMask = new float[c];
    for (int i = 0; i < c; i++) validMask[i] = pool.valid()[i] ? 1f : 0f;

    float[] base;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, 96);
      try {
        var scS = in2(ctx, dim, bn);
        var scE = in2(ctx, dim, bn);
        var sIdx = Ggml.newTensor1d(ctx, w.typeI32, c);
        var eIdx = Ggml.newTensor1d(ctx, w.typeI32, c);
        Ggml.setInput(sIdx);
        Ggml.setInput(eIdx);
        var lenT = in2(ctx, 3, c);
        var compatT = in2(ctx, 1, c);
        var contentT = in2(ctx, contentDim, c);
        var maskT = in2(ctx, 1, c);
        var qprojT = in2(ctx, dim, q);
        var gammaT = in2(ctx, dim, q);
        var betaT = in2(ctx, dim, q);

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
        ); // (D, C)
        var dots = Ggml.scale(
          ctx,
          Ggml.mulMat(ctx, cand, qprojT),
          (float) (1.0 / Math.sqrt(dim))
        ); // (C, Q)
        var tmpl = Ggml.newTensor3d(ctx, w.typeF32, dim, q, c);
        var cand3 = Ggml.repeat(
          ctx,
          Ggml.reshape3d(ctx, cand, dim, 1, c),
          tmpl
        ); // (D, Q, C)
        var cond = Ggml.add(
          ctx,
          Ggml.mul(ctx, cand3, Ggml.reshape3d(ctx, gammaT, dim, q, 1)),
          Ggml.reshape3d(ctx, betaT, dim, q, 1)
        );
        var fo = w.linear(
          ctx,
          Ggml.geluErf(
            ctx,
            w.linear(
              ctx,
              Ggml.reshape2d(ctx, cond, dim, (long) q * c),
              "scorer.film_output.0"
            )
          ),
          "scorer.film_output.3"
        ); // (1, Q·C)
        var filmT = Ggml.cont(
          ctx,
          Ggml.permute(ctx, Ggml.reshape2d(ctx, fo, q, c), 1, 0, 2, 3)
        ); // (C, Q)
        var total = Ggml.add(ctx, dots, filmT);
        var graph = Ggml.newGraph(ctx, 96);
        w.compute(ctx, graph, total);
        Ggml.setFloats(scS, call, a.scS());
        Ggml.setFloats(scE, call, a.scE());
        Ggml.setInts(sIdx, call, pool.s());
        Ggml.setInts(eIdx, call, pool.e());
        Ggml.setFloats(lenT, call, lenFeat);
        Ggml.setFloats(compatT, call, pool.compat());
        Ggml.setFloats(contentT, call, content);
        Ggml.setFloats(maskT, call, validMask);
        Ggml.setFloats(qprojT, call, a.qproj());
        Ggml.setFloats(gammaT, call, gammaP1);
        Ggml.setFloats(betaT, call, beta);
        w.run(graph);
        base = Ggml.getFloats(total, call, c * q); // [c + C*q]
      } finally {
        Ggml.free(ctx);
      }
    }
    // inside evidence prefix (fp32, mean-centred) per query
    var out = new float[q][c];
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
          out[qq][i] = MASK_LOGIT;
          continue;
        }
        int s = pool.s()[i],
          e = pool.e()[i];
        double score =
          base[i + c * qq] +
          a.startLogits()[s + bn * qq] +
          a.endLogits()[e + bn * qq];
        double interval = insidePrefix[e] - insidePrefix[s] + mean * (e - s);
        score += interval / Math.sqrt(Math.max(e - s, 1));
        out[qq][i] = (float) score;
      }
    }
    return out;
  }

  /** NER for one unpadded row. */
  public synchronized NerResult scoreEntities(
    long[] ids,
    int[] wordPositions,
    int[] queryPositions
  ) {
    var a = stageA(ids, wordPositions, queryPositions, false);
    var pool = buildPool(a);
    var logits = scorePool(a, pool);
    var cands = new int[poolSize][2];
    for (int i = 0; i < poolSize; i++) {
      cands[i][0] = pool.s()[i];
      cands[i][1] = pool.e()[i];
    }
    return new NerResult(logits, cands, pool.valid(), a.nullLogits());
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
