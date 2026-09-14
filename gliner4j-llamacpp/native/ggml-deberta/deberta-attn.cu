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

// Fused DeBERTa disentangled attention (after He et al., https://github.com/microsoft/DeBERTa, MIT;
// reimplemented from the paper, no code shared):
//
//   out_i = sum_j softmax_j( scale * q_i.k_j + q_i.P[i-j] + k_j.Qp[i-j] ) * v_j
//
// with P / Qp the per-offset position tables (row PAD + i - j + n - 1, already scaled by
// 1/sqrt(3*hd), padded by PAD zero-or-clamped rows on both sides so every 16-row window a tile pair
// touches is in bounds). Nothing n^2 is ever written to global memory: the ggml graph used to build
// the two (2n-1)*n*heads position-score matrices with SGEMMs, read them back as strided bands and
// add them into an f16 mask for flash attention; here the same band trick happens inside a tile in
// shared memory.
//
// Tiled kernel: one block per (query tile of 64, head, batch row), 8 warps. Two tile
// configurations are compiled in (64x32 keys / 62 KB for Turing, 64x64 keys / 94 KB for Ampere and
// later) and one is chosen per device at first use from its shared-memory limit. Per key tile the
// block stages K, V and the windows of P / Qp the tile pair touches (f16, coalesced 16-byte loads, padded rows) and computes on the tensor cores
//   S = Q.K^T,  G = Q.Pwindow^T,  H = K.Qpwindow^T
// (f32 accumulate, stored f16), every 16x16 tile spread over the 8 warps. The two position terms
// are then read as shifted rows of G / H (the band views), the online softmax runs per row
// (lane = key) and P.V runs on the tensor cores; the running-max rescale of the P.V accumulator
// goes through a small shared-memory round trip (the wmma fragment layout is opaque).
//
// Reference kernel (GLINER4J_DEBERTA_REF=1): one warp per query row, plain dot products.

#include "deberta-attn.h"

#include "ggml.h"
#include "ggml-impl.h"

#include <cuda_fp16.h>
#include <mma.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using namespace nvcuda;

namespace {

constexpr int HD  = 64;                        // head dim — every DeBERTa-v3 size
constexpr int LD  = HD + 8;                    // smem row stride (halves): 144 B, conflict-free wmma loads
constexpr int PAD = GLINER4J_DEBERTA_POS_PAD;

constexpr int round16(int x) { return (x + 15) / 16 * 16; }

// One tile configuration = one compiled kernel. TQ queries per block over 8 warps; TK keys per
// key tile (a multiple of 32: each lane owns TK/32 keys in the softmax step). Everything else —
// the offset window, the shared-memory layout, the tensor-core tile counts — follows from those
// two numbers, so a variant for another GPU generation is a one-line instantiation below.
template <int TQ_, int TK_>
struct Cfg {
    static constexpr int TQ       = TQ_;
    static constexpr int TK       = TK_;
    static constexpr int NWARPS   = 8;
    static constexpr int NTHREADS = NWARPS * 32;
    static constexpr int ROWS_PW  = TQ / NWARPS;      // softmax rows per warp
    static constexpr int KPL      = TK / 32;          // keys per lane in the softmax step
    static constexpr int RW       = TQ + TK - 1;      // distinct offsets i - j inside a tile pair
    static constexpr int RWP      = round16(RW);      // padded for wmma
    static constexpr int RT       = TQ / 16;          // query row tiles
    static constexpr int KT       = TK / 16;          // key row tiles
    static constexpr int WT       = RWP / 16;         // window column tiles
    // shared memory layout (bytes)
    static constexpr size_t SM_Q       = 0;                                   // TQ  x LD  f16
    static constexpr size_t SM_K       = SM_Q + (size_t) TQ * LD * 2;         // TK  x LD  f16
    static constexpr size_t SM_V       = SM_K + (size_t) TK * LD * 2;         // TK  x LD  f16
    static constexpr size_t SM_W       = SM_V + (size_t) TK * LD * 2;         // RWP x LD  f16: P window, then Qp window
    static constexpr size_t SM_G       = SM_W + (size_t) RWP * LD * 2;        // TQ  x RWP f16 (O staging at the end)
    static constexpr size_t SM_H       = SM_G + (size_t) TQ * RWP * 2;        // TK  x RWP f16
    static constexpr size_t SM_S       = SM_H + (size_t) TK * RWP * 2;        // TQ  x TK  f16 (scores, then probabilities)
    static constexpr size_t SM_SCRATCH = SM_S + (size_t) TQ * TK * 2;         // NWARPS x (16 x 16 f32) + alphas
    static constexpr size_t SM_TOTAL   = SM_SCRATCH + (size_t) NWARPS * 256 * 4;

    static_assert(TK % 32 == 0 && TQ % NWARPS == 0 && TQ % 16 == 0, "tile shape");
    static_assert(RT * 2 == NWARPS, "P.V mapping: two warps per query row tile");
    static_assert(PAD >= TK + 16, "table padding must cover the window overhang");
    static_assert((size_t) TQ * HD * 4 <= (size_t) TQ * RWP * 2 + (size_t) TK * RWP * 2, "O staging must fit in G+H");
    static_assert((size_t) TQ * 4 <= (size_t) NWARPS * 256 * 4, "alphas must fit in scratch");
    static_assert(SM_TOTAL <= 227 * 1024, "shared memory (sm_90 opt-in is 227 KB)");
};

// Devices with 64 KB of shared memory per block (Turing class) → 64 x 32 tiles, 62 KB, one block per SM.
using CfgTuring = Cfg<64, 32>;
// Ampere / Ada / Hopper (>= 96 KB opt-in): 64 x 64 tiles, 94 KB — half the key-tile rounds and a
// 2x (instead of 3x) window overhead in the position products; two blocks per SM on sm_80/sm_90.
using CfgAmpere = Cfg<64, 64>;

using attn_args = gliner4j_deberta_attn_args;

__device__ __forceinline__ float warp_max(float x) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, o));
    return x;
}

__device__ __forceinline__ float warp_sum(float x) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) x += __shfl_xor_sync(0xffffffffu, x, o);
    return x;
}

using frag_a     = wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major>;
using frag_b     = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major>;
using frag_b_row = wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major>;
using frag_acc   = wmma::fragment<wmma::accumulator, 16, 16, 16, float>;

// out(16 x 16, f16, row stride ld_out) = A(16 rows at `A`, stride LD) . B(16 rows at `B`, stride LD)^T
__device__ __forceinline__ void mma_tile(const __half * A, const __half * B, float * scratch, __half * out, int ld_out, int lane) {
    frag_acc acc;
    wmma::fill_fragment(acc, 0.0f);
    #pragma unroll
    for (int kk = 0; kk < HD; kk += 16) {
        frag_a a;
        frag_b b;
        wmma::load_matrix_sync(a, A + kk, LD);
        wmma::load_matrix_sync(b, B + kk, LD);
        wmma::mma_sync(acc, a, b, acc);
    }
    wmma::store_matrix_sync(scratch, acc, 16, wmma::mem_row_major);
    __syncwarp();
    #pragma unroll
    for (int e = lane; e < 256; e += 32) {
        out[(e / 16) * ld_out + e % 16] = __float2half(scratch[e]);
    }
    __syncwarp();
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *(const uint32_t *) &h;
}

// stage `n_rows` rows (HD elements each, f16 — or f32 converted on the way, which is what lets the
// graph skip its q / k / v casts) starting at global row `row0` into smem (stride LD); rows >= valid
// are zero
template <int NTHREADS, bool F32>
__device__ __forceinline__ void stage_rows(const char * src, long nb1, int row0, int valid, int n_rows, __half * dst) {
    for (int idx = threadIdx.x; idx < n_rows * 8; idx += NTHREADS) {
        const int r = idx / 8, c = idx % 8;
        uint4 val = make_uint4(0, 0, 0, 0);
        const int row = row0 + r;
        if (row < valid) {
            if constexpr (F32) {
                const float4 * f = (const float4 *) (src + (size_t) row * nb1 + c * 32);
                const float4 a = f[0], b = f[1];
                val = make_uint4(pack_half2(a.x, a.y), pack_half2(a.z, a.w), pack_half2(b.x, b.y), pack_half2(b.z, b.w));
            } else {
                val = *(const uint4 *) (src + (size_t) row * nb1 + c * 16);
            }
        }
        *(uint4 *) (dst + r * LD + c * 8) = val;
    }
}

struct tile_ctx {
    __half * Qs; __half * Ks; __half * Vs; __half * Ws; __half * Gs; __half * Hs; __half * Ss; float * scratch;
    const char * kh; const char * vh; const char * Ph; const char * Qph;
    long k_nb1, v_nb1, p_nb1, qp_nb1;
    int n, n_tab, len, i0, warp, lane;
};

// Stages K (and V) for key tile j0 and leaves S, G and H in shared memory; ends synchronised.
template <class C, bool F32>
__device__ __forceinline__ void compute_tile(const tile_ctx & c, int j0, bool want_v) {
    __syncthreads(); // previous tile fully consumed
    stage_rows<C::NTHREADS, F32>(c.kh, c.k_nb1, j0, c.len, C::TK, c.Ks);
    if (want_v) {
        stage_rows<C::NTHREADS, F32>(c.vh, c.v_nb1, j0, c.len, C::TK, c.Vs);
    }
    // window of the position tables for this (query tile, key tile): local offset
    // r = i_l - j_l + (TK - 1) ∈ [0, RW) ↔ table row PAD + i - j + n_tab - 1 (always in bounds;
    // the tables may be built for a larger length n_tab >= n — per-length cache buckets)
    const int base = PAD + c.i0 - j0 + c.n_tab - C::TK;
    stage_rows<C::NTHREADS, false>(c.Ph, c.p_nb1, base, 1 << 30, C::RWP, c.Ws);
    __syncthreads();

    // G = Q . P^T : RT row tiles x WT col tiles over the warps
    for (int t = c.warp; t < C::RT * C::WT; t += C::NWARPS) {
        const int rf = t % C::RT, nb = t / C::RT;
        mma_tile(c.Qs + rf * 16 * LD, c.Ws + nb * 16 * LD, c.scratch, c.Gs + rf * 16 * C::RWP + nb * 16, C::RWP, c.lane);
    }
    // S = Q . K^T : RT x KT tiles
    for (int t = c.warp; t < C::RT * C::KT; t += C::NWARPS) {
        const int rf = t % C::RT, nb = t / C::RT;
        mma_tile(c.Qs + rf * 16 * LD, c.Ks + nb * 16 * LD, c.scratch, c.Ss + rf * 16 * C::TK + nb * 16, C::TK, c.lane);
    }
    __syncthreads(); // P window consumed
    stage_rows<C::NTHREADS, false>(c.Qph, c.qp_nb1, base, 1 << 30, C::RWP, c.Ws);
    __syncthreads();
    // H = K . Qp^T : KT x WT tiles
    for (int t = c.warp; t < C::KT * C::WT; t += C::NWARPS) {
        const int rf = t % C::KT, nb = t / C::KT;
        mma_tile(c.Ks + rf * 16 * LD, c.Ws + nb * 16 * LD, c.scratch, c.Hs + rf * 16 * C::RWP + nb * 16, C::RWP, c.lane);
    }
    __syncthreads(); // S, G, H complete
}

// score of (row i_l, key j_l) for the current tile, -inf when masked
template <class C>
__device__ __forceinline__ float tile_score(const tile_ctx & c, int i_l, int j_l, bool valid, float scale) {
    const int rr = i_l - j_l + (C::TK - 1);
    const float s = scale * __half2float(c.Ss[i_l * C::TK + j_l])
                  + __half2float(c.Gs[i_l * C::RWP + rr])
                  + __half2float(c.Hs[j_l * C::RWP + rr]);
    return valid ? s : -INFINITY;
}

// One pass over the key tiles with the online softmax; P.V on the tensor cores, its accumulator
// rescaled through a small shared-memory round trip (the wmma fragment layout is opaque).
template <class C, bool F32>
__global__ void __launch_bounds__(C::NTHREADS) deberta_attn_tiled(const attn_args args) {
    extern __shared__ __align__(256) unsigned char smem[];
    tile_ctx c;
    c.Qs      = (__half *) (smem + C::SM_Q);
    c.Ks      = (__half *) (smem + C::SM_K);
    c.Vs      = (__half *) (smem + C::SM_V);
    c.Ws      = (__half *) (smem + C::SM_W);
    c.Gs      = (__half *) (smem + C::SM_G);
    c.Hs      = (__half *) (smem + C::SM_H);
    c.Ss      = (__half *) (smem + C::SM_S);
    c.scratch = (float  *) (smem + C::SM_SCRATCH) + (threadIdx.x / 32) * 256;

    c.i0    = blockIdx.x * C::TQ;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    c.n     = args.n;
    c.n_tab = (args.W - 2 * PAD + 1) / 2;
    c.len   = args.lens[b];
    c.warp  = threadIdx.x / 32;
    c.lane  = threadIdx.x % 32;
    const int warp = c.warp, lane = c.lane, n = c.n, len = c.len, i0 = c.i0;

    const char * qh = args.q + (size_t) h * args.q_nb2 + (size_t) b * args.q_nb3;
    c.kh     = args.k  + (size_t) h * args.k_nb2 + (size_t) b * args.k_nb3;
    c.vh     = args.v  + (size_t) h * args.v_nb2 + (size_t) b * args.v_nb3;
    c.Ph     = args.P  + (size_t) h * args.p_nb2;
    c.Qph    = args.Qp + (size_t) h * args.qp_nb2;
    c.k_nb1  = args.k_nb1;
    c.v_nb1  = args.v_nb1;
    c.p_nb1  = args.p_nb1;
    c.qp_nb1 = args.qp_nb1;

    stage_rows<C::NTHREADS, F32>(qh, args.q_nb1, i0, n, C::TQ, c.Qs); // Q tile, resident for the whole kernel

    const int n_tiles = (len + C::TK - 1) / C::TK;

    float m[C::ROWS_PW], l[C::ROWS_PW];
    #pragma unroll
    for (int r = 0; r < C::ROWS_PW; r++) { m[r] = -INFINITY; l[r] = 0.0f; }
    // warp w accumulates row tile w/2 x dim tiles {2*(w%2), 2*(w%2)+1}
    const int o_rf = warp / 2, o_nd = (warp % 2) * 2;
    frag_acc oacc[2];
    wmma::fill_fragment(oacc[0], 0.0f);
    wmma::fill_fragment(oacc[1], 0.0f);
    float * alphas = (float *) (smem + C::SM_SCRATCH);           // [TQ], aliases the (idle) scratch
    float * Ow     = (float *) (smem + C::SM_G) + warp * 16 * 32; // per-warp 16 x 32 f32 in the G/H region

    for (int t = 0; t < n_tiles; t++) {
        const int j0 = t * C::TK;
        compute_tile<C, F32>(c, j0, true);
        #pragma unroll
        for (int r = 0; r < C::ROWS_PW; r++) {
            const int i_l = warp * C::ROWS_PW + r;
            float sv[C::KPL];
            float tmax = -INFINITY;
            #pragma unroll
            for (int k = 0; k < C::KPL; k++) {
                const int j_l = lane + 32 * k;
                sv[k] = tile_score<C>(c, i_l, j_l, j0 + j_l < len, args.scale);
                tmax  = fmaxf(tmax, sv[k]);
            }
            const float m_new = fmaxf(m[r], warp_max(tmax));
            const float alpha = __expf(m[r] - m_new);   // 0 on the first tile (m = -inf)
            float psum = 0.0f;
            #pragma unroll
            for (int k = 0; k < C::KPL; k++) {
                const int j_l = lane + 32 * k;
                const float p = (j0 + j_l < len) ? __expf(sv[k] - m_new) : 0.0f;
                psum += p;
                c.Ss[i_l * C::TK + j_l] = __float2half(p);
            }
            l[r] = l[r] * alpha + warp_sum(psum);
            m[r] = m_new;
            if (lane == 0) alphas[i_l] = alpha;
        }
        __syncthreads(); // probabilities, alphas complete; G / H no longer needed
        // rescale the accumulator rows: fragments → smem → scale → fragments
        wmma::store_matrix_sync(Ow,      oacc[0], 32, wmma::mem_row_major);
        wmma::store_matrix_sync(Ow + 16, oacc[1], 32, wmma::mem_row_major);
        __syncwarp();
        #pragma unroll
        for (int e = lane; e < 16 * 32; e += 32) {
            Ow[e] *= alphas[o_rf * 16 + e / 32];
        }
        __syncwarp();
        wmma::load_matrix_sync(oacc[0], Ow,      32, wmma::mem_row_major);
        wmma::load_matrix_sync(oacc[1], Ow + 16, 32, wmma::mem_row_major);
        #pragma unroll
        for (int kk = 0; kk < C::KT; kk++) {
            frag_a pa;
            wmma::load_matrix_sync(pa, c.Ss + o_rf * 16 * C::TK + kk * 16, C::TK);
            #pragma unroll
            for (int d = 0; d < 2; d++) {
                frag_b_row vb;
                wmma::load_matrix_sync(vb, c.Vs + kk * 16 * LD + (o_nd + d) * 16, LD);
                wmma::mma_sync(oacc[d], pa, vb, oacc[d]);
            }
        }
    }
    // 1 / row sum, published for the epilogue
    __syncthreads();
    float * inv = alphas; // reuse
    #pragma unroll
    for (int r = 0; r < C::ROWS_PW; r++) {
        if (lane == 0) inv[warp * C::ROWS_PW + r] = l[r] > 0.0f ? 1.0f / l[r] : 0.0f;
    }

    // ---- epilogue: O fragments → smem (G/H region as TQ x 64 f32) → global, coalesced ----
    __syncthreads();
    float * Os = (float *) (smem + C::SM_G);
    wmma::store_matrix_sync(Os + o_rf * 16 * HD + o_nd * 16,       oacc[0], HD, wmma::mem_row_major);
    wmma::store_matrix_sync(Os + o_rf * 16 * HD + (o_nd + 1) * 16, oacc[1], HD, wmma::mem_row_major);
    __syncthreads();
    for (int idx = threadIdx.x; idx < C::TQ * (HD / 4); idx += C::NTHREADS) {
        const int r = idx / (HD / 4), c4 = idx % (HD / 4);
        const int i = i0 + r;
        if (i >= n) continue;
        const float sc = inv[r];
        float4 o = *(const float4 *) (Os + r * HD + c4 * 4);
        o.x *= sc; o.y *= sc; o.z *= sc; o.w *= sc;
        char * row = args.dst + (size_t) h * args.d_nb1 + (size_t) i * args.d_nb2 + (size_t) b * args.d_nb3;
        *(float4 *) (row + c4 * 16) = o;
    }
}

template <bool F32>
__device__ __forceinline__ float ld_elem(const char * row, int i) {
    return F32 ? ((const float *) row)[i] : __half2float(((const __half *) row)[i]);
}

// Reference: one warp per query row, everything in f32, no shared memory.
template <bool F32>
__global__ void deberta_attn_reference(const attn_args args) {
    const int i    = blockIdx.x * 4 + threadIdx.x / 32;
    const int h    = blockIdx.y;
    const int b    = blockIdx.z;
    const int lane = threadIdx.x % 32;
    const int n    = args.n;
    if (i >= n) return;
    const int len = args.lens[b];

    const char * q = args.q + (size_t) i * args.q_nb1 + (size_t) h * args.q_nb2 + (size_t) b * args.q_nb3;
    const float q0 = ld_elem<F32>(q, lane), q1 = ld_elem<F32>(q, lane + 32);

    float m = -INFINITY, l = 0.0f, o0 = 0.0f, o1 = 0.0f;
    for (int j = 0; j < len; j++) {
        const char * k  = args.k + (size_t) j * args.k_nb1 + (size_t) h * args.k_nb2 + (size_t) b * args.k_nb3;
        const char * v  = args.v + (size_t) j * args.v_nb1 + (size_t) h * args.v_nb2 + (size_t) b * args.v_nb3;
        const int     d  = PAD + i - j + (args.W - 2 * PAD + 1) / 2 - 1;
        const __half * P  = (const __half *) (args.P  + (size_t) d * args.p_nb1  + (size_t) h * args.p_nb2);
        const __half * Qp = (const __half *) (args.Qp + (size_t) d * args.qp_nb1 + (size_t) h * args.qp_nb2);
        const float k0 = ld_elem<F32>(k, lane), k1 = ld_elem<F32>(k, lane + 32);
        float s = args.scale * (q0 * k0 + q1 * k1)
                + (q0 * __half2float(P[lane]) + q1 * __half2float(P[lane + 32]))
                + (k0 * __half2float(Qp[lane]) + k1 * __half2float(Qp[lane + 32]));
        s = warp_sum(s);
        const float m_new = fmaxf(m, s);
        const float alpha = __expf(m - m_new);
        const float p     = __expf(s - m_new);
        l  = l * alpha + p;
        o0 = o0 * alpha + p * ld_elem<F32>(v, lane);
        o1 = o1 * alpha + p * ld_elem<F32>(v, lane + 32);
        m  = m_new;
    }
    const float inv = l > 0.0f ? 1.0f / l : 0.0f;
    float * row = (float *) (args.dst + (size_t) h * args.d_nb1 + (size_t) i * args.d_nb2 + (size_t) b * args.d_nb3);
    row[lane]      = o0 * inv;
    row[lane + 32] = o1 * inv;
}

// ---- register-resident variant --------------------------------------------------------------
// Same maths as deberta_attn_tiled, FlashAttention-2 structure: each warp owns 16 query rows; the
// q.k scores, the probabilities and the output accumulator never leave the registers (raw
// mma.sync m16n8k16 + ldmatrix, sm_75+). Only the two position products G = Q.P^T and H = K.Qp^T
// go through shared memory, because the score of (i, j) reads them on the diagonal i - j. That
// removes the f16 score tile, the probability tile and the accumulator rescale round trips of the
// wmma kernel, and keeps the scores in f32.
template <int NW_, int TK_>
struct RegCfg {
    static constexpr int NWARPS   = NW_;
    static constexpr int NTHREADS = NWARPS * 32;
    static constexpr int TQ       = NWARPS * 16;      // one m16 tile of queries per warp
    static constexpr int TK       = TK_;              // keys per tile, multiple of 16
    static constexpr int RW       = TQ + TK - 1;      // distinct offsets i - j inside a tile pair
    static constexpr int RWP      = round16(RW);
    static constexpr int LDG      = RWP + 8;          // G / H row stride (halves): conflict-free diagonal reads
    static constexpr int NT_W     = RWP / 8;          // n8 tiles across the window
    static constexpr int NT_K     = TK / 8;           // n8 tiles across the keys
    static constexpr int KT       = TK / 16;          // k16 steps over the keys (P.V) = K row tiles (H)
    static constexpr int ND       = HD / 8;           // n8 tiles across the head dim
    // shared memory layout (bytes); Q is staged into the G region before the first tile
    static constexpr size_t SM_G     = 0;                                  // TQ  x LDG f16
    static constexpr size_t SM_H     = SM_G + (size_t) TQ * LDG * 2;      // TK  x LDG f16
    static constexpr size_t SM_K     = SM_H + (size_t) TK * LDG * 2;      // TK  x LD  f16
    static constexpr size_t SM_V     = SM_K + (size_t) TK * LD * 2;       // TK  x LD  f16
    static constexpr size_t SM_WP    = SM_V + (size_t) TK * LD * 2;       // RWP x LD  f16: P window
    static constexpr size_t SM_WQ    = SM_WP + (size_t) RWP * LD * 2;     // RWP x LD  f16: Qp window
    static constexpr size_t SM_TOTAL = SM_WQ + (size_t) RWP * LD * 2;

    static constexpr int NW_PER_MT = NWARPS >= KT ? NWARPS / KT : 1;   // warps sharing one K row tile in H

    static_assert(TK % 16 == 0 && (NWARPS % KT == 0 || KT % NWARPS == 0), "tile shape");
    static_assert(LD <= LDG, "Q staging must fit in the G region");
    static_assert(SM_TOTAL <= 227 * 1024, "shared memory (sm_90 opt-in is 227 KB)");
};

__device__ __forceinline__ uint32_t smem_addr(const void * p) {
    return (uint32_t) __cvta_generic_to_shared(p);
}

__device__ __forceinline__ void ldsm_x4(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const __half * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
}

__device__ __forceinline__ void ldsm_x4_trans(uint32_t & r0, uint32_t & r1, uint32_t & r2, uint32_t & r3, const __half * p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(smem_addr(p)));
}

// c(16 x 8, f32) += a(16 x 16, f16 row) . b(16 x 8, f16 col); Turing has no k16 shape, two k8 steps there
__device__ __forceinline__ void mma_16816(float * c, const uint32_t * a, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]) : "r"(a[2]), "r"(a[3]), "r"(b1));
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

// A fragments (4 k16 steps over the head dim) of the 16 rows at `rows` (stride LD)
__device__ __forceinline__ void load_a_frags(uint32_t (&a)[4][4], const __half * rows, int lane) {
    const __half * p = rows + (lane % 16) * LD + (lane / 16) * 8;
    #pragma unroll
    for (int ks = 0; ks < HD / 16; ks++) {
        ldsm_x4(a[ks][0], a[ks][1], a[ks][2], a[ks][3], p + ks * 16);
    }
}

// out(16 x 16 f16 at `out`, stride ld_out) = A(16 x HD, fragments) . B(16 rows at `brows`, stride LD)^T
__device__ __forceinline__ void mma_rows16(const uint32_t (&a)[4][4], const __half * brows, __half * out, int ld_out, int lane) {
    float c0[4] = { 0.0f, 0.0f, 0.0f, 0.0f }, c1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    const __half * p = brows + ((lane % 8) + (lane / 16) * 8) * LD + ((lane / 8) % 2) * 8;
    #pragma unroll
    for (int ks = 0; ks < HD / 16; ks++) {
        uint32_t b0, b1, b2, b3;
        ldsm_x4(b0, b1, b2, b3, p + ks * 16);
        mma_16816(c0, a[ks], b0, b1);
        mma_16816(c1, a[ks], b2, b3);
    }
    const int g = lane >> 2, qd = lane & 3;
    *(uint32_t *) (out + g * ld_out + qd * 2)           = pack_half2(c0[0], c0[1]);
    *(uint32_t *) (out + (g + 8) * ld_out + qd * 2)     = pack_half2(c0[2], c0[3]);
    *(uint32_t *) (out + g * ld_out + 8 + qd * 2)       = pack_half2(c1[0], c1[1]);
    *(uint32_t *) (out + (g + 8) * ld_out + 8 + qd * 2) = pack_half2(c1[2], c1[3]);
}

// stage `n_rows` table rows starting at `row0` into smem (stride LD), row index clamped to [0, W):
// rows outside the table only ever pair with query rows >= n (discarded), clamping keeps them finite
template <int NTHREADS>
__device__ __forceinline__ void stage_rows_clamped(const char * src, long nb1, int row0, int W, int n_rows, __half * dst) {
    for (int idx = threadIdx.x; idx < n_rows * 8; idx += NTHREADS) {
        const int r = idx / 8, c = idx % 8;
        const int row = min(max(row0 + r, 0), W - 1);
        *(uint4 *) (dst + r * LD + c * 8) = *(const uint4 *) (src + (size_t) row * nb1 + c * 16);
    }
}

template <class C, bool F32>
__global__ void __launch_bounds__(C::NTHREADS) deberta_attn_reg(const attn_args args) {
    extern __shared__ __align__(256) unsigned char smem[];
    __half * Gs  = (__half *) (smem + C::SM_G);
    __half * Hs  = (__half *) (smem + C::SM_H);
    __half * Ks  = (__half *) (smem + C::SM_K);
    __half * Vs  = (__half *) (smem + C::SM_V);
    __half * WPs = (__half *) (smem + C::SM_WP);
    __half * WQs = (__half *) (smem + C::SM_WQ);

    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32, g = lane >> 2, qd = lane & 3;
    const int i0 = blockIdx.x * C::TQ, h = blockIdx.y, b = blockIdx.z;
    const int n = args.n, W = args.W, n_tab = (W - 2 * PAD + 1) / 2;
    const int len = args.lens[b];

    const char * qh  = args.q  + (size_t) h * args.q_nb2  + (size_t) b * args.q_nb3;
    const char * kh  = args.k  + (size_t) h * args.k_nb2  + (size_t) b * args.k_nb3;
    const char * vh  = args.v  + (size_t) h * args.v_nb2  + (size_t) b * args.v_nb3;
    const char * Ph  = args.P  + (size_t) h * args.p_nb2;
    const char * Qph = args.Qp + (size_t) h * args.qp_nb2;

    // Q: staged once into the (still idle) G region, the warp's A fragments stay in registers
    stage_rows<C::NTHREADS, F32>(qh, args.q_nb1, i0, n, C::TQ, Gs);
    __syncthreads();
    uint32_t qa[4][4];
    load_a_frags(qa, Gs + warp * 16 * LD, lane);

    float o[C::ND][4];
    #pragma unroll
    for (int d = 0; d < C::ND; d++) { o[d][0] = o[d][1] = o[d][2] = o[d][3] = 0.0f; }
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;   // rows g and g + 8 of the warp's tile
    const int r0 = warp * 16 + g, r1 = r0 + 8;

    const int n_tiles = (len + C::TK - 1) / C::TK;
    for (int t = 0; t < n_tiles; t++) {
        const int j0 = t * C::TK;
        __syncthreads(); // previous tile fully consumed (V, G, H; Q fragments already loaded)
        stage_rows<C::NTHREADS, F32>(kh, args.k_nb1, j0, len, C::TK, Ks);
        stage_rows<C::NTHREADS, F32>(vh, args.v_nb1, j0, len, C::TK, Vs);
        // local offset r = i_l - j_l + (TK - 1) ∈ [0, RW) ↔ table row PAD + i - j + n_tab - 1
        const int base = PAD + i0 - j0 + n_tab - C::TK;
        stage_rows_clamped<C::NTHREADS>(Ph,  args.p_nb1,  base, W, C::RWP, WPs);
        stage_rows_clamped<C::NTHREADS>(Qph, args.qp_nb1, base, W, C::RWP, WQs);
        __syncthreads();

        // G rows of this warp = Q_w . P_window^T (f16 into smem)
        #pragma unroll
        for (int np = 0; np < C::NT_W / 2; np++) {
            mma_rows16(qa, WPs + np * 16 * LD, Gs + warp * 16 * C::LDG + np * 16, C::LDG, lane);
        }
        // S = Q_w . K^T stays in registers: s[nt] covers keys nt*8 .. nt*8+7
        float s[C::NT_K][4];
        #pragma unroll
        for (int nt = 0; nt < C::NT_K; nt++) { s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f; }
        {
            const __half * p = Ks + ((lane % 8) + (lane / 16) * 8) * LD + ((lane / 8) % 2) * 8;
            #pragma unroll
            for (int np = 0; np < C::NT_K / 2; np++) {
                #pragma unroll
                for (int ks = 0; ks < HD / 16; ks++) {
                    uint32_t b0, b1, b2, b3;
                    ldsm_x4(b0, b1, b2, b3, p + np * 16 * LD + ks * 16);
                    mma_16816(s[2 * np],     qa[ks], b0, b1);
                    mma_16816(s[2 * np + 1], qa[ks], b2, b3);
                }
            }
        }
        // H = K . Qp_window^T: key row tiles warp % KT (+ NWARPS ...), window column tiles strided
        // over the warps sharing a row tile
        for (int mt = warp % C::KT; mt < C::KT; mt += C::NWARPS) {
            uint32_t ka[4][4];
            load_a_frags(ka, Ks + mt * 16 * LD, lane);
            for (int np = warp / C::KT; np < C::NT_W / 2; np += C::NW_PER_MT) {
                mma_rows16(ka, WQs + np * 16 * LD, Hs + mt * 16 * C::LDG + np * 16, C::LDG, lane);
            }
        }
        __syncthreads(); // G, H complete

        // full scores, online softmax over this tile (rows g / g + 8, the quad shares a row)
        float mx0 = -INFINITY, mx1 = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < C::NT_K; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int j_l = nt * 8 + qd * 2 + (e & 1);
                const int rl  = (e < 2) ? r0 : r1;
                const int rr  = rl - j_l + (C::TK - 1);
                float v = args.scale * s[nt][e] + __half2float(Gs[rl * C::LDG + rr]) + __half2float(Hs[j_l * C::LDG + rr]);
                v = (j0 + j_l < len) ? v : -INFINITY;
                s[nt][e] = v;
                if (e < 2) mx0 = fmaxf(mx0, v); else mx1 = fmaxf(mx1, v);
            }
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float mn0 = fmaxf(m0, mx0), mn1 = fmaxf(m1, mx1);
        const float a0 = __expf(m0 - mn0), a1 = __expf(m1 - mn1);   // 0 on the first tile (m = -inf)
        float ps0 = 0.0f, ps1 = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < C::NT_K; nt++) {
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const int j_l = nt * 8 + qd * 2 + (e & 1);
                const float p = (j0 + j_l < len) ? __expf(s[nt][e] - ((e < 2) ? mn0 : mn1)) : 0.0f;
                s[nt][e] = p;
                if (e < 2) ps0 += p; else ps1 += p;
            }
        }
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        l0 = l0 * a0 + ps0; m0 = mn0;
        l1 = l1 * a1 + ps1; m1 = mn1;
        #pragma unroll
        for (int d = 0; d < C::ND; d++) { o[d][0] *= a0; o[d][1] *= a0; o[d][2] *= a1; o[d][3] *= a1; }

        // O += P . V: the probability accumulators re-pack as f16 A fragments (two n8 tiles = one k16 step)
        #pragma unroll
        for (int ks = 0; ks < C::KT; ks++) {
            uint32_t pa[4];
            pa[0] = pack_half2(s[2 * ks][0],     s[2 * ks][1]);
            pa[1] = pack_half2(s[2 * ks][2],     s[2 * ks][3]);
            pa[2] = pack_half2(s[2 * ks + 1][0], s[2 * ks + 1][1]);
            pa[3] = pack_half2(s[2 * ks + 1][2], s[2 * ks + 1][3]);
            const __half * vp = Vs + (ks * 16 + (lane % 8) + ((lane / 8) % 2) * 8) * LD + (lane / 16) * 8;
            #pragma unroll
            for (int dp = 0; dp < C::ND / 2; dp++) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4_trans(b0, b1, b2, b3, vp + dp * 16);
                mma_16816(o[2 * dp],     pa, b0, b1);
                mma_16816(o[2 * dp + 1], pa, b2, b3);
            }
        }
    }

    // epilogue: normalise, write the two rows of this thread (8-byte stores, dims 8 d + 2 qd)
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    const int i_a = i0 + r0, i_b = i0 + r1;
    if (i_a < n) {
        char * row = args.dst + (size_t) h * args.d_nb1 + (size_t) i_a * args.d_nb2 + (size_t) b * args.d_nb3;
        #pragma unroll
        for (int d = 0; d < C::ND; d++) {
            *(float2 *) (row + (d * 8 + qd * 2) * 4) = make_float2(o[d][0] * inv0, o[d][1] * inv0);
        }
    }
    if (i_b < n) {
        char * row = args.dst + (size_t) h * args.d_nb1 + (size_t) i_b * args.d_nb2 + (size_t) b * args.d_nb3;
        #pragma unroll
        for (int d = 0; d < C::ND; d++) {
            *(float2 *) (row + (d * 8 + qd * 2) * 4) = make_float2(o[d][2] * inv1, o[d][3] * inv1);
        }
    }
}

using CfgReg16x64  = RegCfg<1, 64>;   //  56 KB: one warp, exercises the 64-key path on a 64 KB device (tests only)
using CfgReg64x32  = RegCfg<4, 32>;   //  56 KB: fits Turing's 64 KB (one block per SM)
using CfgReg64x64  = RegCfg<4, 64>;   //  88 KB: sm_80 / sm_90 two blocks per SM, sm_86 / sm_89 one
using CfgReg128x32 = RegCfg<8, 32>;   // 108 KB: two blocks per SM on sm_80 / sm_90
using CfgReg128x64 = RegCfg<8, 64>;   // 147 KB: one block per SM on sm_80 / sm_90

// ---- runtime kernel selection --------------------------------------------------------------
enum kernel_variant { KV_TURING = 0, KV_AMPERE, KV_REG64x32, KV_REG64x64, KV_REG128x32, KV_REG128x64, KV_REG16x64, KV_COUNT };

struct variant_info {
    const char * name;
    size_t       smem;
    bool         tuned;   // candidate for the autotune (false: test-only shapes)
};

const variant_info VARIANTS[KV_COUNT] = {
    { "turing",    CfgTuring::SM_TOTAL,    true  },   // wmma, 64 x 32 tiles
    { "ampere",    CfgAmpere::SM_TOTAL,    true  },   // wmma, 64 x 64 tiles
    { "reg64x32",  CfgReg64x32::SM_TOTAL,  true  },   // register-resident, 4 warps x 32 keys
    { "reg64x64",  CfgReg64x64::SM_TOTAL,  true  },
    { "reg128x32", CfgReg128x32::SM_TOTAL, true  },
    { "reg128x64", CfgReg128x64::SM_TOTAL, true  },
    { "reg16x64",  CfgReg16x64::SM_TOTAL,  false },
};

template <class C, void (*KERNEL)(const attn_args)>
cudaError_t launch_cfg(const attn_args & a, int B, cudaStream_t stream) {
    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t err = cudaFuncSetAttribute(KERNEL, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) C::SM_TOTAL);
        if (err != cudaSuccess) {
            return err;
        }
        attr_set = true;
    }
    const dim3 grid((a.n + C::TQ - 1) / C::TQ, a.heads, B);
    KERNEL<<<grid, C::NTHREADS, C::SM_TOTAL, stream>>>(a);
    return cudaGetLastError();
}

// one instantiation per (variant, operand type): f16 q / k / v, or f32 converted while staging
#define GLINER4J_LAUNCH(CFG, KERNEL) \
    return a.qkv_f32 ? launch_cfg<CFG, KERNEL<CFG, true>>(a, B, stream) : launch_cfg<CFG, KERNEL<CFG, false>>(a, B, stream)

cudaError_t launch_variant(kernel_variant v, const attn_args & a, int B, cudaStream_t stream) {
    switch (v) {
        case KV_AMPERE:    GLINER4J_LAUNCH(CfgAmpere,    deberta_attn_tiled);
        case KV_REG64x32:  GLINER4J_LAUNCH(CfgReg64x32,  deberta_attn_reg);
        case KV_REG64x64:  GLINER4J_LAUNCH(CfgReg64x64,  deberta_attn_reg);
        case KV_REG128x32: GLINER4J_LAUNCH(CfgReg128x32, deberta_attn_reg);
        case KV_REG128x64: GLINER4J_LAUNCH(CfgReg128x64, deberta_attn_reg);
        case KV_REG16x64:  GLINER4J_LAUNCH(CfgReg16x64,  deberta_attn_reg);
        default:           GLINER4J_LAUNCH(CfgTuring,    deberta_attn_tiled);
    }
}
#undef GLINER4J_LAUNCH

// uniform [-scale, scale) f16 from a hash of the index (autotune inputs)
__global__ void fill_hash(__half * dst, size_t count, uint32_t seed, float scale) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    uint32_t x = (uint32_t) i * 2654435761u ^ seed;
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    dst[i] = __float2half(((float) (x & 0xffffff) / 16777216.0f * 2.0f - 1.0f) * scale);
}

char report[1024];   // autotune table + choice of the last resolved device (log / host report)

struct device_choice {
    bool           resolved = false;
    kernel_variant variant  = KV_TURING;
    int            cc       = 0;
    int            smem     = 0;
};

// Times every variant the device can hold on a synthetic (n = 1024, 12 heads) problem, checks it
// against the scalar reference kernel and returns the fastest correct one. Runs once per device,
// on the first call (~100 ms). Every result is logged so a deployment shows which kernel it runs.
kernel_variant autotune(const cudaDeviceProp & prop, int smem_opt_in, kernel_variant fallback) {
    const int n = 1024, heads = 12, B = 1, WP = 2 * n - 1 + 2 * PAD;
    const size_t qkv = (size_t) HD * n * heads, tab = (size_t) HD * WP * heads, out = (size_t) HD * heads * n;
    __half *dq = nullptr, *dk = nullptr, *dv = nullptr, *dP = nullptr, *dQp = nullptr;
    float *ddst = nullptr;
    int32_t * dlens = nullptr;
    bool ok = cudaMalloc(&dq, qkv * 2) == cudaSuccess && cudaMalloc(&dk, qkv * 2) == cudaSuccess && cudaMalloc(&dv, qkv * 2) == cudaSuccess
           && cudaMalloc(&dP, tab * 2) == cudaSuccess && cudaMalloc(&dQp, tab * 2) == cudaSuccess
           && cudaMalloc(&ddst, out * 4) == cudaSuccess && cudaMalloc(&dlens, 4) == cudaSuccess;
    float * ref = ok ? (float *) malloc(out * 4) : nullptr;
    float * got = ok ? (float *) malloc(out * 4) : nullptr;
    kernel_variant best = fallback;
    if (ok && ref && got) {
        const int lens = n - 5;
        cudaMemcpy(dlens, &lens, 4, cudaMemcpyHostToDevice);
        fill_hash<<<(unsigned) ((qkv + 255) / 256), 256>>>(dq, qkv, 1u, 1.0f);
        fill_hash<<<(unsigned) ((qkv + 255) / 256), 256>>>(dk, qkv, 2u, 1.0f);
        fill_hash<<<(unsigned) ((qkv + 255) / 256), 256>>>(dv, qkv, 3u, 1.0f);
        fill_hash<<<(unsigned) ((tab + 255) / 256), 256>>>(dP, tab, 4u, 0.3f);
        fill_hash<<<(unsigned) ((tab + 255) / 256), 256>>>(dQp, tab, 5u, 0.3f);
        attn_args a;
        a.q = (const char *) dq; a.k = (const char *) dk; a.v = (const char *) dv;
        a.P = (const char *) dP; a.Qp = (const char *) dQp; a.lens = dlens; a.dst = (char *) ddst;
        a.n = n; a.heads = heads; a.W = WP; a.scale = 1.0f / sqrtf(3.0f * HD); a.qkv_f32 = false;
        a.q_nb1 = a.k_nb1 = a.v_nb1 = HD * 2; a.q_nb2 = a.k_nb2 = a.v_nb2 = (long) HD * n * 2; a.q_nb3 = a.k_nb3 = a.v_nb3 = (long) HD * n * heads * 2;
        a.p_nb1 = a.qp_nb1 = HD * 2; a.p_nb2 = a.qp_nb2 = (long) HD * WP * 2;
        a.d_nb1 = HD * 4; a.d_nb2 = (long) HD * heads * 4; a.d_nb3 = (long) HD * heads * n * 4;
        deberta_attn_reference<false><<<dim3((n + 3) / 4, heads, B), 128>>>(a);
        ok = cudaDeviceSynchronize() == cudaSuccess && cudaMemcpy(ref, ddst, out * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
        float ref_max = 0.0f;
        for (size_t i = 0; ok && i < out; i++) ref_max = fmaxf(ref_max, fabsf(ref[i]));
        const float tol = 0.02f * fmaxf(1.0f, ref_max);
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0);
        cudaEventCreate(&e1);
        float best_ms = 1e30f;
        char line[512];
        int pos = snprintf(line, sizeof(line), "gliner4j deberta: autotune n=%d heads=%d:", n, heads);
        for (int v = 0; ok && v < KV_COUNT; v++) {
            if (!VARIANTS[v].tuned || VARIANTS[v].smem > (size_t) smem_opt_in) continue;
            cudaMemset(ddst, 0, out * 4);
            cudaError_t err = launch_variant((kernel_variant) v, a, B, 0);
            if (err != cudaSuccess || cudaDeviceSynchronize() != cudaSuccess) {
                cudaGetLastError();
                pos += snprintf(line + pos, sizeof(line) - pos, " %s launch-error", VARIANTS[v].name);
                continue;
            }
            cudaMemcpy(got, ddst, out * 4, cudaMemcpyDeviceToHost);
            float worst = 0.0f;
            for (size_t i = 0; i < out; i++) worst = fmaxf(worst, fabsf(got[i] - ref[i]));
            float ms = 1e30f;
            for (int round = 0; round < 3; round++) {
                cudaEventRecord(e0, 0);
                for (int w = 0; w < 5; w++) launch_variant((kernel_variant) v, a, B, 0);
                cudaEventRecord(e1, 0);
                cudaEventSynchronize(e1);
                float r = 0.0f;
                cudaEventElapsedTime(&r, e0, e1);
                ms = fminf(ms, r / 5);
            }
            const bool correct = worst <= tol;
            pos += snprintf(line + pos, sizeof(line) - pos, " %s %.0f us%s", VARIANTS[v].name, ms * 1000, correct ? "" : " (WRONG)");
            if (correct && ms < best_ms) { best_ms = ms; best = (kernel_variant) v; }
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
        GGML_LOG_INFO("%s\n", line);
        snprintf(report, sizeof(report), "%s", line + std::strlen("gliner4j deberta: "));
    }
    free(ref);
    free(got);
    cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dP); cudaFree(dQp); cudaFree(ddst); cudaFree(dlens);
    cudaGetLastError();
    return best;
}

int variant_by_name(const char * name) {
    for (int v = 0; v < KV_COUNT; v++) {
        if (std::strcmp(name, VARIANTS[v].name) == 0) return v;
    }
    return -1;
}

const char * forced_variant = nullptr;   // set through gliner4j_deberta_attn_force_variant (test harness)

// One choice per CUDA device: GLINER4J_DEBERTA_KERNEL=<variant> forces one (if the device holds
// it), otherwise the autotune picks the fastest correct one; GLINER4J_DEBERTA_AUTOTUNE=0 falls
// back to the static rule (wmma 64x64 where it fits, 64x32 elsewhere). Logged once.
device_choice choices[64];

device_choice & choose_kernel(int device) {
    device_choice & ch = choices[device & 63];
    if (ch.resolved) {
        return ch;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        ch.resolved = true; // fall back to the smallest variant
        return ch;
    }
    ch.cc   = prop.major * 10 + prop.minor;
    ch.smem = (int) prop.sharedMemPerBlockOptin;
    report[0] = 0;
    const kernel_variant rule = ch.smem >= (int) CfgAmpere::SM_TOTAL ? KV_AMPERE : KV_TURING;
    const char * force = forced_variant != nullptr ? forced_variant : std::getenv("GLINER4J_DEBERTA_KERNEL");
    const char * tune  = std::getenv("GLINER4J_DEBERTA_AUTOTUNE");
    const char * how   = "autotune";
    int v = -1;
    if (force != nullptr && std::strcmp(force, "auto") != 0) {
        v = variant_by_name(force);
        if (v < 0) {
            GGML_LOG_WARN("gliner4j deberta: unknown kernel '%s' (turing, ampere, reg64x32, reg64x64, reg128x32, reg128x64, reg16x64, auto)\n", force);
        } else if (VARIANTS[v].smem > (size_t) ch.smem) {
            GGML_LOG_WARN("gliner4j deberta: kernel '%s' needs %zu KB of shared memory, device offers %d KB\n", force, VARIANTS[v].smem / 1024, ch.smem / 1024);
            v = -1;
        } else {
            how = "forced";
        }
    }
    if (v < 0) {
        if (tune != nullptr && std::strcmp(tune, "0") == 0) {
            v = rule;
            how = "static rule";
        } else {
            v = autotune(prop, ch.smem, rule);
        }
    }
    ch.variant = (kernel_variant) v;
    GGML_LOG_INFO("gliner4j deberta: %s (sm_%d, %d KB smem/block) → %s kernel (%s)\n",
        prop.name, ch.cc, ch.smem / 1024, VARIANTS[ch.variant].name, how);
    const size_t used = std::strcmp(how, "autotune") == 0 ? std::strlen(report) : 0;
    snprintf(report + used, sizeof(report) - used, "%s%s (sm_%d, %d KB smem/block) → %s kernel (%s)",
        used ? "; " : "", prop.name, ch.cc, ch.smem / 1024, VARIANTS[ch.variant].name, how);
    ch.resolved = true;
    return ch;
}

} // namespace

int gliner4j_deberta_attn_force_variant(const char * name) {
    if (name != nullptr && std::strcmp(name, "auto") != 0) {
        const int v = variant_by_name(name);
        if (v < 0) {
            return -1;
        }
        int device = 0;
        cudaDeviceProp prop;
        if (cudaGetDevice(&device) == cudaSuccess && cudaGetDeviceProperties(&prop, device) == cudaSuccess
            && VARIANTS[v].smem > prop.sharedMemPerBlockOptin) {
            return -2;
        }
    }
    forced_variant = name;
    for (auto & c : choices) c.resolved = false;   // re-resolved (and logged) on the next launch
    return 0;
}

const char * gliner4j_deberta_attn_report() {
    int device = 0;
    cudaGetDevice(&device);
    choose_kernel(device);
    return report;
}

const char * gliner4j_deberta_attn_variant_name() {
    int device = 0;
    cudaGetDevice(&device);
    return VARIANTS[choose_kernel(device).variant].name;
}

cudaError_t gliner4j_deberta_attn_launch(const ggml_tensor * dst, cudaStream_t stream, bool reference) {
    const ggml_tensor * q    = dst->src[0];
    const ggml_tensor * k    = dst->src[1];
    const ggml_tensor * v    = dst->src[2];
    const ggml_tensor * P    = dst->src[3];
    const ggml_tensor * Qp   = dst->src[4];
    const ggml_tensor * lens = dst->src[5];
    GGML_ASSERT(q && k && v && P && Qp && lens);
    // q / k / v: f16, or the f32 projections as they come out of ggml's matmul (converted while staging)
    const bool qkv_f32 = q->type == GGML_TYPE_F32;
    GGML_ASSERT((q->type == GGML_TYPE_F16 || qkv_f32) && k->type == q->type && v->type == q->type);
    GGML_ASSERT(P->type == GGML_TYPE_F16 && Qp->type == GGML_TYPE_F16 && lens->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(q->ne[0] == HD && k->ne[0] == HD && v->ne[0] == HD && P->ne[0] == HD && Qp->ne[0] == HD);
    const size_t es = qkv_f32 ? 4 : 2;
    GGML_ASSERT(q->nb[0] == es && k->nb[0] == es && v->nb[0] == es && P->nb[0] == 2 && Qp->nb[0] == 2);
    const int n     = (int) q->ne[1];
    const int heads = (int) q->ne[2];
    const int B     = (int) q->ne[3];
    GGML_ASSERT(k->ne[1] == n && k->ne[2] == heads && k->ne[3] == B);
    GGML_ASSERT(v->ne[1] == n && v->ne[2] == heads && v->ne[3] == B);
    // tables cover offsets of a length n_tab >= n: rows = 2*n_tab - 1 + 2*PAD
    GGML_ASSERT(P->ne[1] >= 2 * n - 1 + 2 * PAD && (P->ne[1] - 2 * PAD) % 2 == 1 && P->ne[2] == heads);
    GGML_ASSERT(Qp->ne[1] == P->ne[1] && Qp->ne[2] == heads);
    GGML_ASSERT(ggml_is_contiguous(P) && ggml_is_contiguous(Qp));
    GGML_ASSERT(lens->ne[0] == B);
    GGML_ASSERT(dst->ne[0] == HD && dst->ne[1] == heads && dst->ne[2] == n && dst->ne[3] == B);
    GGML_ASSERT(ggml_is_contiguous(dst));
    // 16-byte vector loads of Q / K / V rows and 32-byte-aligned wmma windows on the tables
    GGML_ASSERT(q->nb[1] % 16 == 0 && k->nb[1] % 16 == 0 && v->nb[1] % 16 == 0);
    GGML_ASSERT(((uintptr_t) q->data) % 16 == 0 && ((uintptr_t) k->data) % 16 == 0 && ((uintptr_t) v->data) % 16 == 0);
    GGML_ASSERT(((uintptr_t) P->data) % 32 == 0 && ((uintptr_t) Qp->data) % 32 == 0);

    attn_args a;
    a.q = (const char *) q->data;  a.k = (const char *) k->data;  a.v = (const char *) v->data;
    a.P = (const char *) P->data;  a.Qp = (const char *) Qp->data;
    a.lens = (const int32_t *) lens->data;
    a.dst = (char *) dst->data;
    a.n = n; a.heads = heads; a.W = (int) P->ne[1];
    a.scale = 1.0f / sqrtf(3.0f * HD);
    a.qkv_f32 = qkv_f32;
    a.q_nb1 = q->nb[1]; a.q_nb2 = q->nb[2]; a.q_nb3 = q->nb[3];
    a.k_nb1 = k->nb[1]; a.k_nb2 = k->nb[2]; a.k_nb3 = k->nb[3];
    a.v_nb1 = v->nb[1]; a.v_nb2 = v->nb[2]; a.v_nb3 = v->nb[3];
    a.p_nb1 = P->nb[1]; a.p_nb2 = P->nb[2];
    a.qp_nb1 = Qp->nb[1]; a.qp_nb2 = Qp->nb[2];
    a.d_nb1 = dst->nb[1]; a.d_nb2 = dst->nb[2]; a.d_nb3 = dst->nb[3];
    return gliner4j_deberta_attn_launch_args(a, B, stream, reference);
}

cudaError_t gliner4j_deberta_attn_launch_args(const gliner4j_deberta_attn_args & a, int B, cudaStream_t stream, bool reference) {
    if (reference) {
        const dim3 grid((a.n + 3) / 4, a.heads, B);
        if (a.qkv_f32) deberta_attn_reference<true><<<grid, 128, 0, stream>>>(a);
        else           deberta_attn_reference<false><<<grid, 128, 0, stream>>>(a);
        return cudaGetLastError();
    }
    int device = 0;
    cudaGetDevice(&device);
    const device_choice & ch = choose_kernel(device);
    return launch_variant(ch.variant, a, B, stream);
}
