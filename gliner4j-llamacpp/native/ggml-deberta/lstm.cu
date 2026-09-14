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
// Bidirectional LSTM recurrence in one launch. The recurrent product W_hh·h(t-1) is the whole
// cost — 4H·H MACs per step and direction, i.e. the 2.4 MB matrix (H = 384) read once per step —
// and it has to be *spread over the GPU*: one block per direction (the serial kernel below,
// kept as the fallback) pulls the matrix through a single SM's L2 path and lands at ~60 ms for
// 400 words. The cooperative kernel gives every block U hidden units, i.e. their 4U gate rows
// (one warp per row, lanes strided over the inputs), keeps the cells in registers of the owning
// threads, exchanges h(t) through the output tensor itself and separates the steps with a
// grid-wide barrier. Both directions run in the same grid (blocks [0, nb) forward, [nb, 2nb)
// backward). The composed ggml graph did the same as ~20 tiny kernels per step and direction.

#include "lstm.h"
#include "ggml.h"
#include "ggml-impl.h"

#include <cooperative_groups.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace cg = cooperative_groups;

namespace {

__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + __expf(-x)); }

// ---- cooperative kernel ------------------------------------------------------------------------
constexpr int U  = 16;   // hidden units per block → 4U gate rows
constexpr int BT = 512;  // 16 warps: each warp handles 4U / 16 = 4 gate rows per step

// Reads 4 consecutive weights of a W_hh row as float4 (f32 rows) or half4 (f16 rows).
__device__ __forceinline__ float4 load4(const float * p) { return *(const float4 *) p; }
__device__ __forceinline__ float4 load4(const __half * p) {
    const __half2 * h2 = (const __half2 *) p;
    const float2 a = __half22float2(h2[0]);
    const float2 b = __half22float2(h2[1]);
    return make_float4(a.x, a.y, b.x, b.y);
}

template <typename TW>
__global__ void __launch_bounds__(BT)
lstm_coop_kernel(const float * __restrict__ xw_f, const float * __restrict__ xw_b,
                 const TW * __restrict__ whh_f, const TW * __restrict__ whh_b,
                 float * __restrict__ out, int H, int W,
                 long xw_nb1, long whh_nb1, long out_nb1) {
    cg::grid_group grid = cg::this_grid();
    __shared__ float gates[4 * U];
    const int  nb      = H / U;
    const bool reverse = (int) blockIdx.x >= nb;
    const int  j0      = (reverse ? (int) blockIdx.x - nb : (int) blockIdx.x) * U;
    const float * xw   = reverse ? xw_b  : xw_f;
    const TW * whh     = reverse ? whh_b : whh_f;
    const int  hoff    = reverse ? H : 0;
    const int  warp    = threadIdx.x >> 5;
    const int  lane    = threadIdx.x & 31;
    const int  H4      = H >> 2;
    float c = 0.0f;  // cell of unit j0 + threadIdx.x (threads < U)

    for (int step = 0; step < W; step++) {
        const int t     = reverse ? W - 1 - step : step;
        const int tprev = reverse ? t + 1 : t - 1;
        const float * xt    = (const float *) ((const char *) xw + (size_t) t * xw_nb1);
        const float * hprev = step == 0 ? nullptr
                            : (const float *) ((const char *) out + (size_t) tprev * out_nb1) + hoff;
        for (int rr = warp; rr < 4 * U; rr += BT / 32) {
            const int gate = rr / U;
            const int u    = rr - gate * U;
            const int r    = gate * H + j0 + u;   // row of W_hh (PyTorch order i, f, g, o)
            float acc = 0.0f;
            if (hprev != nullptr) {
                const TW * row = (const TW *) ((const char *) whh + (size_t) r * whh_nb1);
                const float4 * hv = (const float4 *) hprev;
                for (int k = lane; k < H4; k += 32) {
                    const float4 a = load4(row + 4 * k);
                    const float4 b = __ldcg(hv + k);   // written by other blocks: bypass L1
                    acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
                }
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) gates[rr] = xt[r] + acc;
        }
        __syncthreads();
        if (threadIdx.x < U) {
            const int   u = threadIdx.x;
            const float i = sigmoidf_(gates[u]);
            const float f = sigmoidf_(gates[U + u]);
            const float g = tanhf(gates[2 * U + u]);
            const float o = sigmoidf_(gates[3 * U + u]);
            c = f * c + i * g;
            const float hn = o * tanhf(c);
            __stcg((float *) ((char *) out + (size_t) t * out_nb1) + hoff + j0 + u, hn);
        }
        grid.sync();
    }
}

// ---- serial fallback (one block per direction) --------------------------------------------------
constexpr int NTHREADS = 1024;
constexpr int MAX_H = 1024;   // h + gates in shared memory (5·H floats)

template <typename TW>
__global__ void __launch_bounds__(NTHREADS)
lstm_serial_kernel(const float * __restrict__ xw_f, const float * __restrict__ xw_b,
                   const TW * __restrict__ whh_f, const TW * __restrict__ whh_b,
                   float * __restrict__ out, int H, int W,
                   long xw_nb1, long whh_nb1, long out_nb1) {
    extern __shared__ float smem[];
    float * h     = smem;
    float * gates = smem + H;
    const bool reverse = blockIdx.x == 1;
    const float * xw  = reverse ? xw_b  : xw_f;
    const TW * whh = reverse ? whh_b : whh_f;
    const int G = 4 * H;
    const int tid = threadIdx.x;
    for (int j = tid; j < H; j += NTHREADS) h[j] = 0.0f;
    float c = 0.0f;
    __syncthreads();
    for (int step = 0; step < W; step++) {
        const int t = reverse ? W - 1 - step : step;
        const float * xt = (const float *) ((const char *) xw + (size_t) t * xw_nb1);
        for (int r = tid; r < G; r += NTHREADS) {
            const TW * row = (const TW *) ((const char *) whh + (size_t) r * whh_nb1);
            const float4 * hv = (const float4 *) h;
            float acc = 0.0f;
            const int H4 = H >> 2;
            #pragma unroll 4
            for (int k = 0; k < H4; k++) {
                const float4 a = load4(row + 4 * k);
                const float4 b = hv[k];
                acc += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
            }
            gates[r] = xt[r] + (step == 0 ? 0.0f : acc);
        }
        __syncthreads();
        if (tid < H) {
            const float i = sigmoidf_(gates[tid]);
            const float f = sigmoidf_(gates[H + tid]);
            const float g = tanhf(gates[2 * H + tid]);
            const float o = sigmoidf_(gates[3 * H + tid]);
            c = f * c + i * g;
            const float hn = o * tanhf(c);
            h[tid] = hn;
            *((float *) ((char *) out + (size_t) t * out_nb1) + (reverse ? H : 0) + tid) = hn;
        }
        __syncthreads();
    }
}

struct coop_choice {
    bool checked = false;
    bool ok      = false;
    int  max_blocks = 0;
};
coop_choice g_coop[16];

// Whether the cooperative kernel can run `blocks` co-resident blocks on the current device.
bool coop_available(int blocks) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) return false;
    coop_choice & ch = g_coop[dev];
    if (!ch.checked) {
        ch.checked = true;
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess || !prop.cooperativeLaunch) return false;
        int per_sm = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, lstm_coop_kernel<__half>, BT, 0) != cudaSuccess) return false;
        ch.max_blocks = per_sm * prop.multiProcessorCount;
        ch.ok = ch.max_blocks > 0;
        const char * force = std::getenv("GLINER4J_LSTM_KERNEL");
        if (force != nullptr && std::strcmp(force, "serial") == 0) ch.ok = false;
        GGML_LOG_INFO("gliner4j lstm: cooperative kernel %s (%d co-resident blocks on %s)\n",
                      ch.ok ? "enabled" : "disabled", ch.max_blocks, prop.name);
    }
    return ch.ok && blocks <= ch.max_blocks;
}

} // namespace

template <typename TW>
cudaError_t launch_typed(const ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * xw_f  = dst->src[0];
    const ggml_tensor * xw_b  = dst->src[1];
    const ggml_tensor * whh_f = dst->src[2];
    const ggml_tensor * whh_b = dst->src[3];
    const int H = (int) whh_f->ne[0];
    const int W = (int) xw_f->ne[1];
    const float * a_xw_f = (const float *) xw_f->data;  const float * a_xw_b = (const float *) xw_b->data;
    const TW * a_wf = (const TW *) whh_f->data;         const TW * a_wb = (const TW *) whh_b->data;
    float * a_out = (float *) dst->data;
    long xw_nb1 = (long) xw_f->nb[1], whh_nb1 = (long) whh_f->nb[1], out_nb1 = (long) dst->nb[1];
    int h = H, w = W;
    if (H % U == 0 && coop_available(2 * (H / U))) {
        void * args[] = { &a_xw_f, &a_xw_b, &a_wf, &a_wb, &a_out, &h, &w, &xw_nb1, &whh_nb1, &out_nb1 };
        return cudaLaunchCooperativeKernel((const void *) lstm_coop_kernel<TW>, dim3(2 * (H / U)), dim3(BT), args, 0, stream);
    }
    const size_t smem = (size_t) 5 * H * sizeof(float);
    lstm_serial_kernel<TW><<<2, NTHREADS, smem, stream>>>(a_xw_f, a_xw_b, a_wf, a_wb, a_out, H, W, xw_nb1, whh_nb1, out_nb1);
    return cudaGetLastError();
}

cudaError_t gliner4j_lstm_launch(const ggml_tensor * dst, cudaStream_t stream) {
    const ggml_tensor * xw_f  = dst->src[0];
    const ggml_tensor * xw_b  = dst->src[1];
    const ggml_tensor * whh_f = dst->src[2];
    const ggml_tensor * whh_b = dst->src[3];
    GGML_ASSERT(xw_f && xw_b && whh_f && whh_b);
    GGML_ASSERT(xw_f->type == GGML_TYPE_F32 && xw_b->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT((whh_f->type == GGML_TYPE_F32 || whh_f->type == GGML_TYPE_F16) && whh_b->type == whh_f->type);
    const int H = (int) whh_f->ne[0];
    const int W = (int) xw_f->ne[1];
    const size_t es = ggml_type_size(whh_f->type);
    GGML_ASSERT(H % 4 == 0 && H <= MAX_H);
    GGML_ASSERT(whh_f->ne[1] == 4 * H && whh_b->ne[0] == H && whh_b->ne[1] == 4 * H);
    GGML_ASSERT(xw_f->ne[0] == 4 * H && xw_b->ne[0] == 4 * H && xw_b->ne[1] == W);
    GGML_ASSERT(dst->ne[0] == 2 * H && dst->ne[1] == W);
    GGML_ASSERT(xw_f->nb[0] == 4 && xw_b->nb[0] == 4 && whh_f->nb[0] == es && whh_b->nb[0] == es);
    GGML_ASSERT(xw_f->nb[1] == xw_b->nb[1] && whh_f->nb[1] == whh_b->nb[1]);
    GGML_ASSERT(whh_f->nb[1] % 16 == 0 && ((uintptr_t) whh_f->data) % 16 == 0 && ((uintptr_t) whh_b->data) % 16 == 0);
    GGML_ASSERT(dst->nb[1] % 16 == 0 && ((uintptr_t) dst->data) % 16 == 0);
    if (W == 0) {
        return cudaSuccess;
    }
    return whh_f->type == GGML_TYPE_F16 ? launch_typed<__half>(dst, stream) : launch_typed<float>(dst, stream);
}
