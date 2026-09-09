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
// Standalone check of the fused kernels against a double-precision CPU reference, every kernel
// variant the device can hold, with per-launch timing (best of 5 rounds).
//   nvcc -O2 -arch=native -I$LLAMA_CPP_DIR/ggml/include -I$LLAMA_CPP_DIR/ggml/src test_attn.cu deberta-attn.cu \
//        -L$LLAMA_CPP_DIR/build/bin -lggml-base -o test_attn && HEADS=12 ./test_attn
// HEADS (default 2) sets the head count of every case, N adds one (n, B=1) case, VARIANTS limits
// the kernels tried (comma separated names, default all).
#include "deberta-attn.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <cstring>
#include <cuda_fp16.h>
#include <cstdint>

static const int HD = 64;

static float frand() { return (float) rand() / RAND_MAX * 2.0f - 1.0f; }

static std::vector<uint16_t> to_f16(const std::vector<float> & v) {
    std::vector<uint16_t> out(v.size());
    for (size_t i = 0; i < v.size(); i++) { __half h = __float2half(v[i]); out[i] = *(uint16_t *) &h; }
    return out;
}
static float f16_round(float x) { return __half2float(__float2half(x)); }

static std::vector<std::string> split(const char * s) {
    std::vector<std::string> out;
    std::string cur;
    for (; *s; s++) { if (*s == ',') { if (!cur.empty()) out.push_back(cur); cur.clear(); } else cur += *s; }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

int main() {
    const int PAD = GLINER4J_DEBERTA_POS_PAD;
    std::vector<std::pair<int, int>> cases = { {5, 1}, {131, 1}, {512, 1}, {512, 2}, {1024, 1} };
    if (getenv("N")) cases.push_back({ atoi(getenv("N")), 1 });
    const int heads = getenv("HEADS") ? atoi(getenv("HEADS")) : 2;
    const std::vector<std::string> variants = split(getenv("VARIANTS") ? getenv("VARIANTS") : "turing,ampere,reg64x32,reg64x64,reg128x32,reg128x64,reg16x64");
    int failures = 0;
    for (auto & c : cases) {
        const int n = c.first, B = c.second, W = 2 * n - 1, WP = W + 2 * PAD;
        std::vector<float> q((size_t) HD * n * heads * B), k(q.size()), v(q.size());
        std::vector<float> P((size_t) HD * WP * heads, 0.0f), Qp(P.size(), 0.0f);
        std::vector<int32_t> lens(B);
        for (auto & x : q) x = f16_round(frand()); for (auto & x : k) x = f16_round(frand()); for (auto & x : v) x = f16_round(frand());
        // layouts: q/k/v (hd, n, heads, B) contiguous; P (hd, WP, heads) with the table at rows [PAD, PAD+W); dst (hd, heads, n, B)
        auto qi = [&](int d, int i, int h, int b) { return ((size_t) b * heads * n + (size_t) h * n + i) * HD + d; };
        auto pi = [&](int d, int r, int h) { return ((size_t) h * WP + PAD + r) * HD + d; };
        auto di = [&](int d, int h, int i, int b) { return ((size_t) b * n * heads + (size_t) i * heads + h) * HD + d; };
        for (int h = 0; h < heads; h++) for (int r = 0; r < W; r++) for (int d = 0; d < HD; d++) { P[pi(d,r,h)] = f16_round(frand() * 0.3f); Qp[pi(d,r,h)] = f16_round(frand() * 0.3f); }
        // clamped padding rows, as the Java side builds them
        for (int h = 0; h < heads; h++) for (int r = 0; r < PAD; r++) for (int d = 0; d < HD; d++) {
            P[pi(d, -1 - r, h)] = P[pi(d, 0, h)]; Qp[pi(d, -1 - r, h)] = Qp[pi(d, 0, h)];
            P[pi(d, W + r, h)] = P[pi(d, W - 1, h)]; Qp[pi(d, W + r, h)] = Qp[pi(d, W - 1, h)];
        }
        for (int b = 0; b < B; b++) lens[b] = n - 7 * b;
        const float scale = 1.0f / sqrtf(3.0f * HD);
        std::vector<double> ref((size_t) HD * heads * n * B, 0.0);
        for (int b = 0; b < B; b++) for (int h = 0; h < heads; h++) for (int i = 0; i < n; i++) {
            std::vector<double> s(lens[b]);
            double m = -1e300;
            for (int j = 0; j < lens[b]; j++) {
                double acc = 0;
                const int r = i - j + n - 1;
                for (int d = 0; d < HD; d++) {
                    acc += scale * q[qi(d,i,h,b)] * k[qi(d,j,h,b)] + q[qi(d,i,h,b)] * P[pi(d,r,h)] + k[qi(d,j,h,b)] * Qp[pi(d,r,h)];
                }
                s[j] = acc; m = fmax(m, acc);
            }
            double l = 0; for (int j = 0; j < lens[b]; j++) { s[j] = exp(s[j] - m); l += s[j]; }
            for (int d = 0; d < HD; d++) { double o = 0; for (int j = 0; j < lens[b]; j++) o += s[j] * v[qi(d,j,h,b)]; ref[di(d,h,i,b)] = o / l; }
        }
        auto q16 = to_f16(q), k16 = to_f16(k), v16 = to_f16(v), P16 = to_f16(P), Qp16 = to_f16(Qp);
        uint16_t *dq, *dk, *dv, *dP, *dQp; float * ddst; int32_t * dlens;
        cudaMalloc(&dq, q16.size() * 2); cudaMalloc(&dk, k16.size() * 2); cudaMalloc(&dv, v16.size() * 2);
        cudaMalloc(&dP, P16.size() * 2); cudaMalloc(&dQp, Qp16.size() * 2); cudaMalloc(&ddst, ref.size() * 4); cudaMalloc(&dlens, B * 4);
        cudaMemcpy(dq, q16.data(), q16.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dk, k16.data(), k16.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dv, v16.data(), v16.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dP, P16.data(), P16.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(dQp, Qp16.data(), Qp16.size() * 2, cudaMemcpyHostToDevice); cudaMemcpy(dlens, lens.data(), B * 4, cudaMemcpyHostToDevice);
        // the same q / k / v as f32 (already f16-rounded values): the op converts them while staging
        float *dq32, *dk32, *dv32;
        cudaMalloc(&dq32, q.size() * 4); cudaMalloc(&dk32, k.size() * 4); cudaMalloc(&dv32, v.size() * 4);
        cudaMemcpy(dq32, q.data(), q.size() * 4, cudaMemcpyHostToDevice); cudaMemcpy(dk32, k.data(), k.size() * 4, cudaMemcpyHostToDevice);
        cudaMemcpy(dv32, v.data(), v.size() * 4, cudaMemcpyHostToDevice);
        gliner4j_deberta_attn_args a{};
        a.q = (const char *) dq; a.k = (const char *) dk; a.v = (const char *) dv; a.P = (const char *) dP; a.Qp = (const char *) dQp;
        a.lens = dlens; a.dst = (char *) ddst; a.n = n; a.heads = heads; a.W = WP; a.scale = scale;
        a.q_nb1 = a.k_nb1 = a.v_nb1 = HD * 2; a.q_nb2 = a.k_nb2 = a.v_nb2 = (long) HD * n * 2; a.q_nb3 = a.k_nb3 = a.v_nb3 = (long) HD * n * heads * 2;
        a.p_nb1 = a.qp_nb1 = HD * 2; a.p_nb2 = a.qp_nb2 = (long) HD * WP * 2;
        a.d_nb1 = HD * 4; a.d_nb2 = (long) HD * heads * 4; a.d_nb3 = (long) HD * heads * n * 4;
        gliner4j_deberta_attn_args a32 = a;
        a32.q = (const char *) dq32; a32.k = (const char *) dk32; a32.v = (const char *) dv32; a32.qkv_f32 = true;
        a32.q_nb1 = a32.k_nb1 = a32.v_nb1 = HD * 4; a32.q_nb2 = a32.k_nb2 = a32.v_nb2 = (long) HD * n * 4; a32.q_nb3 = a32.k_nb3 = a32.v_nb3 = (long) HD * n * heads * 4;
        auto check = [&](const char * label, bool reference, const gliner4j_deberta_attn_args & a) {
            cudaMemset(ddst, 0, ref.size() * 4);
            cudaError_t err = gliner4j_deberta_attn_launch_args(a, B, 0, reference);
            cudaDeviceSynchronize();
            if (err != cudaSuccess || cudaGetLastError() != cudaSuccess) { printf("n=%d B=%d %-9s launch error\n", n, B, label); failures++; return false; }
            std::vector<float> out(ref.size()); cudaMemcpy(out.data(), ddst, out.size() * 4, cudaMemcpyDeviceToHost);
            double worst = 0, worst_ref = 0; size_t wi = 0;
            for (size_t i = 0; i < out.size(); i++) { double d = fabs(out[i] - ref[i]); if (d > worst) { worst = d; wi = i; } worst_ref = fmax(worst_ref, fabs(ref[i])); }
            const bool ok = worst < 0.02 * fmax(1.0, worst_ref);
            printf("n=%4d B=%d %-13s max|diff|=%.5f (max|ref|=%.3f) at %zu %s", n, B, label, worst, worst_ref, wi, ok ? "OK" : "FAIL");
            if (!ok) failures++;
            return ok;
        };
        check("ref", true, a);
        printf("\n");
        check("ref/f32", true, a32);
        printf("\n");
        for (const auto & name : variants) {
            const int rc = gliner4j_deberta_attn_force_variant(name.c_str());
            if (rc == -1) { printf("unknown variant %s\n", name.c_str()); failures++; continue; }
            if (rc == -2) { printf("n=%4d B=%d %-13s skipped (does not fit this device)\n", n, B, name.c_str()); continue; }
            const bool ok = check(name.c_str(), false, a);
            if (ok && n >= 131) {
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                for (int w = 0; w < 3; w++) gliner4j_deberta_attn_launch_args(a, B, 0, false);
                float best = 1e30f;
                for (int round = 0; round < 5; round++) {
                    cudaEventRecord(e0);
                    for (int w = 0; w < 10; w++) gliner4j_deberta_attn_launch_args(a, B, 0, false);
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                    best = fminf(best, ms / 10);
                }
                printf("  %8.1f us/launch (heads=%d)", best * 1000, heads);
            }
            printf("\n");
            const std::string label32 = name + "/f32";
            const bool ok32 = check(label32.c_str(), false, a32);
            if (ok32 && n >= 131) {
                cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
                for (int w = 0; w < 3; w++) gliner4j_deberta_attn_launch_args(a32, B, 0, false);
                float best = 1e30f;
                for (int round = 0; round < 5; round++) {
                    cudaEventRecord(e0);
                    for (int w = 0; w < 10; w++) gliner4j_deberta_attn_launch_args(a32, B, 0, false);
                    cudaEventRecord(e1); cudaEventSynchronize(e1);
                    float ms = 0; cudaEventElapsedTime(&ms, e0, e1);
                    best = fminf(best, ms / 10);
                }
                printf("  %8.1f us/launch (heads=%d)", best * 1000, heads);
            }
            printf("\n");
        }
        cudaFree(dq32); cudaFree(dk32); cudaFree(dv32);
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(dP); cudaFree(dQp); cudaFree(ddst); cudaFree(dlens);
    }
    printf("%s\n", failures ? "FAILURES" : "ALL OK");
    return failures ? 1 : 0;
}
