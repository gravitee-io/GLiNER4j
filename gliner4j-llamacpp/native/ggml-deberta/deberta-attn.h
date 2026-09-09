/**
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
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

struct ggml_tensor;

// Rows of clamped padding on each side of the per-offset position tables handed to the kernel
// (P / Qp are (hd, 2n-1 + 2*PAD, heads) f16; row PAD + i - j + n - 1 holds offset i - j). Mirrored
// by GgmlDebertaV3.POS_PAD on the Java side.
#define GLINER4J_DEBERTA_POS_PAD 96

// Validates the node's operands (see ggml-deberta.h for the contract) and launches the fused
// attention on `stream`. `reference` selects the scalar one-warp-per-query kernel used to bisect
// numerical issues in the tiled one. Returns cudaSuccess or the launch error; shape/type
// violations abort through GGML_ABORT.
cudaError_t gliner4j_deberta_attn_launch(const ggml_tensor * dst, cudaStream_t stream, bool reference);

// Raw operand description (device pointers + byte strides) — what the ggml launcher extracts
// from the node; also used directly by the standalone test harness (test_attn.cu).
struct gliner4j_deberta_attn_args {
    const char *    q;   const char * k;   const char * v;
    const char *    P;   const char * Qp;
    const int32_t * lens;
    char *          dst;
    int   n;            // padded sequence length (tokens per batch row)
    int   heads;
    int   W;            // 2n - 1 rows in P / Qp
    float scale;        // 1/sqrt(3*hd), applied to q.k (P / Qp are pre-scaled)
    bool  qkv_f32;      // q / k / v rows are f32 (converted to f16 while staging) instead of f16
    long q_nb1, q_nb2, q_nb3;
    long k_nb1, k_nb2, k_nb3;
    long v_nb1, v_nb2, v_nb3;
    long p_nb1, p_nb2;
    long qp_nb1, qp_nb2;
    long d_nb1, d_nb2, d_nb3;   // dst (hd, heads, n, B): nb1 = head, nb2 = token, nb3 = batch
};

cudaError_t gliner4j_deberta_attn_launch_args(const gliner4j_deberta_attn_args & a, int B, cudaStream_t stream, bool reference);

// Kernel variant control (test harness / diagnostics). force_variant selects one by name
// ("turing", "ampere", "reg64x32", "reg64x64", "reg128x32", "reg128x64", "reg16x64", "auto" for
// the autotune, nullptr for the GLINER4J_DEBERTA_KERNEL environment rule); returns -1 for an
// unknown name, -2 when the current device cannot hold it. The choice is re-resolved on the next
// launch. variant_name reports the kernel in use on the current
// device.
int          gliner4j_deberta_attn_force_variant(const char * name);
const char * gliner4j_deberta_attn_variant_name();
// One line: the autotune table (when it ran) and the kernel chosen for the current device; resolves
// the choice (runs the autotune) if the first launch has not happened yet.
const char * gliner4j_deberta_attn_report();
