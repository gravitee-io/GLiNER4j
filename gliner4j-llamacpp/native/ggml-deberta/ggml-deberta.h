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

// gliner4j "DEBERTA" ggml backend plugin: one fused disentangled-attention op for the
// DeBERTa-v2/v3 encoder (GLiNER2 / GLiNER2.5 / GLiNER uni / bi backbones), cooperating with the
// stock CUDA backend on the same device through ggml_backend_sched (zero-copy: the plugin uses the
// CUDA buffer type). Disentangled attention after DeBERTa (He et al.),
// https://github.com/microsoft/DeBERTa (MIT) — see the README's References section.
//
// Loaded with ggml_backend_load(<path>/libggml-deberta.so). Java obtains the op through
// ggml_backend_reg_get_proc_address(reg, GLINER4J_DEBERTA_ATTN_FN) and builds the node with
//
//   ggml_custom_4d(ctx, GGML_TYPE_F32, hd, heads, n, B, args, 6, fn, 1, NULL)
//
// args = { q (hd, n, heads, B) f16 | k same | v same | P (hd, 2n-1+2*PAD, heads) f16 | Qp same |
//          lens (B) i32 } — q/k/v may be strided views (16-byte aligned rows), P/Qp are the
// per-offset position tables (row PAD + i - j + n - 1, pre-scaled by 1/sqrt(3*hd), PAD =
// GLINER4J_DEBERTA_POS_PAD clamped rows on each side), lens the valid tokens per batch row.
// Result (hd, heads, n, B) f32 = softmax(scale*q.k + q.P[i-j] + k.Qp[i-j]) . v, the permuted
// layout ggml_flash_attn_ext returns.

#include "ggml.h"
#include "ggml-backend.h"

#ifdef __cplusplus
extern "C" {
#endif

#define GLINER4J_DEBERTA_ATTN_FN "gliner4j_deberta_attn_fn"   // ggml_custom_op_t marker
#define GLINER4J_DEBERTA_VERSION "gliner4j_deberta_version"   // const char * (*)(void)
// Second op: the bidirectional word LSTM of the GLiNER uni-encoder span head (see lstm.h).
#define GLINER4J_LSTM_FN "gliner4j_lstm_fn"                     // ggml_custom_op_t marker
// void (*)(bool on): host-side synchronisation around the fused op. On by default; the host turns
// it off when its ggml_backend_sched orders cross-backend splits with events (patched ggml,
// parallel=true) — see patches/0001-ggml-sched-cross-backend-events.patch.
#define GLINER4J_DEBERTA_SET_HOST_SYNC "gliner4j_deberta_set_host_sync"
// const char * (*)(void): the attention kernel chosen for the current CUDA device, with the
// autotune table that chose it (ggml logs at INFO are usually filtered by the host; this lets it
// log the choice itself). Resolves the choice on first call.
#define GLINER4J_DEBERTA_KERNEL_REPORT "gliner4j_deberta_kernel_report"
// Present (non-null) when the attention op accepts f32 q / k / v operands and converts them while
// staging: the graph then feeds the projections straight in and skips three cast ops per layer.
#define GLINER4J_DEBERTA_QKV_F32 "gliner4j_deberta_qkv_f32"

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_deberta_reg(void);

#ifdef __cplusplus
}
#endif
