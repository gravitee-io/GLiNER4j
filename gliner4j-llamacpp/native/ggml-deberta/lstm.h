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

struct ggml_tensor;

// Fused 1-layer bidirectional LSTM recurrence (gliner's LstmSeq2SeqEncoder over the word rows).
// Node: ggml_custom_4d(ctx, F32, 2H, W, 1, 1, {xw_fwd, xw_bwd, whh_fwd, whh_bwd}, 4, fn, 1, NULL)
//   xw_*  (4H, W) f32  input contribution per step, biases folded in: W_ih·x + b_ih + b_hh
//   whh_* (H, 4H) f32  recurrent weights, PyTorch gate order i, f, g, o along the 4H axis
//   dst   (2H, W) f32  [h_fwd(t) ; h_bwd(t)] per column
// One block per direction walks the steps; the composed ggml graph needed ~40 ops per word.
cudaError_t gliner4j_lstm_launch(const ggml_tensor * dst, cudaStream_t stream);
