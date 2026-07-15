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
package io.gravitee.lab.gliner4j.runtime;

import ai.onnxruntime.OrtSession;
import lombok.Builder;
import lombok.Getter;

/**
 * Configuration for ONNX Runtime session resource control.
 *
 * <p>All thread count fields default to {@code null}, which means "auto-calculate"
 * based on {@code Runtime.getRuntime().availableProcessors()}. Set explicit values
 * to cap resource usage in constrained environments (containers, shared runtimes).
 *
 * <p>Usage:
 * <pre>{@code
 * var config = RuntimeConfig.builder()
 *     .encoderIntraOpThreads(2)
 *     .scoringIntraOpThreads(1)
 *     .build();
 * var gliner = GLiNER4jNER.load(modelDir, entities, config);
 * }</pre>
 */
@Getter
@Builder
public class RuntimeConfig {

  /** Intra-op threads for encoder and span_rep sessions. null = auto (availableProcessors). */
  @Builder.Default
  private final Integer encoderIntraOpThreads = null;

  /** Inter-op threads for encoder and span_rep sessions. null = auto (max(2, cpus/2)). */
  @Builder.Default
  private final Integer encoderInterOpThreads = null;

  /** Intra-op threads for scoring_head session. null = auto (max(2, cpus/4)). */
  @Builder.Default
  private final Integer scoringIntraOpThreads = null;

  /** Inter-op threads for scoring_head session. null = auto (1). */
  @Builder.Default
  private final Integer scoringInterOpThreads = null;

  /**
   * Whether ORT's intra-op thread pool may busy-wait (spin) between ops. Spinning trims latency
   * but pins CPU cores at 100% under load; set false to yield instead — usually a large CPU
   * reduction with little throughput cost when the GPU is the compute path. Default true (ORT's
   * default behaviour).
   */
  @Builder.Default
  private final boolean intraOpSpinning = true;

  /** ORT graph optimization level. */
  @Builder.Default
  private final OrtSession.SessionOptions.OptLevel optimizationLevel =
    OrtSession.SessionOptions.OptLevel.ALL_OPT;

  /** Whether to cache optimized model graphs to disk. */
  @Builder.Default
  private final boolean optimizedModelCacheEnabled = true;

  /**
   * Execution provider (hardware backend) for all ONNX sessions. Defaults to {@link ExecutionProvider#AUTO},
   * which detects the best provider compiled into the loaded native runtime (CUDA &gt; OpenVINO &gt; CPU).
   * Non-CPU providers require a matching native runtime; when unavailable the runtime warns and falls back to CPU.
   */
  @Builder.Default
  private final ExecutionProvider executionProvider = ExecutionProvider.AUTO;

  /** GPU device ordinal used by the {@link ExecutionProvider#CUDA} provider. */
  @Builder.Default
  private final int gpuDeviceId = 0;

  /**
   * OpenVINO device_type for the {@link ExecutionProvider#OPENVINO} provider — one of
   * "CPU", "GPU", "NPU", "AUTO" (or "GPU.0", HETERO/MULTI variants). OpenVINO rejects an empty
   * value, so a blank here is treated as "CPU".
   */
  @Builder.Default
  private final String openVinoDeviceType = "CPU";

  /**
   * Directory for ONNX Runtime built-in profiling traces. When set, every session records
   * per-node timings and writes a Chrome-trace JSON (open in Perfetto / chrome://tracing)
   * prefixed with the model file name into this directory when the session closes.
   * {@code null} (default) disables profiling.
   */
  @Builder.Default
  private final String profilingDir = null;

  /**
   * When profiling is enabled, flush the profiling traces to disk this many seconds after the
   * sessions are loaded (via {@code OrtSession.endProfiling()}) instead of waiting for session
   * close. Useful on ephemeral hosts where the filesystem does not survive shutdown.
   * {@code null} (default) keeps flush-on-close only.
   */
  @Builder.Default
  private final Integer profilingSeconds = null;

  /**
   * Max allowed (longest / shortest) sequence-length ratio inside one encoder sub-batch.
   * Batched calls sort texts by token length and split them into sub-batches so short texts
   * are not padded to the longest text in the batch. Values ≤ 1 disable bucketing (single
   * sub-batch, previous behavior).
   */
  @Builder.Default
  private final double batchLengthRatio = 1.3;

  /** Max texts per encoder sub-batch when bucketing. null = unbounded. */
  @Builder.Default
  private final Integer maxSubBatchSize = null;

  /**
   * Cache per-call override assemblers (LRU, keyed on the override definition list) so a
   * repeated label set skips re-tokenizing the whole schema prompt. Enabled by default;
   * disable to rebuild the assembler on every override call (the previous behavior).
   */
  @Builder.Default
  private final boolean overrideCacheEnabled = true;

  /** Max cached per-call override assemblers when the cache is enabled. */
  @Builder.Default
  private final int overrideCacheSize = 32;

  /**
   * The effective override-cache capacity: {@link #getOverrideCacheSize()} when enabled,
   * 0 (disabled) otherwise.
   */
  public int effectiveOverrideCacheSize() {
    return overrideCacheEnabled ? overrideCacheSize : 0;
  }

  /**
   * Coalesce concurrent single-text extract() calls into batched runs via a micro-batching
   * collector. Adds up to {@link #getMicroBatchMaxWaitMicros()} latency per call in exchange
   * for concurrent throughput; off by default.
   */
  @Builder.Default
  private final boolean microBatchingEnabled = false;

  /** Max texts coalesced into one micro-batch. */
  @Builder.Default
  private final int microBatchMaxSize = 16;

  /** Max time a micro-batched call waits for companions, in microseconds. */
  @Builder.Default
  private final long microBatchMaxWaitMicros = 2000;

  /**
   * Returns the default configuration (identical to the current hardcoded behavior).
   *
   * @return default runtime configuration
   */
  public static RuntimeConfig defaults() {
    return RuntimeConfig.builder().build();
  }
}
