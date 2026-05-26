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

  /** ORT graph optimization level. */
  @Builder.Default
  private final OrtSession.SessionOptions.OptLevel optimizationLevel =
    OrtSession.SessionOptions.OptLevel.ALL_OPT;

  /** Whether to cache optimized model graphs to disk. */
  @Builder.Default
  private final boolean optimizedModelCacheEnabled = true;

  /**
   * Execution provider (hardware backend) for all ONNX sessions. Defaults to {@link ExecutionProvider#CPU}.
   * Non-CPU providers require a matching native runtime; when unavailable the runtime warns and falls back to CPU.
   */
  @Builder.Default
  private final ExecutionProvider executionProvider = ExecutionProvider.CPU;

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
   * Returns the default configuration (identical to the current hardcoded behavior).
   *
   * @return default runtime configuration
   */
  public static RuntimeConfig defaults() {
    return RuntimeConfig.builder().build();
  }
}
