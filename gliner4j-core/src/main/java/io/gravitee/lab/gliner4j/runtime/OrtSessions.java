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

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.LongBuffer;
import java.nio.file.Path;
import lombok.extern.slf4j.Slf4j;

/**
 * Shared ONNX Runtime helpers: session-option construction, execution-provider registration (with
 * graceful CPU fallback), and direct-buffer allocation.
 *
 * <p>Extracted from {@link BaseRuntime} so that runtimes which are <em>not</em> single-encoder —
 * and therefore don't extend {@link BaseRuntime} — can reuse the same logic. {@code BaseRuntime}
 * (GLiNER2, classifier, GLiClass) and standalone {@link ArchitectureRuntime} implementations
 * (the GLiNER uni-encoder monolithic-graph runtime; bi-encoder/decoder later) all go through here.
 */
@Slf4j
public final class OrtSessions {

  private OrtSessions() {}

  /**
   * Builds session options with the given threading, execution mode, optional optimized-model
   * cache path, and the configured execution provider.
   */
  public static OrtSession.SessionOptions createSessionOptions(
    int intraOpThreads,
    int interOpThreads,
    OrtSession.SessionOptions.ExecutionMode executionMode,
    RuntimeConfig config,
    Path cacheDir,
    String modelFileName
  ) throws OrtException {
    var opts = new OrtSession.SessionOptions();
    opts.setIntraOpNumThreads(intraOpThreads);
    opts.setInterOpNumThreads(interOpThreads);
    opts.setExecutionMode(executionMode);
    opts.setOptimizationLevel(config.getOptimizationLevel());
    // Disable ORT's memory-pattern planner. It pre-plans and reuses activation buffers from the
    // first inference, which is unsafe for our fully dynamic-shape graphs (the micro-batcher feeds
    // varying (batch, num_spans) buckets, and two lanes share one session): a buffer planned for a
    // small bucket cannot be reused for a larger one, surfacing as
    // "Shape mismatch attempting to re-use buffer. {1,42,768} != {2,2376,768}". Turning it off makes
    // each Run allocate from the arena independently — required for correctness under bucketing +
    // concurrent lanes; negligible cost on the GPU/dynamic path.
    opts.setMemoryPatternOptimization(false);
    log.info(
      "ORT mem_pattern DISABLED for {} (dynamic-shape + concurrent-lane safety)",
      modelFileName
    );
    if (!config.isIntraOpSpinning()) {
      // Stop the intra-op pool from busy-waiting between ops — yields the cores instead of
      // pinning them at 100%. Worth it when the heavy math is on the GPU (CUDA EP).
      opts.addConfigEntry("session.intra_op.allow_spinning", "0");
      opts.addConfigEntry("session.inter_op.allow_spinning", "0");
    }
    if (cacheDir != null) {
      opts.setOptimizedModelFilePath(
        cacheDir.resolve(modelFileName).toString()
      );
    }
    if (config.getProfilingDir() != null) {
      opts.enableProfiling(
        Path.of(config.getProfilingDir(), modelFileName).toString()
      );
      log.info(
        "ORT profiling enabled for {} — trace JSON written to {} on session close",
        modelFileName,
        config.getProfilingDir()
      );
    }
    applyExecutionProvider(opts, config, modelFileName);
    return opts;
  }

  /**
   * Registers the configured execution provider on the session options. Falls back to CPU
   * (the ORT default) with a warning if the provider is unavailable in the loaded native runtime,
   * so a misconfigured GPU/OpenVINO build degrades gracefully instead of failing the model load.
   */
  public static void applyExecutionProvider(
    OrtSession.SessionOptions opts,
    RuntimeConfig config,
    String modelFileName
  ) {
    var ep = ExecutionProvider.resolve(config.getExecutionProvider());
    if (ep == ExecutionProvider.CPU) {
      return;
    }
    try {
      ep.configure(opts, config);
      log.info("Registered {} execution provider for {}", ep, modelFileName);
    } catch (OrtException | RuntimeException | UnsatisfiedLinkError e) {
      log.warn(
        "Could not register {} execution provider for {} — falling back to CPU. Cause: {}",
        ep,
        modelFileName,
        e.getMessage()
      );
    }
  }

  public static LongBuffer allocateDirectLongBuffer(long[] data) {
    return ByteBuffer.allocateDirect(data.length * Long.BYTES)
      .order(ByteOrder.nativeOrder())
      .asLongBuffer()
      .put(data)
      .rewind();
  }

  public static LongBuffer allocateDirectLongBuffer(int capacity) {
    return ByteBuffer.allocateDirect(capacity * Long.BYTES)
      .order(ByteOrder.nativeOrder())
      .asLongBuffer();
  }

  public static int getOrDefault(Integer value, int defaultValue) {
    return value != null ? value : defaultValue;
  }
}
