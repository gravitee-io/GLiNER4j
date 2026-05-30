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
    if (cacheDir != null) {
      opts.setOptimizedModelFilePath(
        cacheDir.resolve(modelFileName).toString()
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
    var ep = config.getExecutionProvider();
    if (ep == null || ep == ExecutionProvider.CPU) {
      return;
    }
    try {
      ep.configure(
        opts,
        config.getGpuDeviceId(),
        config.getOpenVinoDeviceType()
      );
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
