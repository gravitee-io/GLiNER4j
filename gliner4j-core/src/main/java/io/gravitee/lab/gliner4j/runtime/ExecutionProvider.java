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
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * ONNX Runtime execution provider (the hardware backend that runs the graph).
 *
 * <p>The provider must be compiled into the native ONNX Runtime library on the classpath:
 * <ul>
 *   <li>{@link #CPU} — always available (the default {@code com.microsoft.onnxruntime:onnxruntime} jar).</li>
 *   <li>{@link #CUDA} — requires the {@code com.microsoft.onnxruntime:onnxruntime_gpu} jar and a CUDA-capable GPU
 *       with matching CUDA/cuDNN runtime libraries installed. Build the project with {@code -Pcuda}.</li>
 *   <li>{@link #OPENVINO} — requires an ONNX Runtime build compiled with the OpenVINO execution provider
 *       (no official Maven artifact ships it; supply your own native library / jar).</li>
 *   <li>{@link #COREML} — Apple's accelerator, bundled in the standard macOS {@code onnxruntime} jar.
 *       This is the ONNX Runtime equivalent of "use the Apple Neural Engine / GPU"; Apple MLX is a separate
 *       array framework and is <em>not</em> an ONNX Runtime execution provider.</li>
 * </ul>
 *
 * <p>If the requested provider cannot be registered (e.g. CUDA requested without the GPU runtime),
 * {@link BaseRuntime} logs a warning and falls back to CPU rather than failing the load.
 */
@Slf4j
public enum ExecutionProvider {
  /** Default CPU provider. Always available. */
  CPU,
  /** NVIDIA CUDA provider. Requires the {@code onnxruntime_gpu} native library. */
  CUDA,
  /** Intel OpenVINO provider. Requires an OpenVINO-enabled ONNX Runtime build. */
  OPENVINO,
  /** Apple CoreML provider. Bundled in the standard macOS {@code onnxruntime} jar. */
  COREML;

  /**
   * Resolves a provider from a case-insensitive string (CLI arg, JMH param, env var).
   * Unknown, blank, or {@code null} values resolve to {@link #CPU}.
   *
   * @param value the provider name (e.g. "cuda", "openvino", "coreml", "cpu")
   * @return the matching provider, or {@link #CPU} when unrecognised
   */
  public static ExecutionProvider fromString(String value) {
    if (value == null || value.isBlank()) {
      return CPU;
    }
    return switch (value.trim().toLowerCase()) {
      case "cuda", "gpu", "nvidia" -> CUDA;
      case "openvino", "vino", "intel" -> OPENVINO;
      // MLX is not an ORT execution provider — CoreML is the Apple-accelerated backend.
      case "coreml", "mlx", "apple", "ane" -> COREML;
      default -> CPU;
    };
  }

  /**
   * Whether ONNX Runtime can serialize this provider's optimized graph to disk
   * (via {@code setOptimizedModelFilePath}). Providers that compile/fuse nodes — CUDA, OpenVINO,
   * CoreML — produce graphs ORT refuses to serialize ("contains compiled nodes"), so the optimized-
   * model cache is only meaningful for {@link #CPU}.
   *
   * @return {@code true} if the optimized-model cache can be written for this provider
   */
  public boolean supportsOptimizedModelCache() {
    return this == CPU;
  }

  /** Lower-case token used to namespace the optimized-model cache directory per backend. */
  public String cacheTag() {
    return name().toLowerCase();
  }

  /**
   * Registers this provider on the given session options. The CPU provider is the ORT default
   * and registers nothing. Throws if the native library lacks the provider — callers are expected
   * to catch and fall back to CPU.
   *
   * @param opts               the session options to mutate
   * @param gpuDeviceId        device ordinal for CUDA (ignored by other providers)
   * @param openVinoDeviceType OpenVINO device hint (e.g. "", "CPU", "GPU", "NPU"); ignored by other providers
   * @throws OrtException if the provider is not available in the loaded native runtime
   */
  void configure(
    OrtSession.SessionOptions opts,
    int gpuDeviceId,
    String openVinoDeviceType
  ) throws OrtException {
    switch (this) {
      case CPU -> log.debug(
        "Using default CPU execution provider — nothing to register"
      );
      case CUDA -> {
        log.info(
          "Registering CUDA execution provider (deviceId={})",
          gpuDeviceId
        );
        opts.addCUDA(gpuDeviceId);
        log.info("CUDA execution provider registered");
      }
      case OPENVINO -> {
        // OpenVINO requires a concrete device_type (CPU/GPU/NPU/AUTO/...); empty is rejected.
        var device = openVinoDeviceType == null || openVinoDeviceType.isBlank()
          ? "CPU"
          : openVinoDeviceType;
        log.info("Registering OpenVINO execution provider (device={})", device);
        registerOpenVino(opts, device);
        log.info("OpenVINO execution provider registered");
      }
      case COREML -> {
        log.info("Registering CoreML execution provider");
        opts.addCoreML();
        log.info("CoreML execution provider registered");
      }
    }
  }

  /**
   * Registers OpenVINO through ONNX Runtime's modern map-based EP API rather than the deprecated
   * {@link OrtSession.SessionOptions#addOpenVINO(String)}.
   *
   * <p>{@code addOpenVINO(String)} routes through ORT 1.21's legacy-to-V2 options adapter
   * ({@code OrtOpenVINOProviderOptionsToOrtOpenVINOProviderOptionsV2} in {@code provider_bridge_ort.cc}),
   * which has a bug: it assigns the {@code enable_opencl_throttling} bool straight into a
   * {@code std::string}, producing a one-byte NUL string instead of {@code "false"}. The provider
   * factory then rejects it with "enable_opencl_throttling should be a boolean" and the whole
   * registration fails. The generic V2 API ({@code SessionOptionsAppendExecutionProvider} with
   * provider name {@code "OpenVINO"}) bypasses that adapter and consumes our options map directly;
   * we pass only {@code device_type}, so {@code enable_opencl_throttling} stays unset and keeps its
   * default.
   *
   * <p>Neither {@code OnnxRuntime.extractOpenVINO()} nor {@code SessionOptions.addExecutionProvider}
   * is public in the ORT Java bindings, so we reach them reflectively. Any failure is surfaced as an
   * {@link OrtException} so {@link BaseRuntime} can fall back to CPU.
   */
  private static void registerOpenVino(
    OrtSession.SessionOptions opts,
    String device
  ) throws OrtException {
    try {
      // The generic V2 registration path (unlike the legacy addOpenVINO) does not extract the
      // OpenVINO shared provider library from the jar / native path — do it ourselves first.
      Method extractOpenVINO = Class.forName(
        "ai.onnxruntime.OnnxRuntime"
      ).getDeclaredMethod("extractOpenVINO");
      extractOpenVINO.setAccessible(true);
      if (Boolean.FALSE.equals(extractOpenVINO.invoke(null))) {
        throw new OrtException(
          "OpenVINO provider library (onnxruntime_providers_openvino) not found on the classpath or native path"
        );
      }

      Method addExecutionProvider = OrtSession
        .SessionOptions.class.getDeclaredMethod(
        "addExecutionProvider",
        String.class,
        Map.class
      );
      addExecutionProvider.setAccessible(true);
      addExecutionProvider.invoke(
        opts,
        "OpenVINO",
        Map.of("device_type", device)
      );
    } catch (InvocationTargetException e) {
      if (e.getCause() instanceof OrtException ortException) {
        throw ortException;
      }
      throw new OrtException(
        "Failed to register OpenVINO execution provider: " + e.getCause()
      );
    } catch (ReflectiveOperationException e) {
      throw new OrtException(
        "Failed to register OpenVINO execution provider via the V2 API: " + e
      );
    }
  }
}
