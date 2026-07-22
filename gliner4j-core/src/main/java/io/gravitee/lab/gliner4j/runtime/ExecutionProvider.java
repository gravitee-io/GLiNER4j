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

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProvider;
import ai.onnxruntime.OrtSession;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
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
 * </ul>
 *
 * <p>If the requested provider cannot be registered (e.g. CUDA requested without the GPU runtime),
 * {@link BaseRuntime} logs a warning and falls back to CPU rather than failing the load.
 *
 * <p>The default is {@link #AUTO}, which inspects the providers compiled into the loaded native
 * runtime ({@link OrtEnvironment#getAvailableProviders()}) and picks the best accelerator in the
 * order CUDA &gt; OpenVINO, falling back to CPU when none is present.
 */
@Slf4j
public enum ExecutionProvider {
  /**
   * Auto-detect the best provider compiled into the loaded native runtime. Resolves to the first
   * available of CUDA, OpenVINO, then CPU. This is the default; see {@link #resolve}.
   */
  AUTO,
  /** Default CPU provider. Always available. */
  CPU,
  /** NVIDIA CUDA provider. Requires the {@code onnxruntime_gpu} native library. */
  CUDA,
  /** Intel OpenVINO provider. Requires an OpenVINO-enabled ONNX Runtime build. */
  OPENVINO;

  /**
   * Preference order used by {@link #AUTO} resolution — best accelerator first. CPU is the implicit
   * final fallback and is therefore not listed.
   */
  private static final List<ExecutionProvider> AUTO_PREFERENCE = List.of(
    CUDA,
    OPENVINO
  );

  /**
   * Memoized result of {@link #AUTO} resolution. The set of providers compiled into the native
   * runtime is fixed for the JVM's lifetime, so we detect (and log) once and reuse the answer.
   */
  private static volatile ExecutionProvider autoResolved;

  /**
   * Resolves a provider from a case-insensitive string (CLI arg, JMH param, env var).
   * Blank, {@code null}, or unrecognised values resolve to {@link #AUTO} (auto-detect the backend).
   *
   * @param value the provider name (e.g. "auto", "cuda", "openvino", "cpu")
   * @return the matching provider, or {@link #AUTO} when unrecognised
   */
  public static ExecutionProvider fromString(String value) {
    if (value == null || value.isBlank()) {
      return AUTO;
    }
    return switch (value.trim().toLowerCase()) {
      case "cpu" -> CPU;
      case "cuda", "gpu", "nvidia" -> CUDA;
      case "openvino", "vino", "intel" -> OPENVINO;
      default -> AUTO;
    };
  }

  /**
   * The execution providers that are actually compiled into the loaded native ONNX Runtime, mapped
   * onto this enum. {@code CPU} is always present. Note this reflects what the native library
   * <em>supports</em>, not whether the matching hardware is physically present — e.g. the GPU jar
   * reports CUDA even on a machine without an NVIDIA GPU (registration then fails and the runtime
   * falls back to CPU).
   *
   * @return the set of available providers (always contains {@link #CPU})
   */
  public static Set<ExecutionProvider> available() {
    var providers = EnumSet.of(CPU);
    for (OrtProvider ortProvider : OrtEnvironment.getAvailableProviders()) {
      switch (ortProvider) {
        // TensorRT ships in the same GPU build as CUDA; treat it as CUDA availability.
        case CUDA, TENSOR_RT -> providers.add(CUDA);
        case OPEN_VINO -> providers.add(OPENVINO);
        default -> {
          /* provider not exposed by this enum */
        }
      }
    }
    return providers;
  }

  /**
   * Resolves a requested provider to a concrete one. Non-{@link #AUTO} values (and {@code null},
   * treated as {@code AUTO}) are returned unchanged; {@code AUTO} is resolved to the best available
   * accelerator (CUDA &gt; OpenVINO), or {@link #CPU} when none is compiled in.
   *
   * <p>The {@code AUTO} decision is detected once and memoized for the JVM lifetime, so it is logged
   * only the first time.
   *
   * @param requested the requested provider (may be {@code null})
   * @return a concrete provider, never {@code AUTO} or {@code null}
   */
  public static ExecutionProvider resolve(ExecutionProvider requested) {
    if (requested != null && requested != AUTO) {
      return requested;
    }
    var resolved = autoResolved;
    if (resolved == null) {
      var available = available();
      resolved = AUTO_PREFERENCE.stream()
        .filter(available::contains)
        .findFirst()
        .orElse(CPU);
      autoResolved = resolved;
      if (resolved == CPU) {
        log.info(
          "AUTO execution provider resolved to CPU — no accelerator EP compiled into the loaded ONNX Runtime (available: {}). " +
            "To enable an accelerator, rebuild with the matching Maven profile: CUDA -> -Pcuda (also source scripts/cuda_env.sh on Linux), " +
            "OpenVINO -> -Popenvino (run task build:openvino first).",
          available
        );
      } else {
        log.info(
          "AUTO execution provider resolved to {} (available: {})",
          resolved,
          available
        );
      }
    }
    return resolved;
  }

  /**
   * Whether ONNX Runtime can serialize this provider's optimized graph to disk
   * (via {@code setOptimizedModelFilePath}). Providers that compile/fuse nodes — CUDA, OpenVINO —
   * produce graphs ORT refuses to serialize ("contains compiled nodes"), so the optimized-
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
   * @param opts   the session options to mutate
   * @param config the runtime config supplying provider-specific settings (GPU device id,
   *               OpenVINO device type)
   * @throws OrtException if the provider is not available in the loaded native runtime
   */
  void configure(OrtSession.SessionOptions opts, RuntimeConfig config)
    throws OrtException {
    int gpuDeviceId = config.getGpuDeviceId();
    String openVinoDeviceType = config.getOpenVinoDeviceType();
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
