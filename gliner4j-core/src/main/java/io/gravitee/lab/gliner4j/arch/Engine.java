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
package io.gravitee.lab.gliner4j.arch;

import java.util.Locale;

/**
 * The inference engine a bundle runs on. Selected by the top-level {@code "engine"} key of
 * {@code gliner4j_config.json}; absent means {@link #ONNX}.
 *
 * <p>A model family may be implemented on more than one engine: the ONNX Runtime implementation
 * ships in {@code gliner4j-core}, the llama.cpp/ggml one in {@code gliner4j-llamacpp}. Unlike
 * {@link Architecture#fromConfigValue}, an unknown engine value is an error — silently falling back
 * to another engine would load the wrong runtime for the bundle's files.
 */
public enum Engine {
  /** ONNX Runtime graphs under {@code onnx}, {@code onnx_quantized}, … (the default). */
  ONNX("onnx"),
  /** llama.cpp backbone and/or ggml graphs under {@code gguf/} (module {@code gliner4j-llamacpp}). */
  LLAMACPP("llamacpp");

  private final String configValue;

  Engine(String configValue) {
    this.configValue = configValue;
  }

  public String configValue() {
    return configValue;
  }

  /**
   * Parses the {@code engine} config value.
   *
   * @param value the raw value, may be null or blank
   * @return the engine; {@link #ONNX} when absent
   * @throws IllegalArgumentException on an unknown value
   */
  public static Engine fromConfigValue(String value) {
    if (value == null || value.isBlank()) {
      return ONNX;
    }
    var v = value
      .trim()
      .toLowerCase(Locale.ROOT)
      .replace('-', '_')
      .replace('.', '_');
    for (var e : values()) {
      if (
        e.configValue.equals(v) ||
        e.name().toLowerCase(Locale.ROOT).equals(v) ||
        ("llama_cpp".equals(v) && e == LLAMACPP)
      ) {
        return e;
      }
    }
    throw new IllegalArgumentException(
      "Unknown engine '" +
        value +
        "' in gliner4j_config.json; expected one of onnx, llamacpp"
    );
  }
}
