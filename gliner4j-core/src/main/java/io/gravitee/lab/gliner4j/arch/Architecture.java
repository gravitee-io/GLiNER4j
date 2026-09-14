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

/**
 * The GLiNER model family a bundle belongs to. Selected automatically at load time from the
 * {@code "architecture"} key in {@code gliner4j_config.json}.
 *
 * <p>Each family differs only in how the prompt/inputs are assembled and how the ONNX graphs are
 * run and scored; the tokenizer, ORT runtime/execution-provider layer, and result types are shared
 * (see {@code docs/model-architecture-spi.md}).
 */
public enum Architecture {
  /** fastino GLiNER2: encoder → span_rep → count-aware scoring_head → classifier_head. */
  GLINER2("gliner2"),
  /**
   * fastino GLiNER2.5 (boundary architecture): encoder → boundary encoder → start/end/inside
   * marginals → shared candidate pool → pooled pair scoring, in one merged {@code ner_full.onnx}.
   */
  GLINER2DOT5("gliner2dot5"),
  /** Original GLiNER uni-encoder (e.g. gliner-multitask, UTC-DeBERTa, gliner-x). */
  GLINER_UNI("gliner-uni"),
  /** GLiNER bi-encoder / poly-encoder (e.g. gliner-bi, modern-gliner-bi, relex, linker). */
  GLINER_BI("gliner-bi"),
  /** GLiClass zero-shot sequence classification. */
  GLICLASS("gliclass"),
  /**
   * GLiClass {@code decoder-kv}: a causal Qwen3 backbone (run by llama.cpp through llamaj.cpp, KV
   * cache per session) plus a small ONNX scorer. Implemented by the {@code gliner4j-llamacpp}
   * module and registered through {@link java.util.ServiceLoader}.
   */
  GLICLASS_DECODER_KV("gliclass-decoder-kv"),
  /**
   * GLiNER {@code gliner_streaming_span}: a causal Qwen3 backbone (llama.cpp via llamaj.cpp, KV
   * cache per session) with a DeBERTa labels encoder and {@code markerV2} span head run as ggml
   * graphs. Streaming NER with rolling span revision. Implemented by {@code gliner4j-llamacpp}.
   */
  GLINER_STREAMING_SPAN("gliner-streaming-span"),
  /** GLiNER encoder-decoder for scalable open-ontology NER. */
  GLINER_DECODER("gliner-decoder");

  /** The value used in {@code gliner4j_config.json}. */
  private final String configValue;

  Architecture(String configValue) {
    this.configValue = configValue;
  }

  public String configValue() {
    return configValue;
  }

  /**
   * Resolves the family from the config value. A {@code null}, blank, or unknown value defaults to
   * {@link #GLINER2} so that bundles exported before the {@code architecture} key existed keep
   * loading as GLiNER2.
   *
   * @param value the {@code "architecture"} string from the config, may be null
   * @return the resolved family, never null
   */
  public static Architecture fromConfigValue(String value) {
    if (value == null || value.isBlank()) {
      return GLINER2;
    }
    var normalized = value.trim().toLowerCase();
    for (var arch : values()) {
      if (arch.configValue.equals(normalized)) {
        return arch;
      }
    }
    return GLINER2;
  }
}
