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
package io.gravitee.lab.gliner4j;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.gravitee.lab.gliner4j.arch.Architecture;
import io.gravitee.lab.gliner4j.arch.Engine;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Map;
import lombok.Builder;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

/**
 * Configuration for a GLiNER4j instance, loaded from gliner4j_config.json.
 */
@Slf4j
@Getter
@Builder
public class GLiNER4jConfig {

  public static final float DEFAULT_THRESHOLD = 0.5f;
  public static final int DEFAULT_MAX_WIDTH = 12;

  private final Path modelPath;

  /** The GLiNER family this bundle belongs to. Absent in the config ⇒ {@link Architecture#GLINER2}. */
  @Builder.Default
  private final Architecture architecture = Architecture.GLINER2;

  /** The inference engine this bundle's files target. Absent in the config ⇒ {@link Engine#ONNX}. */
  @Builder.Default
  private final Engine engine = Engine.ONNX;

  @Builder.Default
  private final float defaultThreshold = DEFAULT_THRESHOLD;

  @Builder.Default
  private final int maxWidth = DEFAULT_MAX_WIDTH;

  @Builder.Default
  private final int hiddenSize = 768;

  @Builder.Default
  private final int maxCount = 20;

  @Builder.Default
  private final boolean usesSpanIdx = true;

  @Builder.Default
  private final String tokenPooling = "first";

  @Builder.Default
  private final Map<String, Long> specialTokenIds = Map.of();

  /**
   * Family-specific config keys, kept under the {@code "architecture_config"} block so each model
   * family can carry its own knobs without forking the shared schema. Empty for bundles that
   * declare none. Read it through {@link #archInt}/{@link #archLong}/{@link #archBoolean}/{@link #archString}.
   */
  @Builder.Default
  private final Map<String, Object> architectureConfig = Map.of();

  /** Returns the {@code architecture_config} value for {@code key} as an int, or {@code def} if absent. */
  public int archInt(String key, int def) {
    var v = architectureConfig.get(key);
    return v instanceof Number n ? n.intValue() : def;
  }

  /** Returns the {@code architecture_config} value for {@code key} as a long, or {@code def} if absent. */
  public long archLong(String key, long def) {
    var v = architectureConfig.get(key);
    return v instanceof Number n ? n.longValue() : def;
  }

  /** Returns the {@code architecture_config} value for {@code key} as a double, or {@code def} if absent. */
  public double archDouble(String key, double def) {
    var v = architectureConfig.get(key);
    return v instanceof Number n ? n.doubleValue() : def;
  }

  /** Returns the {@code architecture_config} value for {@code key} as a boolean, or {@code def} if absent. */
  public boolean archBoolean(String key, boolean def) {
    var v = architectureConfig.get(key);
    return v instanceof Boolean b ? b : def;
  }

  /** Returns the {@code architecture_config} value for {@code key} as a String, or {@code def} if absent. */
  public String archString(String key, String def) {
    var v = architectureConfig.get(key);
    return v != null ? v.toString() : def;
  }

  /**
   * Loads configuration from gliner4j_config.json in the model directory.
   *
   * @param modelDir path to the model directory
   * @return loaded configuration
   */
  public static GLiNER4jConfig load(Path modelDir) {
    var configFile = modelDir.resolve("gliner4j_config.json");
    var mapper = new ObjectMapper();
    try {
      var json = mapper.readValue(configFile.toFile(), JsonConfig.class);
      var architecture = Architecture.fromConfigValue(json.architecture);
      var engine = Engine.fromConfigValue(json.engine);
      log.info(
        "Loaded config: architecture={}, engine={}, hiddenSize={}, maxWidth={}, maxCount={}, usesSpanIdx={}, tokenPooling={}",
        architecture.configValue(),
        engine.configValue(),
        json.hiddenSize,
        json.maxWidth,
        json.maxCount,
        json.usesSpanIdx,
        json.tokenPooling
      );
      return GLiNER4jConfig.builder()
        .modelPath(modelDir)
        .architecture(architecture)
        .engine(engine)
        .hiddenSize(json.hiddenSize)
        .maxWidth(json.maxWidth)
        .maxCount(json.maxCount)
        .usesSpanIdx(json.usesSpanIdx)
        .tokenPooling(json.tokenPooling)
        .specialTokenIds(json.specialTokenIds)
        .architectureConfig(
          json.architectureConfig != null ? json.architectureConfig : Map.of()
        )
        .build();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load config from " + configFile, e);
    }
  }

  @JsonIgnoreProperties(ignoreUnknown = true)
  private static class JsonConfig {

    @JsonProperty("architecture")
    String architecture = null;

    @JsonProperty("engine")
    String engine = null;

    @JsonProperty("hidden_size")
    int hiddenSize = 768;

    @JsonProperty("max_width")
    int maxWidth = DEFAULT_MAX_WIDTH;

    @JsonProperty("max_count")
    int maxCount = 20;

    @JsonProperty("uses_span_idx")
    boolean usesSpanIdx = true;

    @JsonProperty("token_pooling")
    String tokenPooling = "first";

    @JsonProperty("special_token_ids")
    Map<String, Long> specialTokenIds = Map.of();

    @JsonProperty("architecture_config")
    Map<String, Object> architectureConfig = Map.of();
  }
}
