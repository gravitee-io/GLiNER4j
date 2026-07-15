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
package io.gravitee.lab.gliner4j.benchmark;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.gravitee.lab.gliner4j.GLiNER4jNER;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH benchmark for per-call entity override extraction.
 *
 * <p>Rotates through a small set of override label lists so the assembler LRU cache
 * ({@code RuntimeConfig.overrideCacheEnabled}) is exercised the way a real workload would:
 * a handful of recurring label sets rather than one fixed schema. Compare
 * {@code overrideCache=false} (old behavior, schema re-tokenized every call) against the
 * default to measure the cache win.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(1)
public class OverrideBenchmark {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final int CORPUS_SAMPLES = 16;
  private static final long CORPUS_SEED = 0xC0FFEEL;
  private static final int OVERRIDE_SETS = 4;

  @Param({ "base" })
  private String profile;

  @Param({ "onnx_quantized" })
  private String variant;

  @Param({ "short" })
  private String textLength;

  /** false disables the override assembler cache (previous behavior). */
  @Param({ "false", "true" })
  private boolean overrideCache;

  private GLiNER4jNER gliner;
  private List<String> corpus;
  private List<List<EntityDefinition>> overrideSets;
  private int cursor;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    var resource = "/profiles/" + profile + ".json";
    String modelDir;
    var allEntities = new ArrayList<EntityDefinition>();
    try (var in = OverrideBenchmark.class.getResourceAsStream(resource)) {
      if (in == null) {
        throw new IllegalStateException(
          "Profile resource not found: " + resource
        );
      }
      var json = MAPPER.readTree(in);
      modelDir = json.get("modelDir").asText();
      for (var node : json.get("entities")) {
        allEntities.add(
          new EntityDefinition(
            node.get("name").asText(),
            node.path("description").asText("")
          )
        );
      }
    }

    // Rotating override sets: distinct sublists of the profile's entities
    overrideSets = new ArrayList<>(OVERRIDE_SETS);
    for (int i = 0; i < OVERRIDE_SETS; i++) {
      int size = Math.max(2, allEntities.size() - (i % 3));
      overrideSets.add(
        List.copyOf(allEntities.subList(0, Math.min(size, allEntities.size())))
      );
    }

    int targetTokens = switch (textLength) {
      case "tiny" -> 16;
      case "short" -> 64;
      case "medium" -> 256;
      case "long" -> 512;
      default -> throw new IllegalStateException(
        "Unknown textLength: " + textLength
      );
    };
    corpus = new CorpusGenerator(CORPUS_SEED).generate(
      profile,
      targetTokens,
      CORPUS_SAMPLES
    );

    var runtimeConfig = RuntimeConfig.builder()
      .overrideCacheEnabled(overrideCache)
      .build();
    gliner = GLiNER4jNER.load(
      Path.of(modelDir),
      overrideSets.get(0),
      variant,
      runtimeConfig
    );
    cursor = 0;
  }

  @TearDown(Level.Trial)
  public void teardown() {
    gliner.close();
  }

  @Benchmark
  public void extractWithOverride(Blackhole bh) {
    int i = cursor++;
    var text = corpus.get(i % corpus.size());
    var overrides = overrideSets.get(i % overrideSets.size());
    bh.consume(gliner.extract(text, overrides, 0.5f));
  }
}
