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
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH benchmark for GLiNER4j inference.
 *
 * <p>Parameterized over:
 *   - profile: which model + entity vocabulary to use (loaded from /profiles/{name}.json)
 *   - textLength: token-length bucket (tiny ≈ 16, short ≈ 64, medium ≈ 256, long ≈ 512)
 *   - entityCount: number of entity labels to keep (a "common app" small set vs a wider one).
 *     Falls through to "use all" when the requested count exceeds the profile's available
 *     entities (e.g., base has only 6 — both 8 and 16 run with all 6).
 *   - batchSize: number of texts per extractBatch() call
 *
 * <p>The corpus is generated at trial setup from {@link CorpusGenerator}, which composes
 * datafaker output with simple templates. No hand-written sample text ships in the repo;
 * the seed is fixed so different benchmark runs see the same content.
 *
 * <p>To reduce JIT specialization on a single batch, a wrapping cursor advances by
 * {@code batchSize} per invocation, so each {@code @Benchmark} call sees a different slice
 * of the {@value CORPUS_SAMPLES}-sample corpus.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(1)
public class InferenceBenchmark {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final int CORPUS_SAMPLES = 16;
  private static final long CORPUS_SEED = 0xC0FFEEL;
  private static final Map<String, Integer> TARGET_TOKENS = Map.of(
    "tiny",
    16,
    "short",
    64,
    "medium",
    256,
    "long",
    512
  );

  @Param({ "base", "pii" })
  private String profile;

  @Param({ "onnx_quantized" })
  private String variant;

  @Param({ "tiny", "short", "medium", "long" })
  private String textLength;

  @Param({ "1", "4", "8" })
  private int batchSize;

  @Param({ "8", "16" })
  private int entityCount;

  private GLiNER4jNER gliner;
  private List<String> corpus;
  private int cursor;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    var resource = "/profiles/" + profile + ".json";
    String modelDir;
    var allEntities = new ArrayList<EntityDefinition>();
    try (var in = InferenceBenchmark.class.getResourceAsStream(resource)) {
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

    var entities = entityCount < allEntities.size()
      ? allEntities.subList(0, entityCount)
      : allEntities;

    var targetTokens = TARGET_TOKENS.get(textLength);
    if (targetTokens == null) {
      throw new IllegalStateException("Unknown textLength: " + textLength);
    }
    corpus = new CorpusGenerator(CORPUS_SEED).generate(
      profile,
      targetTokens,
      CORPUS_SAMPLES
    );

    if (batchSize > corpus.size()) {
      throw new IllegalStateException(
        "batchSize (%d) exceeds corpus size (%d)".formatted(
          batchSize,
          corpus.size()
        )
      );
    }

    // Execution provider is auto-detected from the native runtime on the classpath:
    // a default build runs on CPU, a -Pcuda build picks CUDA, a -Popenvino build picks OpenVINO.
    var runtimeConfig = RuntimeConfig.builder().build();
    gliner = GLiNER4jNER.load(
      Path.of(modelDir),
      entities,
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
  public void extractBatch(Blackhole bh) {
    var n = corpus.size();
    var start = cursor;
    cursor = (cursor + batchSize) % n;

    List<String> slice;
    if (start + batchSize <= n) {
      slice = corpus.subList(start, start + batchSize);
    } else {
      slice = new ArrayList<>(batchSize);
      slice.addAll(corpus.subList(start, n));
      slice.addAll(corpus.subList(0, batchSize - (n - start)));
    }
    bh.consume(gliner.extractBatch(slice));
  }
}
