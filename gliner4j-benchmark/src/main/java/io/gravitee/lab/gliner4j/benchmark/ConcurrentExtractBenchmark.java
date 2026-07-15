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
import java.util.concurrent.atomic.AtomicInteger;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH benchmark for concurrent single-text extract() against one shared facade.
 *
 * <p>Measures throughput of N caller threads hammering {@code extract()} on the same
 * loaded model — the "server" usage pattern. Compare {@code microBatching=false}
 * (per-call inference, safe since the runtime buffer pool) against
 * {@code microBatching=true} (calls coalesced into batched runs).
 */
@BenchmarkMode(Mode.Throughput)
@OutputTimeUnit(TimeUnit.SECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 2)
@Measurement(iterations = 8, time = 2)
@Fork(1)
@Threads(4)
public class ConcurrentExtractBenchmark {

  private static final ObjectMapper MAPPER = new ObjectMapper();
  private static final int CORPUS_SAMPLES = 16;
  private static final long CORPUS_SEED = 0xC0FFEEL;

  @Param({ "base" })
  private String profile;

  @Param({ "onnx_quantized" })
  private String variant;

  @Param({ "short" })
  private String textLength;

  @Param({ "false", "true" })
  private boolean microBatching;

  private GLiNER4jNER gliner;
  private List<String> corpus;
  private final AtomicInteger cursor = new AtomicInteger();

  @Setup(Level.Trial)
  public void setup() throws IOException {
    var resource = "/profiles/" + profile + ".json";
    String modelDir;
    var entities = new ArrayList<EntityDefinition>();
    try (
      var in = ConcurrentExtractBenchmark.class.getResourceAsStream(resource)
    ) {
      if (in == null) {
        throw new IllegalStateException(
          "Profile resource not found: " + resource
        );
      }
      var json = MAPPER.readTree(in);
      modelDir = json.get("modelDir").asText();
      for (var node : json.get("entities")) {
        entities.add(
          new EntityDefinition(
            node.get("name").asText(),
            node.path("description").asText("")
          )
        );
      }
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
      .microBatchingEnabled(microBatching)
      .build();
    gliner = GLiNER4jNER.load(
      Path.of(modelDir),
      entities,
      variant,
      runtimeConfig
    );
  }

  @TearDown(Level.Trial)
  public void teardown() {
    gliner.close();
  }

  @Benchmark
  public void extractConcurrently(Blackhole bh) {
    var text = corpus.get(
      Math.floorMod(cursor.getAndIncrement(), corpus.size())
    );
    bh.consume(gliner.extract(text));
  }
}
