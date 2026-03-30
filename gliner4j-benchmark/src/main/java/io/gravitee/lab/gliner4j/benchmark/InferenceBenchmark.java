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

import io.gravitee.lab.gliner4j.GLiNER4j;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 5, time = 1)
@Fork(3)
public class InferenceBenchmark {

  @Param({ "1", "4", "8" })
  private int batchSize;

  @Param({ "4", "8" })
  private int entityCount;

  @Param({ "models/gliner2-base-onnx" })
  private String modelPath;

  @Param({ "gliner4j-benchmark/src/main/resources/corpus.txt" })
  private String corpusPath;

  private static final List<EntityDefinition> ENTITIES_4 = List.of(
    new EntityDefinition("person"),
    new EntityDefinition("organization"),
    new EntityDefinition("location"),
    new EntityDefinition("date", "Calendar date or time expression")
  );

  private static final List<EntityDefinition> ENTITIES_8 = List.of(
    new EntityDefinition("person"),
    new EntityDefinition("organization"),
    new EntityDefinition("location"),
    new EntityDefinition("date", "Calendar date or time expression"),
    new EntityDefinition("currency", "Monetary value or currency name"),
    new EntityDefinition("job title", "Professional role or job position"),
    new EntityDefinition("product", "Commercial product or service name"),
    new EntityDefinition("email")
  );

  private GLiNER4j gliner;
  private List<String> batch;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    var corpus = Files
      .readAllLines(Path.of(corpusPath))
      .stream()
      .filter(line -> !line.isBlank())
      .toList();
    if (corpus.isEmpty()) {
      throw new IllegalStateException("Corpus file is empty: " + corpusPath);
    }
    if (batchSize > corpus.size()) {
      throw new IllegalStateException(
        "batchSize (%d) exceeds corpus size (%d)".formatted(
            batchSize,
            corpus.size()
          )
      );
    }

    var entities = entityCount == 4 ? ENTITIES_4 : ENTITIES_8;
    gliner = GLiNER4j.load(Path.of(modelPath), entities);
    batch = corpus.subList(0, batchSize);
  }

  @TearDown(Level.Trial)
  public void teardown() {
    gliner.close();
  }

  @Benchmark
  public void extractBatch(Blackhole bh) {
    bh.consume(gliner.extractBatch(batch));
  }
}
