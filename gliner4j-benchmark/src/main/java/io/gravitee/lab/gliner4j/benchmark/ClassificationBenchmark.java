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
import io.gravitee.lab.gliner4j.GLiNER4jClassifier;
import io.gravitee.lab.gliner4j.runtime.ExecutionProvider;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

/**
 * JMH benchmark for GLiNER4j zero-shot text classification ({@link GLiNER4jClassifier}).
 *
 * <p>Counterpart to {@link InferenceBenchmark} (which covers NER). Parameterized over:
 *   - profile: which classification model + label set to use (loaded from /profiles/{name}.json).
 *     Defaults to {@code gliclass} (GLiClass / ModernBERT). Add {@code -p profile=gliguard} to
 *     compare against the GLiNER2 / DeBERTa classifier head.
 *   - labelCount: number of labels kept (a small "common app" set vs the full set)
 *   - textLength / batchSize: as in {@link InferenceBenchmark}
 *
 * <p>The corpus uses the {@code base} (news-flavoured) generator — generic, topic-bearing text is
 * representative for classification latency; the labels, not the text content, drive the work.
 */
@BenchmarkMode(Mode.AverageTime)
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@State(Scope.Benchmark)
@Warmup(iterations = 2, time = 1)
@Measurement(iterations = 10, time = 1)
@Fork(1)
public class ClassificationBenchmark {

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

  @Param({ "gliclass" })
  private String profile;

  /** ONNX: the {@code onnx*} sub-directory. llama.cpp: the GGUF quantization ({@code f16}, {@code q8_0}); other values use the bundle default. */
  @Param({ "onnx_quantized" })
  private String variant;

  /** {@code onnx} (ONNX Runtime bundle at {@code modelDir}) or {@code llamacpp} (bundle at {@code llamacppModelDir}). */
  @Param({ "onnx" })
  private String engine;

  /** {@code auto}, {@code cpu}, {@code cuda}, {@code openvino}; {@code auto} = Metal/CUDA for llama.cpp, best ORT EP for onnx. */
  @Param({ "auto" })
  private String executionProvider;

  @Param({ "tiny", "short", "medium", "long" })
  private String textLength;

  @Param({ "1", "4", "8" })
  private int batchSize;

  @Param({ "4", "8" })
  private int labelCount;

  private GLiNER4jClassifier classifier;
  private List<String> corpus;
  private int cursor;

  @Setup(Level.Trial)
  public void setup() throws IOException {
    var resource = "/profiles/" + profile + ".json";
    String modelDir;
    var allLabels = new ArrayList<ClassificationLabel>();
    try (var in = ClassificationBenchmark.class.getResourceAsStream(resource)) {
      if (in == null) {
        throw new IllegalStateException(
          "Profile resource not found: " + resource
        );
      }
      var json = MAPPER.readTree(in);
      modelDir = "llamacpp".equals(engine)
        ? json.path("llamacppModelDir").asText(null)
        : json.get("modelDir").asText();
      if (modelDir == null) {
        throw new IllegalStateException(
          "Profile " +
            profile +
            " has no llamacppModelDir (export it with task <family>:llamacpp)"
        );
      }
      for (var node : json.get("labels")) {
        allLabels.add(
          new ClassificationLabel(
            node.get("name").asText(),
            node.path("description").asText("")
          )
        );
      }
    }

    var labels = labelCount < allLabels.size()
      ? allLabels.subList(0, labelCount)
      : allLabels;

    var targetTokens = TARGET_TOKENS.get(textLength);
    if (targetTokens == null) {
      throw new IllegalStateException("Unknown textLength: " + textLength);
    }
    // Generic news-flavoured text; classification latency is driven by the labels, not content.
    corpus = new CorpusGenerator(CORPUS_SEED).generate(
      "base",
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

    // ONNX: the EP must also be compiled in (-Pcuda / -Popenvino); llama.cpp: auto picks Metal/CUDA.
    var runtimeConfig = RuntimeConfig.builder()
      .executionProvider(
        ExecutionProvider.valueOf(
          executionProvider.toUpperCase(java.util.Locale.ROOT)
        )
      )
      .build();
    classifier = GLiNER4jClassifier.load(
      Path.of(modelDir),
      labels,
      variant,
      runtimeConfig
    );
    cursor = 0;
  }

  @TearDown(Level.Trial)
  public void teardown() {
    classifier.close();
  }

  @Benchmark
  public void classifyBatch(Blackhole bh) {
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
    bh.consume(classifier.classifyBatch(slice));
  }
}
