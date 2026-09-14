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
package io.gravitee.lab.gliner4j.llamacpp;

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.strategy.ClassificationStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.utils.LinAlg;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GLiClass uni-encoder classification on the llama.cpp engine: prompt
 * {@code <<LABEL>>l1<<LABEL>>l2…<<SEP>>text} through a llama.cpp-native encoder
 * ({@link LlamaEncoderBackbone}), CLS + class-token rows through the ggml head
 * ({@link GgmlGliclassHead}), sigmoid + threshold. Same prompt and decode as the ONNX
 * {@code GliclassClassificationStrategy}; batches fan out on virtual threads so the encoder
 * dispatcher packs them into shared {@code llama_encode} calls.
 */
public final class LlamaGliclassClassificationStrategy
  implements ClassificationStrategy {

  private static final Logger log = LoggerFactory.getLogger(
    LlamaGliclassClassificationStrategy.class
  );

  public static final int DEFAULT_N_UBATCH = 2048;
  public static final int DEFAULT_N_SEQ_MAX = 8;

  private final DjlTokenizerWrapper tokenizer;
  private final LlamaEncoderBackbone encoder;
  private final GgmlGliclassHead head;
  private final List<ClassificationLabel> labels;
  private final GLiNER4jTelemetry telemetry = new GLiNER4jTelemetry("classify");
  private final long classTokenIndex;
  private final String labelToken;
  private final String sepToken;
  private final boolean promptFirst;

  LlamaGliclassClassificationStrategy(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    LlamaEncoderBackbone encoder,
    GgmlGliclassHead head,
    List<ClassificationLabel> labels
  ) {
    this.tokenizer = tokenizer;
    this.encoder = encoder;
    this.head = head;
    this.labels = labels;
    this.classTokenIndex = config.archLong("class_token_index", -1);
    this.labelToken = config.archString("label_token", "<<LABEL>>");
    this.sepToken = config.archString("sep_token", "<<SEP>>");
    this.promptFirst = config.archBoolean("prompt_first", true);
    if (classTokenIndex < 0) {
      throw new IllegalStateException(
        "GLiClass bundle is missing architecture_config.class_token_index"
      );
    }
  }

  /** Loads the encoder GGUF and head for a {@code gliclass} bundle with {@code engine=llamacpp}. */
  public static LlamaGliclassClassificationStrategy create(
    LoadContext ctx,
    List<ClassificationLabel> labels
  ) {
    var config = ctx.config();
    var rc = ctx.runtimeConfig();
    var pooling = config.archString("pooling_strategy", "first");
    if (
      !"first".equals(pooling) ||
      config.archBoolean("extract_text_features", false) ||
      !config.archBoolean("embed_class_token", true)
    ) {
      throw new UnsupportedOperationException(
        "GLiClass on llama.cpp supports pooling_strategy=first, extract_text_features=false, " +
          "embed_class_token=true"
      );
    }
    var ggufDir = ctx.modelDir().resolve("gguf");
    var backbone = resolveBackbone(ggufDir, config, ctx.variant());
    int threads = LlamaBackbone.threads(rc);
    int gpuLayers = gpuLayers(rc);
    var encoder = new LlamaEncoderBackbone(
      backbone,
      gpuLayers,
      config.archInt("n_ubatch", DEFAULT_N_UBATCH),
      config.archInt("n_seq_max", DEFAULT_N_SEQ_MAX),
      threads
    );
    var head = new GgmlGliclassHead(
      ggufDir.resolve("heads.gguf"),
      gpuLayers > 0,
      threads,
      config.getHiddenSize(),
      config.archString("scorer_type", "simple"),
      config.archBoolean("normalize_features", false)
    );
    log.info(
      "GLiClass llama.cpp strategy loaded (backbone={}, scorer={})",
      backbone.getFileName(),
      config.archString("scorer_type", "simple")
    );
    return new LlamaGliclassClassificationStrategy(
      config,
      ctx.tokenizer(),
      encoder,
      head,
      labels
    );
  }

  /**
   * The encoder GGUF for a load: {@code variant} names a quantization ({@code f16}, {@code q8_0},
   * … → {@code gguf/backbone-<variant>.gguf}) or a file name; anything else (including the ONNX
   * default variant) falls back to {@code architecture_config.backbone_gguf}.
   */
  static java.nio.file.Path resolveBackbone(
    java.nio.file.Path ggufDir,
    GLiNER4jConfig config,
    String variant
  ) {
    if (variant != null && !variant.isBlank()) {
      for (var candidate : List.of(variant, "backbone-" + variant + ".gguf")) {
        var p = ggufDir.resolve(candidate);
        if (Files.isRegularFile(p)) {
          return p;
        }
      }
    }
    return ggufDir.resolve(
      config.archString("backbone_gguf", "backbone-q8_0.gguf")
    );
  }

  static int gpuLayers(RuntimeConfig rc) {
    return switch (rc.getExecutionProvider()) {
      case CPU, OPENVINO -> 0;
      case AUTO, CUDA -> 99; // llama.cpp picks Metal / CUDA / Vulkan from the loaded backends
    };
  }

  @Override
  public List<ClassificationResult> classify(String text, float threshold) {
    return classifyWith(text, labels, threshold);
  }

  @Override
  public List<ClassificationResult> classify(
    String text,
    List<ClassificationLabel> overrideLabels,
    float threshold
  ) {
    return classifyWith(text, overrideLabels, threshold);
  }

  @Override
  public List<List<ClassificationResult>> classifyBatch(
    List<String> texts,
    float threshold
  ) {
    if (texts == null || texts.isEmpty()) {
      return List.of();
    }
    if (texts.size() == 1) {
      return List.of(classifyWith(texts.get(0), labels, threshold));
    }
    // Concurrent submissions coalesce in the encoder dispatcher into shared llama_encode calls.
    try (var pool = Executors.newVirtualThreadPerTaskExecutor()) {
      var futures = new ArrayList<
        CompletableFuture<List<ClassificationResult>>
      >();
      for (var text : texts) {
        futures.add(
          CompletableFuture.supplyAsync(
            () -> classifyWith(text, labels, threshold),
            pool
          )
        );
      }
      var out = new ArrayList<List<ClassificationResult>>(texts.size());
      for (var f : futures) {
        out.add(f.join());
      }
      return out;
    }
  }

  private List<ClassificationResult> classifyWith(
    String text,
    List<ClassificationLabel> activeLabels,
    float threshold
  ) {
    long startNanos = System.nanoTime();
    if (text == null || text.isBlank() || activeLabels.isEmpty()) {
      telemetry.record(0.0, 1, 0);
      return List.of();
    }
    var sb = new StringBuilder();
    for (var label : activeLabels) {
      sb.append(labelToken).append(label.name());
    }
    sb.append(sepToken);
    var prompt = promptFirst ? sb + text : text + sb;
    long[] ids = tokenizer.encodeWithSpecialTokens(prompt);

    var hidden = encoder.encode(ids);

    var classPositions = new ArrayList<Integer>(activeLabels.size());
    for (int i = 0; i < ids.length; i++) {
      if (ids[i] == classTokenIndex) {
        classPositions.add(i);
      }
    }
    if (classPositions.isEmpty()) {
      telemetry.record(0.0, 1, 0);
      return List.of();
    }
    if (classPositions.size() != activeLabels.size()) {
      log.warn(
        "GLiClass class-token count ({}) does not match label count ({}); scoring the first {} label(s) only",
        classPositions.size(),
        activeLabels.size(),
        Math.min(classPositions.size(), activeLabels.size())
      );
    }
    int n = Math.min(classPositions.size(), activeLabels.size());
    var classRows = new float[n][];
    for (int k = 0; k < n; k++) {
      classRows[k] = hidden[classPositions.get(k)];
    }
    float[] logits = head.score(hidden[0], classRows);

    var results = new ArrayList<ClassificationResult>(n);
    for (int k = 0; k < n; k++) {
      float score = LinAlg.sigmoid(logits[k]);
      if (score >= threshold) {
        results.add(
          new ClassificationResult(activeLabels.get(k).name(), score)
        );
      }
    }
    results.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));
    telemetry.record(
      (System.nanoTime() - startNanos) / 1_000_000.0,
      1,
      results.size()
    );
    return results;
  }

  @Override
  public void close() {
    head.close();
    encoder.close();
    tokenizer.close();
  }
}
