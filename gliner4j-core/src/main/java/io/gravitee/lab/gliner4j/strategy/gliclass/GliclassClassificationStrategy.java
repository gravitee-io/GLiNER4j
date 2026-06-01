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
package io.gravitee.lab.gliner4j.strategy.gliclass;

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.runtime.GliclassRuntime;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.strategy.ClassificationStrategy;
import io.gravitee.lab.gliner4j.telemetry.GLiNER4jTelemetry;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import lombok.extern.slf4j.Slf4j;

/**
 * GLiClass zero-shot classification strategy (uni-encoder, {@code simple} dot-product scorer).
 *
 * <p>Pipeline for {@code gliclass-modern-base-v3.0}:
 * <pre>
 *   prompt = "&lt;&lt;LABEL&gt;&gt;l1&lt;&lt;LABEL&gt;&gt;l2...&lt;&lt;LABEL&gt;&gt;lN&lt;&lt;SEP&gt;&gt;" + text   (prompt_first)
 *   H      = encoder(prompt)                                   // [CLS] at position 0
 *   classEmbs = H[ &lt;&lt;LABEL&gt;&gt; positions ]                       // one per label, in order
 *   textEmb   = H[0]                                           // CLS (extract_text_features=false, pooling=first)
 *   logits    = score_head(textEmb, classEmbs)                // projectors + dot product
 *   scores    = sigmoid(logits)                               // multi-label
 * </pre>
 *
 * <p>Selected by {@link io.gravitee.lab.gliner4j.arch.gliclass.GliclassArchitecture}.
 */
@Slf4j
public final class GliclassClassificationStrategy
  implements ClassificationStrategy {

  private final GLiNER4jConfig config;
  private final DjlTokenizerWrapper tokenizer;
  private final GliclassRuntime runtime;
  private final List<ClassificationLabel> labels;
  private final GLiNER4jTelemetry telemetry;

  private final long classTokenIndex;
  private final String labelToken;
  private final String sepToken;
  private final boolean promptFirst;

  private GliclassClassificationStrategy(
    GLiNER4jConfig config,
    DjlTokenizerWrapper tokenizer,
    GliclassRuntime runtime,
    List<ClassificationLabel> labels
  ) {
    this.config = config;
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.labels = labels;
    this.telemetry = new GLiNER4jTelemetry("classify");

    this.classTokenIndex = config.archLong("class_token_index", -1);
    this.labelToken = config.archString("label_token", "<<LABEL>>");
    this.sepToken = config.archString("sep_token", "<<SEP>>");
    this.promptFirst = config.archBoolean("prompt_first", true);

    var pooling = config.archString("pooling_strategy", "first");
    boolean extractText = config.archBoolean("extract_text_features", false);
    if (!"first".equals(pooling) || extractText) {
      throw new UnsupportedOperationException(
        "GLiClass strategy currently supports pooling_strategy=first + " +
          "extract_text_features=false (text rep = CLS); got pooling=" +
          pooling +
          ", extract_text_features=" +
          extractText
      );
    }
    if (classTokenIndex < 0) {
      throw new IllegalStateException(
        "GLiClass bundle is missing architecture_config.class_token_index"
      );
    }
  }

  /**
   * Builds the GLiClass classification strategy from the family-agnostic load context.
   *
   * @param ctx the load context
   * @param labels the load-time classification labels
   * @return a ready strategy
   */
  public static GliclassClassificationStrategy create(
    LoadContext ctx,
    List<ClassificationLabel> labels
  ) {
    var runtime = new GliclassRuntime(
      ctx.modelDir(),
      ctx.variant(),
      ctx.runtimeConfig()
    );
    return new GliclassClassificationStrategy(
      ctx.config(),
      ctx.tokenizer(),
      runtime,
      labels
    );
  }

  // ---- ClassificationStrategy ---------------------------------------------

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
    if (texts == null) {
      return List.of();
    }
    // First cut: per-text encode. The encoder is the cost; batching it (padded
    // runEncoderBatch + per-row gather) is a follow-up optimization.
    var results = new ArrayList<List<ClassificationResult>>(texts.size());
    for (var text : texts) {
      results.add(classifyWith(text, labels, threshold));
    }
    return results;
  }

  // ---- core ----------------------------------------------------------------

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

    var prompt = buildPrompt(activeLabels, text);
    long[] ids = tokenizer.encodeWithSpecialTokens(prompt);
    long[] mask = new long[ids.length];
    Arrays.fill(mask, 1L);

    var hidden = runtime.runEncoderFull(ids, mask)[0]; // [seq][hidden]

    // Gather class-token positions in order; the k-th <<LABEL>> marker ↔ activeLabels[k].
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
        "GLiClass class-token count ({}) does not match label count ({}); scoring the first {} " +
          "label(s) only — check that no label name contains the {} marker or tokenizes unexpectedly",
        classPositions.size(),
        activeLabels.size(),
        Math.min(classPositions.size(), activeLabels.size()),
        labelToken
      );
    }
    int n = Math.min(classPositions.size(), activeLabels.size());
    var classEmbs = new float[n][];
    for (int k = 0; k < n; k++) {
      classEmbs[k] = hidden[classPositions.get(k)];
    }
    float[] textEmb = hidden[0]; // CLS

    float[] logits = runtime.runScoreHead(textEmb, classEmbs);

    var results = new ArrayList<ClassificationResult>(n);
    for (int k = 0; k < n; k++) {
      float score = sigmoid(logits[k]);
      if (score >= threshold) {
        results.add(
          new ClassificationResult(activeLabels.get(k).name(), score)
        );
      }
    }
    results.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));

    double durationMs = (System.nanoTime() - startNanos) / 1_000_000.0;
    telemetry.record(durationMs, 1, results.size());
    return results;
  }

  private String buildPrompt(
    List<ClassificationLabel> activeLabels,
    String text
  ) {
    var sb = new StringBuilder();
    for (var label : activeLabels) {
      sb.append(labelToken).append(label.name());
    }
    sb.append(sepToken);
    return promptFirst ? sb + text : text + sb;
  }

  private static float sigmoid(float x) {
    return 1.0f / (1.0f + (float) Math.exp(-x));
  }

  @Override
  public void close() {
    runtime.close();
    tokenizer.close();
    log.info("GliclassClassificationStrategy closed");
  }
}
