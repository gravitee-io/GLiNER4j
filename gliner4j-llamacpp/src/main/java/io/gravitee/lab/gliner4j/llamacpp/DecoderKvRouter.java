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
import io.gravitee.lab.gliner4j.arch.Architecture;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Path;
import java.util.List;

/**
 * Zero-shot classification and LLM routing with a GLiClass {@code decoder-kv} bundle
 * (e.g. {@code scx-admin/scx-router-v0.1}): stateless calls with any label set, single- or
 * multi-label, and {@link #openSession streaming sessions} that re-score a growing conversation
 * per turn while encoding only the new tokens.
 *
 * <p>The same bundle also loads through {@link io.gravitee.lab.gliner4j.GLiNER4jClassifier}
 * (multi-label only) via the {@code gliclass-decoder-kv} architecture registration.
 */
public final class DecoderKvRouter implements AutoCloseable {

  private final DecoderKvEngine engine;
  private final float defaultThreshold;

  private DecoderKvRouter(DecoderKvEngine engine, float defaultThreshold) {
    this.engine = engine;
    this.defaultThreshold = defaultThreshold;
  }

  public static DecoderKvRouter load(Path modelDir) {
    return load(
      modelDir,
      BaseRuntime.DEFAULT_VARIANT,
      RuntimeConfig.defaults()
    );
  }

  public static DecoderKvRouter load(
    Path modelDir,
    RuntimeConfig runtimeConfig
  ) {
    return load(modelDir, BaseRuntime.DEFAULT_VARIANT, runtimeConfig);
  }

  public static DecoderKvRouter load(
    Path modelDir,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    var config = GLiNER4jConfig.load(modelDir);
    if (config.getArchitecture() != Architecture.GLICLASS_DECODER_KV) {
      throw new IllegalArgumentException(
        modelDir +
          " is a '" +
          config.getArchitecture().configValue() +
          "' bundle; DecoderKvRouter needs architecture=gliclass-decoder-kv"
      );
    }
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var ctx = new LoadContext(
      modelDir,
      variant,
      runtimeConfig,
      config,
      tokenizer
    );
    return new DecoderKvRouter(
      DecoderKvEngine.load(ctx),
      config.getDefaultThreshold()
    );
  }

  /** Multi-label classification with the bundle's default threshold. */
  public List<ClassificationResult> classify(String text, List<String> labels) {
    return classify(text, labels, defaultThreshold);
  }

  /** Multi-label classification: sigmoid per label, labels scoring {@code >= threshold}, best first. */
  public List<ClassificationResult> classify(
    String text,
    List<String> labels,
    float threshold
  ) {
    var logits = logits(text, labels);
    return logits == null
      ? List.of()
      : DecoderKvScores.multiLabel(logits, labels, threshold);
  }

  /** Single-label classification: softmax over {@code labels}, every label returned, best first. */
  public List<ClassificationResult> classifySingleLabel(
    String text,
    List<String> labels
  ) {
    var logits = logits(text, labels);
    return logits == null
      ? List.of()
      : DecoderKvScores.singleLabel(logits, labels);
  }

  /** Raw logits, one per label, or {@code null} for an empty text or label list. */
  public float[] logits(String text, List<String> labels) {
    if (text == null || text.isBlank() || labels == null || labels.isEmpty()) {
      return null;
    }
    var ids = engine.encodeText(text);
    if (ids.length == 0) {
      return null;
    }
    return engine.score(ids, engine.section(labels));
  }

  /** Opens a streaming session (one llama.cpp sequence); close it when the conversation ends. */
  public DecoderKvSession openSession(String sessionId) {
    return new DecoderKvSession(engine, sessionId);
  }

  /** Context window of the backbone in tokens (text + label section must fit). */
  public int contextWindow() {
    return engine.nCtx();
  }

  @Override
  public void close() {
    engine.close();
  }
}
