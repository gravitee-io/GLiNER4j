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
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Path;
import java.util.List;

/**
 * Streaming zero-shot NER with a GLiNER {@code gliner_streaming_span} bundle (e.g.
 * {@code knowledgator/gliner-stream-pii-v1.0}): stateless extraction with any label set, and
 * {@link #openSession sessions} that accept text chunk by chunk and return the revised entity
 * snapshot after each append while encoding only the new tokens.
 *
 * <p>The same bundle also loads through {@link io.gravitee.lab.gliner4j.GLiNER4jNER} once
 * {@code gliner4j-llamacpp} is on the classpath.
 */
public final class StreamingSpanNer implements AutoCloseable {

  private final StreamingSpanEngine engine;
  private final float defaultThreshold;

  private StreamingSpanNer(StreamingSpanEngine engine, float defaultThreshold) {
    this.engine = engine;
    this.defaultThreshold = defaultThreshold;
  }

  public static StreamingSpanNer load(Path modelDir) {
    return load(
      modelDir,
      BaseRuntime.DEFAULT_VARIANT,
      RuntimeConfig.defaults()
    );
  }

  public static StreamingSpanNer load(
    Path modelDir,
    RuntimeConfig runtimeConfig
  ) {
    return load(modelDir, BaseRuntime.DEFAULT_VARIANT, runtimeConfig);
  }

  public static StreamingSpanNer load(
    Path modelDir,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    var config = GLiNER4jConfig.load(modelDir);
    if (config.getArchitecture() != Architecture.GLINER_STREAMING_SPAN) {
      throw new IllegalArgumentException(
        modelDir +
          " is a '" +
          config.getArchitecture().configValue() +
          "' bundle; StreamingSpanNer needs architecture=gliner-streaming-span"
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
    return new StreamingSpanNer(
      StreamingSpanEngine.load(ctx),
      config.getDefaultThreshold()
    );
  }

  /** Stateless extraction with the bundle's default threshold. */
  public List<EntitySpan> extract(String text, List<String> labels) {
    return extract(text, labels, defaultThreshold);
  }

  /** Stateless extraction: one cold pass over the whole text, every span scored once. */
  public List<EntitySpan> extract(
    String text,
    List<String> labels,
    float threshold
  ) {
    if (text == null || text.isBlank() || labels == null || labels.isEmpty()) {
      return List.of();
    }
    try (var session = new StreamingSpanSession(engine, "stateless", labels)) {
      return session.append(text, threshold);
    }
  }

  /** Opens a streaming session (one llama.cpp sequence) for {@code labels}; close it when done. */
  public StreamingSpanSession openSession(
    String sessionId,
    List<String> labels
  ) {
    return new StreamingSpanSession(engine, sessionId, labels);
  }

  public int contextWindow() {
    return engine.nCtx();
  }

  @Override
  public void close() {
    engine.close();
  }
}
