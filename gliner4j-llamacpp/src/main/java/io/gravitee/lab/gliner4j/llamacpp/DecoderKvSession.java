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

import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import java.util.List;

/**
 * A streaming classification session: text chunks accumulate in one llama.cpp KV sequence, and
 * every {@link #classify} scores the whole conversation so far by decoding only the label section
 * on top of the cache (the label tokens are dropped again afterwards, so labels can change between
 * calls). Mirrors gliclass's {@code StreamingZeroShotClassificationPipeline}.
 *
 * <p>Not thread-safe; one session per conversation. Close it to release the sequence slot.
 */
public final class DecoderKvSession implements AutoCloseable {

  private final DecoderKvEngine engine;
  private final String id;
  private final int seq;
  private int textLen;
  private boolean closed;

  DecoderKvSession(DecoderKvEngine engine, String id) {
    this.engine = engine;
    this.id = id;
    this.seq = engine.acquireSequence();
  }

  public String id() {
    return id;
  }

  /** Number of text tokens currently cached. */
  public int cachedTokens() {
    return textLen;
  }

  /** Appends a chunk to the cached text (chunks are concatenated verbatim, keep your own spaces). */
  public DecoderKvSession append(String chunk) {
    checkOpen();
    if (chunk == null || chunk.isEmpty()) {
      return this;
    }
    var ids = engine.encodeText(chunk);
    if (ids.length == 0) {
      return this;
    }
    engine.append(seq, ids, textLen);
    textLen += ids.length;
    return this;
  }

  /** Multi-label scores (sigmoid, {@code >= threshold}) for the cached text. */
  public List<ClassificationResult> classify(
    List<String> labels,
    float threshold
  ) {
    return DecoderKvScores.multiLabel(logits(labels), labels, threshold);
  }

  /** Single-label scores (softmax over {@code labels}, best first) for the cached text. */
  public List<ClassificationResult> classifySingleLabel(List<String> labels) {
    return DecoderKvScores.singleLabel(logits(labels), labels);
  }

  private float[] logits(List<String> labels) {
    checkOpen();
    if (labels.isEmpty()) {
      return new float[0];
    }
    return engine.scoreCached(seq, textLen, engine.section(labels));
  }

  private void checkOpen() {
    if (closed) {
      throw new IllegalStateException("session " + id + " is closed");
    }
  }

  @Override
  public void close() {
    if (!closed) {
      closed = true;
      engine.clearSequence(seq);
      engine.releaseSequence(seq);
    }
  }
}
