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

import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.TreeMap;

/**
 * One streaming NER session: text arrives in chunks, the backbone KV cache and word states grow
 * with it, and after every append the spans ending in the new words (plus those ending within
 * {@code right_context_width} of the latest word) are rescored and merged into the session's span
 * history. {@link #snapshot} decodes the whole history, so every result is the complete current
 * entity set over the accumulated text — revisions included. Mirrors
 * {@code StreamingSpanGLiNER.inference(session_id=...)}.
 *
 * <p>Chunks are concatenated verbatim and each chunk is word-split on its own, exactly like the
 * Python session. Not thread-safe.
 *
 * <p><b>Failures.</b> An append that would overflow the context window is rejected before anything
 * changes, so the session stays usable. Any failure once the chunk reached the backbone closes the
 * session: the KV cache may already hold the chunk, and the Java view could no longer match it.
 */
public final class StreamingSpanSession implements AutoCloseable {

  private final StreamingSpanBackend engine;
  private final String id;
  private final List<String> labels;
  private final int seq;
  private final StringBuilder text = new StringBuilder();
  private final List<Integer> charStarts = new ArrayList<>();
  private final List<Integer> charEnds = new ArrayList<>();
  private final List<float[]> wordStates = new ArrayList<>();
  private final TreeMap<Long, float[]> spanLogits = new TreeMap<>();
  private float[][] labelReps;
  private int tokenLen;
  private boolean closed;
  private RuntimeException failure;

  StreamingSpanSession(
    StreamingSpanBackend engine,
    String id,
    List<String> labels
  ) {
    if (labels.isEmpty()) {
      throw new IllegalArgumentException("at least one label is required");
    }
    this.engine = engine;
    this.id = id;
    this.labels = List.copyOf(labels);
    this.seq = engine.acquireSequence();
  }

  public String id() {
    return id;
  }

  public List<String> labels() {
    return labels;
  }

  /** The accumulated session text. */
  public String text() {
    return text.toString();
  }

  public int cachedTokens() {
    return tokenLen;
  }

  public int words() {
    return wordStates.size();
  }

  /**
   * Appends a chunk, rescoring the affected spans, and returns the full snapshot at {@code threshold}.
   *
   * @throws IllegalStateException if the chunk would overflow the context window (the session is
   *     left unchanged), or if the session is closed
   */
  public List<EntitySpan> append(String chunk, float threshold) {
    checkOpen();
    if (chunk == null || chunk.isEmpty()) {
      return snapshot(threshold);
    }
    var words = engine.words(chunk);
    if (words.words().isEmpty()) {
      text.append(chunk);
      return snapshot(threshold);
    }

    // Cold pass: prompt + first chunk in one decode; label vectors from the prompt rows.
    var prompt = labelReps == null ? engine.prompt(labels) : null;
    int startPos = prompt == null ? tokenLen : 0;
    long[] ids = prompt == null
      ? words.ids()
      : concat(prompt.ids(), words.ids());
    if (startPos + ids.length > engine.nCtx()) {
      throw new IllegalStateException(
        "append would exceed the context window: " +
          (startPos + ids.length) +
          " > nCtx=" +
          engine.nCtx() +
          " — session " +
          id +
          " is unchanged; start a new session to continue"
      );
    }

    try {
      var rows = engine.decode(seq, ids, startPos);
      int wordRowOffset = 0;
      if (prompt != null) {
        labelReps = engine.labelEmbeddings(
          rows.subList(0, prompt.ids().length),
          prompt
        );
        wordRowOffset = prompt.ids().length;
      }
      // The KV cache now holds the chunk: bring the Java side level with it.
      int charOffset = text.length();
      text.append(chunk);
      for (int i = 0; i < words.words().size(); i++) {
        charStarts.add(charOffset + words.starts()[i]);
        charEnds.add(charOffset + words.ends()[i]);
      }
      tokenLen = startPos + ids.length;
      int past = wordStates.size();
      for (int first : words.firstSubtoken()) {
        wordStates.add(rows.get(wordRowOffset + first));
      }
      rescore(past);
    } catch (RuntimeException e) {
      fail(e);
      throw e;
    }
    return snapshot(threshold);
  }

  /** Rescores the candidate spans for the words added after the first {@code past}. */
  private void rescore(int past) {
    int total = wordStates.size();
    var cands = candidates(
      past,
      total - past,
      engine.maxWidth(),
      engine.rightContextWidth()
    );
    if (cands.length == 0) {
      return;
    }
    int firstStart = Integer.MAX_VALUE;
    for (long key : cands)
      firstStart = Math.min(firstStart, (int) (key >>> 32));
    var window = wordStates.subList(firstStart, total);
    var starts = new int[cands.length];
    var ends = new int[cands.length];
    for (int i = 0; i < cands.length; i++) {
      starts[i] = (int) (cands[i] >>> 32) - firstStart;
      ends[i] = (int) (cands[i] & 0xffffffffL) - firstStart;
    }
    var logits = engine.spanLogits(
      window,
      starts,
      ends,
      total - 1 - firstStart,
      labelReps
    );
    for (int i = 0; i < cands.length; i++) {
      spanLogits.put(cands[i], logits[i]);
    }
  }

  private static long[] concat(long[] a, long[] b) {
    var out = new long[a.length + b.length];
    System.arraycopy(a, 0, out, 0, a.length);
    System.arraycopy(b, 0, out, a.length, b.length);
    return out;
  }

  /**
   * gliner's {@code prepare_streaming_span_idx}: every span ending in the new words, plus spans
   * ending within {@code rightContextWidth} of the latest word, as {@code start<<32 | end}
   * (inclusive word indices), widths up to {@code maxWidth}.
   */
  static long[] candidates(
    int past,
    int added,
    int maxWidth,
    int rightContextWidth
  ) {
    int total = past + added;
    if (total == 0 || added == 0) {
      return new long[0];
    }
    int latest = total - 1;
    int minimumEnd = Math.min(past, Math.max(0, latest - rightContextWidth));
    int firstStart = Math.max(0, minimumEnd - (maxWidth - 1));
    var out = new ArrayList<Long>();
    for (int s = firstStart; s < total; s++) {
      for (int w = 0; w < maxWidth; w++) {
        int e = s + w;
        if (e < total && e >= minimumEnd) {
          out.add(((long) s << 32) | e);
        }
      }
    }
    return out.stream().mapToLong(Long::longValue).toArray();
  }

  /** Decodes the span history: sigmoid, threshold, greedy flat non-overlap, sorted by start. */
  public List<EntitySpan> snapshot(float threshold) {
    checkOpen();
    var candidates = new ArrayList<EntitySpan>();
    for (var entry : spanLogits.entrySet()) {
      int start = (int) (entry.getKey() >>> 32);
      int end = (int) (entry.getKey() & 0xffffffffL);
      var logits = entry.getValue();
      for (int c = 0; c < labels.size(); c++) {
        float p = DecoderKvScores.sigmoid(logits[c]);
        if (p > threshold) {
          candidates.add(new EntitySpan(labels.get(c), null, p, start, end));
        }
      }
    }
    // Greedy: highest score first (stable), keep spans that do not overlap an accepted one.
    candidates.sort(
      Comparator.comparingDouble((EntitySpan s) -> -s.confidence())
    );
    var accepted = new ArrayList<EntitySpan>();
    for (var cand : candidates) {
      boolean overlaps = false;
      for (var kept : accepted) {
        if (!(cand.start() > kept.end() || kept.start() > cand.end())) {
          overlaps = true;
          break;
        }
      }
      if (!overlaps) accepted.add(cand);
    }
    var out = new ArrayList<EntitySpan>(accepted.size());
    var full = text.toString();
    for (var span : accepted) {
      int cs = charStarts.get(span.start());
      int ce = charEnds.get(span.end());
      out.add(
        new EntitySpan(
          span.type(),
          full.substring(cs, ce),
          span.confidence(),
          cs,
          ce
        )
      );
    }
    out.sort(Comparator.comparingInt(EntitySpan::start));
    return out;
  }

  /** Closes the session after a failure past the capacity check; later calls report the cause. */
  private void fail(RuntimeException cause) {
    failure = cause;
    try {
      close();
    } catch (RuntimeException e) {
      cause.addSuppressed(e);
    }
  }

  private void checkOpen() {
    if (failure != null) {
      throw new IllegalStateException(
        "session " + id + " was closed after a failed append",
        failure
      );
    }
    if (closed) {
      throw new IllegalStateException("session " + id + " is closed");
    }
  }

  @Override
  public void close() {
    if (!closed) {
      closed = true;
      engine.releaseSequence(seq);
    }
  }
}
