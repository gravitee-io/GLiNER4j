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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import io.gravitee.lab.gliner4j.processor.TextEncoder;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Session state under context-window rejection and backbone failure, on a fake backend (no model). */
class StreamingSpanSessionTest {

  private static final List<String> LABELS = List.of("person", "city");

  /** One token per word, prompt = one token per label + SEP, every span scores "person". */
  static final class FakeBackend extends StreamingSpanBackend {

    final int nCtx;
    int decodes;
    int released = -1;
    RuntimeException decodeFailure;

    FakeBackend(int nCtx) {
      this.nCtx = nCtx;
    }

    @Override
    StreamingSpanEngine.Words words(String chunk) {
      var enc = new TextEncoder(chunk);
      int n = enc.getTextLen();
      var ids = new long[n];
      var first = new int[n];
      for (int i = 0; i < n; i++) {
        ids[i] = 100 + i;
        first[i] = i;
      }
      return new StreamingSpanEngine.Words(
        enc.getWords(),
        enc.getWordStartChars(),
        enc.getWordEndChars(),
        ids,
        first
      );
    }

    @Override
    StreamingSpanEngine.Prompt prompt(List<String> labels) {
      var ids = new long[labels.size() + 1];
      var positions = new int[labels.size()];
      for (int i = 0; i < labels.size(); i++) positions[i] = i;
      return new StreamingSpanEngine.Prompt(labels, ids, positions);
    }

    @Override
    List<float[]> decode(int seq, long[] ids, int startPos) {
      if (decodeFailure != null) throw decodeFailure;
      decodes++;
      var rows = new ArrayList<float[]>();
      for (int i = 0; i < ids.length; i++) rows.add(
        new float[] { startPos + i }
      );
      return rows;
    }

    @Override
    float[][] labelEmbeddings(
      List<float[]> promptRows,
      StreamingSpanEngine.Prompt prompt
    ) {
      return new float[prompt.labels().size()][1];
    }

    @Override
    float[][] spanLogits(
      List<float[]> window,
      int[] starts,
      int[] ends,
      int latest,
      float[][] labels
    ) {
      var out = new float[starts.length][];
      for (int i = 0; i < starts.length; i++) {
        // Single-word spans are confident persons; wider spans are not.
        out[i] = new float[] { starts[i] == ends[i] ? 5f : -5f, -5f };
      }
      return out;
    }

    @Override
    int maxWidth() {
      return 2;
    }

    @Override
    int rightContextWidth() {
      return 1;
    }

    @Override
    int nCtx() {
      return nCtx;
    }

    @Override
    int acquireSequence() {
      return 7;
    }

    @Override
    void releaseSequence(int seq) {
      released = seq;
    }
  }

  @Test
  void rejectedWarmAppendLeavesSessionUnchangedAndUsable() {
    // prompt = 3 tokens; "alice bob" = 2 → 5 cached; nCtx 6 leaves room for exactly one word.
    var backend = new FakeBackend(6);
    try (var session = new StreamingSpanSession(backend, "s", LABELS)) {
      var before = session.append("alice bob", 0.5f);
      assertThat(session.cachedTokens()).isEqualTo(5);

      assertThatThrownBy(() -> session.append(" carol dave", 0.5f))
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("context window")
        .hasMessageContaining("unchanged");

      assertThat(backend.decodes).isEqualTo(1);
      assertThat(session.text()).isEqualTo("alice bob");
      assertThat(session.words()).isEqualTo(2);
      assertThat(session.cachedTokens()).isEqualTo(5);
      assertThat(session.snapshot(0.5f)).isEqualTo(before);

      var after = session.append(" carol", 0.5f);
      assertThat(session.text()).isEqualTo("alice bob carol");
      assertThat(session.words()).isEqualTo(3);
      assertThat(session.cachedTokens()).isEqualTo(6);
      assertThat(after)
        .extracting(s -> s.text())
        .containsExactly("alice", "bob", "carol");
    }
  }

  @Test
  void rejectedColdAppendLeavesSessionEmptyAndUsable() {
    var backend = new FakeBackend(4);
    try (var session = new StreamingSpanSession(backend, "s", LABELS)) {
      assertThatThrownBy(() -> session.append("alice bob", 0.5f)).isInstanceOf(
        IllegalStateException.class
      );
      assertThat(backend.decodes).isZero();
      assertThat(session.text()).isEmpty();
      assertThat(session.words()).isZero();
      assertThat(session.cachedTokens()).isZero();
      assertThat(session.snapshot(0.5f)).isEmpty();

      assertThat(session.append("alice", 0.5f))
        .extracting(s -> s.text())
        .containsExactly("alice");
    }
  }

  @Test
  void backboneFailureClosesTheSession() {
    var backend = new FakeBackend(64);
    var session = new StreamingSpanSession(backend, "s", LABELS);
    session.append("alice", 0.5f);
    backend.decodeFailure = new IllegalStateException("llama_decode failed");

    assertThatThrownBy(() -> session.append(" bob", 0.5f)).hasMessage(
      "llama_decode failed"
    );
    assertThat(backend.released).isEqualTo(7);
    assertThatThrownBy(() -> session.snapshot(0.5f))
      .isInstanceOf(IllegalStateException.class)
      .hasMessageContaining("failed append")
      .hasCauseReference(backend.decodeFailure);
    session.close();
  }
}
