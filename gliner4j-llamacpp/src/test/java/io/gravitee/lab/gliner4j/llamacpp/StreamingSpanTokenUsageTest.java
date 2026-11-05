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

import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.NerOutput;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

/**
 * Pins {@code inputTokens} of {@link StreamingSpanNerStrategy} on a fake backend (no model): the
 * count must equal both the hand-computed prompt + text length and the number of tokens the
 * backend was asked to decode for that text.
 */
class StreamingSpanTokenUsageTest {

  private static final List<EntityDefinition> ENTITIES = List.of(
    new EntityDefinition("person"),
    new EntityDefinition("city")
  );

  /** Two tokens per label plus its {@code <<LABEL>>}, one {@code <<SEP>>}, one token per word. */
  static final class RecordingBackend extends StreamingSpanBackend {

    final List<Integer> decoded = new ArrayList<>();
    int released;

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
      var ids = new long[labels.size() * 3 + 1];
      var positions = new int[labels.size()];
      for (int i = 0; i < labels.size(); i++) positions[i] = i * 3 + 2;
      return new StreamingSpanEngine.Prompt(labels, ids, positions);
    }

    @Override
    List<float[]> decode(int seq, long[] ids, int startPos) {
      decoded.add(ids.length);
      var rows = new ArrayList<float[]>();
      for (int i = 0; i < ids.length; i++) rows.add(new float[] { 0f });
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
      var out = new float[starts.length][labels.length];
      for (var row : out) Arrays.fill(row, -5f);
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
      return 4096;
    }

    @Override
    int acquireSequence() {
      return 0;
    }

    @Override
    void releaseSequence(int seq) {
      released++;
    }
  }

  @Test
  void countsPromptAndWordTokensDecodedForTheText() {
    var backend = new RecordingBackend();
    var strategy = new StreamingSpanNerStrategy(backend, ENTITIES, () -> {});
    // prompt 2 * 3 + 1 = 7, words John / lives / in / Paris / . = 5
    var out = strategy.extractWithUsage("John lives in Paris.", 0.5f);
    assertThat(out.inputTokens()).isEqualTo(12);
    assertThat(backend.decoded).containsExactly(12);
    assertThat(out.spans()).isEqualTo(
      strategy.extract("John lives in Paris.", 0.5f)
    );
  }

  @Test
  void perCallEntitiesChangeThePrompt() {
    var backend = new RecordingBackend();
    var strategy = new StreamingSpanNerStrategy(backend, ENTITIES, () -> {});
    var out = strategy.extractWithUsage(
      "John lives in Paris.",
      List.of(new EntityDefinition("person")),
      0.5f
    );
    assertThat(out.inputTokens()).isEqualTo(4 + 5);
    assertThat(
      strategy
        .extractWithUsage("John lives in Paris.", List.of(), 0.5f)
        .inputTokens()
    ).isZero();
  }

  @Test
  void batchMatchesSingleAndRedecodesThePromptPerText() {
    var backend = new RecordingBackend();
    var strategy = new StreamingSpanNerStrategy(backend, ENTITIES, () -> {});
    var texts = Arrays.asList("John lives in Paris.", "", null, "Paris", " . ");
    var batch = strategy.extractBatchWithUsage(texts, 0.5f);
    assertThat(batch)
      .extracting(NerOutput::inputTokens)
      .containsExactly(12, 0, 0, 8, 8);
    // Every non-empty text decoded the whole prompt again on a fresh sequence.
    assertThat(backend.decoded).containsExactly(12, 8, 8);
    for (int i = 0; i < texts.size(); i++) {
      assertThat(batch.get(i)).isEqualTo(
        strategy.extractWithUsage(texts.get(i), 0.5f)
      );
    }
    assertThat(batch.stream().map(NerOutput::spans).toList()).isEqualTo(
      strategy.extractBatch(texts, 0.5f)
    );
  }
}
