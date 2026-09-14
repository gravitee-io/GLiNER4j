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
import static org.assertj.core.api.Assertions.within;

import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import java.util.List;
import org.junit.jupiter.api.Test;

class DecoderKvScoresTest {

  @Test
  void multiLabelThresholdsAndSorts() {
    var out = DecoderKvScores.multiLabel(
      new float[] { -2f, 0f, 2f },
      List.of("a", "b", "c"),
      0.5f
    );
    assertThat(out)
      .extracting(ClassificationResult::label)
      .containsExactly("c", "b");
    assertThat(out.get(1).confidence()).isCloseTo(0.5f, within(1e-6f));
  }

  @Test
  void singleLabelSoftmaxSumsToOneAndSorts() {
    var out = DecoderKvScores.singleLabel(
      new float[] { 1f, 3f, 2f },
      List.of("a", "b", "c")
    );
    assertThat(out)
      .extracting(ClassificationResult::label)
      .containsExactly("b", "c", "a");
    double sum = out
      .stream()
      .mapToDouble(ClassificationResult::confidence)
      .sum();
    assertThat(sum).isCloseTo(1.0, within(1e-5));
  }
}
