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
package io.gravitee.lab.gliner4j.postprocess;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.lab.gliner4j.postprocess.BoundaryDecoder.Candidate;
import java.util.List;
import org.junit.jupiter.api.Test;

class BoundaryDecoderTest {

  @Test
  void flatPolicyMaximizesTotalScoreNotGreedyBest() {
    // One wide span (0.6) overlapping two narrow ones (0.5 + 0.5 = 1.0 > 0.6): the
    // weighted-interval-scheduling answer keeps the two narrow spans, greedy would keep the wide.
    var resolved = BoundaryDecoder.resolveFlat(
      List.of(
        new Candidate(0.6f, 0, 4),
        new Candidate(0.5f, 0, 2),
        new Candidate(0.5f, 2, 4)
      )
    );
    assertThat(resolved).containsExactly(
      new Candidate(0.5f, 0, 2),
      new Candidate(0.5f, 2, 4)
    );
  }

  @Test
  void touchingHalfOpenSpansDoNotOverlap() {
    var resolved = BoundaryDecoder.resolveFlat(
      List.of(new Candidate(0.9f, 0, 3), new Candidate(0.8f, 3, 5))
    );
    assertThat(resolved).hasSize(2);
  }

  @Test
  void exactDuplicatesCollapseToTheHighestScore() {
    var resolved = BoundaryDecoder.resolveFlat(
      List.of(new Candidate(0.7f, 1, 2), new Candidate(0.9f, 1, 2))
    );
    assertThat(resolved).containsExactly(new Candidate(0.9f, 1, 2));
  }

  @Test
  void outputIsRankedByScoreThenStart() {
    var resolved = BoundaryDecoder.resolveFlat(
      List.of(
        new Candidate(0.6f, 5, 6),
        new Candidate(0.95f, 0, 1),
        new Candidate(0.6f, 2, 3)
      )
    );
    assertThat(resolved).containsExactly(
      new Candidate(0.95f, 0, 1),
      new Candidate(0.6f, 2, 3),
      new Candidate(0.6f, 5, 6)
    );
  }
}
