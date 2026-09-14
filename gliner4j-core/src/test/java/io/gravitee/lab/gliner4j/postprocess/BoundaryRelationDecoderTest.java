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

import io.gravitee.lab.gliner4j.postprocess.BoundaryRelationDecoder.Edge;
import io.gravitee.lab.gliner4j.postprocess.BoundaryRelationDecoder.Mention;
import java.util.List;
import org.junit.jupiter.api.Test;

class BoundaryRelationDecoderTest {

  private static Edge edge(
    float s,
    String h,
    int h0,
    int h1,
    String t,
    int t0,
    int t1
  ) {
    return new Edge(s, new Mention(h, h0, h1), new Mention(t, t0, t1));
  }

  @Test
  void partialMentionsCollapseIntoTheContainingMention() {
    // "John" ⊂ "John Smith": the partial head is canonicalized to the longer mention and the
    // two edges collapse to one, keeping the higher score.
    var edges = List.of(
      edge(0.9f, "John Smith", 0, 10, "Apple", 21, 26),
      edge(0.7f, "John", 0, 4, "Apple", 21, 26)
    );
    var out = BoundaryRelationDecoder.deduplicate(edges);
    assertThat(out).hasSize(1);
    assertThat(out.get(0).head().text()).isEqualTo("John Smith");
    assertThat(out.get(0).score()).isEqualTo(0.9f);
  }

  @Test
  void repeatedMentionsKeepTheClosestPair() {
    // Same (head text, tail text) twice: the closer occurrence pair wins even with a lower score.
    var edges = List.of(
      edge(0.95f, "Mary", 0, 4, "Google", 60, 66),
      edge(0.80f, "Mary", 40, 44, "Google", 60, 66)
    );
    var out = BoundaryRelationDecoder.deduplicate(edges);
    assertThat(out).hasSize(1);
    assertThat(out.get(0).head().start()).isEqualTo(40);
  }

  @Test
  void strictTokenSubsetWithSameTailIsDropped() {
    var edges = List.of(
      edge(0.9f, "San Francisco", 30, 43, "California", 50, 60),
      edge(0.8f, "Francisco", 34, 43, "California", 50, 60)
    );
    // Different coordinates but "Francisco" ⊂ "San Francisco": canonicalization already merges
    // them (contained span); the semantic-subset rule guards the non-contained case too.
    var out = BoundaryRelationDecoder.deduplicate(edges);
    assertThat(out)
      .extracting(e -> e.head().text())
      .containsExactly("San Francisco");
  }

  @Test
  void outputSortedByHeadThenTailStart() {
    var edges = List.of(
      edge(0.9f, "Bob", 50, 53, "Apple", 60, 65),
      edge(0.9f, "John", 0, 4, "Microsoft", 15, 24),
      edge(0.9f, "Mary", 30, 34, "Google", 45, 51)
    );
    var out = BoundaryRelationDecoder.deduplicate(edges);
    assertThat(out)
      .extracting(e -> e.head().text())
      .containsExactly("John", "Mary", "Bob");
  }
}
