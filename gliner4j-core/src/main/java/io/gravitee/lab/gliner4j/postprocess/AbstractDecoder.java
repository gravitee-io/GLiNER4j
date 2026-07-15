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

import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Comparator;
import java.util.List;

abstract class AbstractDecoder {

  List<EntitySpan> getEntitySpans(ArrayList<EntitySpan> candidates) {
    candidates.sort(
      Comparator.comparingDouble(EntitySpan::confidence).reversed()
    );
    var result = new ArrayList<EntitySpan>();
    // Greedy non-overlapping selection in confidence order. Accepted spans are marked in a
    // bitset over character offsets, so the overlap test ([start,end) intersection) is a
    // word-masked scan instead of an O(accepted) pass per candidate.
    var occupied = new BitSet();
    for (int i = 0; i < candidates.size(); i++) {
      var candidate = candidates.get(i);
      int firstSet = occupied.nextSetBit(candidate.start());
      if (firstSet == -1 || firstSet >= candidate.end()) {
        result.add(candidate);
        occupied.set(candidate.start(), candidate.end());
      }
    }
    result.sort(Comparator.comparingInt(EntitySpan::start));
    return result;
  }
}
