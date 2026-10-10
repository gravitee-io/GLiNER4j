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
package io.gravitee.lab.gliner4j.schema;

import java.util.Map;

/**
 * The answer to one {@link DecisionQuestion}: a probability per option (softmax over the
 * options, in option order). The accessors read it the way the question's type is meant to be
 * read.
 *
 * @param id the question id
 * @param probabilities option key → probability, in option order, summing to 1
 */
public record Decision(String id, Map<String, Float> probabilities) {
  /** The most probable option (the answer to a {@link DecisionQuestion.Choice}). */
  public String choice() {
    String best = null;
    float bestP = Float.NEGATIVE_INFINITY;
    for (var e : probabilities.entrySet()) {
      if (e.getValue() > bestP) {
        best = e.getKey();
        bestP = e.getValue();
      }
    }
    return best;
  }

  /** The probability of {@link #choice()}. */
  public float confidence() {
    return probabilities.values().stream().max(Float::compare).orElse(0f);
  }

  /**
   * The expected level index, {@code Σ i·p(i)} over the options in order (the answer to a
   * {@link DecisionQuestion.Score}: {@code 0} is the lowest level).
   */
  public float score() {
    float score = 0f;
    int i = 0;
    for (float p : probabilities.values()) {
      score += i++ * p;
    }
    return score;
  }

  /** The probability of yes (the answer to a {@link DecisionQuestion.Noul}). */
  public float probabilityTrue() {
    return probabilities.getOrDefault(DecisionQuestion.Noul.YES, 0f);
  }
}
