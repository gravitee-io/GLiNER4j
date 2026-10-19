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
package io.gravitee.lab.gliner4j;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.Decision;
import io.gravitee.lab.gliner4j.schema.DecisionQuestion.Choice;
import io.gravitee.lab.gliner4j.schema.DecisionQuestion.Noul;
import io.gravitee.lab.gliner4j.schema.DecisionQuestion.Score;
import java.util.LinkedHashMap;
import java.util.List;
import org.junit.jupiter.api.Test;

/** Model-free tests of the decision questions, their task prompt and the answer accessors. */
class GLiNER4jDeciderTest {

  @Test
  void score_rejectsRepeatedLevels() {
    assertThatThrownBy(() ->
      new Score("q", "Which level?", List.of("same", "same"))
    )
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("repeats option 'same'");
  }

  @Test
  void choice_rejectsRepeatedOptionNames() {
    var options = List.of(
      new ClassificationLabel("refund", "money back"),
      new ClassificationLabel("refund", "store credit")
    );
    assertThatThrownBy(() -> new Choice("intent", "", options))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("repeats option 'refund'");
  }

  @Test
  void questions_rejectFewerThanTwoOptions() {
    assertThatThrownBy(() -> new Score("q", "", List.of("only")))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("at least two options");
    assertThatThrownBy(() -> new Choice("q", "", List.of())).isInstanceOf(
      IllegalArgumentException.class
    );
  }

  @Test
  void questions_rejectBlankIdsAndOptions() {
    assertThatThrownBy(() -> new Noul(" ", "Escalate?")).isInstanceOf(
      IllegalArgumentException.class
    );
    assertThatThrownBy(() ->
      new Score(null, "", List.of("low", "high"))
    ).isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new Score("q", "", List.of("low", "")))
      .isInstanceOf(IllegalArgumentException.class)
      .hasMessageContaining("blank option");
  }

  @Test
  void options_keepDeclaredOrder() {
    assertThat(new Score("u", "", List.of("low", "normal", "high")).options())
      .extracting(ClassificationLabel::name)
      .containsExactly("low", "normal", "high");
    assertThat(
      new Noul("handoff", "", "needs a human", "").options()
    ).containsExactly(
      new ClassificationLabel("yes", "needs a human"),
      new ClassificationLabel("no", "")
    );
  }

  @Test
  void task_isIdAloneOrIdColonInstructions() {
    assertThat(GLiNER4jDecider.task(new Noul("handoff", " "))).isEqualTo(
      "handoff"
    );
    assertThat(
      GLiNER4jDecider.task(new Noul("handoff", "Should a human take over?"))
    ).isEqualTo("handoff: Should a human take over?");
  }

  @Test
  void decision_readsProbabilitiesInOptionOrder() {
    var probabilities = new LinkedHashMap<String, Float>();
    probabilities.put("low", 0.2f);
    probabilities.put("normal", 0.3f);
    probabilities.put("high", 0.5f);
    var decision = new Decision("urgency", probabilities);

    assertThat(decision.choice()).isEqualTo("high");
    assertThat(decision.confidence()).isEqualTo(0.5f);
    assertThat(decision.score()).isCloseTo(1.3f, within(1e-6f));
    assertThat(decision.probabilityTrue()).isZero();
  }
}
