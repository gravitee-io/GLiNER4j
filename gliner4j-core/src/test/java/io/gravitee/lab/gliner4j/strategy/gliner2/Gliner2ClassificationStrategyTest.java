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
package io.gravitee.lab.gliner4j.strategy.gliner2;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.lab.gliner4j.schema.DecisionQuestion;
import org.junit.jupiter.api.Test;

/** The task-conditioned schema tokens against the upstream {@code _transform_schema} layout. */
class Gliner2ClassificationStrategyTest {

  @Test
  void taskSchema_putsTheTaskPromptInThePSlot() {
    var question = new DecisionQuestion.Noul(
      "handoff",
      "Should a human take over?"
    );
    var encoder = Gliner2ClassificationStrategy.labelSchemaEncoder(
      "handoff: Should a human take over?",
      question.options()
    );
    assertThat(encoder.getSchemaTokens()).containsExactly(
      "(",
      "[P]",
      "handoff: Should a human take over?",
      "(",
      "[L]",
      "yes",
      "[L]",
      "no",
      ")",
      ")"
    );
  }
}
