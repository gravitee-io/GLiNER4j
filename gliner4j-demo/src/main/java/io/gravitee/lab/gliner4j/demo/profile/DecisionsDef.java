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
package io.gravitee.lab.gliner4j.demo.profile;

import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.DecisionQuestion;
import java.util.List;

/**
 * Decisions section of a demo profile: System One questions asked about every sample.
 *
 * @param questions the typed questions
 * @param samples the states to decide on
 */
public record DecisionsDef(List<Question> questions, List<String> samples) {
  public List<DecisionQuestion> toQuestions() {
    return questions.stream().map(Question::toQuestion).toList();
  }

  /**
   * One question, in the System One JSON shape.
   *
   * @param type {@code choice} (options), {@code score} (levels, lowest first) or {@code noul} (yes / no)
   */
  public record Question(
    String id,
    String type,
    String instructions,
    List<NamedDescription> options,
    List<String> levels,
    String whenTrue,
    String whenFalse
  ) {
    DecisionQuestion toQuestion() {
      var text = instructions == null ? "" : instructions;
      return switch (type) {
        case "choice" -> new DecisionQuestion.Choice(
          id,
          text,
          options
            .stream()
            .map(o -> new ClassificationLabel(o.name(), o.description()))
            .toList()
        );
        case "score" -> new DecisionQuestion.Score(id, text, levels);
        case "noul" -> new DecisionQuestion.Noul(id, text, whenTrue, whenFalse);
        default -> throw new IllegalArgumentException("Unknown question type '" + type + "' for " + id);
      };
    }
  }
}
