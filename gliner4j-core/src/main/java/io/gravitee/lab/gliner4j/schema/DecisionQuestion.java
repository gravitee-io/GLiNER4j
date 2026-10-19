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

import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * A typed question asked about a state, in the System One shape: a {@link Choice} between
 * options, an ordinal {@link Score} over levels, or a yes/no {@link Noul}. Each question is
 * answered by one {@link Decision}.
 *
 * <p>A question becomes one classification task: {@code id} is the task name, {@code
 * instructions} the task prompt, and the options are its labels.
 */
public sealed interface DecisionQuestion {
  /** The question id, used as the task name and as the key of its answer. */
  String id();

  /** What is asked; blank sends the task name alone. */
  String instructions();

  /** The answer options, in the order of the answer's probabilities. */
  List<ClassificationLabel> options();

  /**
   * Picks one option.
   *
   * @param options the options, at least two with distinct names; a description, when present,
   *     is part of the prompt
   */
  record Choice(
    String id,
    String instructions,
    List<ClassificationLabel> options
  ) implements DecisionQuestion {
    public Choice {
      requireId(id);
      Objects.requireNonNull(options, "options");
      requireDistinct(
        id,
        options.stream().map(ClassificationLabel::name).toList()
      );
    }
  }

  /**
   * Picks a level on an ordinal scale; the answer's {@link Decision#score()} is the expected level.
   *
   * @param levels the levels, lowest first, at least two and distinct
   */
  record Score(String id, String instructions, List<String> levels) implements
    DecisionQuestion {
    public Score {
      requireId(id);
      Objects.requireNonNull(levels, "levels");
      requireDistinct(id, levels);
    }

    @Override
    public List<ClassificationLabel> options() {
      return levels.stream().map(ClassificationLabel::new).toList();
    }
  }

  /**
   * Answers yes or no; the answer's {@link Decision#probabilityTrue()} is the probability of yes.
   *
   * @param whenTrue optional description of when the answer is yes
   * @param whenFalse optional description of when the answer is no
   */
  record Noul(
    String id,
    String instructions,
    String whenTrue,
    String whenFalse
  ) implements DecisionQuestion {
    public static final String YES = "yes";
    public static final String NO = "no";

    public Noul {
      requireId(id);
    }

    public Noul(String id, String instructions) {
      this(id, instructions, "", "");
    }

    @Override
    public List<ClassificationLabel> options() {
      return List.of(
        new ClassificationLabel(YES, whenTrue == null ? "" : whenTrue),
        new ClassificationLabel(NO, whenFalse == null ? "" : whenFalse)
      );
    }
  }

  private static void requireId(String id) {
    if (id == null || id.isBlank()) {
      throw new IllegalArgumentException("question id must not be blank");
    }
  }

  /** Options key the answer's probabilities, so a repeated name would overwrite another's mass. */
  private static void requireDistinct(String id, List<String> names) {
    if (names.size() < 2) {
      throw new IllegalArgumentException(
        "question '" + id + "' needs at least two options, got " + names.size()
      );
    }
    var seen = new HashSet<String>();
    for (var name : names) {
      if (name == null || name.isBlank()) {
        throw new IllegalArgumentException(
          "question '" + id + "' has a blank option"
        );
      }
      if (!seen.add(name)) {
        throw new IllegalArgumentException(
          "question '" + id + "' repeats option '" + name + "'"
        );
      }
    }
  }
}
