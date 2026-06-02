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
package io.gravitee.lab.gliner4j.arch;

import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.strategy.ClassificationStrategy;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import java.util.List;

/**
 * A GLiNER model family. Builds the task strategies that own a family's
 * {@code assemble → forward → decode} pipeline, and declares which tasks it supports.
 *
 * <p>Resolved from a bundle's {@code architecture} config key via {@link ModelArchitectures}.
 * Implementations are stateless singletons; per-load state lives in the strategies they create.
 *
 * <p>As more families are added, task-strategy factory methods ({@code newClassificationStrategy},
 * {@code newRelationStrategy}, …) are added here alongside {@link #newNerStrategy}. A facade that
 * does not yet route through a strategy calls {@link #requireSupported} to fail fast on a bundle
 * whose family cannot serve the requested task.
 */
public interface ModelArchitecture {
  /** The family this implementation handles. */
  Architecture id();

  /** Whether this family can serve {@code task}. */
  boolean supports(TaskType task);

  /**
   * Builds the NER strategy for this family. Default-throws so families that do not serve NER
   * (e.g. a classification-only family) need not implement it.
   *
   * @param ctx the family-agnostic load inputs
   * @param entities the load-time entity schema
   * @return a ready NER strategy
   * @throws UnsupportedOperationException if this family does not support NER
   */
  default NerStrategy newNerStrategy(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    throw unsupported(TaskType.NER);
  }

  /**
   * Builds the text-classification strategy for this family. Default-throws so families that do
   * not serve classification need not implement it.
   *
   * @param ctx the family-agnostic load inputs
   * @param labels the load-time classification labels
   * @return a ready classification strategy
   * @throws UnsupportedOperationException if this family does not support classification
   */
  default ClassificationStrategy newClassificationStrategy(
    LoadContext ctx,
    List<ClassificationLabel> labels
  ) {
    throw unsupported(TaskType.CLASSIFICATION);
  }

  private UnsupportedOperationException unsupported(TaskType task) {
    return new UnsupportedOperationException(
      "Model family '" +
        id().configValue() +
        "' does not support task " +
        task +
        " in this version of GLiNER4j."
    );
  }

  /**
   * Throws if this family cannot serve {@code task}. Used by facades whose task is not yet routed
   * through a strategy, so loading a mismatched bundle fails clearly at load time.
   *
   * @param task the task the caller is about to perform
   * @throws UnsupportedOperationException if {@link #supports(TaskType)} is false
   */
  default void requireSupported(TaskType task) {
    if (!supports(task)) {
      throw new UnsupportedOperationException(
        "Model family '" +
          id().configValue() +
          "' does not support task " +
          task +
          " in this version of GLiNER4j."
      );
    }
  }
}
