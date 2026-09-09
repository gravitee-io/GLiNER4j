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
package io.gravitee.lab.gliner4j.arch.gliner2;

import io.gravitee.lab.gliner4j.arch.Architecture;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.arch.ModelArchitecture;
import io.gravitee.lab.gliner4j.arch.TaskType;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.strategy.ClassificationStrategy;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import io.gravitee.lab.gliner4j.strategy.RelationStrategy;
import io.gravitee.lab.gliner4j.strategy.gliner2.Gliner2ClassificationStrategy;
import io.gravitee.lab.gliner4j.strategy.gliner2.Gliner2NerStrategy;
import io.gravitee.lab.gliner4j.strategy.gliner2.Gliner2RelationStrategy;
import java.util.List;

/**
 * The fastino GLiNER2 family. Supports every task GLiNER4j implements (NER, classification,
 * relation, structure) via the encoder → span_rep → count-aware scoring_head / classifier_head
 * graph set.
 */
public final class Gliner2Architecture implements ModelArchitecture {

  @Override
  public Architecture id() {
    return Architecture.GLINER2;
  }

  @Override
  public boolean supports(TaskType task) {
    // GLiNER2 is the original family and serves all currently implemented tasks.
    return true;
  }

  @Override
  public NerStrategy newNerStrategy(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    return Gliner2NerStrategy.create(ctx, entities);
  }

  @Override
  public ClassificationStrategy newClassificationStrategy(
    LoadContext ctx,
    List<ClassificationLabel> labels
  ) {
    return Gliner2ClassificationStrategy.create(ctx, labels);
  }

  @Override
  public RelationStrategy newRelationStrategy(
    LoadContext ctx,
    List<RelationDefinition> relations
  ) {
    return Gliner2RelationStrategy.create(ctx, relations);
  }

  @Override
  public io.gravitee.lab.gliner4j.runtime.Gliner2SpanRuntime newSpanRuntime(
    LoadContext ctx
  ) {
    return new io.gravitee.lab.gliner4j.runtime.GLiNER4jNERRuntime(
      ctx.modelDir(),
      ctx.variant(),
      ctx.runtimeConfig()
    );
  }
}
