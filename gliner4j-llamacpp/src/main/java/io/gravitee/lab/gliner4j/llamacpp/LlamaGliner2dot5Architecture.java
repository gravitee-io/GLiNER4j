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
package io.gravitee.lab.gliner4j.llamacpp;

import io.gravitee.lab.gliner4j.arch.Architecture;
import io.gravitee.lab.gliner4j.arch.Engine;
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
import io.gravitee.lab.gliner4j.strategy.gliner2dot5.Gliner2dot5NerStrategy;
import io.gravitee.lab.gliner4j.strategy.gliner2dot5.Gliner2dot5RelationStrategy;
import java.util.List;

/** GLiNER2.5 on the ggml engine: the core boundary strategies over the ggml runtimes. */
public final class LlamaGliner2dot5Architecture implements ModelArchitecture {

  @Override
  public Architecture id() {
    return Architecture.GLINER2DOT5;
  }

  @Override
  public Engine engine() {
    return Engine.LLAMACPP;
  }

  @Override
  public boolean supports(TaskType task) {
    return (
      task == TaskType.NER ||
      task == TaskType.CLASSIFICATION ||
      task == TaskType.RELATION
    );
  }

  @Override
  public NerStrategy newNerStrategy(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    return Gliner2dot5NerStrategy.create(
      ctx,
      entities,
      GgmlGliner2dot5NerRuntime.load(ctx)
    );
  }

  @Override
  public ClassificationStrategy newClassificationStrategy(
    LoadContext ctx,
    List<ClassificationLabel> labels
  ) {
    return Gliner2ClassificationStrategy.create(
      ctx,
      labels,
      GgmlGliner2dot5NerRuntime.load(ctx)
    );
  }

  @Override
  public RelationStrategy newRelationStrategy(
    LoadContext ctx,
    List<RelationDefinition> relations
  ) {
    return Gliner2dot5RelationStrategy.create(
      ctx,
      relations,
      GgmlGliner2dot5RelationRuntime.load(ctx)
    );
  }
}
