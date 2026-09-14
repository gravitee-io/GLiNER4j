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
import io.gravitee.lab.gliner4j.runtime.Gliner2SpanRuntime;
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
 * GLiNER2 on the ggml engine: the core GLiNER2 strategies over a {@link GgmlGliner2Runtime}.
 * Serves every GLiNER2 task (NER, classification, relations, structures, the unified facade).
 */
public final class LlamaGliner2Architecture implements ModelArchitecture {

  @Override
  public Architecture id() {
    return Architecture.GLINER2;
  }

  @Override
  public Engine engine() {
    return Engine.LLAMACPP;
  }

  @Override
  public boolean supports(TaskType task) {
    return true;
  }

  @Override
  public NerStrategy newNerStrategy(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    return Gliner2NerStrategy.create(
      ctx,
      entities,
      GgmlGliner2Runtime.load(ctx)
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
      GgmlGliner2Runtime.load(ctx)
    );
  }

  @Override
  public RelationStrategy newRelationStrategy(
    LoadContext ctx,
    List<RelationDefinition> relations
  ) {
    return Gliner2RelationStrategy.create(
      ctx,
      relations,
      GgmlGliner2Runtime.load(ctx)
    );
  }

  @Override
  public Gliner2SpanRuntime newSpanRuntime(LoadContext ctx) {
    return GgmlGliner2Runtime.load(ctx);
  }
}
