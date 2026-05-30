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
package io.gravitee.lab.gliner4j.arch.glinerbi;

import io.gravitee.lab.gliner4j.arch.Architecture;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.arch.ModelArchitecture;
import io.gravitee.lab.gliner4j.arch.TaskType;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import io.gravitee.lab.gliner4j.strategy.glinerbi.GlinerBiNerStrategy;
import java.util.List;

/**
 * The original-GLiNER bi-encoder family (e.g. gliner-bi-small, modern-gliner-bi): separate text and
 * label encoders fused into a single span model.onnx, span_mode=markerV0. NER only.
 */
public final class GlinerBiArchitecture implements ModelArchitecture {

  @Override
  public Architecture id() {
    return Architecture.GLINER_BI;
  }

  @Override
  public boolean supports(TaskType task) {
    return task == TaskType.NER;
  }

  @Override
  public NerStrategy newNerStrategy(
    LoadContext ctx,
    List<EntityDefinition> entities
  ) {
    return GlinerBiNerStrategy.create(ctx, entities);
  }
}
