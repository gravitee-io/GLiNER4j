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
package io.gravitee.lab.gliner4j.arch.gliclass;

import io.gravitee.lab.gliner4j.arch.Architecture;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.arch.ModelArchitecture;
import io.gravitee.lab.gliner4j.arch.TaskType;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.strategy.ClassificationStrategy;
import io.gravitee.lab.gliner4j.strategy.gliclass.GliclassClassificationStrategy;
import java.util.List;

/**
 * The GLiClass family: zero-shot text classification via a uni-encoder + dot-product score head
 * (e.g. gliclass-modern-base-v3.0, ModernBERT backbone). Classification only.
 */
public final class GliclassArchitecture implements ModelArchitecture {

  @Override
  public Architecture id() {
    return Architecture.GLICLASS;
  }

  @Override
  public boolean supports(TaskType task) {
    return task == TaskType.CLASSIFICATION;
  }

  @Override
  public ClassificationStrategy newClassificationStrategy(
    LoadContext ctx,
    List<ClassificationLabel> labels
  ) {
    return GliclassClassificationStrategy.create(ctx, labels);
  }
}
