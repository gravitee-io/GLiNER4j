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
package io.gravitee.lab.gliner4j.arch.glineruni;

import io.gravitee.lab.gliner4j.arch.Architecture;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.arch.ModelArchitecture;
import io.gravitee.lab.gliner4j.arch.TaskType;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.strategy.NerStrategy;
import io.gravitee.lab.gliner4j.strategy.glineruni.GlinerUniNerStrategy;
import io.gravitee.lab.gliner4j.strategy.glineruni.GlinerUniTokenNerStrategy;
import java.util.List;

/**
 * The original-GLiNER uni-encoder family (e.g. gliner-pii-base, gliner-multitask): a single
 * monolithic span model.onnx, span_mode=markerV0 (token_level later). NER only.
 */
public final class GlinerUniArchitecture implements ModelArchitecture {

  @Override
  public Architecture id() {
    return Architecture.GLINER_UNI;
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
    // span_mode selects the decode: markerV0 enumerates spans (SpanDecoder); token_level emits
    // BIO start/end/inside per word·class (TokenSpanDecoder, no span_idx inputs).
    var spanMode = ctx.config().archString("span_mode", "markerV0");
    if ("token_level".equals(spanMode)) {
      return GlinerUniTokenNerStrategy.create(ctx, entities);
    }
    return GlinerUniNerStrategy.create(ctx, entities);
  }
}
