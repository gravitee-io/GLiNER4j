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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;

/**
 * The GLiClass uni-encoder scoring head as a ggml graph: text (CLS) and class-token rows through
 * their {@code FeaturesProjector}s (Linear → GELU → Linear), then the {@code simple} dot-product
 * scorer or the {@code mlp} pair scorer (concat → 256 → ReLU → 128 → ReLU → 1). Weights come from
 * {@code gguf/heads.gguf} written by {@code export_llamacpp.py gliclass}.
 */
final class GgmlGliclassHead implements AutoCloseable {

  private final GgmlWeights w;
  private final int hidden;
  private final boolean mlp;
  private final boolean normalizeFeatures;
  private final float logitScale;

  GgmlGliclassHead(
    Path gguf,
    boolean useGpu,
    int cpuThreads,
    int hidden,
    String scorerType,
    boolean normalizeFeatures
  ) {
    this.w = new GgmlWeights(gguf, useGpu, cpuThreads, "ggml GLiClass head");
    this.hidden = hidden;
    this.mlp = switch (scorerType) {
      case "mlp" -> true;
      case "simple" -> false;
      default -> throw new UnsupportedOperationException(
        "GLiClass scorer_type=" +
          scorerType +
          " is not supported on the llama.cpp engine"
      );
    };
    this.normalizeFeatures = normalizeFeatures;
    float scale = 1f;
    if (normalizeFeatures) {
      try (var call = Arena.ofConfined()) {
        scale = Ggml.getFloats(w.get("logit_scale"), call, 1)[0];
      }
    }
    this.logitScale = scale;
  }

  /** Raw logits, one per class row. */
  synchronized float[] score(float[] textRow, float[][] classRows) {
    int n = classRows.length;
    if (n == 0) {
      return new float[0];
    }
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call);
      try {
        var text = Ggml.newTensor2d(ctx, w.typeF32, hidden, 1);
        var classes = Ggml.newTensor2d(ctx, w.typeF32, hidden, n);
        Ggml.setInput(text);
        Ggml.setInput(classes);

        var t = projector(ctx, text, "text_projector");
        var c = projector(ctx, classes, "classes_projector");
        if (normalizeFeatures) {
          t = Ggml.l2Norm(ctx, t, 1e-8f);
          c = Ggml.l2Norm(ctx, c, 1e-8f);
        }
        MemorySegment logits;
        if (mlp) {
          var pair = Ggml.concat(ctx, Ggml.repeat(ctx, t, c), c, 0); // (2·hidden, n)
          var m = Ggml.relu(ctx, w.linear(ctx, pair, "scorer.mlp.0"));
          m = Ggml.relu(ctx, w.linear(ctx, m, "scorer.mlp.2"));
          logits = w.linear(ctx, m, "scorer.mlp.4"); // (1, n)
        } else {
          logits = Ggml.mulMat(ctx, c, t); // (n, 1)
        }
        if (normalizeFeatures) {
          logits = Ggml.scale(ctx, logits, logitScale);
        }
        var graph = Ggml.newGraph(ctx, GgmlWeights.GRAPH_SIZE);
        w.compute(ctx, graph, logits);

        Ggml.setFloats(text, call, textRow);
        var flat = new float[hidden * n];
        for (int i = 0; i < n; i++) {
          System.arraycopy(classRows[i], 0, flat, i * hidden, hidden);
        }
        Ggml.setFloats(classes, call, flat);
        w.run(graph);
        return Ggml.getFloats(logits, call, n);
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /** GLiClass {@code FeaturesProjector}: linear_1 → GELU → linear_2. */
  private MemorySegment projector(
    MemorySegment ctx,
    MemorySegment x,
    String prefix
  ) {
    return w.linear(
      ctx,
      Ggml.geluErf(ctx, w.linear(ctx, x, prefix + ".linear_1")),
      prefix + ".linear_2"
    );
  }

  @Override
  public synchronized void close() {
    w.close();
  }
}
