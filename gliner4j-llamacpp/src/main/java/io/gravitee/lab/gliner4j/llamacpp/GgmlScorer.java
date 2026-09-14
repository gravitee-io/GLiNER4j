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
import java.nio.file.Path;
import java.util.List;

/**
 * The GLiClass {@code DecoderKVScorer} as a ggml compute graph on the same backend family as the
 * llama.cpp backbone: the 2-layer DeBERTa encoder ({@link GgmlDeberta}) over the label-section
 * hidden states, then the text/label projectors and the pair MLP. Weights come from
 * {@code gguf/scorer.gguf} written by {@code export_gliclass_decoder_kv.py}.
 */
public final class GgmlScorer implements LabelScorer {

  private final GgmlWeights w;
  private final GgmlDeberta encoder;
  private final int hidden;
  private final boolean normalizeFeatures;
  private final float logitScale;

  public GgmlScorer(
    Path gguf,
    boolean useGpu,
    int cpuThreads,
    boolean normalizeFeatures,
    float logitScale,
    float layerNormEps
  ) {
    this.w = new GgmlWeights(gguf, useGpu, cpuThreads, "ggml scorer");
    this.encoder = new GgmlDeberta(
      w,
      "scorer_encoder.layer.",
      "scorer_encoder.rel_embeddings.weight",
      layerNormEps
    );
    this.hidden = encoder.hidden();
    this.normalizeFeatures = normalizeFeatures;
    this.logitScale = logitScale;
  }

  @Override
  public synchronized float[][] score(
    List<List<float[]>> labelHidden,
    LabelSection section
  ) {
    var out = new float[labelHidden.size()][];
    for (int b = 0; b < labelHidden.size(); b++) {
      out[b] = scoreOne(labelHidden.get(b), section);
    }
    return out;
  }

  private float[] scoreOne(List<float[]> rows, LabelSection section) {
    int ls = rows.size();
    if (ls != section.bodyLength()) {
      throw new IllegalArgumentException(
        "expected " + section.bodyLength() + " rows, got " + ls
      );
    }
    int numLabels = section.labelPositions().length;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call);
      try {
        var x = Ggml.newTensor2d(ctx, w.typeF32, hidden, ls);
        Ggml.setInput(x);
        var labelIdx = Ggml.newTensor1d(ctx, w.typeI32, numLabels);
        var sepIdx = Ggml.newTensor1d(ctx, w.typeI32, 1);
        Ggml.setInput(labelIdx);
        Ggml.setInput(sepIdx);

        var enc = encoder.build(ctx, x, ls);
        var labelRows = Ggml.getRows(ctx, enc.output(), labelIdx); // (hidden, L)
        var textRow = Ggml.getRows(ctx, enc.output(), sepIdx); // (hidden, 1)
        var text = w.linear(ctx, textRow, "text_projector");
        var labels = w.linear(ctx, labelRows, "label_projector");
        if (normalizeFeatures) {
          text = Ggml.l2Norm(ctx, text, 1e-8f);
          labels = Ggml.l2Norm(ctx, labels, 1e-8f);
        }
        var combined = Ggml.concat(
          ctx,
          Ggml.repeat(ctx, text, labels),
          labels,
          0
        ); // (2·hidden, L)
        var m = Ggml.relu(ctx, w.linear(ctx, combined, "mlp.0"));
        m = Ggml.relu(ctx, w.linear(ctx, m, "mlp.2"));
        var logits = w.linear(ctx, m, "mlp.4"); // (1, L)
        if (normalizeFeatures) {
          logits = Ggml.scale(ctx, logits, logitScale);
        }
        var graph = Ggml.newGraph(ctx, GgmlWeights.GRAPH_SIZE);
        w.compute(ctx, graph, logits);

        var flat = new float[hidden * ls];
        for (int i = 0; i < ls; i++) System.arraycopy(
          rows.get(i),
          0,
          flat,
          i * hidden,
          hidden
        );
        Ggml.setFloats(x, call, flat);
        encoder.feed(enc, call);
        var li = new int[numLabels];
        for (int i = 0; i < numLabels; i++) li[i] =
          (int) section.labelPositions()[i];
        Ggml.setInts(labelIdx, call, li);
        Ggml.setInts(sepIdx, call, new int[] { (int) section.sepPosition() });
        w.run(graph);
        return Ggml.getFloats(logits, call, numLabels);
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  @Override
  public synchronized void close() {
    w.close();
  }
}
