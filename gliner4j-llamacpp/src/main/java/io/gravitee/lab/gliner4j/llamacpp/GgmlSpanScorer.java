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
 * The GLiNER streaming-span head as ggml graphs: the DeBERTa labels encoder over the prompt rows
 * (one vector per {@code <<LABEL>>}, then {@code prompt_rep_layer}) and the {@code markerV2} span
 * representation ({@code project_start ⊕ project_end ⊕ project_context(latest)} → ReLU →
 * {@code out_project}) scored by dot product against the label vectors.
 */
public final class GgmlSpanScorer implements AutoCloseable {

  private final GgmlWeights w;
  private final GgmlDeberta labelsEncoder;
  private final int hidden;

  public GgmlSpanScorer(
    Path gguf,
    boolean useGpu,
    int cpuThreads,
    float layerNormEps
  ) {
    this.w = new GgmlWeights(gguf, useGpu, cpuThreads, "ggml span scorer");
    this.labelsEncoder = new GgmlDeberta(
      w,
      "labels_encoder.layer.",
      "labels_encoder.rel_embeddings.weight",
      layerNormEps
    );
    this.hidden = labelsEncoder.hidden();
  }

  public int hidden() {
    return hidden;
  }

  /**
   * Label vectors from the backbone states of the prompt {@code label1 <<LABEL>> … <<SEP>>}.
   *
   * @param promptRows backbone hidden state of every prompt token, through the separator
   * @param labelPositions index of each {@code <<LABEL>>} token within {@code promptRows}
   * @return {@code [labels][hidden]} projected label representations
   */
  public synchronized float[][] labelEmbeddings(
    List<float[]> promptRows,
    int[] labelPositions
  ) {
    int n = promptRows.size();
    int c = labelPositions.length;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call);
      try {
        var x = Ggml.newTensor2d(ctx, w.typeF32, hidden, n);
        Ggml.setInput(x);
        var labelIdx = Ggml.newTensor1d(ctx, w.typeI32, c);
        Ggml.setInput(labelIdx);
        var enc = labelsEncoder.build(ctx, x, n);
        var labelRows = Ggml.getRows(ctx, enc.output(), labelIdx); // (hidden, C)
        var out = w.projection(ctx, labelRows, "prompt_rep_layer"); // (hidden, C)
        var graph = Ggml.newGraph(ctx, GgmlWeights.GRAPH_SIZE);
        w.compute(ctx, graph, out);

        Ggml.setFloats(x, call, flatten(promptRows, hidden));
        labelsEncoder.feed(enc, call);
        Ggml.setInts(labelIdx, call, labelPositions);
        w.run(graph);
        return unflatten(Ggml.getFloats(out, call, c * hidden), c, hidden);
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /**
   * Scores candidate spans over a window of word states.
   *
   * @param wordRows backbone word states (first sub-token) of the window, in order
   * @param spanStart per candidate, start word index within {@code wordRows}
   * @param spanEnd per candidate, inclusive end word index within {@code wordRows}
   * @param latest index within {@code wordRows} of the latest word (the context feature)
   * @param labels {@code [labels][hidden]} label vectors from {@link #labelEmbeddings}
   * @return {@code [candidates][labels]} raw logits
   */
  public synchronized float[][] spanLogits(
    List<float[]> wordRows,
    int[] spanStart,
    int[] spanEnd,
    int latest,
    float[][] labels
  ) {
    int n = wordRows.size();
    int s = spanStart.length;
    int c = labels.length;
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call);
      try {
        var h = Ggml.newTensor2d(ctx, w.typeF32, hidden, n);
        var lab = Ggml.newTensor2d(ctx, w.typeF32, hidden, c);
        var startIdx = Ggml.newTensor1d(ctx, w.typeI32, s);
        var endIdx = Ggml.newTensor1d(ctx, w.typeI32, s);
        var latestIdx = Ggml.newTensor1d(ctx, w.typeI32, 1);
        for (var t : new java.lang.foreign.MemorySegment[] {
          h,
          lab,
          startIdx,
          endIdx,
          latestIdx,
        })
          Ggml.setInput(t);

        var ps = w.projection(ctx, h, "span_rep.project_start"); // (hidden, n)
        var pe = w.projection(ctx, h, "span_rep.project_end");
        var pc = w.projection(
          ctx,
          Ggml.getRows(ctx, h, latestIdx),
          "span_rep.project_context"
        ); // (hidden, 1)
        var startRep = Ggml.getRows(ctx, ps, startIdx); // (hidden, S)
        var endRep = Ggml.getRows(ctx, pe, endIdx);
        var ctxRep = Ggml.repeat(ctx, pc, startRep);
        var feats = Ggml.relu(
          ctx,
          Ggml.concat(ctx, Ggml.concat(ctx, startRep, endRep, 0), ctxRep, 0)
        ); // (3·hidden, S)
        var spanRep = w.projection(ctx, feats, "span_rep.out_project"); // (hidden, S)
        var scores = Ggml.mulMat(ctx, lab, spanRep); // (C, S)
        var graph = Ggml.newGraph(ctx, GgmlWeights.GRAPH_SIZE);
        w.compute(ctx, graph, scores);

        Ggml.setFloats(h, call, flatten(wordRows, hidden));
        Ggml.setFloats(lab, call, flatten(List.of(labels), hidden));
        Ggml.setInts(startIdx, call, spanStart);
        Ggml.setInts(endIdx, call, spanEnd);
        Ggml.setInts(latestIdx, call, new int[] { latest });
        w.run(graph);
        return unflatten(Ggml.getFloats(scores, call, s * c), s, c);
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  private static float[] flatten(List<float[]> rows, int width) {
    var flat = new float[rows.size() * width];
    for (int i = 0; i < rows.size(); i++) System.arraycopy(
      rows.get(i),
      0,
      flat,
      i * width,
      width
    );
    return flat;
  }

  private static float[][] unflatten(float[] flat, int rows, int width) {
    var out = new float[rows][width];
    for (int i = 0; i < rows; i++) System.arraycopy(
      flat,
      i * width,
      out[i],
      0,
      width
    );
    return out;
  }

  @Override
  public synchronized void close() {
    w.close();
  }
}
