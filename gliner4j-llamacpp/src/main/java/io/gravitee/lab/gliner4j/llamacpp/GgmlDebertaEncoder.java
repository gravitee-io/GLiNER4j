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

/**
 * Runs {@link GgmlDebertaV3} end to end: token ids in, {@code last_hidden_state} rows out. One
 * graph per call (built for the exact sequence length); calls are serialised on the backend.
 */
public final class GgmlDebertaEncoder implements AutoCloseable {

  private final GgmlWeights w;
  private final GgmlDebertaV3 encoder;

  public GgmlDebertaEncoder(Path gguf, boolean useGpu, int cpuThreads) {
    this.w = new GgmlWeights(gguf, useGpu, cpuThreads, "ggml DeBERTa encoder");
    this.encoder = new GgmlDebertaV3(w);
  }

  public int hiddenSize() {
    return encoder.hidden();
  }

  public int layers() {
    return encoder.layers();
  }

  /** Hidden states {@code [n][hidden]} for one unpadded sequence. */
  public synchronized float[][] encode(long[] ids) {
    int n = ids.length;
    int hidden = encoder.hidden();
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call);
      try {
        var built = encoder.build(ctx, n);
        var graph = Ggml.newGraph(ctx, GgmlWeights.GRAPH_SIZE);
        w.compute(ctx, graph, built.output());
        encoder.feed(built, call, ids);
        w.run(graph);
        var flat = Ggml.getFloats(built.output(), call, n * hidden);
        var rows = new float[n][hidden];
        for (int i = 0; i < n; i++) {
          System.arraycopy(flat, i * hidden, rows[i], 0, hidden);
        }
        return rows;
      } finally {
        Ggml.free(ctx);
      }
    }
  }

  /** Hidden states for several rows encoded together (padded batch); row {@code r} gets {@code rows[r].length} vectors. */
  public synchronized float[][][] encodeBatch(long[][] rows) {
    int batch = rows.length;
    int n = 0;
    for (var r : rows) n = Math.max(n, r.length);
    int hidden = encoder.hidden();
    try (var call = Arena.ofConfined()) {
      var ctx = w.newGraphContext(call, encoder.graphNodes() + 8);
      try {
        var built = encoder.build(ctx, n, batch);
        var graph = Ggml.newGraph(ctx, encoder.graphNodes() + 8);
        w.compute(ctx, graph, built.output());
        encoder.feed(built, call, rows);
        w.run(graph);
        var flat = Ggml.getFloats(built.output(), call, n * batch * hidden);
        var out = new float[batch][][];
        for (int r = 0; r < batch; r++) {
          out[r] = new float[rows[r].length][hidden];
          for (int i = 0; i < rows[r].length; i++) {
            System.arraycopy(flat, (r * n + i) * hidden, out[r][i], 0, hidden);
          }
        }
        return out;
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
