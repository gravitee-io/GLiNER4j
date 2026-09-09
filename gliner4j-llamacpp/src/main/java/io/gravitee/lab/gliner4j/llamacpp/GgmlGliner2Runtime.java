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

import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.runtime.FlatBatchScoringResult;
import io.gravitee.lab.gliner4j.runtime.FloatTensor;
import io.gravitee.lab.gliner4j.runtime.Gliner2ClassifierRuntime;
import io.gravitee.lab.gliner4j.runtime.Gliner2SpanRuntime;
import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * The GLiNER2 runtimes ({@link Gliner2SpanRuntime}, {@link Gliner2ClassifierRuntime}) on the ggml
 * engine: one {@link GgmlGliner2Model} serves the merged span graph and the classifier graph, so
 * the core GLiNER2 strategies, extractors and the unified facade run unchanged on top of it.
 * Batches are scored row by row (the ggml graph is built per sequence length).
 */
public final class GgmlGliner2Runtime
  implements Gliner2SpanRuntime, Gliner2ClassifierRuntime {

  private static final Logger log = LoggerFactory.getLogger(
    GgmlGliner2Runtime.class
  );

  /**
   * {@code -Dgliner4j.ggml.replicas=N}: model instances per runtime. Each replica is a full copy of
   * the weights with its own CUDA stream and scheduler, so N rows can be in flight on the same
   * card at once — a single 1 400-token row leaves most of a data-center GPU idle and serialises
   * on one graph at a time, so throughput stays flat with the card mostly unused. Callers must be
   * concurrent for it to matter; the pool hands one replica per call.
   */
  public static final String REPLICAS_PROPERTY = "gliner4j.ggml.replicas";

  private final GgmlGliner2Model[] models;
  private final BlockingQueue<GgmlGliner2Model> idle;

  public GgmlGliner2Runtime(GgmlGliner2Model model) {
    this(new GgmlGliner2Model[] { model });
  }

  public GgmlGliner2Runtime(GgmlGliner2Model[] models) {
    if (models.length == 0) throw new IllegalArgumentException("no model");
    this.models = models;
    this.idle = new ArrayBlockingQueue<>(models.length);
    this.idle.addAll(Arrays.asList(models));
  }

  public static GgmlGliner2Runtime load(LoadContext ctx) {
    var rc = ctx.runtimeConfig();
    int gpuLayers = LlamaGliclassClassificationStrategy.gpuLayers(rc);
    int replicas = Math.max(1, Integer.getInteger(REPLICAS_PROPERTY, 1));
    var gguf = GgmlWeights.resolveModelGguf(ctx.modelDir(), ctx.variant());
    var models = new GgmlGliner2Model[replicas];
    for (int i = 0; i < replicas; i++) {
      models[i] = new GgmlGliner2Model(
        gguf,
        gpuLayers > 0,
        LlamaBackbone.threads(rc)
      );
    }
    log.info(
      "GLiNER2 ggml runtime loaded (hidden={}, maxCount={}, replicas={})",
      models[0].hiddenSize(),
      models[0].maxCount(),
      replicas
    );
    return new GgmlGliner2Runtime(models);
  }

  /** Blocks until a replica is free. */
  private GgmlGliner2Model acquire() {
    try {
      return idle.take();
    } catch (InterruptedException e) {
      Thread.currentThread().interrupt();
      throw new IllegalStateException(
        "interrupted waiting for a GLiNER2 replica",
        e
      );
    }
  }

  private void release(GgmlGliner2Model model) {
    idle.add(model);
  }

  @Override
  public FlatBatchScoringResult runNerFullBatch(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] wordPositionsFlat,
    int batchSize,
    int maxTextLen,
    long pPosition,
    long[] fieldPositions,
    long[] spanIdxFlat,
    int maxWidth,
    long count
  ) {
    var model = acquire();
    try {
      int fields = fieldPositions.length;
      int cnt = (int) Math.max(0, Math.min(count, model.maxCount()));
      var fieldPos = new int[fields];
      for (int i = 0; i < fields; i++) fieldPos[i] = (int) fieldPositions[i];
      long perRow = (long) cnt * fields * maxTextLen * maxWidth;
      var scores = new float[(int) (perRow * batchSize)];
      var rows = new long[batchSize][];
      var wordRows = new int[batchSize][];
      for (int b = 0; b < batchSize; b++) {
        rows[b] = unpadded(inputIds[b], attentionMask[b]);
        wordRows[b] = wordPositions(wordPositionsFlat, b, maxTextLen);
      }
      // CPU / Metal: rows are scored one at a time — the padded batched graph is numerically the
      // same but measured slower there (padding + 4-D attention tensors). CUDA: one batched graph,
      // which is what keeps the GPU busy on short rows (see GgmlGliner2Model.prefersBatchedScoring).
      GgmlGliner2Model.UnitScores[] units;
      int longest = 0;
      for (var r : rows) longest = Math.max(longest, r.length);
      if (batchSize > 1 && model.prefersBatchedScoring(longest)) {
        units = model.scoreUnitBatch(
          rows,
          wordRows,
          (int) pPosition,
          fieldPos,
          cnt,
          maxWidth
        );
      } else {
        units = new GgmlGliner2Model.UnitScores[batchSize];
        for (int b = 0; b < batchSize; b++) {
          units[b] = model.scoreUnit(
            rows[b],
            wordRows[b],
            (int) pPosition,
            fieldPos,
            cnt,
            maxWidth
          );
        }
      }
      var countLogits = new float[batchSize][];
      for (int b = 0; b < batchSize; b++) {
        var unit = units[b];
        countLogits[b] = unit.countLogits();
        long base = b * perRow;
        for (int c = 0; c < cnt; c++) {
          for (int f = 0; f < fields; f++) {
            for (int t = 0; t < wordRows[b].length; t++) {
              System.arraycopy(
                unit.spans()[c][f][t],
                0,
                scores,
                (int) (base +
                  (((long) c * fields + f) * maxTextLen + t) * maxWidth),
                maxWidth
              );
            }
          }
        }
      }
      var tensor = new FloatTensor(
        FloatBuffer.wrap(scores),
        new long[] { batchSize, cnt, fields, maxTextLen, maxWidth }
      );
      return new FlatBatchScoringResult(countLogits, tensor, () -> {});
    } finally {
      release(model);
    }
  }

  @Override
  public float[][] runClassifierFullBatch(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] labelPositions,
    int batchSize
  ) {
    var model = acquire();
    try {
      var pos = new int[labelPositions.length];
      for (int i = 0; i < pos.length; i++) pos[i] = (int) labelPositions[i];
      var rows = new long[batchSize][];
      for (int b = 0; b < batchSize; b++) rows[b] = unpadded(
        inputIds[b],
        attentionMask[b]
      );
      int longest = 0;
      for (var r : rows) longest = Math.max(longest, r.length);
      // CUDA: short rows are launch-bound, so pad them into one graph (same rule as the NER path)
      if (batchSize > 1 && model.prefersBatchedScoring(longest)) {
        return model.classifyBatch(rows, pos);
      }
      var out = new float[batchSize][];
      for (int b = 0; b < batchSize; b++) out[b] = model.classify(rows[b], pos);
      return out;
    } finally {
      release(model);
    }
  }

  private static long[] unpadded(long[] ids, long[] mask) {
    int n = ids.length;
    while (n > 0 && (n > mask.length || mask[n - 1] == 0)) n--;
    return n == ids.length ? ids : Arrays.copyOf(ids, n);
  }

  private static int[] wordPositions(long[] flat, int row, int maxTextLen) {
    int base = row * maxTextLen;
    int n = 0;
    while (n < maxTextLen && flat[base + n] >= 0) n++;
    var out = new int[n];
    for (int i = 0; i < n; i++) out[i] = (int) flat[base + i];
    return out;
  }

  @Override
  public void close() {
    for (var m : models) m.close();
  }
}
