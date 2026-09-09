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
import io.gravitee.lab.gliner4j.runtime.Gliner2ClassifierRuntime;
import io.gravitee.lab.gliner4j.runtime.Gliner2dot5NerRuntime;
import java.util.Arrays;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GLiNER2.5 NER (and the GLiNER2 classifier contract) on the ggml engine over one
 * {@link GgmlGliner2dot5Model}; rows are scored one at a time, or as one padded batch on CUDA.
 */
public final class GgmlGliner2dot5NerRuntime
  implements Gliner2dot5NerRuntime, Gliner2ClassifierRuntime {

  private static final Logger log = LoggerFactory.getLogger(
    GgmlGliner2dot5NerRuntime.class
  );

  private final GgmlGliner2dot5Model model;

  public GgmlGliner2dot5NerRuntime(GgmlGliner2dot5Model model) {
    this.model = model;
  }

  static GgmlGliner2dot5Model loadModel(LoadContext ctx) {
    var rc = ctx.runtimeConfig();
    int gpuLayers = LlamaGliclassClassificationStrategy.gpuLayers(rc);
    var model = new GgmlGliner2dot5Model(
      GgmlWeights.resolveModelGguf(ctx.modelDir(), ctx.variant()),
      gpuLayers > 0,
      LlamaBackbone.threads(rc)
    );
    log.info(
      "GLiNER2.5 ggml model loaded (pool={}, pairCap={})",
      model.poolSize(),
      model.pairCap()
    );
    return model;
  }

  public static GgmlGliner2dot5NerRuntime load(LoadContext ctx) {
    return new GgmlGliner2dot5NerRuntime(loadModel(ctx));
  }

  @Override
  public Gliner2dot5NerRuntime.Scoring run(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] wordPositionsFlat,
    int batchSize,
    int maxTextLen,
    long[] queryPositions
  ) {
    int q = queryPositions.length;
    int c = model.poolSize();
    var qp = toInts(queryPositions);
    var pairLogits = new float[batchSize * q * c];
    var candidates = new long[batchSize * c * 2];
    var nullLogits = new float[batchSize * q];
    var rows = new long[batchSize][];
    var words = new int[batchSize][];
    int longest = 0;
    for (int b = 0; b < batchSize; b++) {
      rows[b] = unpadded(inputIds[b], attentionMask[b]);
      words[b] = wordPositions(wordPositionsFlat, b, maxTextLen);
      longest = Math.max(longest, rows[b].length);
    }
    // CUDA: short rows are launch-bound, so pad them into one graph (GgmlGliner2Runtime does the same)
    GgmlGliner2dot5Model.NerResult[] batched = batchSize > 1 &&
      model.prefersBatchedScoring(longest)
      ? model.scoreEntitiesBatch(rows, words, qp)
      : null;
    for (int b = 0; b < batchSize; b++) {
      var res = batched != null
        ? batched[b]
        : model.scoreEntities(rows[b], words[b], qp);
      for (int qq = 0; qq < q; qq++) {
        System.arraycopy(
          res.pairLogits()[qq],
          0,
          pairLogits,
          (b * q + qq) * c,
          c
        );
        nullLogits[b * q + qq] = res.nullLogits()[qq];
      }
      for (int i = 0; i < c; i++) {
        candidates[(b * c + i) * 2] = res.candidates()[i][0];
        candidates[(b * c + i) * 2 + 1] = res.candidates()[i][1];
      }
    }
    return new Gliner2dot5NerRuntime.Scoring(
      pairLogits,
      candidates,
      nullLogits,
      batchSize,
      q,
      c
    );
  }

  @Override
  public float[][] runClassifierFullBatch(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] labelPositions,
    int batchSize
  ) {
    var pos = toInts(labelPositions);
    var out = new float[batchSize][];
    for (int b = 0; b < batchSize; b++) {
      out[b] = model.classify(unpadded(inputIds[b], attentionMask[b]), pos);
    }
    return out;
  }

  private static int[] toInts(long[] a) {
    var out = new int[a.length];
    for (int i = 0; i < a.length; i++) out[i] = (int) a[i];
    return out;
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
    model.close();
  }
}
