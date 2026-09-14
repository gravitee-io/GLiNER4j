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
import io.gravitee.lab.gliner4j.runtime.Gliner2dot5RelationRuntime;
import java.util.Arrays;

/** GLiNER2.5 relation extraction on the ggml engine over one {@link GgmlGliner2dot5Model}. */
public final class GgmlGliner2dot5RelationRuntime
  implements Gliner2dot5RelationRuntime {

  private final GgmlGliner2dot5Model model;

  public GgmlGliner2dot5RelationRuntime(GgmlGliner2dot5Model model) {
    this.model = model;
  }

  public static GgmlGliner2dot5RelationRuntime load(LoadContext ctx) {
    return new GgmlGliner2dot5RelationRuntime(
      GgmlGliner2dot5NerRuntime.loadModel(ctx)
    );
  }

  /** {@code queryPositions} = head_0, tail_0, head_1, tail_1, …. */
  @Override
  public Gliner2dot5RelationRuntime.Scoring run(
    long[][] inputIds,
    long[][] attentionMask,
    int maxSeqLen,
    long[] wordPositionsFlat,
    int batchSize,
    int maxTextLen,
    long[] queryPositions
  ) {
    int r = queryPositions.length / 2;
    int p = model.pairCap();
    var qp = toInts(queryPositions);
    var logits = new float[batchSize * r * p];
    var pairs = new long[batchSize * r * p * 4];
    for (int b = 0; b < batchSize; b++) {
      var res = model.scoreRelations(
        unpadded(inputIds[b], attentionMask[b]),
        wordPositions(wordPositionsFlat, b, maxTextLen),
        qp
      );
      for (int rr = 0; rr < r; rr++) {
        System.arraycopy(res.logits()[rr], 0, logits, (b * r + rr) * p, p);
        for (int j = 0; j < p; j++) {
          for (int k = 0; k < 4; k++) {
            pairs[((b * r + rr) * p + j) * 4 + k] = res.pairs()[rr][j][k];
          }
        }
      }
    }
    return new Gliner2dot5RelationRuntime.Scoring(
      logits,
      pairs,
      batchSize,
      r,
      p
    );
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
