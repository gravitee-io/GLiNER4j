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

import java.util.List;

/**
 * What a {@link StreamingSpanSession} needs from its engine: word splitting, the prompt, the
 * KV-cached backbone and the span scorer. {@link StreamingSpanEngine} is the only production
 * implementation; the seam lets session state handling be tested without a model bundle.
 */
abstract class StreamingSpanBackend {

  abstract StreamingSpanEngine.Words words(String chunk);

  abstract StreamingSpanEngine.Prompt prompt(List<String> labels);

  /** Decodes {@code ids} at positions {@code startPos…} into {@code seq} and returns every token's state. */
  abstract List<float[]> decode(int seq, long[] ids, int startPos);

  abstract float[][] labelEmbeddings(
    List<float[]> promptRows,
    StreamingSpanEngine.Prompt prompt
  );

  abstract float[][] spanLogits(
    List<float[]> window,
    int[] starts,
    int[] ends,
    int latest,
    float[][] labels
  );

  abstract int maxWidth();

  abstract int rightContextWidth();

  abstract int nCtx();

  abstract int acquireSequence();

  abstract void releaseSequence(int seq);
}
