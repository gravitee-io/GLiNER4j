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

import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import java.util.ArrayList;
import java.util.List;

/** Logit post-processing mirroring gliclass's {@code _postprocess_logits}. */
final class DecoderKvScores {

  private DecoderKvScores() {}

  /** Multi-label: sigmoid per label, keep {@code >= threshold}, descending confidence. */
  static List<ClassificationResult> multiLabel(
    float[] logits,
    List<String> labels,
    float threshold
  ) {
    var out = new ArrayList<ClassificationResult>(labels.size());
    for (int i = 0; i < labels.size(); i++) {
      float score = sigmoid(logits[i]);
      if (score >= threshold) {
        out.add(new ClassificationResult(labels.get(i), score));
      }
    }
    out.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));
    return out;
  }

  /** Single-label: softmax over the labels, every label returned, best first. */
  static List<ClassificationResult> singleLabel(
    float[] logits,
    List<String> labels
  ) {
    var probs = softmax(logits, labels.size());
    var out = new ArrayList<ClassificationResult>(labels.size());
    for (int i = 0; i < labels.size(); i++) {
      out.add(new ClassificationResult(labels.get(i), probs[i]));
    }
    out.sort((a, b) -> Float.compare(b.confidence(), a.confidence()));
    return out;
  }

  static float sigmoid(float x) {
    return (float) (1.0 / (1.0 + Math.exp(-x)));
  }

  static float[] softmax(float[] logits, int n) {
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < n; i++) max = Math.max(max, logits[i]);
    double sum = 0;
    var out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = (float) Math.exp(logits[i] - max);
      sum += out[i];
    }
    for (int i = 0; i < n; i++) out[i] = (float) (out[i] / sum);
    return out;
  }
}
