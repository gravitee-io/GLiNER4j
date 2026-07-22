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
package io.gravitee.lab.gliner4j.processor;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * Groups batch slots into length-homogeneous sub-batches so the encoder does not pad every
 * text to the batch-wide max sequence length.
 *
 * <p>Slots are sorted by sequence length and cut greedily: a new bucket starts when adding
 * the next slot would exceed {@code lengthRatio} between the bucket's longest and shortest
 * member, or when the bucket reaches {@code maxBucketSize}.
 */
public final class BatchBucketer {

  private BatchBucketer() {}

  /**
   * Buckets slot indices by sequence length.
   *
   * @param seqLens       per-slot sequence lengths
   * @param lengthRatio   max allowed (longest / shortest) ratio inside a bucket; ≤ 1 disables
   *                      bucketing and returns a single bucket in original slot order
   * @param maxBucketSize max slots per bucket; null or ≤ 0 = unbounded
   * @return buckets of slot indices, each sorted by ascending length
   */
  public static List<int[]> bucketize(
    int[] seqLens,
    double lengthRatio,
    Integer maxBucketSize
  ) {
    int n = seqLens.length;
    if (n == 0) {
      return List.of();
    }
    int cap = maxBucketSize == null || maxBucketSize <= 0
      ? Integer.MAX_VALUE
      : maxBucketSize;
    if (lengthRatio <= 1.0 && cap >= n) {
      var all = new int[n];
      for (int i = 0; i < n; i++) {
        all[i] = i;
      }
      return List.of(all);
    }

    var order = new Integer[n];
    for (int i = 0; i < n; i++) {
      order[i] = i;
    }
    Arrays.sort(order, Comparator.comparingInt(s -> seqLens[s]));

    var buckets = new ArrayList<int[]>();
    int start = 0;
    while (start < n) {
      int bucketMin = seqLens[order[start]];
      int end = start + 1;
      while (
        end < n &&
        end - start < cap &&
        (lengthRatio <= 1.0 || seqLens[order[end]] <= bucketMin * lengthRatio)
      ) {
        end++;
      }
      var bucket = new int[end - start];
      for (int i = start; i < end; i++) {
        bucket[i - start] = order[i];
      }
      buckets.add(bucket);
      start = end;
    }
    return buckets;
  }
}
