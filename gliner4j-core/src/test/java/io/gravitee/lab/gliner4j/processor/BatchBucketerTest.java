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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.Arrays;
import java.util.List;
import org.junit.jupiter.api.Test;

class BatchBucketerTest {

  @Test
  void emptyInputYieldsNoBuckets() {
    assertEquals(List.of(), BatchBucketer.bucketize(new int[0], 1.3, null));
  }

  @Test
  void disabledRatioYieldsSingleBucketInOriginalOrder() {
    var buckets = BatchBucketer.bucketize(new int[] { 50, 500, 60 }, 1.0, null);
    assertEquals(1, buckets.size());
    assertEquals("[0, 1, 2]", Arrays.toString(buckets.get(0)));
  }

  @Test
  void splitsSkewedLengthsByRatio() {
    var seqLens = new int[] { 50, 500, 60, 55, 480 };
    var buckets = BatchBucketer.bucketize(seqLens, 1.3, null);
    assertEquals(2, buckets.size());
    assertEquals("[0, 3, 2]", Arrays.toString(buckets.get(0)));
    assertEquals("[4, 1]", Arrays.toString(buckets.get(1)));
  }

  @Test
  void everySlotAppearsExactlyOnce() {
    var seqLens = new int[] { 10, 200, 40, 40, 900, 15, 300, 55 };
    var buckets = BatchBucketer.bucketize(seqLens, 1.5, null);
    var seen = new boolean[seqLens.length];
    for (var bucket : buckets) {
      for (int s : bucket) {
        assertTrue(!seen[s], "slot " + s + " appeared twice");
        seen[s] = true;
      }
    }
    for (boolean b : seen) {
      assertTrue(b);
    }
  }

  @Test
  void respectsRatioWithinEachBucket() {
    var seqLens = new int[] { 10, 200, 40, 40, 900, 15, 300, 55 };
    var buckets = BatchBucketer.bucketize(seqLens, 1.5, null);
    for (var bucket : buckets) {
      int min = Integer.MAX_VALUE;
      int max = 0;
      for (int s : bucket) {
        min = Math.min(min, seqLens[s]);
        max = Math.max(max, seqLens[s]);
      }
      assertTrue(
        max <= min * 1.5,
        "bucket violates ratio: " + min + ".." + max
      );
    }
  }

  @Test
  void respectsMaxBucketSize() {
    var seqLens = new int[] { 10, 10, 10, 10, 10 };
    var buckets = BatchBucketer.bucketize(seqLens, 1.3, 2);
    assertEquals(3, buckets.size());
    for (var bucket : buckets) {
      assertTrue(bucket.length <= 2);
    }
  }

  @Test
  void sizeCapAppliesEvenWhenRatioDisabled() {
    var buckets = BatchBucketer.bucketize(new int[] { 5, 5, 5 }, 1.0, 2);
    assertEquals(2, buckets.size());
  }

  @Test
  void singleSlot() {
    var buckets = BatchBucketer.bucketize(new int[] { 42 }, 1.3, null);
    assertEquals(1, buckets.size());
    assertEquals("[0]", Arrays.toString(buckets.get(0)));
  }
}
