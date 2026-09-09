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
package io.gravitee.lab.gliner4j.postprocess;

import io.gravitee.lab.gliner4j.runtime.Gliner2dot5NerRuntime.Scoring;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.utils.LinAlg;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;

/**
 * Decodes GLiNER2.5 pooled pair logits into entity spans.
 *
 * <p>Mirrors {@code BoundaryExtractor._decode_entities}: per query, sigmoid the pair logits
 * (divided by the pair temperature), keep pool candidates at or above the threshold, drop the
 * whole query when the abstention head fires, resolve overlaps with the {@code flat} policy
 * (maximum-total-score non-overlapping set via weighted interval scheduling, deterministic
 * ties), then map half-open word boundaries {@code [start, end)} to characters as
 * {@code wordStartChars[start]} / {@code wordEndChars[end - 1]}.
 */
public final class BoundaryDecoder {

  /** A thresholded pool candidate: probability plus half-open word boundaries. */
  public record Candidate(float prob, int start, int end) {}

  private final float pairTemperature;
  private final boolean abstentionEnabled;
  private final float abstentionThreshold;

  public BoundaryDecoder(
    float pairTemperature,
    boolean abstentionEnabled,
    float abstentionThreshold
  ) {
    this.pairTemperature = pairTemperature;
    this.abstentionEnabled = abstentionEnabled;
    this.abstentionThreshold = abstentionThreshold;
  }

  /**
   * Decodes one batch row.
   *
   * @param scoring the merged-graph outputs
   * @param row this text's row within the sub-batch
   * @param fieldNames entity type per query, in query order
   * @param wordStartChars character start offset per word
   * @param wordEndChars character end offset per word
   * @param text the original text
   * @param textLen number of words for this row (may be shorter than the padded tensor)
   * @param threshold minimum probability for inclusion
   * @return the resolved spans, sorted by character start
   */
  public List<EntitySpan> decode(
    Scoring scoring,
    int row,
    List<String> fieldNames,
    int[] wordStartChars,
    int[] wordEndChars,
    String text,
    int textLen,
    float threshold
  ) {
    var out = new ArrayList<EntitySpan>();
    int numQueries = Math.min(scoring.numQueries(), fieldNames.size());
    for (int q = 0; q < numQueries; q++) {
      if (
        abstentionEnabled &&
        LinAlg.sigmoid(scoring.nullLogit(row, q)) > abstentionThreshold
      ) {
        continue;
      }
      var scored = new ArrayList<Candidate>();
      for (int c = 0; c < scoring.poolSize(); c++) {
        float prob = LinAlg.sigmoid(
          scoring.pairLogit(row, q, c) / pairTemperature
        );
        if (prob < threshold) {
          continue;
        }
        int start = scoring.candidateStart(row, c);
        int end = scoring.candidateEnd(row, c);
        if (start < 0 || start >= end || end > textLen) {
          continue;
        }
        scored.add(new Candidate(prob, start, end));
      }
      if (scored.isEmpty()) {
        continue;
      }
      var type = fieldNames.get(q);
      for (var cand : resolveFlat(scored)) {
        int charStart = wordStartChars[cand.start()];
        int charEnd = Math.min(wordEndChars[cand.end() - 1], text.length());
        if (charEnd <= charStart) {
          continue;
        }
        var surface = text.substring(charStart, charEnd);
        if (surface.isBlank()) {
          continue;
        }
        out.add(
          new EntitySpan(type, surface.strip(), cand.prob(), charStart, charEnd)
        );
      }
    }
    out.sort(Comparator.comparingInt(EntitySpan::start));
    return out;
  }

  // Deterministic ranking shared by dedup, DP tie-breaks and the output order:
  // descending score, ascending start, ascending end.
  private static final Comparator<Candidate> RANK = Comparator.comparingDouble(
    (Candidate c) -> -c.prob()
  )
    .thenComparingInt(Candidate::start)
    .thenComparingInt(Candidate::end);

  /**
   * fastino's {@code flat} overlap policy: collapse exact-boundary duplicates to the best-ranked
   * one, then return the maximum-total-score set of pairwise non-overlapping spans. Dynamic-program
   * ties prefer the larger set, then the lexicographically better ranking of the selection.
   *
   * @param scored thresholded candidates for one query
   * @return the selected candidates, ranked by descending score then start/end
   */
  public static List<Candidate> resolveFlat(List<Candidate> scored) {
    if (scored.isEmpty()) {
      return List.of();
    }
    var ranked = new ArrayList<>(scored);
    ranked.sort(RANK);
    var distinct = new ArrayList<Candidate>(ranked.size());
    var seen = new HashSet<Long>();
    for (var c : ranked) {
      if (seen.add(((long) c.start() << 32) | (c.end() & 0xffffffffL))) {
        distinct.add(c);
      }
    }
    // rankIndex[i] = position of distinct[i] in the ranking (distinct is already rank-ordered).
    int n = distinct.size();
    var order = new Integer[n];
    for (int i = 0; i < n; i++) order[i] = i;
    // Schedule by end asc, start asc, then ranking.
    java.util.Arrays.sort(order, (a, b) -> {
      var ca = distinct.get(a);
      var cb = distinct.get(b);
      if (ca.end() != cb.end()) return Integer.compare(ca.end(), cb.end());
      if (ca.start() != cb.start()) return Integer.compare(
        ca.start(),
        cb.start()
      );
      return Integer.compare(a, b);
    });
    var ends = new int[n];
    for (int i = 0; i < n; i++) ends[i] = distinct.get(order[i]).end();

    // best[i] = optimal selection over the first i scheduled items (as rank indices).
    var bestScore = new double[n + 1];
    @SuppressWarnings("unchecked")
    var bestSel = (List<Integer>[]) new List[n + 1];
    bestSel[0] = List.of();
    for (int i = 0; i < n; i++) {
      var item = distinct.get(order[i]);
      // Last scheduled item ending at or before this item's start (half-open: end <= start).
      int pred = upperBound(ends, item.start(), i) - 1;
      double withScore = bestScore[pred + 1] + item.prob();
      var withSel = new ArrayList<>(bestSel[pred + 1]);
      withSel.add(order[i]);
      double withoutScore = bestScore[i];
      var withoutSel = bestSel[i];
      boolean takeWith;
      if (withScore > withoutScore) {
        takeWith = true;
      } else if (withScore < withoutScore) {
        takeWith = false;
      } else if (withSel.size() != withoutSel.size()) {
        takeWith = withSel.size() > withoutSel.size();
      } else {
        takeWith = compareSelections(withSel, withoutSel) < 0;
      }
      bestScore[i + 1] = takeWith ? withScore : withoutScore;
      bestSel[i + 1] = takeWith ? withSel : withoutSel;
    }
    var selected = new ArrayList<Candidate>(bestSel[n].size());
    for (int idx : bestSel[n]) selected.add(distinct.get(idx));
    selected.sort(RANK);
    return selected;
  }

  /** Lexicographic comparison of two selections by their sorted rank indices (lower = better). */
  private static int compareSelections(List<Integer> a, List<Integer> b) {
    var sa = new ArrayList<>(a);
    var sb = new ArrayList<>(b);
    sa.sort(null);
    sb.sort(null);
    for (int i = 0; i < Math.min(sa.size(), sb.size()); i++) {
      int cmp = Integer.compare(sa.get(i), sb.get(i));
      if (cmp != 0) return cmp;
    }
    return Integer.compare(sa.size(), sb.size());
  }

  /** First index in {@code [0, hi)} whose value is strictly greater than {@code key}. */
  private static int upperBound(int[] sorted, int key, int hi) {
    int lo = 0;
    while (lo < hi) {
      int mid = (lo + hi) >>> 1;
      if (sorted[mid] <= key) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }
}
