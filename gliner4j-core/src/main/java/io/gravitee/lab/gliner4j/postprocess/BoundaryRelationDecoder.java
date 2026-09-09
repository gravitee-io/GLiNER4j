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

import io.gravitee.lab.gliner4j.runtime.Gliner2dot5RelationRuntime.Scoring;
import io.gravitee.lab.gliner4j.schema.FieldSpan;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import io.gravitee.lab.gliner4j.utils.LinAlg;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

/**
 * Decodes GLiNER2.5 relation-graph outputs into {@link RelationInstance}s.
 *
 * <p>Mirrors {@code BoundaryExtractor._decode_relations}: sigmoid the pair logits (divided by the
 * relation temperature), keep pairs at or above the threshold, map half-open word boundaries to
 * characters, then collapse the head×tail cross product into semantic edges the way
 * {@code _deduplicate_relation_edges} does — canonicalize mentions to their longest containing
 * mention, keep the best score per exact coordinates, keep the closest/highest edge per
 * case-folded (head, tail) text, and drop edges whose one argument is a strict token subset of
 * another edge with the same opposite argument.
 */
public final class BoundaryRelationDecoder {

  /** A thresholded relation edge with character-level head and tail mentions. */
  public record Edge(float score, Mention head, Mention tail) {}

  /** A stripped mention surface with its character offsets. */
  public record Mention(String text, int start, int end) {}

  private final float relationTemperature;

  public BoundaryRelationDecoder(float relationTemperature) {
    this.relationTemperature = relationTemperature;
  }

  /**
   * Decodes one batch row into a per-relation map that always lists every requested relation.
   *
   * @param scoring the merged-graph outputs
   * @param row this text's row within the sub-batch
   * @param relationNames relation type per relation index (result keys)
   * @param headField name of the head field in the returned {@link RelationInstance}
   * @param tailField name of the tail field
   */
  public Map<String, List<RelationInstance>> decode(
    Scoring scoring,
    int row,
    List<String> relationNames,
    String headField,
    String tailField,
    int[] wordStartChars,
    int[] wordEndChars,
    String text,
    int textLen,
    float threshold
  ) {
    var out = new LinkedHashMap<String, List<RelationInstance>>();
    int numRelations = Math.min(scoring.numRelations(), relationNames.size());
    for (int r = 0; r < numRelations; r++) {
      var edges = new ArrayList<Edge>();
      for (int p = 0; p < scoring.pairCap(); p++) {
        float score = LinAlg.sigmoid(
          scoring.logit(row, r, p) / relationTemperature
        );
        if (score < threshold) continue;
        int hs = scoring.boundary(row, r, p, 0);
        int he = scoring.boundary(row, r, p, 1);
        int ts = scoring.boundary(row, r, p, 2);
        int te = scoring.boundary(row, r, p, 3);
        if (
          !(0 <= hs && hs < he && he <= textLen) ||
          !(0 <= ts && ts < te && te <= textLen)
        ) {
          continue;
        }
        var head = mention(hs, he, wordStartChars, wordEndChars, text);
        var tail = mention(ts, te, wordStartChars, wordEndChars, text);
        if (head == null || tail == null) continue;
        edges.add(new Edge(score, head, tail));
      }
      var name = relationNames.get(r);
      var instances = new ArrayList<RelationInstance>();
      for (var edge : deduplicate(edges)) {
        var fields = new LinkedHashMap<String, FieldSpan>(2);
        fields.put(headField, fieldSpan(headField, edge.head(), edge.score()));
        fields.put(tailField, fieldSpan(tailField, edge.tail(), edge.score()));
        instances.add(new RelationInstance(name, fields, edge.score()));
      }
      out.put(name, instances);
    }
    for (int r = numRelations; r < relationNames.size(); r++) {
      out.put(relationNames.get(r), List.of());
    }
    return out;
  }

  private static Mention mention(
    int start,
    int end,
    int[] wordStartChars,
    int[] wordEndChars,
    String text
  ) {
    int charStart = wordStartChars[start];
    int charEnd = Math.min(wordEndChars[end - 1], text.length());
    if (charEnd <= charStart) return null;
    var surface = text.substring(charStart, charEnd).strip();
    return surface.isEmpty() ? null : new Mention(surface, charStart, charEnd);
  }

  private static FieldSpan fieldSpan(String field, Mention m, float score) {
    return new FieldSpan(field, m.text(), score, m.start(), m.end());
  }

  /**
   * fastino's {@code _deduplicate_relation_edges}, edge for edge.
   *
   * @param edges thresholded edges of one relation type
   * @return semantic edges sorted by head start, tail start, descending score
   */
  public static List<Edge> deduplicate(List<Edge> edges) {
    if (edges.size() < 2) {
      return edges;
    }
    var headCanonical = canonicalMentions(edges, true);
    var tailCanonical = canonicalMentions(edges, false);

    // Exact coordinates (after canonicalization): best score wins.
    var exact = new LinkedHashMap<List<Integer>, Edge>();
    for (var edge : edges) {
      var head = headCanonical.get(coords(edge.head()));
      var tail = tailCanonical.get(coords(edge.tail()));
      var normalized = new Edge(edge.score(), head, tail);
      var key = List.of(head.start(), head.end(), tail.start(), tail.end());
      var previous = exact.get(key);
      if (previous == null || edge.score() > previous.score()) {
        exact.put(key, normalized);
      }
    }

    // Same (head text, tail text): keep the closest pair, then the highest score.
    var semantic = new LinkedHashMap<List<String>, Edge>();
    for (var edge : exact.values()) {
      var key = List.of(
        semanticText(edge.head().text()),
        semanticText(edge.tail().text())
      );
      var previous = semantic.get(key);
      if (previous == null || compareRank(edge, previous) < 0) {
        semantic.put(key, edge);
      }
    }

    // Drop edges whose one argument is a strict token subset of another edge's, with the same
    // opposite argument.
    var values = new ArrayList<>(semantic.values());
    var kept = new ArrayList<Edge>();
    for (var edge : values) {
      var headTokens = tokens(edge.head().text());
      var tailTokens = tokens(edge.tail().text());
      boolean dominated = false;
      for (var other : values) {
        if (other == edge) continue;
        var otherHead = tokens(other.head().text());
        var otherTail = tokens(other.tail().text());
        if (
          (strictSubset(headTokens, otherHead) &&
            tailTokens.equals(otherTail)) ||
          (strictSubset(tailTokens, otherTail) && headTokens.equals(otherHead))
        ) {
          dominated = true;
          break;
        }
      }
      if (!dominated) kept.add(edge);
    }
    kept.sort(
      Comparator.comparingInt((Edge e) -> e.head().start())
        .thenComparingInt(e -> e.tail().start())
        .thenComparingDouble(e -> -e.score())
    );
    return kept;
  }

  /** For each distinct mention, the longest (then leftmost) mention containing it. */
  private static Map<List<Integer>, Mention> canonicalMentions(
    List<Edge> edges,
    boolean head
  ) {
    var mentions = new LinkedHashMap<List<Integer>, Mention>();
    for (var edge : edges) {
      var m = head ? edge.head() : edge.tail();
      mentions.put(coords(m), m);
    }
    var canonical = new LinkedHashMap<List<Integer>, Mention>();
    for (var entry : mentions.entrySet()) {
      var m = entry.getValue();
      Mention best = null;
      for (var candidate : mentions.values()) {
        if (candidate.start() <= m.start() && candidate.end() >= m.end()) {
          if (
            best == null ||
            candidate.end() - candidate.start() > best.end() - best.start() ||
            (candidate.end() - candidate.start() == best.end() - best.start() &&
              candidate.start() < best.start())
          ) {
            best = candidate;
          }
        }
      }
      canonical.put(entry.getKey(), best);
    }
    return canonical;
  }

  private static List<Integer> coords(Mention m) {
    return List.of(m.start(), m.end());
  }

  private static String semanticText(String value) {
    return String.join(
      " ",
      value.toLowerCase(Locale.ROOT).trim().split("\\s+")
    );
  }

  private static Set<String> tokens(String value) {
    return new HashSet<>(Arrays.asList(semanticText(value).split(" ")));
  }

  private static boolean strictSubset(Set<String> a, Set<String> b) {
    return a.size() < b.size() && b.containsAll(a);
  }

  /** (distance, -score, head start, tail start) — lower ranks first. */
  private static int compareRank(Edge a, Edge b) {
    int cmp = Integer.compare(distance(a), distance(b));
    if (cmp != 0) return cmp;
    cmp = Double.compare(b.score(), a.score());
    if (cmp != 0) return cmp;
    cmp = Integer.compare(a.head().start(), b.head().start());
    if (cmp != 0) return cmp;
    return Integer.compare(a.tail().start(), b.tail().start());
  }

  private static int distance(Edge e) {
    return Math.max(
      Math.max(
        e.head().start() - e.tail().end(),
        e.tail().start() - e.head().end()
      ),
      0
    );
  }
}
