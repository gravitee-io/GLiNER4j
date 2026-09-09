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

import static org.assertj.core.api.Assertions.assertThat;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The ggml DeBERTa-v3 encoder vs HF {@code last_hidden_state} on the rows the export script
 * recorded (5 / 131 / 512 tokens — the last two cross the 128-token identity range of the
 * log-bucketed relative positions). Skipped unless
 * {@code uv run scripts/export_llamacpp.py deberta-encoder --model-path microsoft/deberta-v3-small
 * --output-dir models/deberta-v3-small-ggml} has run.
 */
@EnabledIf("bundleExists")
class GgmlDebertaV3ParityTest {

  private static final Path DIR = Path.of("../models/deberta-v3-small-ggml");

  static boolean bundleExists() {
    return Files.exists(DIR.resolve("gguf/encoder.gguf"));
  }

  record Entry(
    String name,
    List<Long> input_ids,
    Map<String, List<Double>> rows,
    double max_abs
  ) {}

  @Test
  void paddedBatchMatchesSingleRows() throws Exception {
    var reference = List.of(
      new ObjectMapper().readValue(
        DIR.resolve("reference.json").toFile(),
        Entry[].class
      )
    );
    var rows = new long[4][];
    rows[0] = reference
      .get(0)
      .input_ids()
      .stream()
      .mapToLong(Long::longValue)
      .toArray(); // 5 tokens
    rows[1] = reference
      .get(1)
      .input_ids()
      .stream()
      .mapToLong(Long::longValue)
      .toArray(); // 131 tokens
    rows[2] = java.util.Arrays.copyOf(
      reference
        .get(2)
        .input_ids()
        .stream()
        .mapToLong(Long::longValue)
        .toArray(),
      200
    );
    rows[3] = java.util.Arrays.copyOf(rows[1], 77);
    for (boolean gpu : new boolean[] { true, false }) {
      try (
        var enc = new GgmlDebertaEncoder(
          DIR.resolve("gguf/encoder.gguf"),
          gpu,
          4
        )
      ) {
        var batched = enc.encodeBatch(rows);
        for (int r = 0; r < 4; r++) {
          var single = enc.encode(rows[r]);
          double worst = 0;
          for (int i = 0; i < single.length; i++) for (
            int d = 0;
            d < single[i].length;
            d++
          ) {
            worst = Math.max(worst, Math.abs(single[i][d] - batched[r][i][d]));
          }
          System.out.printf(
            "batch row %d (n=%d) gpu=%b max|single-batched|=%.5f%n",
            r,
            rows[r].length,
            gpu,
            worst
          );
          assertThat(worst).as("row %d gpu=%b", r, gpu).isLessThan(0.02);
        }
      }
    }
  }

  @Test
  void matchesHuggingFaceHiddenStates() throws Exception {
    var reference = List.of(
      new ObjectMapper().readValue(
        DIR.resolve("reference.json").toFile(),
        Entry[].class
      )
    );
    for (boolean gpu : new boolean[] { true, false }) {
      try (
        var enc = new GgmlDebertaEncoder(
          DIR.resolve("gguf/encoder.gguf"),
          gpu,
          4
        )
      ) {
        for (var e : reference) {
          var ids = e.input_ids().stream().mapToLong(Long::longValue).toArray();
          var h = enc.encode(ids);
          long t0 = System.nanoTime();
          enc.encode(ids); // warm timing (first call pays shader/graph setup)
          double ms = (System.nanoTime() - t0) / 1e6;
          double worst = 0;
          for (var row : e.rows().entrySet()) {
            int i = Integer.parseInt(row.getKey());
            for (int d = 0; d < h[i].length; d++) {
              worst = Math.max(
                worst,
                Math.abs(h[i][d] - row.getValue().get(d))
              );
            }
          }
          System.out.printf(
            "deberta-v3-small %s n=%d gpu=%b: max|diff|=%.5f (max|h|=%.2f) %.1f ms%n",
            e.name(),
            ids.length,
            gpu,
            worst,
            e.max_abs(),
            ms
          );
          // f16 weights on both backends; tolerance relative to the activation scale
          assertThat(worst)
            .as("%s gpu=%b", e.name(), gpu)
            .isLessThan(0.02 * Math.max(1.0, e.max_abs()));
        }
      }
    }
  }
}
