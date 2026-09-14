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
import static org.assertj.core.api.Assertions.within;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.gravitee.lab.gliner4j.GLiNER4jNER;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Parity of the llama.cpp + ggml streaming-span pipeline with the gliner Python library on the
 * entities and session snapshots the export script wrote into {@code reference.json}. Skipped
 * unless {@code task stream-pii} has produced {@code models/gliner-stream-pii-onnx}.
 */
@EnabledIf("bundleExists")
class StreamingSpanIntegrationTest {

  private static final Path MODEL_DIR = Path.of(
    "../models/gliner-stream-pii-onnx"
  );
  private static final float SCORE_TOLERANCE = 0.05f; // q8_0 backbone vs fp32 PyTorch

  static boolean bundleExists() {
    return (
      Files.exists(MODEL_DIR.resolve("gguf/backbone-q8_0.gguf")) &&
      Files.exists(MODEL_DIR.resolve("gguf/scorer.gguf"))
    );
  }

  private static StreamingSpanNer ner;
  private static JsonNode reference;

  @BeforeAll
  static void load() throws Exception {
    ner = StreamingSpanNer.load(MODEL_DIR);
    reference = new ObjectMapper().readTree(
      MODEL_DIR.resolve("reference.json").toFile()
    );
  }

  @AfterAll
  static void close() {
    if (ner != null) ner.close();
  }

  private static List<String> labels(JsonNode node) {
    var out = new ArrayList<String>();
    node.forEach(n -> out.add(n.asText()));
    return out;
  }

  private static void assertMatches(
    List<EntitySpan> got,
    JsonNode expected,
    String context
  ) {
    assertThat(got).as("%s: %s", context, got).hasSize(expected.size());
    for (int i = 0; i < expected.size(); i++) {
      var e = expected.get(i);
      var g = got.get(i);
      assertThat(g.text())
        .as("%s [%d] text", context, i)
        .isEqualTo(e.get("text").asText());
      assertThat(g.type())
        .as("%s [%d] label", context, i)
        .isEqualTo(e.get("label").asText());
      assertThat(g.start())
        .as("%s [%d] start", context, i)
        .isEqualTo(e.get("start").asInt());
      assertThat(g.end())
        .as("%s [%d] end", context, i)
        .isEqualTo(e.get("end").asInt());
      assertThat(g.confidence())
        .as("%s [%d] score", context, i)
        .isCloseTo((float) e.get("score").asDouble(), within(SCORE_TOLERANCE));
    }
  }

  @Test
  void statelessMatchesPythonReference() {
    for (var entry : reference.get("stateless")) {
      var got = ner.extract(
        entry.get("text").asText(),
        labels(entry.get("labels")),
        0.5f
      );
      assertMatches(got, entry.get("entities"), entry.get("text").asText());
    }
  }

  @Test
  void sessionSnapshotsMatchPythonReference() {
    int s = 0;
    for (var entry : reference.get("sessions")) {
      try (
        var session = ner.openSession("ref-" + s++, labels(entry.get("labels")))
      ) {
        var chunks = entry.get("chunks");
        var snapshots = entry.get("snapshots");
        for (int i = 0; i < chunks.size(); i++) {
          var got = session.append(chunks.get(i).asText(), 0.5f);
          assertMatches(
            got,
            snapshots.get(i),
            "session " +
              s +
              " after chunk " +
              i +
              " '" +
              chunks.get(i).asText() +
              "'"
          );
        }
      }
    }
  }

  @Test
  void facadeDispatchesThroughServiceLoader() {
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("email address"),
      new EntityDefinition("phone number")
    );
    try (var gliner = GLiNER4jNER.load(MODEL_DIR, entities)) {
      var results = gliner.extract(
        "Customer Alice Johnson can be reached at alice@example.com or +1 202-555-0147.",
        0.5f
      );
      assertThat(results.get("person"))
        .extracting(EntitySpan::text)
        .containsExactly("Alice Johnson");
      assertThat(results.get("email address"))
        .extracting(EntitySpan::text)
        .containsExactly("alice@example.com");
      assertThat(results.get("phone number"))
        .extracting(EntitySpan::text)
        .containsExactly("+1 202-555-0147");
    }
  }
}
