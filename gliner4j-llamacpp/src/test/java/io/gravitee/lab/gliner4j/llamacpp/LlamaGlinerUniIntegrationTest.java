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

import com.fasterxml.jackson.databind.ObjectMapper;
import io.gravitee.lab.gliner4j.GLiNER4jNER;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Original-GLiNER uni-encoder on the ggml engine vs the gliner-library predictions recorded by the
 * export script, and vs the ONNX engine when that bundle is present. Skipped unless
 * {@code task gliner-pii:llamacpp} has produced {@code models/gliner-pii-base-llamacpp}.
 */
@EnabledIf("bundleExists")
class LlamaGlinerUniIntegrationTest {

  /** ggml bundle → ONNX bundle: uni markerV0 (pii), bi-encoder (MiniLM labels), token_level (multitask-large). */
  static java.util.stream.Stream<
    org.junit.jupiter.params.provider.Arguments
  > bundles() {
    return java.util.stream.Stream.of(
      org.junit.jupiter.params.provider.Arguments.of(
        "../models/gliner-pii-base-llamacpp",
        "../models/gliner-pii-base-onnx"
      ),
      org.junit.jupiter.params.provider.Arguments.of(
        "../models/gliner-bi-small-llamacpp",
        "../models/gliner-bi-small-onnx"
      ),
      org.junit.jupiter.params.provider.Arguments.of(
        "../models/gliner-multitask-large-llamacpp",
        "../models/gliner-multitask-large-onnx"
      )
    ).filter(a ->
      Files.exists(Path.of((String) a.get()[0]).resolve("gguf/model.gguf"))
    );
  }

  static boolean bundleExists() {
    return bundles().findAny().isPresent();
  }

  record RefEntity(
    int start,
    int end,
    String text,
    String label,
    double score
  ) {}

  record Entry(
    String text,
    List<String> labels,
    float threshold,
    List<RefEntity> entities
  ) {}

  static List<EntityDefinition> defs(List<String> names) {
    return names.stream().map(EntityDefinition::new).toList();
  }

  /** (start,end,label) → confidence */
  static Map<String, Float> flatten(Map<String, List<EntitySpan>> byType) {
    var out = new TreeMap<String, Float>();
    byType.forEach((type, spans) ->
      spans.forEach(s ->
        out.put(s.start() + "-" + s.end() + ":" + type, s.confidence())
      )
    );
    return out;
  }

  @org.junit.jupiter.params.ParameterizedTest(name = "{0}")
  @org.junit.jupiter.params.provider.MethodSource("bundles")
  void matchesGlinerLibraryPredictions(String ggmlDir, String onnxDir)
    throws Exception {
    var dir = Path.of(ggmlDir);
    var reference = List.of(
      new ObjectMapper().readValue(
        dir.resolve("reference.json").toFile(),
        Entry[].class
      )
    );
    assertThat(reference).isNotEmpty();
    try (var ner = GLiNER4jNER.load(dir, defs(List.of("person")))) {
      for (var e : reference) {
        var got = flatten(
          ner.extract(e.text(), defs(e.labels()), e.threshold())
        );
        var expected = new TreeMap<String, Double>();
        for (var r : e.entities())
          expected.put(r.start() + "-" + r.end() + ":" + r.label(), r.score());
        assertThat(got.keySet()).as(e.text()).isEqualTo(expected.keySet());
        for (var k : expected.keySet()) {
          assertThat(got.get(k))
            .as("%s / %s", e.text(), k)
            .isCloseTo(expected.get(k).floatValue(), within(0.03f));
        }
      }
    }
  }

  @org.junit.jupiter.params.ParameterizedTest(name = "{0}")
  @org.junit.jupiter.params.provider.MethodSource("bundles")
  void matchesOnnxEngine(String ggmlDir, String onnxDir) throws Exception {
    var reference = List.of(
      new ObjectMapper().readValue(
        Path.of(ggmlDir).resolve("reference.json").toFile(),
        Entry[].class
      )
    );
    org.junit.jupiter.api.Assumptions.assumeTrue(
      Files.exists(Path.of(onnxDir).resolve("onnx/model.onnx")),
      "no ONNX bundle " + onnxDir
    );
    try (
      var ner = GLiNER4jNER.load(Path.of(ggmlDir), defs(List.of("person")));
      var onnx = GLiNER4jNER.load(Path.of(onnxDir), defs(List.of("person")))
    ) {
      var texts = new ArrayList<String>();
      for (var e : reference) texts.add(e.text());
      texts.add(
        "Dr. Alice Wong (alice.wong@hospital.org, +1 415 555 0199) prescribed 20 mg of atorvastatin to Robert Miller, 67, of 14 Elm Street, San Jose, on March 3rd; the follow-up at Kaiser Permanente is booked for April 1st with card 4111 1111 1111 1111 on file."
      );
      var labels = defs(
        List.of(
          "person",
          "email",
          "phone number",
          "address",
          "city",
          "organization",
          "date",
          "medication",
          "credit card number"
        )
      );
      for (var text : texts) {
        var a = flatten(ner.extract(text, labels, 0.3f));
        var b = flatten(onnx.extract(text, labels, 0.3f));
        assertThat(a.keySet()).as(text).isEqualTo(b.keySet());
        for (var k : a.keySet()) {
          assertThat(a.get(k))
            .as("%s / %s", text, k)
            .isCloseTo(b.get(k), within(0.03f));
        }
      }
    }
  }
}
