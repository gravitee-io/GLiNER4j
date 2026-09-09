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

import io.gravitee.lab.gliner4j.GLiNER4jClassifier;
import io.gravitee.lab.gliner4j.GLiNER4jNER;
import io.gravitee.lab.gliner4j.extractor.RelationExtractor;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * GLiNER2.5 on the ggml engine vs the ONNX engine (entities, relations, classification). Skipped
 * unless {@code models/gliner2dot5-small-llamacpp} and {@code models/gliner2dot5-small-onnx} exist.
 */
@EnabledIf("bundlesExist")
class LlamaGliner2dot5IntegrationTest {

  private static final Path GGML = Path.of(
    "../models/gliner2dot5-small-llamacpp"
  );
  private static final Path ONNX = Path.of("../models/gliner2dot5-small-onnx");

  static boolean bundlesExist() {
    return (
      Files.exists(GGML.resolve("gguf/model.gguf")) &&
      Files.exists(ONNX.resolve("onnx/ner_full.onnx"))
    );
  }

  static final List<String> TEXTS = List.of(
    "John Smith works at Acme Corp in Berlin since 2019.",
    "Contact Dr. Maria Lopez (maria.lopez@clinic.org, +34 600 123 456) at Hospital del Mar before 12 March 2025.",
    "Apple unveiled the iPhone 16 in Cupertino, and Tim Cook said sales in China rose 8% last quarter; Google and Microsoft followed with their own launches in Seattle and Mountain View.",
    "John works for Apple and lives in San Francisco. Mary works for Google.",
    "The committee met on Tuesday to review the quarterly figures, which had come in well below the forecast published in March; several members argued for a revised outlook while others preferred to wait for the audited statements due next month. " +
      "In the northern valleys the snow rarely melts before May, and the roads that link the villages stay closed for most of the winter, so supplies are brought in by sled or, when the weather allows, by a small aircraft that lands on the frozen lake. " +
      "Our engineers in Toulouse and Montreal shipped the new release on 3 September 2025 after Airbus and Bombardier signed the framework agreement with Siemens."
  );

  static List<EntityDefinition> defs(List<String> names) {
    return names.stream().map(EntityDefinition::new).toList();
  }

  static Map<String, Float> flatten(Map<String, List<EntitySpan>> byType) {
    var out = new TreeMap<String, Float>();
    byType.forEach((type, spans) ->
      spans.forEach(s ->
        out.put(s.start() + "-" + s.end() + ":" + type, s.confidence())
      )
    );
    return out;
  }

  static List<String> renderRelations(
    Map<String, List<RelationInstance>> rels
  ) {
    var out = new ArrayList<String>();
    rels.forEach((type, list) ->
      list.forEach(r -> {
        var sb = new StringBuilder(type);
        new TreeMap<>(r.fields()).forEach((f, span) ->
          sb
            .append(' ')
            .append(f)
            .append('=')
            .append(span.start())
            .append('-')
            .append(span.end())
        );
        out.add(sb.toString());
      })
    );
    Collections.sort(out);
    return out;
  }

  @Test
  void entitiesMatchOnnxEngine() {
    var entities = defs(
      List.of(
        "person",
        "organization",
        "location",
        "date",
        "product",
        "email",
        "phone number",
        "percentage"
      )
    );
    try (
      var a = GLiNER4jNER.load(GGML, entities);
      var b = GLiNER4jNER.load(ONNX, entities)
    ) {
      for (var text : TEXTS) {
        var x = flatten(a.extract(text, 0.3f));
        var y = flatten(b.extract(text, 0.3f));
        assertThat(x.keySet()).as(text).isEqualTo(y.keySet());
        for (var k : x.keySet())
          assertThat(x.get(k))
            .as("%s / %s", text, k)
            .isCloseTo(y.get(k), within(0.03f));
      }
      var batch = a.extractBatch(TEXTS, 0.3f);
      for (int i = 0; i < TEXTS.size(); i++) {
        assertThat(flatten(batch.get(i)).keySet())
          .as("batch " + TEXTS.get(i))
          .isEqualTo(flatten(a.extract(TEXTS.get(i), 0.3f)).keySet());
      }
    }
  }

  @Test
  void relationsMatchOnnxEngine() {
    var relations = List.of(
      new RelationDefinition("works_for", "", List.of("head", "tail")),
      new RelationDefinition("lives_in", "", List.of("head", "tail"))
    );
    try (
      var a = RelationExtractor.load(GGML, relations);
      var b = RelationExtractor.load(ONNX, relations)
    ) {
      for (var text : TEXTS) {
        assertThat(renderRelations(a.extract(text)))
          .as(text)
          .isEqualTo(renderRelations(b.extract(text)));
      }
    }
  }

  @Test
  void classificationMatchesOnnxEngine() {
    var labels = List.of(
      new ClassificationLabel("technology"),
      new ClassificationLabel("business"),
      new ClassificationLabel("healthcare"),
      new ClassificationLabel("sports")
    );
    try (
      var a = GLiNER4jClassifier.load(GGML, labels);
      var b = GLiNER4jClassifier.load(ONNX, labels)
    ) {
      for (var text : TEXTS) {
        var x = a
          .classify(text, 0.0f)
          .stream()
          .collect(
            Collectors.toMap(
              ClassificationResult::label,
              ClassificationResult::confidence
            )
          );
        var y = b
          .classify(text, 0.0f)
          .stream()
          .collect(
            Collectors.toMap(
              ClassificationResult::label,
              ClassificationResult::confidence
            )
          );
        for (var k : y.keySet())
          assertThat(x.get(k))
            .as("%s / %s", text, k)
            .isCloseTo(y.get(k), within(0.03f));
      }
    }
  }
}
