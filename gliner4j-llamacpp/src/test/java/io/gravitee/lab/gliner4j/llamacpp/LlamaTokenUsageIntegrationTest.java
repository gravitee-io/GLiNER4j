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

import io.gravitee.lab.gliner4j.GLiNER4jClassifier;
import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.GLiNER4jNER;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationOutput;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.NerOutput;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.utils.WhitespaceWordSplitter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Pins {@code inputTokens} of the llama.cpp strategies against real bundles: the count must equal
 * the sequence rebuilt from the bundle's own tokenizer. Each test runs only when its bundle is
 * present under {@code ../models/}.
 */
class LlamaTokenUsageIntegrationTest {

  private static final Path STREAMING = Path.of(
    "../models/gliner-stream-pii-onnx"
  );
  private static final Path GLICLASS = Path.of(
    "../models/gliclass-edge-llamacpp"
  );
  private static final Path DECODER_KV = Path.of("../models/scx-router-onnx");

  private static final String TEXT =
    "John Smith works at Microsoft in Seattle and emails john@example.com";
  private static final List<String> BATCH = Arrays.asList(
    TEXT,
    "",
    "Marie Curie lived in Paris.",
    null
  );
  private static final List<EntityDefinition> ENTITIES = List.of(
    new EntityDefinition("person"),
    new EntityDefinition("organization"),
    new EntityDefinition("location")
  );
  private static final List<ClassificationLabel> LABELS = List.of(
    new ClassificationLabel("technology"),
    new ClassificationLabel("sports"),
    new ClassificationLabel("politics")
  );

  static Stream<String> uniBundles() {
    return Stream.of(
      "../models/gliner-pii-base-llamacpp",
      "../models/gliner-bi-small-llamacpp",
      "../models/gliner-multitask-large-llamacpp"
    ).filter(d -> Files.exists(Path.of(d).resolve("gguf/model.gguf")));
  }

  static boolean hasUniBundle() {
    return uniBundles().findAny().isPresent();
  }

  static boolean hasStreaming() {
    return (
      Files.exists(STREAMING.resolve("gguf/backbone-q8_0.gguf")) &&
      Files.exists(STREAMING.resolve("gguf/scorer.gguf"))
    );
  }

  static boolean hasGliclass() {
    return Files.exists(GLICLASS.resolve("gguf/heads.gguf"));
  }

  static boolean hasDecoderKv() {
    return (
      Files.exists(DECODER_KV.resolve("gguf/backbone-q8_0.gguf")) &&
      Files.exists(DECODER_KV.resolve("onnx/scorer.onnx"))
    );
  }

  // ---- NER -----------------------------------------------------------------

  @Test
  @EnabledIf("hasStreaming")
  void streamingSpanNer() {
    var config = GLiNER4jConfig.load(STREAMING);
    var tokenizer = new DjlTokenizerWrapper(STREAMING);
    try (var ner = GLiNER4jNER.load(STREAMING, ENTITIES)) {
      // label1 <<LABEL>> label2 <<LABEL>> ... <<SEP>> word subtokens, decoded from scratch.
      int expected = ENTITIES.size() + 1;
      for (var entity : ENTITIES) {
        expected += tokens(tokenizer, entity.name());
      }
      for (var word : new TextEncoder(TEXT).getWords()) {
        expected += tokens(tokenizer, word);
      }
      assertThat(config.getArchitecture().configValue()).isEqualTo(
        "gliner-streaming-span"
      );
      var single = ner.extractWithUsage(TEXT, 0.5f);
      assertThat(single.inputTokens()).isEqualTo(expected);
      assertThat(single.spans()).isEqualTo(ner.extract(TEXT, 0.5f));
      assertBatchMatchesSingle(ner);
    }
  }

  @ParameterizedTest
  @MethodSource("uniBundles")
  @EnabledIf("hasUniBundle")
  void glinerUniAndBiNer(String bundle) {
    var dir = Path.of(bundle);
    var config = GLiNER4jConfig.load(dir);
    var tokenizer = new DjlTokenizerWrapper(dir);
    var labelsGguf = config.archString("labels_gguf", null);
    var words = WhitespaceWordSplitter.split(TEXT)
      .stream()
      .map(WhitespaceWordSplitter.Word::text)
      .toList();
    int textRow = 2;
    for (var word : words) textRow += tokens(tokenizer, word);

    try (var ner = GLiNER4jNER.load(dir, ENTITIES)) {
      if (labelsGguf == null) {
        int expected =
          textRow +
          tokens(tokenizer, config.archString("sep_token", "<<SEP>>"));
        for (var entity : ENTITIES) {
          expected +=
            tokens(tokenizer, config.archString("ent_token", "<<ENT>>")) +
            tokens(tokenizer, entity.name());
        }
        assertThat(ner.extractWithUsage(TEXT, 0.5f).inputTokens()).isEqualTo(
          expected
        );
        assertBatchMatchesSingle(ner);
      } else {
        // Bi-encoder: label rows count only on the call that encodes them (label cache).
        var labelTokenizer = new DjlTokenizerWrapper(
          dir.resolve("labels_tokenizer")
        );
        int labelRows = 0;
        for (var entity : ENTITIES) {
          labelRows += labelTokenizer.encodeWithSpecialTokens(
            entity.name()
          ).length;
        }
        assertThat(ner.extractWithUsage(TEXT, 0.5f).inputTokens()).isEqualTo(
          textRow + labelRows
        );
        assertThat(ner.extractWithUsage(TEXT, 0.5f).inputTokens()).isEqualTo(
          textRow
        );
      }
    }
  }

  // ---- classification ------------------------------------------------------

  @Test
  @EnabledIf("hasGliclass")
  void gliclassClassification() {
    var config = GLiNER4jConfig.load(GLICLASS);
    var tokenizer = new DjlTokenizerWrapper(GLICLASS);
    var sb = new StringBuilder();
    for (var label : LABELS) {
      sb
        .append(config.archString("label_token", "<<LABEL>>"))
        .append(label.name());
    }
    sb.append(config.archString("sep_token", "<<SEP>>"));
    var prompt = config.archBoolean("prompt_first", true)
      ? sb + TEXT
      : TEXT + sb;
    try (var classifier = GLiNER4jClassifier.load(GLICLASS, LABELS)) {
      var single = classifier.classifyWithUsage(TEXT, 0.3f);
      assertThat(single.inputTokens()).isEqualTo(
        tokenizer.encodeWithSpecialTokens(prompt).length
      );
      assertBatchMatchesSingle(classifier);
    }
  }

  @Test
  @EnabledIf("hasDecoderKv")
  void decoderKvClassification() {
    var config = GLiNER4jConfig.load(DECODER_KV);
    var tokenizer = new DjlTokenizerWrapper(DECODER_KV);
    var labelToken = config.archString("label_token", "<<LABEL>>");
    var sepToken = config.archString("sep_token", "<<SEP>>");
    var section = new StringBuilder(sepToken);
    for (var label : LABELS) section.append(label.name()).append(labelToken);
    section.append(sepToken);
    try (var classifier = GLiNER4jClassifier.load(DECODER_KV, LABELS)) {
      var single = classifier.classifyWithUsage(TEXT, 0.3f);
      assertThat(single.inputTokens()).isEqualTo(
        tokenizer.encodeWithSpecialTokens(TEXT).length +
          tokenizer.encodeWithSpecialTokens(section.toString()).length
      );
      assertBatchMatchesSingle(classifier);
    }
  }

  // ---- helpers -------------------------------------------------------------

  private static int tokens(DjlTokenizerWrapper tokenizer, String word) {
    return tokenizer.tokenizeWithIds(word).ids().length;
  }

  private static void assertBatchMatchesSingle(GLiNER4jNER ner) {
    var batch = ner.extractBatchWithUsage(BATCH, 0.5f);
    var counts = new ArrayList<Integer>();
    for (var text : BATCH) {
      counts.add(ner.extractWithUsage(text, 0.5f).inputTokens());
    }
    assertThat(batch).extracting(NerOutput::inputTokens).isEqualTo(counts);
    assertThat(batch.get(1)).isEqualTo(NerOutput.empty());
    assertThat(batch.get(3)).isEqualTo(NerOutput.empty());
  }

  private static void assertBatchMatchesSingle(GLiNER4jClassifier classifier) {
    var batch = classifier.classifyBatchWithUsage(BATCH, 0.3f);
    var counts = new ArrayList<Integer>();
    for (var text : BATCH) {
      counts.add(classifier.classifyWithUsage(text, 0.3f).inputTokens());
    }
    assertThat(batch)
      .extracting(ClassificationOutput::inputTokens)
      .isEqualTo(counts);
    assertThat(batch.get(1)).isEqualTo(ClassificationOutput.empty());
    assertThat(batch.get(3)).isEqualTo(ClassificationOutput.empty());
  }
}
