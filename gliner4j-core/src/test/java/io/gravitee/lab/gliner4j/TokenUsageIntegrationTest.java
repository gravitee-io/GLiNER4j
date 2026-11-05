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
package io.gravitee.lab.gliner4j;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.processor.TextPreprocessing;
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
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Pins {@code inputTokens} against real bundles, one test per architecture: the count must equal
 * the prompt row rebuilt from the bundle's own tokenizer, and a batch must report what each text
 * reports alone. Each test runs only when its bundle is present under the repository-root
 * {@code models/} directory the Taskfile exports to (resolved as {@code ../models/} from the
 * module directory Surefire runs in).
 */
class TokenUsageIntegrationTest {

  private static final Path GLINER2 = Path.of("../models/gliner2-base-onnx");
  private static final Path GLINER2DOT5 = Path.of(
    "../models/gliner2dot5-small-onnx"
  );
  private static final Path GLINER_UNI = Path.of(
    "../models/gliner-pii-base-onnx"
  );
  private static final Path GLINER_UNI_TOKEN = Path.of(
    "../models/gliner-multitask-large-onnx"
  );
  private static final Path GLINER_BI = Path.of(
    "../models/gliner-bi-small-onnx"
  );
  private static final Path GLICLASS = Path.of("../models/gliclass-edge-onnx");

  private static final String TEXT =
    "John Smith works at Microsoft in Seattle and emails john@example.com";
  private static final String SHORT_TEXT = "Marie Curie lived in Paris.";
  private static final List<String> BATCH = Arrays.asList(
    TEXT,
    "",
    SHORT_TEXT,
    null
  );

  private static final List<EntityDefinition> ENTITIES = List.of(
    new EntityDefinition("person"),
    new EntityDefinition("organization"),
    new EntityDefinition("location")
  );
  private static final List<EntityDefinition> OVERRIDE_ENTITIES = List.of(
    new EntityDefinition("email address")
  );
  private static final List<ClassificationLabel> LABELS = List.of(
    new ClassificationLabel("technology"),
    new ClassificationLabel("sports"),
    new ClassificationLabel("politics")
  );
  private static final List<ClassificationLabel> OVERRIDE_LABELS = List.of(
    new ClassificationLabel("business")
  );

  static boolean hasGliner2Ner() {
    return Files.exists(GLINER2.resolve("onnx/ner_full.onnx"));
  }

  static boolean hasGliner2Classifier() {
    return Files.exists(GLINER2.resolve("onnx/classifier_full.onnx"));
  }

  static boolean hasGliner2dot5Ner() {
    return Files.exists(GLINER2DOT5.resolve("onnx/ner_full.onnx"));
  }

  static boolean hasGliner2dot5Classifier() {
    return Files.exists(GLINER2DOT5.resolve("onnx/classifier_full.onnx"));
  }

  static boolean hasGlinerUni() {
    return Files.exists(GLINER_UNI.resolve("onnx/model.onnx"));
  }

  static boolean hasGlinerUniToken() {
    return Files.exists(GLINER_UNI_TOKEN.resolve("onnx/model.onnx"));
  }

  static boolean hasGlinerBi() {
    return (
      Files.exists(GLINER_BI.resolve("onnx/model.onnx")) &&
      Files.exists(GLINER_BI.resolve("labels_tokenizer/tokenizer.json"))
    );
  }

  static boolean hasGliclass() {
    return Files.exists(GLICLASS.resolve("onnx/encoder.onnx"));
  }

  // ---- NER -----------------------------------------------------------------

  @Test
  @EnabledIf("hasGliner2Ner")
  void gliner2Ner() {
    var tokenizer = new DjlTokenizerWrapper(GLINER2);
    assertNer(GLINER2, entities ->
      gliner2Row(
        tokenizer,
        "entities",
        "[E]",
        names(entities),
        words(TextPreprocessing.DEFAULT, TEXT)
      )
    );
  }

  @Test
  @EnabledIf("hasGliner2dot5Ner")
  void gliner2dot5Ner() {
    var tokenizer = new DjlTokenizerWrapper(GLINER2DOT5);
    var preprocessing = TextPreprocessing.whitespaceSplitter(
      GLiNER4jConfig.load(GLINER2DOT5)
    );
    assertNer(GLINER2DOT5, entities ->
      gliner2Row(
        tokenizer,
        "entities",
        "[E]",
        names(entities),
        words(preprocessing, TEXT)
      )
    );
  }

  @Test
  @EnabledIf("hasGlinerUni")
  void glinerUniNer() {
    assertNer(GLINER_UNI, entities -> glinerUniRow(GLINER_UNI, entities));
  }

  @Test
  @EnabledIf("hasGlinerUniToken")
  void glinerUniTokenNer() {
    assertNer(GLINER_UNI_TOKEN, entities ->
      glinerUniRow(GLINER_UNI_TOKEN, entities)
    );
  }

  @Test
  @EnabledIf("hasGlinerBi")
  void glinerBiNer() {
    var tokenizer = new DjlTokenizerWrapper(GLINER_BI);
    var labelTokenizer = new DjlTokenizerWrapper(
      GLINER_BI.resolve("labels_tokenizer")
    );
    assertNer(GLINER_BI, entities -> {
      // [CLS] words [SEP] + one unpadded label-encoder row per entity type.
      int count = 2 + wordTokens(tokenizer, splitWords(TEXT));
      for (var entity : entities) {
        count += labelTokenizer.encodeWithSpecialTokens(entity.name()).length;
      }
      return count;
    });
  }

  // ---- classification ------------------------------------------------------

  @Test
  @EnabledIf("hasGliner2Classifier")
  void gliner2Classification() {
    var tokenizer = new DjlTokenizerWrapper(GLINER2);
    var preprocessing = TextPreprocessing.gliner2(GLiNER4jConfig.load(GLINER2));
    assertClassification(GLINER2, labels ->
      gliner2Row(
        tokenizer,
        "classify",
        "[L]",
        labelNames(labels),
        words(preprocessing, TEXT)
      )
    );
  }

  @Test
  @EnabledIf("hasGliner2dot5Classifier")
  void gliner2dot5Classification() {
    var tokenizer = new DjlTokenizerWrapper(GLINER2DOT5);
    var preprocessing = TextPreprocessing.whitespaceSplitter(
      GLiNER4jConfig.load(GLINER2DOT5)
    );
    assertClassification(GLINER2DOT5, labels ->
      gliner2Row(
        tokenizer,
        "classify",
        "[L]",
        labelNames(labels),
        words(preprocessing, TEXT)
      )
    );
  }

  @Test
  @EnabledIf("hasGliclass")
  void gliclassClassification() {
    var config = GLiNER4jConfig.load(GLICLASS);
    var tokenizer = new DjlTokenizerWrapper(GLICLASS);
    var labelToken = config.archString("label_token", "<<LABEL>>");
    var sepToken = config.archString("sep_token", "<<SEP>>");
    boolean promptFirst = config.archBoolean("prompt_first", true);
    assertClassification(GLICLASS, labels -> {
      var sb = new StringBuilder();
      for (var label : labels) {
        sb.append(labelToken).append(label.name());
      }
      sb.append(sepToken);
      var prompt = promptFirst ? sb + TEXT : TEXT + sb;
      return tokenizer.encodeWithSpecialTokens(prompt).length;
    });
  }

  // ---- shared assertions ---------------------------------------------------

  private interface ExpectedNer {
    int forEntities(List<EntityDefinition> entities);
  }

  private interface ExpectedClassification {
    int forLabels(List<ClassificationLabel> labels);
  }

  private static void assertNer(Path modelDir, ExpectedNer expected) {
    try (var ner = GLiNER4jNER.load(modelDir, ENTITIES)) {
      var single = ner.extractWithUsage(TEXT, 0.5f);
      assertThat(single.inputTokens()).isEqualTo(
        expected.forEntities(ENTITIES)
      );
      assertThat(single.spans()).isEqualTo(ner.extract(TEXT, 0.5f));

      var override = ner.extractWithUsage(TEXT, OVERRIDE_ENTITIES, 0.5f);
      assertThat(override.inputTokens()).isEqualTo(
        expected.forEntities(OVERRIDE_ENTITIES)
      );
      assertThat(override.spans()).isEqualTo(
        ner.extract(TEXT, OVERRIDE_ENTITIES, 0.5f)
      );

      var batch = ner.extractBatchWithUsage(BATCH, 0.5f);
      assertThat(batch).hasSize(BATCH.size());
      for (int i = 0; i < BATCH.size(); i++) {
        assertThat(batch.get(i).inputTokens()).isEqualTo(
          ner.extractWithUsage(BATCH.get(i), 0.5f).inputTokens()
        );
      }
      assertThat(batch.get(0).inputTokens()).isEqualTo(single.inputTokens());
      assertThat(batch.get(1)).isEqualTo(NerOutput.empty());
      assertThat(batch.get(3)).isEqualTo(NerOutput.empty());
      assertThat(batch.get(2).inputTokens()).isPositive();
      assertThat(batch.stream().map(NerOutput::spans).toList()).isEqualTo(
        ner.extractBatch(BATCH, 0.5f)
      );
    }
  }

  private static void assertClassification(
    Path modelDir,
    ExpectedClassification expected
  ) {
    try (var classifier = GLiNER4jClassifier.load(modelDir, LABELS)) {
      var single = classifier.classifyWithUsage(TEXT, 0.3f);
      assertThat(single.inputTokens()).isEqualTo(expected.forLabels(LABELS));
      assertThat(single.results()).isEqualTo(classifier.classify(TEXT, 0.3f));

      var override = classifier.classifyWithUsage(TEXT, OVERRIDE_LABELS, 0.3f);
      assertThat(override.inputTokens()).isEqualTo(
        expected.forLabels(OVERRIDE_LABELS)
      );

      var batch = classifier.classifyBatchWithUsage(BATCH, 0.3f);
      assertThat(batch).hasSize(BATCH.size());
      for (int i = 0; i < BATCH.size(); i++) {
        assertThat(batch.get(i).inputTokens()).isEqualTo(
          classifier.classifyWithUsage(BATCH.get(i), 0.3f).inputTokens()
        );
      }
      assertThat(batch.get(0).inputTokens()).isEqualTo(single.inputTokens());
      assertThat(batch.get(1)).isEqualTo(ClassificationOutput.empty());
      assertThat(batch.get(3)).isEqualTo(ClassificationOutput.empty());
      assertThat(
        batch
          .stream()
          .map(ClassificationOutput::results)
          .map(List::size)
          .toList()
      ).isEqualTo(
        classifier.classifyBatch(BATCH, 0.3f).stream().map(List::size).toList()
      );
    }
  }

  // ---- expected-row builders -----------------------------------------------

  /** {@code ( [P] task ( marker n1 marker n2 ... ) ) [SEP_TEXT] words}, no special tokens. */
  private static int gliner2Row(
    DjlTokenizerWrapper tokenizer,
    String task,
    String marker,
    List<String> names,
    List<String> words
  ) {
    var schema = new ArrayList<String>(List.of("(", "[P]", task, "("));
    for (var name : names) {
      schema.add(marker);
      schema.add(name);
    }
    schema.add(")");
    schema.add(")");
    schema.add("[SEP_TEXT]");
    return wordTokens(tokenizer, schema) + wordTokens(tokenizer, words);
  }

  /** {@code [CLS] <<ENT>> t1 ... <<SEP>> words [SEP]}. */
  private static int glinerUniRow(
    Path modelDir,
    List<EntityDefinition> entities
  ) {
    var config = GLiNER4jConfig.load(modelDir);
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var prompt = new ArrayList<String>();
    for (var entity : entities) {
      prompt.add(config.archString("ent_token", "<<ENT>>"));
      prompt.add(entity.name());
    }
    prompt.add(config.archString("sep_token", "<<SEP>>"));
    return (
      2 +
      wordTokens(tokenizer, prompt) +
      wordTokens(tokenizer, splitWords(TEXT))
    );
  }

  private static int wordTokens(
    DjlTokenizerWrapper tokenizer,
    List<String> words
  ) {
    int count = 0;
    for (var word : words) {
      count += tokenizer.tokenizeWithIds(word).ids().length;
    }
    return count;
  }

  private static List<String> words(
    TextPreprocessing preprocessing,
    String text
  ) {
    TextEncoder encoder = preprocessing.encode(text);
    return encoder.getWords();
  }

  private static List<String> splitWords(String text) {
    return WhitespaceWordSplitter.split(text)
      .stream()
      .map(WhitespaceWordSplitter.Word::text)
      .toList();
  }

  private static List<String> names(List<EntityDefinition> entities) {
    return entities.stream().map(EntityDefinition::name).toList();
  }

  private static List<String> labelNames(List<ClassificationLabel> labels) {
    return labels.stream().map(ClassificationLabel::name).toList();
  }
}
