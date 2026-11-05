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
package io.gravitee.lab.gliner4j.strategy;

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.Architecture;
import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.processor.TextPreprocessing;
import io.gravitee.lab.gliner4j.runtime.FlatBatchScoringResult;
import io.gravitee.lab.gliner4j.runtime.Gliner2ClassifierRuntime;
import io.gravitee.lab.gliner4j.runtime.Gliner2SpanRuntime;
import io.gravitee.lab.gliner4j.runtime.Gliner2dot5NerRuntime;
import io.gravitee.lab.gliner4j.runtime.GlinerBiSpanRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationOutput;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.NerOutput;
import io.gravitee.lab.gliner4j.strategy.gliner2.Gliner2ClassificationStrategy;
import io.gravitee.lab.gliner4j.strategy.gliner2.Gliner2NerStrategy;
import io.gravitee.lab.gliner4j.strategy.gliner2dot5.Gliner2dot5NerStrategy;
import io.gravitee.lab.gliner4j.strategy.glinerbi.GlinerBiNerStrategy;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.net.URISyntaxException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import org.junit.jupiter.api.Test;

/**
 * Pins {@code inputTokens} of the GLiNER2-family and GLiNER bi-encoder strategies without a model:
 * a word-level test tokenizer (one id per word, punctuation split off; the label-encoder variant
 * wraps each label in {@code [CLS] ... [SEP]}) and fake runtimes that record the rows they are fed. Each count must equal both the hand-computed prompt length and the non-padding length
 * of the row the runtime actually received.
 */
class TokenUsageStrategyTest {

  private static final String TEXT = "John lives in Paris";

  private static final List<EntityDefinition> ENTITIES = List.of(
    new EntityDefinition("person"),
    new EntityDefinition("location")
  );

  private static final List<ClassificationLabel> LABELS = List.of(
    new ClassificationLabel("positive"),
    new ClassificationLabel("negative")
  );

  // ( [P] entities ( [E] person [E] location ) ) = 10, [SEP_TEXT] = 1.
  private static final int NER_PREFIX = 11;
  // ( [P] classify ( [L] positive [L] negative ) ) = 10, [SEP_TEXT] = 1.
  private static final int CLS_PREFIX = 11;

  // ---- GLiNER2 NER ---------------------------------------------------------

  @Test
  void gliner2Ner_countsSchemaSeparatorAndTextTokens() {
    var runtime = new RecordingSpanRuntime();
    try (
      var strategy = Gliner2NerStrategy.create(
        ctx(RuntimeConfig.defaults(), Architecture.GLINER2),
        ENTITIES,
        runtime
      )
    ) {
      var out = strategy.extractWithUsage(TEXT, 0.5f);

      assertThat(out.inputTokens()).isEqualTo(NER_PREFIX + 4);
      assertThat(out.inputTokens()).isEqualTo(runtime.rowLengths.get(0));
      assertThat(out.spans()).isEqualTo(strategy.extract(TEXT, 0.5f));
    }
  }

  @Test
  void gliner2Ner_batchMatchesSingleAndExcludesEmptyTexts() {
    var runtime = new RecordingSpanRuntime();
    try (
      var strategy = Gliner2NerStrategy.create(
        ctx(RuntimeConfig.defaults(), Architecture.GLINER2),
        ENTITIES,
        runtime
      )
    ) {
      var texts = Arrays.asList(TEXT, "", null, "Paris.");
      var batch = strategy.extractBatchWithUsage(texts, 0.5f);

      assertThat(batch)
        .extracting(NerOutput::inputTokens)
        .containsExactly(NER_PREFIX + 4, 0, 0, NER_PREFIX + 2);
      for (int i = 0; i < texts.size(); i++) {
        assertThat(batch.get(i)).isEqualTo(
          strategy.extractWithUsage(texts.get(i), 0.5f)
        );
      }
      assertThat(batch.stream().map(NerOutput::spans).toList()).isEqualTo(
        strategy.extractBatch(texts, 0.5f)
      );
      // Rows of unequal length go through one bucket unpadded: padding is the runtime's.
      assertThat(runtime.rowLengths.subList(0, 2)).containsExactlyInAnyOrder(
        NER_PREFIX + 4,
        NER_PREFIX + 2
      );
    }
  }

  @Test
  void gliner2Ner_perCallEntitiesAndDescriptionsChangeTheCount() {
    var runtime = new RecordingSpanRuntime();
    try (
      var strategy = Gliner2NerStrategy.create(
        ctx(RuntimeConfig.defaults(), Architecture.GLINER2),
        ENTITIES,
        runtime
      )
    ) {
      // ( [P] entities ( [E] person ) ) = 8, [SEP_TEXT] = 1.
      var single = strategy.extractWithUsage(
        TEXT,
        List.of(new EntityDefinition("person")),
        0.5f
      );
      assertThat(single.inputTokens()).isEqualTo(9 + 4);

      // + [DESCRIPTION] "person:" (2) "a human" (2) = 5.
      var described = strategy.extractWithUsage(
        TEXT,
        List.of(
          new EntityDefinition("person", "a human"),
          new EntityDefinition("location")
        ),
        0.5f
      );
      assertThat(described.inputTokens()).isEqualTo(NER_PREFIX + 5 + 4);
      assertThat(runtime.rowLengths).containsExactly(13, NER_PREFIX + 5 + 4);
    }
  }

  @Test
  void gliner2Ner_microBatchedSingleCallReportsItsOwnRow() {
    var runtime = new RecordingSpanRuntime();
    var config = RuntimeConfig.builder().microBatchingEnabled(true).build();
    try (
      var strategy = Gliner2NerStrategy.create(
        ctx(config, Architecture.GLINER2),
        ENTITIES,
        runtime
      )
    ) {
      assertThat(strategy.extractWithUsage(TEXT, 0.5f).inputTokens()).isEqualTo(
        NER_PREFIX + 4
      );
      assertThat(strategy.extractWithUsage("  ", 0.5f)).isEqualTo(
        NerOutput.empty()
      );
    }
  }

  // ---- GLiNER2.5 NER -------------------------------------------------------

  @Test
  void gliner2dot5Ner_countsTheAppendedPeriod() {
    var runtime = new RecordingBoundaryRuntime();
    try (
      var strategy = Gliner2dot5NerStrategy.create(
        ctx(RuntimeConfig.defaults(), Architecture.GLINER2DOT5),
        ENTITIES,
        runtime
      )
    ) {
      // john lives in paris + appended "."
      var out = strategy.extractWithUsage(TEXT, 0.5f);
      assertThat(out.inputTokens()).isEqualTo(NER_PREFIX + 5);
      assertThat(out.inputTokens()).isEqualTo(runtime.rowLengths.get(0));

      // Already ends in "." : nothing appended.
      assertThat(
        strategy.extractWithUsage("Paris.", 0.5f).inputTokens()
      ).isEqualTo(NER_PREFIX + 2);
      assertThat(
        strategy
          .extractWithUsage(TEXT, List.of(new EntityDefinition("person")), 0.5f)
          .inputTokens()
      ).isEqualTo(9 + 5);
      assertThat(strategy.extractWithUsage(TEXT, List.of(), 0.5f)).isEqualTo(
        NerOutput.empty()
      );
    }
  }

  @Test
  void gliner2dot5Ner_batchMatchesSingle() {
    var runtime = new RecordingBoundaryRuntime();
    try (
      var strategy = Gliner2dot5NerStrategy.create(
        ctx(RuntimeConfig.defaults(), Architecture.GLINER2DOT5),
        ENTITIES,
        runtime
      )
    ) {
      var texts = Arrays.asList("Paris.", TEXT, " ");
      var batch = strategy.extractBatchWithUsage(texts, 0.5f);
      assertThat(batch)
        .extracting(NerOutput::inputTokens)
        .containsExactly(NER_PREFIX + 2, NER_PREFIX + 5, 0);
      for (int i = 0; i < texts.size(); i++) {
        assertThat(batch.get(i)).isEqualTo(
          strategy.extractWithUsage(texts.get(i), 0.5f)
        );
      }
      assertThat(batch.stream().map(NerOutput::spans).toList()).isEqualTo(
        strategy.extractBatch(texts, 0.5f)
      );
    }
  }

  // ---- GLiNER2 / GLiNER2.5 classification ----------------------------------

  @Test
  void gliner2Classification_countsSchemaSeparatorAndTextTokens() {
    var runtime = new RecordingClassifierRuntime();
    try (
      var strategy = Gliner2ClassificationStrategy.create(
        ctx(RuntimeConfig.defaults(), Architecture.GLINER2),
        LABELS,
        runtime
      )
    ) {
      var out = strategy.classifyWithUsage(TEXT, 0.5f);
      assertThat(out.inputTokens()).isEqualTo(CLS_PREFIX + 4);
      assertThat(out.inputTokens()).isEqualTo(runtime.rowLengths.get(0));
      assertThat(out.results()).isEqualTo(strategy.classify(TEXT, 0.5f));

      // ( [P] classify ( [L] positive ) ) = 8, [SEP_TEXT] = 1.
      var override = strategy.classifyWithUsage(
        TEXT,
        List.of(new ClassificationLabel("positive")),
        0.5f
      );
      assertThat(override.inputTokens()).isEqualTo(9 + 4);
    }
  }

  @Test
  void gliner2Classification_batchMatchesSingle() {
    var runtime = new RecordingClassifierRuntime();
    try (
      var strategy = Gliner2ClassificationStrategy.create(
        ctx(RuntimeConfig.defaults(), Architecture.GLINER2),
        LABELS,
        runtime
      )
    ) {
      var texts = Arrays.asList(TEXT, null, "Paris.");
      var batch = strategy.classifyBatchWithUsage(texts, 0.5f);
      assertThat(batch)
        .extracting(ClassificationOutput::inputTokens)
        .containsExactly(CLS_PREFIX + 4, 0, CLS_PREFIX + 2);
      for (int i = 0; i < texts.size(); i++) {
        assertThat(batch.get(i)).isEqualTo(
          strategy.classifyWithUsage(texts.get(i), 0.5f)
        );
      }
      assertThat(
        batch.stream().map(ClassificationOutput::results).toList()
      ).isEqualTo(strategy.classifyBatch(texts, 0.5f));
    }
  }

  @Test
  void gliner2dot5Classification_usesTheWhitespaceSplitterConventions() {
    var runtime = new RecordingClassifierRuntime();
    var ctx = ctx(RuntimeConfig.defaults(), Architecture.GLINER2DOT5);
    try (
      var strategy = Gliner2ClassificationStrategy.create(
        ctx,
        LABELS,
        runtime,
        TextPreprocessing.whitespaceSplitter(ctx.config())
      )
    ) {
      assertThat(
        strategy.classifyWithUsage(TEXT, 0.5f).inputTokens()
      ).isEqualTo(CLS_PREFIX + 5);
    }
  }

  // ---- GLiNER bi-encoder NER ----------------------------------------------

  // [CLS] John lives in Paris [SEP] = 6; label rows [CLS] person [SEP] + [CLS] location [SEP] = 6.
  private static final int BI_TEXT_ROW = 6;
  private static final int BI_LABEL_ROWS = 6;

  @Test
  void glinerBiNer_countsTextRowAndUnpaddedLabelRows() {
    var runtime = new RecordingBiRuntime();
    try (var strategy = glinerBi(runtime)) {
      var out = strategy.extractWithUsage(TEXT, 0.5f);

      assertThat(out.inputTokens()).isEqualTo(BI_TEXT_ROW + BI_LABEL_ROWS);
      assertThat(runtime.textRows).containsExactly(BI_TEXT_ROW);
      assertThat(runtime.labelRows).containsExactly(BI_LABEL_ROWS);
      assertThat(out.spans()).isEqualTo(strategy.extract(TEXT, 0.5f));

      // Labels of unequal length are padded to the longest: [CLS] person name [SEP] = 4 and
      // [CLS] location [SEP] = 3 go out as a 2 x 4 matrix, but only the 7 real tokens count.
      var override = strategy.extractWithUsage(
        TEXT,
        List.of(
          new EntityDefinition("person name"),
          new EntityDefinition("location")
        ),
        0.5f
      );
      assertThat(
        runtime.labelWidths.get(runtime.labelWidths.size() - 1)
      ).isEqualTo(4);
      assertThat(override.inputTokens()).isEqualTo(BI_TEXT_ROW + 7);
      assertThat(runtime.labelRows.get(runtime.labelRows.size() - 1)).isEqualTo(
        7
      );
    }
  }

  @Test
  void glinerBiNer_batchCountsEveryLabelRowForEveryText() {
    var runtime = new RecordingBiRuntime();
    try (var strategy = glinerBi(runtime)) {
      var texts = Arrays.asList(TEXT, "", null, "Paris.");
      var batch = strategy.extractBatchWithUsage(texts, 0.5f);

      // No label cache: the label encoder reads every label for every text it runs on.
      // "Paris." splits into two words: [CLS] Paris . [SEP] = 4.
      assertThat(batch)
        .extracting(NerOutput::inputTokens)
        .containsExactly(BI_TEXT_ROW + BI_LABEL_ROWS, 0, 0, 4 + BI_LABEL_ROWS);
      assertThat(runtime.labelRows).containsExactly(
        BI_LABEL_ROWS,
        BI_LABEL_ROWS
      );
      for (int i = 0; i < texts.size(); i++) {
        assertThat(batch.get(i)).isEqualTo(
          strategy.extractWithUsage(texts.get(i), 0.5f)
        );
      }
      assertThat(batch.stream().map(NerOutput::spans).toList()).isEqualTo(
        strategy.extractBatch(texts, 0.5f)
      );
    }
  }

  @Test
  void glinerBiNer_noEntityTypesSkipsTheModel() {
    var runtime = new RecordingBiRuntime();
    try (var strategy = glinerBi(runtime)) {
      assertThat(strategy.extractWithUsage(TEXT, List.of(), 0.5f)).isEqualTo(
        NerOutput.empty()
      );
      assertThat(runtime.textRows).isEmpty();
    }
  }

  private static GlinerBiNerStrategy glinerBi(RecordingBiRuntime runtime) {
    return GlinerBiNerStrategy.create(
      ctx(RuntimeConfig.defaults(), Architecture.GLINER_BI),
      ENTITIES,
      runtime,
      new DjlTokenizerWrapper(resourceDir("/tiny-label-tokenizer"))
    );
  }

  // ---- fixtures ------------------------------------------------------------

  private static LoadContext ctx(
    RuntimeConfig runtimeConfig,
    Architecture architecture
  ) {
    var dir = tokenizerDir();
    var config = GLiNER4jConfig.builder()
      .modelPath(dir)
      .architecture(architecture)
      .maxWidth(4)
      .build();
    return new LoadContext(
      dir,
      "onnx",
      runtimeConfig,
      config,
      new DjlTokenizerWrapper(dir)
    );
  }

  private static Path tokenizerDir() {
    return resourceDir("/tiny-tokenizer");
  }

  private static Path resourceDir(String name) {
    try {
      return Path.of(
        Objects.requireNonNull(
          TokenUsageStrategyTest.class.getResource(name)
        ).toURI()
      );
    } catch (URISyntaxException e) {
      throw new IllegalStateException(e);
    }
  }

  /** Records the non-padding length of every row; asserts rows arrive unpadded. */
  private static void record(
    List<Integer> rowLengths,
    long[][] inputIds,
    long[][] attentionMask
  ) {
    for (int r = 0; r < inputIds.length; r++) {
      long real = Arrays.stream(attentionMask[r]).sum();
      assertThat(attentionMask[r]).hasSize(inputIds[r].length);
      rowLengths.add((int) real);
    }
  }

  private static final class RecordingSpanRuntime
    implements Gliner2SpanRuntime {

    final List<Integer> rowLengths = new ArrayList<>();

    @Override
    public FlatBatchScoringResult runNerFullBatch(
      long[][] inputIds,
      long[][] attentionMask,
      int maxSeqLen,
      long[] wordPositionsFlat,
      int batchSize,
      int maxTextLen,
      long pPosition,
      long[] fieldPositions,
      long[] spanIdxFlat,
      int maxWidth,
      long count
    ) {
      record(rowLengths, inputIds, attentionMask);
      // Count head says "no entity" for every row, so decode never reads span scores.
      return new FlatBatchScoringResult(
        new float[][] { { 1f, 0f } },
        null,
        () -> {}
      );
    }

    @Override
    public void close() {}
  }

  private static final class RecordingBoundaryRuntime
    implements Gliner2dot5NerRuntime {

    final List<Integer> rowLengths = new ArrayList<>();

    @Override
    public Scoring run(
      long[][] inputIds,
      long[][] attentionMask,
      int maxSeqLen,
      long[] wordPositionsFlat,
      int batchSize,
      int maxTextLen,
      long[] queryPositions
    ) {
      record(rowLengths, inputIds, attentionMask);
      int queries = queryPositions.length;
      var nullLogits = new float[batchSize * queries];
      Arrays.fill(nullLogits, 10f); // abstain on every query
      return new Scoring(
        new float[0],
        new long[0],
        nullLogits,
        batchSize,
        queries,
        0
      );
    }

    @Override
    public void close() {}
  }

  /** Records the text row and the label rows' real (non-padding) token counts of every run. */
  private static final class RecordingBiRuntime implements GlinerBiSpanRuntime {

    final List<Integer> textRows = new ArrayList<>();
    final List<Integer> labelRows = new ArrayList<>();
    final List<Integer> labelWidths = new ArrayList<>();

    @Override
    public float[][][] run(
      long[] inputIds,
      long[] wordsMask,
      int textLen,
      long[][] spanIdx,
      boolean[] spanMask,
      long[][] labelsInputIds,
      long[][] labelsAttentionMask
    ) {
      textRows.add(inputIds.length);
      int real = 0;
      for (var row : labelsAttentionMask) {
        real += (int) Arrays.stream(row).sum();
      }
      labelRows.add(real);
      labelWidths.add(labelsInputIds[0].length);
      int width = spanIdx.length / textLen;
      var logits = new float[textLen][width][labelsInputIds.length];
      for (var start : logits) {
        for (var row : start) {
          Arrays.fill(row, -10f); // sigmoid ~0: no span passes the threshold
        }
      }
      return logits;
    }

    @Override
    public void close() {}
  }

  private static final class RecordingClassifierRuntime
    implements Gliner2ClassifierRuntime {

    final List<Integer> rowLengths = new ArrayList<>();

    @Override
    public float[][] runClassifierFullBatch(
      long[][] inputIds,
      long[][] attentionMask,
      int maxSeqLen,
      long[] labelPositions,
      int batchSize
    ) {
      record(rowLengths, inputIds, attentionMask);
      var logits = new float[batchSize][labelPositions.length];
      for (var row : logits) {
        Arrays.fill(row, 2f); // sigmoid ~0.88: every label passes 0.5
      }
      return logits;
    }

    @Override
    public void close() {}
  }
}
