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

import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationOutput;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import java.util.List;

/**
 * Per-family text-classification implementation: owns {@code assemble → forward → decode} for one
 * model family (GLiNER2's classifier head, GLiClass's scaled dot-product head, …).
 *
 * <p>The public {@link io.gravitee.lab.gliner4j.GLiNER4jClassifier} facade selects an implementation
 * at load time from the bundle's {@code architecture} and delegates to it, applying the default
 * threshold and null-guards.
 *
 * <p>The {@code ...WithUsage} variants return the same labels as their plain counterparts plus
 * the number of tokens the model read per text ({@link ClassificationOutput#inputTokens()}).
 * They are default methods that throw {@link UnsupportedOperationException} so that
 * implementations outside this module keep compiling; every built-in strategy overrides them.
 * The llama.cpp strategies in {@code gliner4j-llamacpp} document their own token layouts on
 * their class Javadoc.
 */
public interface ClassificationStrategy extends AutoCloseable {
  /**
   * Classifies text against the load-time labels.
   *
   * @param text the input text
   * @param threshold minimum confidence (0..1) for a label to be returned
   * @return labels above threshold, sorted by confidence descending
   */
  List<ClassificationResult> classify(String text, float threshold);

  /**
   * Classifies text against per-call labels (overriding the load-time labels).
   *
   * @param text the input text
   * @param labels the labels for this call
   * @param threshold minimum confidence (0..1) for a label to be returned
   * @return labels above threshold, sorted by confidence descending
   */
  List<ClassificationResult> classify(
    String text,
    List<ClassificationLabel> labels,
    float threshold
  );

  /**
   * Classifies multiple texts in one batched pass.
   *
   * @param texts the input texts
   * @param threshold minimum confidence (0..1) for a label to be returned
   * @return per-text results in input order
   */
  List<List<ClassificationResult>> classifyBatch(
    List<String> texts,
    float threshold
  );

  /**
   * Same as {@link #classify(String, float)}, also reporting the tokens the model read.
   *
   * <p>{@link ClassificationOutput#inputTokens()} is the number of non-padding token ids in the
   * encoder input fed to the model for this text (the length of the row's attention mask,
   * special and label prompt tokens included, batch padding excluded). Inputs are never
   * truncated or split into windows: one text is one row. Per built-in architecture:
   * <ul>
   *   <li>{@code gliclass}: the tokenizer's encoding, special tokens included, of
   *       {@code <<LABEL>>label1<<LABEL>>label2...<<SEP>>text} (or text first when the bundle
   *       sets {@code prompt_first=false}).</li>
   *   <li>{@code gliner2} and {@code gliner2dot5}: the single row
   *       {@code ( [P] classify ( [L] label1 [L] label2 ... ) ) [SEP_TEXT] word subtokens}, plus
   *       a {@code [DESCRIPTION] label: description} segment after {@code classify} for every
   *       label that carries a description. The schema prefix tokenization is cached
   *       across calls but the tokens are still fed to the model on every row, so they are
   *       always counted. No {@code [CLS]}/{@code [SEP]} is added. {@code gliner2dot5}, and
   *       {@code gliner2} bundles that set {@code word_splitter=whitespace}, lower-case words and
   *       append a final {@code .} word when the text does not end in {@code .}, {@code !} or
   *       {@code ?} (unless the bundle turns off {@code lowercase_words} / {@code append_period});
   *       that appended word's tokens are counted.</li>
   * </ul>
   * Whenever the model is run for a text its row is counted. The count is {@code 0} only when
   * the model is not run for that text: a null or blank text, a text with no words
   * ({@code gliner2}/{@code gliner2dot5}), or an empty label list on {@code gliclass}, which skips
   * the model in that case ({@code gliner2}/{@code gliner2dot5} still run it and count the
   * row).
   *
   * @param text the input text
   * @param threshold minimum confidence (0..1) for a label to be returned
   * @return the labels and the input token count
   */
  default ClassificationOutput classifyWithUsage(String text, float threshold) {
    throw new UnsupportedOperationException(
      getClass().getName() + " does not report token usage"
    );
  }

  /**
   * Same as {@link #classify(String, List, float)}, also reporting the tokens the model read.
   * The count follows {@link #classifyWithUsage(String, float)}, over the per-call labels.
   *
   * @param text the input text
   * @param labels the labels for this call
   * @param threshold minimum confidence (0..1) for a label to be returned
   * @return the labels and the input token count
   */
  default ClassificationOutput classifyWithUsage(
    String text,
    List<ClassificationLabel> labels,
    float threshold
  ) {
    throw new UnsupportedOperationException(
      getClass().getName() + " does not report token usage"
    );
  }

  /**
   * Same as {@link #classifyBatch(List, float)}, also reporting the tokens the model read per
   * text. Each element's count equals what {@link #classifyWithUsage(String, float)} reports for
   * that text alone: padding added to align rows in a batch is never counted.
   *
   * @param texts the input texts
   * @param threshold minimum confidence (0..1) for a label to be returned
   * @return per-text outputs in input order
   */
  default List<ClassificationOutput> classifyBatchWithUsage(
    List<String> texts,
    float threshold
  ) {
    throw new UnsupportedOperationException(
      getClass().getName() + " does not report token usage"
    );
  }

  @Override
  void close();
}
