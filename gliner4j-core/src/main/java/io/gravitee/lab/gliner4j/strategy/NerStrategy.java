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

import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.schema.NerOutput;
import java.util.List;
import java.util.Map;

/**
 * Per-family NER implementation: owns {@code assemble → forward → decode} for one model family.
 *
 * <p>The public {@link io.gravitee.lab.gliner4j.GLiNER4jNER} facade selects an implementation at
 * load time from the bundle's {@code architecture} (via
 * {@link io.gravitee.lab.gliner4j.arch.ModelArchitecture}) and delegates to it. The facade applies
 * the default threshold and null-guards; the strategy receives an explicit threshold.
 *
 * <p>The {@code ...WithUsage} variants return the same spans as their plain counterparts plus
 * the number of tokens the model read per text ({@link NerOutput#inputTokens()}). They are
 * default methods that throw {@link UnsupportedOperationException} so that implementations
 * outside this module keep compiling; every built-in strategy overrides them.
 * The llama.cpp strategies in {@code gliner4j-llamacpp} document their own token layouts on
 * their class Javadoc.
 */
public interface NerStrategy extends AutoCloseable {
  /**
   * Extracts entities using the load-time entity schema.
   *
   * @param text the input text
   * @param threshold minimum confidence (0..1) for span inclusion
   * @return map of entity type to detected spans
   */
  Map<String, List<EntitySpan>> extract(String text, float threshold);

  /**
   * Extracts entities using per-call entity definitions (overriding the load-time schema).
   *
   * @param text the input text
   * @param entities the entity types to extract for this call
   * @param threshold minimum confidence (0..1) for span inclusion
   * @return map of entity type to detected spans
   */
  Map<String, List<EntitySpan>> extract(
    String text,
    List<EntityDefinition> entities,
    float threshold
  );

  /**
   * Extracts entities from multiple texts in one batched pass.
   *
   * @param texts the input texts
   * @param threshold minimum confidence (0..1) for span inclusion
   * @return per-text results in input order
   */
  List<Map<String, List<EntitySpan>>> extractBatch(
    List<String> texts,
    float threshold
  );

  /**
   * Same as {@link #extract(String, float)}, also reporting the tokens the model read.
   *
   * <p>{@link NerOutput#inputTokens()} is the number of non-padding token ids in the encoder
   * input fed to the model for this text (the length of the row's attention mask, special and
   * schema tokens included, batch padding excluded). Inputs are never truncated or split into
   * windows: one text is one text row. Per built-in architecture:
   * <ul>
   *   <li>{@code glineruni} (span and token heads): the single row
   *       {@code [CLS] <<ENT>> type1 <<ENT>> type2 ... <<SEP>> word subtokens [SEP]}.</li>
   *   <li>{@code glinerbi}: the text row {@code [CLS] word subtokens [SEP]} plus, for every
   *       requested entity type, its label-encoder row {@code [CLS] type [SEP]} (unpadded length,
   *       label-encoder tokenizer). Label rows are re-encoded on every call (there is no label
   *       embedding cache), so each text of a batch counts all label rows.</li>
   *   <li>{@code gliner2} and {@code gliner2dot5}: the single row
   *       {@code ( [P] entities ( [E] type1 [E] type2 ... ) ) [SEP_TEXT] word subtokens}, plus a
   *       {@code [DESCRIPTION] type: description} segment after {@code entities} for every entity
   *       that carries a description. The schema prefix tokenization is cached
   *       across calls but the tokens are still fed to the model on every row, so they are
   *       always counted. No {@code [CLS]}/{@code [SEP]} is added. {@code gliner2dot5}
   *       lower-cases words and appends a final {@code .} word when the text does not end in
   *       {@code .}, {@code !} or {@code ?} (unless the bundle turns off {@code lowercase_words} / {@code append_period});
   *       that appended word's tokens are counted.</li>
   * </ul>
   * Whenever the model is run for a text its row(s) are counted. The count is {@code 0} only
   * when the model is not run for that text: a null or blank text, a text with no words, or an
   * empty entity type list on the strategies that skip the model in that case ({@code glineruni},
   * {@code glinerbi}, {@code gliner2dot5}; {@code gliner2} still runs it over the bare prompt and
   * counts that row).
   *
   * @param text the input text
   * @param threshold minimum confidence (0..1) for span inclusion
   * @return the spans and the input token count
   */
  default NerOutput extractWithUsage(String text, float threshold) {
    throw new UnsupportedOperationException(
      getClass().getName() + " does not report token usage"
    );
  }

  /**
   * Same as {@link #extract(String, List, float)}, also reporting the tokens the model read.
   * The count follows {@link #extractWithUsage(String, float)}, over the per-call entity types.
   *
   * @param text the input text
   * @param entities the entity types to extract for this call
   * @param threshold minimum confidence (0..1) for span inclusion
   * @return the spans and the input token count
   */
  default NerOutput extractWithUsage(
    String text,
    List<EntityDefinition> entities,
    float threshold
  ) {
    throw new UnsupportedOperationException(
      getClass().getName() + " does not report token usage"
    );
  }

  /**
   * Same as {@link #extractBatch(List, float)}, also reporting the tokens the model read per
   * text. Each element's count equals what {@link #extractWithUsage(String, float)} reports for
   * that text alone: padding added to align rows in a batch is never counted.
   *
   * <p>Exception: a strategy that caches label-encoder output across calls (the llama.cpp
   * bi-encoder, {@code LlamaGlinerUniNerStrategy} over a bundle with a label encoder) counts a
   * label row only on the call that actually encodes it. On a cold cache, the first text of the
   * batch that runs the model carries every label row and later texts carry their text row
   * only; on a warm cache no text carries label rows. The same text can therefore report a
   * different count in a batch than alone, and a different count cold than warm. The ONNX
   * {@code glinerbi} strategy has no label cache and is not affected.
   *
   * @param texts the input texts
   * @param threshold minimum confidence (0..1) for span inclusion
   * @return per-text outputs in input order
   */
  default List<NerOutput> extractBatchWithUsage(
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
