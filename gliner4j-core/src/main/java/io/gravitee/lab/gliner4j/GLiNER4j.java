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

import io.gravitee.lab.gliner4j.postprocess.SpanDecoder;
import io.gravitee.lab.gliner4j.processor.InputAssembler;
import io.gravitee.lab.gliner4j.processor.PreprocessedInput;
import io.gravitee.lab.gliner4j.processor.SchemaEncoder;
import io.gravitee.lab.gliner4j.processor.TextEncoder;
import io.gravitee.lab.gliner4j.runtime.GLiNER4jRuntime;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import io.gravitee.lab.gliner4j.tokenizer.TokenMapping;
import io.gravitee.lab.gliner4j.tokenizer.WhitespaceTokenSplitter;
import java.nio.file.Path;
import java.util.*;
import java.util.stream.Collectors;
import lombok.extern.slf4j.Slf4j;

/**
 * Main facade for GLiNER4j — Java-native GLiNER2 NER via ONNX Runtime.
 *
 * <p>Usage:
 * <pre>{@code
 * var entities = List.of(new EntityDefinition("person"), new EntityDefinition("organization"));
 * try (var gliner = GLiNER4j.load(modelDir, entities)) {
 *     Map<String, List<EntitySpan>> results = gliner.extract("John works at Google.");
 * }
 * }</pre>
 */
@Slf4j
public class GLiNER4j implements AutoCloseable {

  private final GLiNER4jConfig config;
  private final List<EntityDefinition> entities;
  private final DjlTokenizerWrapper tokenizer;
  private final GLiNER4jRuntime runtime;
  private final InputAssembler inputAssembler;
  private final WhitespaceTokenSplitter splitter;
  private final SpanDecoder spanDecoder;

  private GLiNER4j(
    GLiNER4jConfig config,
    List<EntityDefinition> entities,
    DjlTokenizerWrapper tokenizer,
    GLiNER4jRuntime runtime,
    InputAssembler inputAssembler
  ) {
    this.config = config;
    this.entities = List.copyOf(entities);
    this.tokenizer = tokenizer;
    this.runtime = runtime;
    this.inputAssembler = inputAssembler;
    this.splitter = new WhitespaceTokenSplitter();
    this.spanDecoder = new SpanDecoder();
  }

  /**
   * Loads a GLiNER model and pre-computes schema encoding for the given entities.
   *
   * @param modelDir path to the directory containing ONNX models and tokenizer files
   * @param entities the entity types to extract
   * @return a ready-to-use GLiNER4j instance
   */
  public static GLiNER4j load(Path modelDir, List<EntityDefinition> entities) {
    log.info(
      "Loading GLiNER4j model from {} with {} entities",
      modelDir,
      entities.size()
    );

    var config = GLiNER4jConfig.load(modelDir);
    var tokenizer = new DjlTokenizerWrapper(modelDir);
    var runtime = new GLiNER4jRuntime(modelDir);
    var schemaEncoder = new SchemaEncoder(entities);
    var inputAssembler = new InputAssembler(tokenizer, schemaEncoder);

    // Pre-allocate encoder buffers with the constant schema prefix
    runtime.initEncoderBuffers(inputAssembler.getSchemaPrefixIds());

    log.info("GLiNER4j model loaded successfully");
    return new GLiNER4j(config, entities, tokenizer, runtime, inputAssembler);
  }

  /**
   * Extracts entities from the given text using the default threshold.
   *
   * @param text the input text to analyze
   * @return map of entity type to list of detected spans
   */
  public Map<String, List<EntitySpan>> extract(String text) {
    return extract(text, config.getDefaultThreshold());
  }

  /**
   * Extracts entities from the given text using the specified threshold.
   *
   * @param text the input text to analyze
   * @param threshold minimum confidence score (0..1) for span inclusion
   * @return map of entity type to list of detected spans
   */
  public Map<String, List<EntitySpan>> extract(String text, float threshold) {
    if (text == null || text.isBlank()) {
      return Map.of();
    }

    // 1. Split text into words with char offsets
    var textEncoder = new TextEncoder(text, splitter);
    if (textEncoder.getTextLen() == 0) {
      return Map.of();
    }

    // 2. Assemble full token sequence
    var input = inputAssembler.assemble(textEncoder);

    // 3. Run encoder
    var hiddenStates = runtime.runEncoder(
      input.inputIds(),
      input.attentionMask()
    );

    // 4. Extract embeddings from hidden states
    var embeddings = extractEmbeddings(hiddenStates[0], input);

    // 5. Build span indices
    int maxWidth = config.getMaxWidth();
    int textLen = input.textLen();
    int numSpans = textLen * maxWidth;
    var spanIdx = new long[1][numSpans][2];
    for (int i = 0; i < textLen; i++) {
      for (int w = 0; w < maxWidth; w++) {
        int idx = i * maxWidth + w;
        int endPos = i + w;
        if (endPos < textLen) {
          spanIdx[0][idx][0] = i;
          spanIdx[0][idx][1] = endPos;
        }
        // else stays (0, 0) — safe default
      }
    }

    // 6. Run span_rep — returns [1][textLen][maxWidth][hidden]
    var textEmbs3d = new float[1][textLen][config.getHiddenSize()];
    System.arraycopy(embeddings.textEmbs, 0, textEmbs3d[0], 0, textLen);
    var spanRep4d = runtime.runSpanRep(textEmbs3d, spanIdx);
    var spanRep = spanRep4d[0]; // [textLen][maxWidth][hidden]

    // 7. Run scoring head once with maxCount — countLogits is independent of the count input
    var scoringResult = runtime.runScoringHead(
      spanRep,
      embeddings.schemaEmbP,
      embeddings.schemaEmbFields,
      (long) config.getMaxCount()
    );

    // 8. Predict count from logits
    int predCount = argmax(scoringResult.countLogits()[0]);
    log.debug("Predicted count: {}", predCount);
    if (predCount == 0) {
      return Map.of();
    }

    // 9. Decode spans (SpanDecoder uses spanScores[0] only)
    var spans = spanDecoder.decode(
      scoringResult.spanScores(),
      input.fieldNames(),
      input.wordStartChars(),
      input.wordEndChars(),
      text,
      textLen,
      threshold
    );

    // 10. Group by type
    return spans
      .stream()
      .collect(
        Collectors.groupingBy(
          EntitySpan::type,
          LinkedHashMap::new,
          Collectors.toList()
        )
      );
  }

  private ExtractedEmbeddings extractEmbeddings(
    float[][] hiddenState,
    PreprocessedInput input
  ) {
    int hiddenSize = config.getHiddenSize();
    var specialTokenIds = config.getSpecialTokenIds();
    long pTokenId = specialTokenIds.getOrDefault("P", -1L);
    long eTokenId = specialTokenIds.getOrDefault("E", -1L);

    // Collect schema embeddings by scanning for special token IDs
    var schemaEmbsList = new ArrayList<float[]>();
    for (int i = 0; i < input.inputIds().length; i++) {
      long tokenId = input.inputIds()[i];
      if (tokenId == pTokenId || tokenId == eTokenId) {
        schemaEmbsList.add(hiddenState[i]);
      }
    }

    // schemaEmbs[0] = [P] embedding, schemaEmbs[1:] = [E] field embeddings
    float[] schemaEmbP = schemaEmbsList.isEmpty()
      ? new float[hiddenSize]
      : schemaEmbsList.get(0);
    float[][] schemaEmbFields = new float[schemaEmbsList.size() -
    1][hiddenSize];
    for (int i = 1; i < schemaEmbsList.size(); i++) {
      schemaEmbFields[i - 1] = schemaEmbsList.get(i);
    }

    // Extract text embeddings using first-subword pooling per word
    int textLen = input.textLen();
    var textEmbs = new float[textLen][hiddenSize];
    var seenWord = new boolean[textLen];
    for (int i = 0; i < input.mappings().length; i++) {
      var mapping = input.mappings()[i];
      if (
        mapping.type() == TokenMapping.SegmentType.TEXT &&
        !seenWord[mapping.origIdx()]
      ) {
        textEmbs[mapping.origIdx()] = hiddenState[i];
        seenWord[mapping.origIdx()] = true;
      }
    }

    return new ExtractedEmbeddings(schemaEmbP, schemaEmbFields, textEmbs);
  }

  private static int argmax(float[] values) {
    int maxIdx = 0;
    float maxVal = values[0];
    for (int i = 1; i < values.length; i++) {
      if (values[i] > maxVal) {
        maxVal = values[i];
        maxIdx = i;
      }
    }
    return maxIdx;
  }

  @Override
  public void close() {
    runtime.close();
    tokenizer.close();
    log.info("GLiNER4j closed");
  }

  private record ExtractedEmbeddings(
    float[] schemaEmbP,
    float[][] schemaEmbFields,
    float[][] textEmbs
  ) {}
}
