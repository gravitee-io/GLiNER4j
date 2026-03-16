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
package io.gravitee.lab.gliner4j.tokenizer;

import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Wraps DJL's HuggingFaceTokenizer for subword tokenization.
 * Loaded from a tokenizer.json file in the model directory.
 */
@Slf4j
public class DjlTokenizerWrapper implements AutoCloseable {

  private final HuggingFaceTokenizer tokenizer;

  /**
   * Loads a HuggingFace tokenizer from the model directory.
   *
   * @param modelDir path to directory containing tokenizer.json
   */
  public DjlTokenizerWrapper(Path modelDir) {
    try {
      this.tokenizer =
        HuggingFaceTokenizer.newInstance(
          modelDir,
          Map.of("padding", "false", "truncation", "false")
        );
      log.info("Loaded HuggingFace tokenizer from {}", modelDir);
    } catch (IOException e) {
      throw new RuntimeException(
        "Failed to load tokenizer from " + modelDir,
        e
      );
    }
  }

  /**
   * Tokenizes a single word into subword tokens.
   *
   * @param word the word to tokenize
   * @return list of subword token strings
   */
  public List<String> tokenize(String word) {
    var encoding = tokenizer.encode(word, false, false);
    var tokens = encoding.getTokens();
    var result = new ArrayList<String>(tokens.length);
    for (var token : tokens) {
      result.add(token);
    }
    return result;
  }

  /**
   * Tokenizes a word into subwords and returns both tokens and IDs in a single call.
   * Avoids the double-encoding overhead of calling tokenize() + convertTokensToIds() separately.
   *
   * @param word the word to tokenize
   * @return tokenization result with both token strings and vocabulary IDs
   */
  public TokenizationResult tokenizeWithIds(String word) {
    var encoding = tokenizer.encode(word, false, false);
    return new TokenizationResult(encoding.getTokens(), encoding.getIds());
  }

  @Override
  public void close() {
    // HuggingFaceTokenizer doesn't have explicit close, but we keep the interface
    log.debug("DjlTokenizerWrapper closed");
  }
}
