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

import ai.djl.huggingface.tokenizers.Encoding;
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
   * Converts subword token strings to their vocabulary IDs.
   *
   * @param tokens list of subword tokens
   * @return array of token IDs
   */
  public long[] convertTokensToIds(List<String> tokens) {
    var ids = new long[tokens.size()];
    for (int i = 0; i < tokens.size(); i++) {
      ids[i] = tokenizer.encode(tokens.get(i), false, false).getIds()[0];
    }
    return ids;
  }

  /**
   * Encodes a pre-tokenized list of words. Each word is tokenized into subwords,
   * and the full sequence of IDs is returned along with word-to-subword mappings.
   *
   * @param words the pre-tokenized words
   * @return encoding with IDs and word mappings
   */
  public Encoding encode(List<String> words) {
    var allTokens = new ArrayList<String>();
    var wordIds = new ArrayList<Integer>();
    for (int w = 0; w < words.size(); w++) {
      var subwords = tokenize(words.get(w));
      for (var sw : subwords) {
        allTokens.add(sw);
        wordIds.add(w);
      }
    }
    // Encode the full flat token list to get proper IDs
    var sb = new StringBuilder();
    for (int i = 0; i < allTokens.size(); i++) {
      if (i > 0) sb.append(" ");
      sb.append(allTokens.get(i));
    }
    return tokenizer.encode(sb.toString(), false, false);
  }

  /**
   * Encodes a raw string (not pre-tokenized).
   *
   * @param text the text to encode
   * @return encoding with IDs
   */
  public Encoding encodeRaw(String text) {
    return tokenizer.encode(text, false, false);
  }

  /**
   * Gets the token ID for a single token string.
   *
   * @param token the token string
   * @return the token ID
   */
  public long tokenToId(String token) {
    var encoding = tokenizer.encode(token, false, false);
    var ids = encoding.getIds();
    if (ids.length == 1) {
      return ids[0];
    }
    // For special tokens that encode to exactly themselves
    for (int i = 0; i < encoding.getTokens().length; i++) {
      if (encoding.getTokens()[i].equals(token)) {
        return ids[i];
      }
    }
    return ids[0];
  }

  @Override
  public void close() {
    // HuggingFaceTokenizer doesn't have explicit close, but we keep the interface
    log.debug("DjlTokenizerWrapper closed");
  }
}
