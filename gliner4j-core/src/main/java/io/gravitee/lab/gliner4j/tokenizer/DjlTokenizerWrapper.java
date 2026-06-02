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
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import lombok.extern.slf4j.Slf4j;

/**
 * Wraps DJL's HuggingFaceTokenizer for subword tokenization.
 * Loaded from a tokenizer.json file in the model directory.
 */
@Slf4j
public class DjlTokenizerWrapper implements AutoCloseable {

  private final HuggingFaceTokenizer tokenizer;
  private final ConcurrentHashMap<String, TokenizationResult> tokenCache =
    new ConcurrentHashMap<>();

  /**
   * Loads a HuggingFace tokenizer from the model directory.
   *
   * @param modelDir path to directory containing tokenizer.json
   */
  public DjlTokenizerWrapper(Path modelDir) {
    try {
      this.tokenizer = HuggingFaceTokenizer.newInstance(
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
  public Encoding tokenize(String word) {
    return tokenizer.encode(word, false, false);
  }

  /**
   * Encodes a full text into token IDs <em>with</em> the model's special tokens (e.g. the leading
   * {@code [CLS]} / trailing {@code [SEP]} from the tokenizer's post-processor), and recognizing
   * any in-string special markers ({@code <<LABEL>>}, {@code <<SEP>>}, …) as single tokens.
   *
   * <p>Used by whole-prompt families like GLiClass, where the class/text markers and the
   * position-0 {@code [CLS]} representation must match training. Distinct from
   * {@link #tokenizeWithIds(String)}, which is per-word and adds no special tokens (GLiNER2 builds
   * its sequence word-by-word).
   *
   * @param text the full prompt to encode
   * @return token IDs including special tokens
   */
  public long[] encodeWithSpecialTokens(String text) {
    return tokenizer.encode(text, true, false).getIds();
  }

  /**
   * Tokenizes a word into subwords and returns both tokens and IDs in a single call.
   * Avoids the double-encoding overhead of calling tokenize() + convertTokensToIds() separately.
   *
   * @param word the word to tokenize
   * @return tokenization result with both token strings and vocabulary IDs
   */
  public TokenizationResult tokenizeWithIds(String word) {
    return tokenCache.computeIfAbsent(word, this::tokenizeUncached);
  }

  private TokenizationResult tokenizeUncached(String word) {
    var encoding = tokenize(word);
    return new TokenizationResult(encoding.getTokens(), encoding.getIds());
  }

  /**
   * Warms {@link #tokenCache} for the given words via a single batched JNI call. After this
   * returns, every {@link #tokenizeWithIds(String)} call for any word in {@code words} is a
   * pure cache hit (no JNI). Callers can then iterate their words and use {@code tokenizeWithIds}
   * inline, avoiding the intermediate result-list materialization.
   *
   * <p>Per-word tokenization semantics are preserved (each word is still encoded independently,
   * matching how GLiNER was trained). Only the JNI dispatch is amortized.
   *
   * @param words the words whose tokenizations should be available from the cache afterwards
   */
  public void prefetchTokens(List<String> words) {
    if (words.isEmpty()) {
      return;
    }

    // Collect unique uncached words while preserving first-seen order.
    var uniqueUncached = new LinkedHashSet<String>();
    for (var w : words) {
      if (!tokenCache.containsKey(w)) {
        uniqueUncached.add(w);
      }
    }

    if (uniqueUncached.isEmpty()) {
      return;
    }

    var uncachedArr = uniqueUncached.toArray(new String[0]);
    Encoding[] encodings = tokenizer.batchEncode(uncachedArr, false, false);
    for (int i = 0; i < uncachedArr.length; i++) {
      var enc = encodings[i];
      // putIfAbsent: a concurrent computeIfAbsent for the same word would arrive at the
      // same value, so duplicating the put is harmless.
      tokenCache.putIfAbsent(
        uncachedArr[i],
        new TokenizationResult(enc.getTokens(), enc.getIds())
      );
    }
  }

  @Override
  public void close() {
    // HuggingFaceTokenizer doesn't have explicit close, but we keep the interface
    tokenCache.clear();
    log.debug("DjlTokenizerWrapper closed");
  }
}
