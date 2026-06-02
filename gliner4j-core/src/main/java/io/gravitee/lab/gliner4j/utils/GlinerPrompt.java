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
package io.gravitee.lab.gliner4j.utils;

import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.util.List;

/**
 * Shared GLiNER prompt-assembly helpers: append a token's (or a text word's) subtoken ids to the
 * {@code input_ids} list while keeping the parallel {@code words_mask} aligned. Used by every
 * original-GLiNER NER strategy (uni markerV0, uni token_level, bi-encoder).
 *
 * <p>The {@code words_mask} marks the first subtoken of each text word with its 1-based word index
 * (0 for everything else, incl. prompt/marker tokens) so the model can pool one representation per
 * word.
 */
public final class GlinerPrompt {

  private GlinerPrompt() {}

  /**
   * Append a prompt token's subtokens; every position gets the same {@code maskValue} (0 for prompt
   * markers and separators).
   */
  public static void appendToken(
    DjlTokenizerWrapper tokenizer,
    String token,
    List<Long> ids,
    List<Long> wordsMask,
    long maskValue
  ) {
    for (long id : tokenizer.tokenizeWithIds(token).ids()) {
      ids.add(id);
      wordsMask.add(maskValue);
    }
  }

  /**
   * Append a text word's subtokens; the first subtoken carries {@code wordIndex1Based}, the rest 0.
   */
  public static void appendWord(
    DjlTokenizerWrapper tokenizer,
    String word,
    List<Long> ids,
    List<Long> wordsMask,
    long wordIndex1Based
  ) {
    long[] sub = tokenizer.tokenizeWithIds(word).ids();
    for (int i = 0; i < sub.length; i++) {
      ids.add(sub[i]);
      wordsMask.add(i == 0 ? wordIndex1Based : 0L);
    }
  }
}
