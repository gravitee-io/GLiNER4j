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

import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.util.ArrayList;
import java.util.List;

/**
 * The transient label part of a decoder-kv sequence, {@code <<SEP>>l1<<LABEL>>l2<<LABEL>>…<<SEP>>},
 * tokenized once per label list.
 *
 * <p>The scorer consumes the hidden states of the section <em>after</em> the opening
 * {@code <<SEP>>} (fastino's {@code _extract_label_section}); {@link #labelPositions()} and
 * {@link #sepPosition()} index into that body. They are the same for every text scored with this
 * label list, which is what lets the ONNX scorer take them as plain inputs.
 *
 * @param labels the label names, in prompt order
 * @param ids token ids of the whole section, opening {@code <<SEP>>} included
 * @param labelPositions index of each {@code <<LABEL>>} marker within the body
 * @param sepPosition index of the closing {@code <<SEP>>} within the body (the text representation)
 */
public record LabelSection(
  List<String> labels,
  long[] ids,
  long[] labelPositions,
  long sepPosition
) {
  /** Number of body tokens the scorer sees (section minus the opening {@code <<SEP>>}). */
  public int bodyLength() {
    return ids.length - 1;
  }

  /**
   * Tokenizes the section for {@code labels} and resolves the marker layout.
   *
   * @throws IllegalArgumentException if a label name tokenizes into a marker token
   */
  public static LabelSection build(
    DjlTokenizerWrapper tokenizer,
    List<String> labels,
    String labelToken,
    String sepToken,
    long classTokenId,
    long sepTokenId
  ) {
    var sb = new StringBuilder(sepToken);
    for (var label : labels) {
      sb.append(label).append(labelToken);
    }
    sb.append(sepToken);
    return of(
      labels,
      tokenizer.encodeWithSpecialTokens(sb.toString()),
      classTokenId,
      sepTokenId
    );
  }

  /** Layout resolution on already-tokenized ids (kept separate so it is unit-testable). */
  public static LabelSection of(
    List<String> labels,
    long[] ids,
    long classTokenId,
    long sepTokenId
  ) {
    if (
      ids.length < 3 ||
      ids[0] != sepTokenId ||
      ids[ids.length - 1] != sepTokenId
    ) {
      throw new IllegalStateException(
        "label section must be <<SEP>> … <<SEP>>; the tokenizer did not emit the marker ids " +
          "(is this the bundle's tokenizer.json with <<LABEL>>/<<SEP>> added tokens?)"
      );
    }
    var positions = new ArrayList<Long>(labels.size());
    for (int i = 1; i < ids.length; i++) {
      if (ids[i] == classTokenId) {
        positions.add((long) (i - 1));
      }
    }
    if (positions.size() != labels.size()) {
      throw new IllegalArgumentException(
        "expected " +
          labels.size() +
          " <<LABEL>> markers in the label section but found " +
          positions.size() +
          " — label names must not contain marker tokens: " +
          labels
      );
    }
    var labelPositions = positions
      .stream()
      .mapToLong(Long::longValue)
      .toArray();
    return new LabelSection(labels, ids, labelPositions, ids.length - 2L);
  }
}
