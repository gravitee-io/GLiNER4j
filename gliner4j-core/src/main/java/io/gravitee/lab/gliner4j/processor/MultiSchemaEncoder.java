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
package io.gravitee.lab.gliner4j.processor;

import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import java.util.ArrayList;
import java.util.List;
import lombok.Getter;

/**
 * Builds a {@link SchemaEncoder} that concatenates one prompt block per structure,
 * joined by {@code [SEP_STRUCT]}, matching the upstream GLiNER2 schema-extraction format:
 *
 * <pre>{@code
 * ( [P] s1 [DESCRIPTION] f1: d1 ( [C] f1 [C] f2 ) ) [SEP_STRUCT] ( [P] s2 ( [C] f3 ) )
 * }</pre>
 *
 * <p>Field names are flattened into a single list across structures. Use {@link #getFieldOffsets()}
 * to recover per-structure field ranges in scoring-head outputs.
 */
public class MultiSchemaEncoder {

  public static final String SEP_STRUCT = "[SEP_STRUCT]";
  public static final String P_TOKEN = "[P]";
  public static final String C_TOKEN = "[C]";
  public static final String DESCRIPTION_TOKEN = "[DESCRIPTION]";

  @Getter
  private final SchemaEncoder combined;

  @Getter
  private final List<StructureDefinition> structures;

  @Getter
  private final int[] fieldOffsets;

  public MultiSchemaEncoder(List<StructureDefinition> structures) {
    if (structures == null || structures.isEmpty()) {
      throw new IllegalArgumentException(
        "structures must contain at least one StructureDefinition"
      );
    }
    this.structures = List.copyOf(structures);

    var combinedTokens = new ArrayList<String>();
    var combinedFieldNames = new ArrayList<String>();
    var offsets = new int[structures.size()];

    for (int s = 0; s < structures.size(); s++) {
      offsets[s] = combinedFieldNames.size();
      var struct = structures.get(s);

      if (s > 0) {
        combinedTokens.add(SEP_STRUCT);
      }

      combinedTokens.add("(");
      combinedTokens.add(P_TOKEN);
      combinedTokens.add(struct.name());
      for (var field : struct.fields()) {
        var desc = field.description();
        if (desc != null && !desc.isBlank()) {
          combinedTokens.add(DESCRIPTION_TOKEN);
          combinedTokens.add(field.name() + ":");
          combinedTokens.add(desc);
        }
      }
      combinedTokens.add("(");
      for (var field : struct.fields()) {
        combinedTokens.add(C_TOKEN);
        combinedTokens.add(field.name());
        combinedFieldNames.add(field.name());
      }
      combinedTokens.add(")");
      combinedTokens.add(")");
    }

    this.fieldOffsets = offsets;
    this.combined = new SchemaEncoder(
      C_TOKEN,
      combinedFieldNames,
      combinedTokens
    );
  }
}
