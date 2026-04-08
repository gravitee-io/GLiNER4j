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

import java.util.ArrayList;
import java.util.List;
import lombok.Getter;

/**
 * Builds the schema token list from field definitions.
 * Schema format: ["(", "[P]", taskKeyword, "[DESCRIPTION]", "name:", "desc", ..., "(", specialToken, "name1", specialToken, "name2", ")", ")"]
 * Description tokens are only emitted for fields with a non-blank description.
 * Immutable after construction — safe to share across threads.
 */
@Getter
public class SchemaEncoder {

  private final List<String> schemaTokens;
  private final List<String> fieldNames;
  private final int numFields;
  private final String specialToken;

  /**
   * Builds schema token list from field names, descriptions, and task-specific parameters.
   *
   * @param taskKeyword the task keyword (e.g. "entities" for NER, "classify" for classification)
   * @param specialToken the special token marking each field (e.g. "[E]" for NER, "[L]" for classification)
   * @param fieldNames the field/label names
   * @param descriptions the field/label descriptions (parallel to fieldNames)
   */
  public SchemaEncoder(
    String taskKeyword,
    String specialToken,
    List<String> fieldNames,
    List<String> descriptions
  ) {
    this.fieldNames = List.copyOf(fieldNames);
    this.numFields = fieldNames.size();
    this.specialToken = specialToken;
    this.schemaTokens = buildSchemaTokens(
      taskKeyword,
      specialToken,
      fieldNames,
      descriptions
    );
  }

  private static List<String> buildSchemaTokens(
    String taskKeyword,
    String specialToken,
    List<String> names,
    List<String> descriptions
  ) {
    var tokens = new ArrayList<String>();
    tokens.add("(");
    tokens.add("[P]");
    tokens.add(taskKeyword);
    for (int i = 0; i < names.size(); i++) {
      var desc = descriptions.get(i);
      if (desc != null && !desc.isBlank()) {
        tokens.add("[DESCRIPTION]");
        tokens.add(names.get(i) + ":");
        tokens.add(desc);
      }
    }
    tokens.add("(");
    for (var name : names) {
      tokens.add(specialToken);
      tokens.add(name);
    }
    tokens.add(")");
    tokens.add(")");
    return List.copyOf(tokens);
  }
}
