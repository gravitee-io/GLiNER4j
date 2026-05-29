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

import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import java.util.ArrayList;
import java.util.List;

/**
 * A single block in a multi-unit GLiNER schema prompt.
 *
 * <p>Each unit emits the same structural format as a single-task prompt:
 * {@code ( [P] <parentLabel> [DESCRIPTION] ... ( <childMarker> child1 <childMarker> child2 ... ) )}
 *
 * <p>For multi-unit prompts (relation extraction with N relation types, or combined extraction
 * with entities + relations), units are joined with {@code [SEP_STRUCT]} between them and a
 * trailing {@code [SEP_TEXT]} before the text tokens.
 *
 * @param kind unit kind for routing decoder logic
 * @param parentLabel parent token text (e.g. "entities", "classify", or a relation name)
 * @param childMarker special marker between children (e.g. "[E]", "[L]", "[R]")
 * @param childNames ordered child names (entity types, label names, or relation field names)
 * @param childDescriptions parallel descriptions (blank string when none)
 */
public record SchemaUnit(
  Kind kind,
  String parentLabel,
  String childMarker,
  List<String> childNames,
  List<String> childDescriptions
) {
  public enum Kind {
    ENTITIES,
    CLASSIFICATIONS,
    RELATION,
    STRUCTURE,
  }

  public SchemaUnit {
    if (childNames.size() != childDescriptions.size()) {
      throw new IllegalArgumentException(
        "childNames and childDescriptions must have the same size"
      );
    }
  }

  /**
   * Builds the entity unit from a list of entity definitions.
   *
   * @param entities the entity types
   * @return the schema unit
   */
  public static SchemaUnit forEntities(List<EntityDefinition> entities) {
    var names = new ArrayList<String>(entities.size());
    var descs = new ArrayList<String>(entities.size());
    for (var e : entities) {
      names.add(e.name());
      descs.add(e.description() == null ? "" : e.description());
    }
    return new SchemaUnit(Kind.ENTITIES, "entities", "[E]", names, descs);
  }

  /**
   * Builds the classification unit from a list of labels.
   *
   * @param labels the classification labels
   * @return the schema unit
   */
  public static SchemaUnit forClassifications(
    List<ClassificationLabel> labels
  ) {
    var names = new ArrayList<String>(labels.size());
    var descs = new ArrayList<String>(labels.size());
    for (var l : labels) {
      names.add(l.name());
      descs.add(l.description() == null ? "" : l.description());
    }
    return new SchemaUnit(
      Kind.CLASSIFICATIONS,
      "classify",
      "[L]",
      names,
      descs
    );
  }

  /**
   * Builds a relation unit from a single relation definition.
   *
   * <p>The parent label is the relation name; child names are the fields (e.g. head, tail).
   * Field-level descriptions are not currently supported by GLiNER2 — descriptions on a
   * {@link RelationDefinition} apply to the relation as a whole, attached to the first field
   * for emission.
   *
   * @param relation the relation definition
   * @return the schema unit
   */
  public static SchemaUnit forRelation(RelationDefinition relation) {
    var fields = relation.fields();
    var descs = new ArrayList<String>(fields.size());
    var desc = relation.description() == null ? "" : relation.description();
    for (int i = 0; i < fields.size(); i++) {
      descs.add(i == 0 ? desc : "");
    }
    return new SchemaUnit(Kind.RELATION, relation.name(), "[R]", fields, descs);
  }

  /**
   * Builds a structure unit from a single {@link StructureDefinition}. Parent label is the
   * structure name; child marker is {@code [C]}; child names and descriptions come from the
   * structure's fields.
   *
   * @param structure the structure definition
   * @return the schema unit
   */
  public static SchemaUnit forStructure(StructureDefinition structure) {
    var fields = structure.fields();
    var names = new ArrayList<String>(fields.size());
    var descs = new ArrayList<String>(fields.size());
    for (var f : fields) {
      names.add(f.name());
      descs.add(f.description() == null ? "" : f.description());
    }
    return new SchemaUnit(
      Kind.STRUCTURE,
      structure.name(),
      "[C]",
      names,
      descs
    );
  }
}
