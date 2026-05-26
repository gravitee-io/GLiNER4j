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
package io.gravitee.lab.gliner4j.schema;

import java.util.ArrayList;
import java.util.List;

/**
 * Composite schema describing what to extract in a single forward pass.
 *
 * <p>A schema is a list of one or more <em>units</em>:
 * <ul>
 *   <li>at most one entity unit (extract named entity spans),</li>
 *   <li>at most one classification unit (multi-label classification),</li>
 *   <li>any number of relation units (one per relation type).</li>
 * </ul>
 *
 * <p>Unit ordering in the prompt mirrors the order they were added to the builder.
 * Use {@link #builder()} to construct.
 */
public final class Schema {

  private final List<EntityDefinition> entities;
  private final List<ClassificationLabel> classifications;
  private final List<RelationDefinition> relations;

  private Schema(
    List<EntityDefinition> entities,
    List<ClassificationLabel> classifications,
    List<RelationDefinition> relations
  ) {
    this.entities = List.copyOf(entities);
    this.classifications = List.copyOf(classifications);
    this.relations = List.copyOf(relations);
  }

  public List<EntityDefinition> entities() {
    return entities;
  }

  public List<ClassificationLabel> classifications() {
    return classifications;
  }

  public List<RelationDefinition> relations() {
    return relations;
  }

  public boolean hasEntities() {
    return !entities.isEmpty();
  }

  public boolean hasClassifications() {
    return !classifications.isEmpty();
  }

  public boolean hasRelations() {
    return !relations.isEmpty();
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {

    private final List<EntityDefinition> entities = new ArrayList<>();
    private final List<ClassificationLabel> classifications = new ArrayList<>();
    private final List<RelationDefinition> relations = new ArrayList<>();

    private Builder() {}

    public Builder entities(List<EntityDefinition> entities) {
      this.entities.clear();
      this.entities.addAll(entities);
      return this;
    }

    public Builder classifications(List<ClassificationLabel> classifications) {
      this.classifications.clear();
      this.classifications.addAll(classifications);
      return this;
    }

    public Builder relations(List<RelationDefinition> relations) {
      this.relations.clear();
      this.relations.addAll(relations);
      return this;
    }

    public Schema build() {
      if (
        entities.isEmpty() && classifications.isEmpty() && relations.isEmpty()
      ) {
        throw new IllegalStateException(
          "Schema must declare at least one entity, classification, or relation"
        );
      }
      return new Schema(entities, classifications, relations);
    }
  }
}
