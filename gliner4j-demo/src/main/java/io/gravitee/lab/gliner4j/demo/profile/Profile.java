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
package io.gravitee.lab.gliner4j.demo.profile;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.databind.ObjectMapper;
import io.gravitee.lab.gliner4j.demo.GLiNER4jDemo;
import java.io.IOException;
import java.util.List;
import java.util.Objects;

public final class Profile {

  public static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
  private String displayName;
  private String modelDir;
  private List<NamedDescription> entities;
  private List<String> nerSamples;
  private List<NamedDescription> labels;
  private List<String> classifySamples;

  private Profile() {}

  @JsonCreator
  private Profile(
    @JsonProperty("displayName") String displayName,
    @JsonProperty("modelDir") String modelDir,
    @JsonProperty("entities") List<NamedDescription> entities,
    @JsonProperty("nerSamples") List<String> nerSamples,
    @JsonProperty("labels") List<NamedDescription> labels,
    @JsonProperty("classifySamples") List<String> classifySamples
  ) {
    this.displayName = displayName;
    this.modelDir = modelDir;
    this.entities = entities;
    this.nerSamples = nerSamples;
    this.labels = labels;
    this.classifySamples = classifySamples;
  }

  private Profile(Profile profile) {
    this(
      profile.displayName,
      profile.modelDir,
      profile.entities,
      profile.nerSamples,
      profile.labels,
      profile.classifySamples
    );
  }

  public Profile(ProfileType profileName) {
    this(loadProfile(profileName));
  }

  private static Profile loadProfile(ProfileType profile) {
    var resourcePath = "/profiles/" + profile.getFilename();
    try (var in = GLiNER4jDemo.class.getResourceAsStream(resourcePath)) {
      if (in == null) {
        throw new IOException("Profile resource not found: " + resourcePath);
      }
      return OBJECT_MAPPER.readValue(in, Profile.class);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public boolean hasLabels() {
    return labels != null && !labels.isEmpty();
  }

  public boolean hasEntities() {
    return entities != null && !entities.isEmpty();
  }

  public String displayName() {
    return displayName;
  }

  public String modelDir() {
    return modelDir;
  }

  public List<NamedDescription> entities() {
    return entities;
  }

  public List<String> nerSamples() {
    return nerSamples;
  }

  public List<NamedDescription> labels() {
    return labels;
  }

  public List<String> classifySamples() {
    return classifySamples;
  }

  @Override
  public boolean equals(Object obj) {
    if (obj == this) return true;
    if (obj == null || obj.getClass() != this.getClass()) return false;
    var that = (Profile) obj;
    return (
      Objects.equals(this.displayName, that.displayName) &&
      Objects.equals(this.modelDir, that.modelDir) &&
      Objects.equals(this.entities, that.entities) &&
      Objects.equals(this.nerSamples, that.nerSamples) &&
      Objects.equals(this.labels, that.labels) &&
      Objects.equals(this.classifySamples, that.classifySamples)
    );
  }

  @Override
  public int hashCode() {
    return Objects.hash(displayName, modelDir, entities, nerSamples, labels, classifySamples);
  }

  @Override
  public String toString() {
    return (
      "Profile[" +
      "displayName=" +
      displayName +
      ", " +
      "modelDir=" +
      modelDir +
      ", " +
      "entities=" +
      entities +
      ", " +
      "nerSamples=" +
      nerSamples +
      ", " +
      "labels=" +
      labels +
      ", " +
      "classifySamples=" +
      classifySamples +
      ']'
    );
  }
}
