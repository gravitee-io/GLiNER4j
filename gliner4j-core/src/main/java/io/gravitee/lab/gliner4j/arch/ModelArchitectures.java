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
package io.gravitee.lab.gliner4j.arch;

import io.gravitee.lab.gliner4j.arch.gliner2.Gliner2Architecture;
import java.util.EnumMap;
import java.util.Map;

/**
 * Registry resolving an {@link Architecture} to its {@link ModelArchitecture} implementation.
 *
 * <p>Families are registered as they are implemented. Resolving a family that has no implementation
 * yet throws {@link UnsupportedOperationException} with a clear message, so loading a bundle whose
 * {@code architecture} GLiNER4j cannot yet handle fails fast at load time rather than producing
 * wrong results.
 */
public final class ModelArchitectures {

  private static final Map<Architecture, ModelArchitecture> REGISTRY =
    new EnumMap<>(Architecture.class);

  static {
    register(new Gliner2Architecture());
  }

  private ModelArchitectures() {}

  private static void register(ModelArchitecture architecture) {
    REGISTRY.put(architecture.id(), architecture);
  }

  /**
   * Resolves the implementation for {@code architecture}.
   *
   * @param architecture the family parsed from the bundle config
   * @return the registered implementation, never null
   * @throws UnsupportedOperationException if no implementation is registered for the family
   */
  public static ModelArchitecture forId(Architecture architecture) {
    var impl = REGISTRY.get(architecture);
    if (impl == null) {
      throw new UnsupportedOperationException(
        "Model family '" +
          architecture.configValue() +
          "' is not yet supported by this version of GLiNER4j. " +
          "Supported families: " +
          REGISTRY.keySet().stream().map(Architecture::configValue).toList()
      );
    }
    return impl;
  }
}
