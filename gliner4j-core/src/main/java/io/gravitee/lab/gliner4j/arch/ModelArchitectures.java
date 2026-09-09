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

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import io.gravitee.lab.gliner4j.arch.gliclass.GliclassArchitecture;
import io.gravitee.lab.gliner4j.arch.gliner2.Gliner2Architecture;
import io.gravitee.lab.gliner4j.arch.gliner2dot5.Gliner2dot5Architecture;
import io.gravitee.lab.gliner4j.arch.glinerbi.GlinerBiArchitecture;
import io.gravitee.lab.gliner4j.arch.glineruni.GlinerUniArchitecture;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.ServiceLoader;

/**
 * Registry resolving an ({@link Architecture}, {@link Engine}) pair to its {@link ModelArchitecture}
 * implementation.
 *
 * <p>Families are registered as they are implemented. Resolving a family that has no implementation
 * yet throws {@link UnsupportedOperationException} with a clear message, so loading a bundle whose
 * {@code architecture} GLiNER4j cannot yet handle fails fast at load time rather than producing
 * wrong results.
 */
public final class ModelArchitectures {

  /** Registry key: a family on an engine. */
  public record Key(Architecture architecture, Engine engine) {
    @Override
    public String toString() {
      return architecture.configValue() + "@" + engine.configValue();
    }
  }

  private static final Map<Key, ModelArchitecture> REGISTRY =
    new LinkedHashMap<>();

  static {
    register(new Gliner2Architecture());
    register(new Gliner2dot5Architecture());
    register(new GliclassArchitecture());
    register(new GlinerUniArchitecture());
    register(new GlinerBiArchitecture());
    // Families shipped by optional modules (e.g. gliner4j-llamacpp) register themselves through
    // META-INF/services/io.gravitee.lab.gliner4j.arch.ModelArchitecture.
    for (var extra : ServiceLoader.load(ModelArchitecture.class)) {
      register(extra);
    }
  }

  private ModelArchitectures() {}

  private static void register(ModelArchitecture architecture) {
    REGISTRY.put(
      new Key(architecture.id(), architecture.engine()),
      architecture
    );
  }

  /** Resolves the implementation for the family and engine declared by {@code config}. */
  public static ModelArchitecture forConfig(GLiNER4jConfig config) {
    return forId(config.getArchitecture(), config.getEngine());
  }

  /** Resolves the ONNX Runtime implementation for {@code architecture}. */
  public static ModelArchitecture forId(Architecture architecture) {
    return forId(architecture, Engine.ONNX);
  }

  /**
   * Resolves the implementation for {@code architecture} on {@code engine}.
   *
   * @param architecture the family parsed from the bundle config
   * @param engine the engine parsed from the bundle config
   * @return the registered implementation, never null
   * @throws UnsupportedOperationException if no implementation is registered for the pair
   */
  public static ModelArchitecture forId(
    Architecture architecture,
    Engine engine
  ) {
    var impl = REGISTRY.get(new Key(architecture, engine));
    if (impl == null) {
      throw new UnsupportedOperationException(
        "Model family '" +
          architecture.configValue() +
          "' on engine '" +
          engine.configValue() +
          "' is not registered in this JVM. Every llamacpp implementation and the families " +
          "gliclass-decoder-kv / gliner-streaming-span ship in the gliner4j-llamacpp module (Java 25), " +
          "which must be on the classpath. Registered: " +
          REGISTRY.keySet().stream().map(Key::toString).toList()
      );
    }
    return impl;
  }
}
