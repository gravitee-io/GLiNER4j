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
package io.gravitee.lab.gliner4j.runtime;

/**
 * Marker for a model family's ONNX-session container.
 *
 * <p>{@link BaseRuntime} is the shared single-encoder implementation (used by GLiNER2 and the
 * GLiNER uni-encoder family). Families with a different graph topology — notably the bi-encoder
 * family, which runs two encoder sessions plus a label-embedding cache — implement this interface
 * directly rather than extending {@link BaseRuntime}, reusing the shared ORT helpers without
 * inheriting the single-encoder buffer model.
 */
public interface ArchitectureRuntime extends AutoCloseable {
  @Override
  void close();
}
