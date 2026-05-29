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
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Path;

/**
 * The family-agnostic inputs a {@link ModelArchitecture} needs to build a task strategy: where the
 * bundle lives, which ONNX variant to load, the resource config, the parsed bundle config, and the
 * shared tokenizer.
 *
 * <p>The {@code tokenizer} is created once by the facade's {@code load(...)} and handed to the
 * strategy, which owns its lifecycle (closes it on {@link AutoCloseable#close()}).
 */
public record LoadContext(
  Path modelDir,
  String variant,
  RuntimeConfig runtimeConfig,
  GLiNER4jConfig config,
  DjlTokenizerWrapper tokenizer
) {}
