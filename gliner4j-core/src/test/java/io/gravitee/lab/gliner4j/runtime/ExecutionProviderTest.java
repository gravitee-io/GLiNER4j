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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

class ExecutionProviderTest {

  @Test
  void fromString_unknownOrBlank_resolvesToAuto() {
    assertThat(ExecutionProvider.fromString(null)).isEqualTo(
      ExecutionProvider.AUTO
    );
    assertThat(ExecutionProvider.fromString("  ")).isEqualTo(
      ExecutionProvider.AUTO
    );
    assertThat(ExecutionProvider.fromString("nonsense")).isEqualTo(
      ExecutionProvider.AUTO
    );
    assertThat(ExecutionProvider.fromString("auto")).isEqualTo(
      ExecutionProvider.AUTO
    );
  }

  @ParameterizedTest
  @ValueSource(strings = { "cpu", "CPU", " Cpu " })
  void fromString_cpuAliases(String value) {
    assertThat(ExecutionProvider.fromString(value)).isEqualTo(
      ExecutionProvider.CPU
    );
  }

  @ParameterizedTest
  @ValueSource(strings = { "cuda", "gpu", "nvidia", "CUDA" })
  void fromString_cudaAliases(String value) {
    assertThat(ExecutionProvider.fromString(value)).isEqualTo(
      ExecutionProvider.CUDA
    );
  }

  @ParameterizedTest
  @ValueSource(strings = { "openvino", "vino", "intel" })
  void fromString_openvinoAliases(String value) {
    assertThat(ExecutionProvider.fromString(value)).isEqualTo(
      ExecutionProvider.OPENVINO
    );
  }

  @ParameterizedTest
  @ValueSource(strings = { "coreml", "mlx", "apple", "ane", "nonsense" })
  void fromString_unsupportedResolvesToAuto(String value) {
    assertThat(ExecutionProvider.fromString(value)).isEqualTo(
      ExecutionProvider.AUTO
    );
  }

  @Test
  void available_alwaysContainsCpu() {
    assertThat(ExecutionProvider.available()).contains(ExecutionProvider.CPU);
  }

  @Test
  void resolve_concreteProvider_returnedUnchanged() {
    assertThat(ExecutionProvider.resolve(ExecutionProvider.CPU)).isEqualTo(
      ExecutionProvider.CPU
    );
    assertThat(ExecutionProvider.resolve(ExecutionProvider.CUDA)).isEqualTo(
      ExecutionProvider.CUDA
    );
  }

  @Test
  void resolve_autoAndNull_pickBestAvailableInPreferenceOrder() {
    var available = ExecutionProvider.available();
    var expected = java.util.stream.Stream.of(
      ExecutionProvider.CUDA,
      ExecutionProvider.OPENVINO
    )
      .filter(available::contains)
      .findFirst()
      .orElse(ExecutionProvider.CPU);

    assertThat(ExecutionProvider.resolve(ExecutionProvider.AUTO)).isEqualTo(
      expected
    );
    // null is treated as AUTO.
    assertThat(ExecutionProvider.resolve(null)).isEqualTo(expected);
  }
}
