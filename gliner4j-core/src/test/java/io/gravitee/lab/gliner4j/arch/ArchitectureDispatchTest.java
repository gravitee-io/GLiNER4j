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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;

/**
 * Unit tests for model-family dispatch: the back-compat default, value normalization, and the
 * fail-fast for families GLiNER4j cannot yet handle. No model bundle required.
 */
class ArchitectureDispatchTest {

  @Test
  void absentOrBlankArchitectureDefaultsToGliner2() {
    assertThat(Architecture.fromConfigValue(null)).isEqualTo(
      Architecture.GLINER2
    );
    assertThat(Architecture.fromConfigValue("")).isEqualTo(
      Architecture.GLINER2
    );
    assertThat(Architecture.fromConfigValue("   ")).isEqualTo(
      Architecture.GLINER2
    );
  }

  @Test
  void knownConfigValuesResolveCaseInsensitively() {
    assertThat(Architecture.fromConfigValue("gliner2")).isEqualTo(
      Architecture.GLINER2
    );
    assertThat(Architecture.fromConfigValue("  GLiNER-Uni  ")).isEqualTo(
      Architecture.GLINER_UNI
    );
    assertThat(Architecture.fromConfigValue("gliner-bi")).isEqualTo(
      Architecture.GLINER_BI
    );
  }

  @Test
  void unknownConfigValueDefaultsToGliner2() {
    assertThat(Architecture.fromConfigValue("totally-unknown")).isEqualTo(
      Architecture.GLINER2
    );
  }

  @Test
  void gliner2IsRegisteredAndSupportsEveryTask() {
    var arch = ModelArchitectures.forId(Architecture.GLINER2);
    assertThat(arch.id()).isEqualTo(Architecture.GLINER2);
    for (var task : TaskType.values()) {
      assertThat(arch.supports(task)).isTrue();
    }
  }

  @Test
  void resolvingAnUnimplementedFamilyFailsFast() {
    // GLINER_DECODER is the remaining unregistered family (GLINER2/GLiClass/uni/bi are implemented).
    assertThatThrownBy(() ->
      ModelArchitectures.forId(Architecture.GLINER_DECODER)
    )
      .isInstanceOf(UnsupportedOperationException.class)
      .hasMessageContaining("gliner-decoder")
      .hasMessageContaining("not yet supported");
  }
}
