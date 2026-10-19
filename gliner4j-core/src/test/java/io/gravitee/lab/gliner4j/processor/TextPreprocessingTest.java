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

import static org.assertj.core.api.Assertions.assertThat;

import io.gravitee.lab.gliner4j.GLiNER4jConfig;
import java.util.Map;
import org.junit.jupiter.api.Test;

class TextPreprocessingTest {

  @Test
  void gliner2_withoutWordSplitter_isDefault() {
    var config = GLiNER4jConfig.builder().build();
    assertThat(TextPreprocessing.gliner2(config)).isEqualTo(
      TextPreprocessing.DEFAULT
    );
  }

  @Test
  void gliner2_withWhitespaceWordSplitter_lowercasesAndAppendsPeriod() {
    var config = GLiNER4jConfig.builder()
      .architectureConfig(Map.of("word_splitter", "whitespace"))
      .build();
    var preprocessing = TextPreprocessing.gliner2(config);
    assertThat(preprocessing.pattern()).isSameAs(
      TextEncoder.WHITESPACE_SPLITTER_PATTERN
    );
    assertThat(preprocessing.lowercaseWords()).isTrue();
    assertThat(preprocessing.appendPeriod()).isTrue();
  }
}
