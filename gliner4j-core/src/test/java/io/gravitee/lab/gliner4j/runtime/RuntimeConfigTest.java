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

import ai.onnxruntime.OrtSession;
import org.junit.jupiter.api.Test;

class RuntimeConfigTest {

  @Test
  void defaultConfig_hasNullThreadCounts() {
    var config = RuntimeConfig.builder().build();
    assertThat(config.getEncoderIntraOpThreads()).isNull();
    assertThat(config.getEncoderInterOpThreads()).isNull();
    assertThat(config.getScoringIntraOpThreads()).isNull();
    assertThat(config.getScoringInterOpThreads()).isNull();
  }

  @Test
  void defaultConfig_hasAllOptLevel() {
    var config = RuntimeConfig.builder().build();
    assertThat(config.getOptimizationLevel())
      .isEqualTo(OrtSession.SessionOptions.OptLevel.ALL_OPT);
  }

  @Test
  void defaultConfig_hasCacheEnabled() {
    var config = RuntimeConfig.builder().build();
    assertThat(config.isOptimizedModelCacheEnabled()).isTrue();
  }

  @Test
  void builder_overridesIndividualFields() {
    var config = RuntimeConfig
      .builder()
      .encoderIntraOpThreads(4)
      .encoderInterOpThreads(2)
      .scoringIntraOpThreads(1)
      .scoringInterOpThreads(1)
      .optimizationLevel(OrtSession.SessionOptions.OptLevel.BASIC_OPT)
      .optimizedModelCacheEnabled(false)
      .build();

    assertThat(config.getEncoderIntraOpThreads()).isEqualTo(4);
    assertThat(config.getEncoderInterOpThreads()).isEqualTo(2);
    assertThat(config.getScoringIntraOpThreads()).isEqualTo(1);
    assertThat(config.getScoringInterOpThreads()).isEqualTo(1);
    assertThat(config.getOptimizationLevel())
      .isEqualTo(OrtSession.SessionOptions.OptLevel.BASIC_OPT);
    assertThat(config.isOptimizedModelCacheEnabled()).isFalse();
  }

  @Test
  void defaults_factoryMethod_matchesEmptyBuilder() {
    var fromDefaults = RuntimeConfig.defaults();
    var fromBuilder = RuntimeConfig.builder().build();

    assertThat(fromDefaults.getEncoderIntraOpThreads())
      .isEqualTo(fromBuilder.getEncoderIntraOpThreads());
    assertThat(fromDefaults.getEncoderInterOpThreads())
      .isEqualTo(fromBuilder.getEncoderInterOpThreads());
    assertThat(fromDefaults.getScoringIntraOpThreads())
      .isEqualTo(fromBuilder.getScoringIntraOpThreads());
    assertThat(fromDefaults.getScoringInterOpThreads())
      .isEqualTo(fromBuilder.getScoringInterOpThreads());
    assertThat(fromDefaults.getOptimizationLevel())
      .isEqualTo(fromBuilder.getOptimizationLevel());
    assertThat(fromDefaults.isOptimizedModelCacheEnabled())
      .isEqualTo(fromBuilder.isOptimizedModelCacheEnabled());
  }
}
