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
package io.gravitee.lab.gliner4j.telemetry;

import static org.assertj.core.api.Assertions.assertThatCode;

import io.opentelemetry.api.GlobalOpenTelemetry;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Verifies that telemetry works as a no-op when no OTel SDK is configured
 * (only opentelemetry-api on the classpath).
 */
class GLiNER4jTelemetryNoopTest {

  @BeforeEach
  void setUp() {
    // Reset to ensure no SDK is registered — GlobalOpenTelemetry returns noop
    GlobalOpenTelemetry.resetForTest();
  }

  @AfterEach
  void tearDown() {
    GlobalOpenTelemetry.resetForTest();
  }

  @Test
  void creatingTelemetryWithoutSdkDoesNotThrow() {
    assertThatCode(() ->
      new GLiNER4jTelemetry("extract")
    ).doesNotThrowAnyException();
  }

  @Test
  void recordingMetricsWithoutSdkDoesNotThrow() {
    var telemetry = new GLiNER4jTelemetry("extract");
    assertThatCode(() ->
      telemetry.record(50.0, 1, 5)
    ).doesNotThrowAnyException();
  }

  @Test
  void multipleRecordingsWithoutSdkDoNotThrow() {
    var telemetry = new GLiNER4jTelemetry("classify");
    assertThatCode(() -> {
      telemetry.record(10.0, 1, 3);
      telemetry.record(20.0, 5, 12);
      telemetry.record(0.0, 0, 0);
    }).doesNotThrowAnyException();
  }
}
