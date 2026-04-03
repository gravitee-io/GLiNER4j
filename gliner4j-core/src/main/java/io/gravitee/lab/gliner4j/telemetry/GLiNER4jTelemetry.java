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

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.api.metrics.DoubleHistogram;
import io.opentelemetry.api.metrics.LongCounter;
import io.opentelemetry.api.metrics.Meter;

/**
 * Initializes and exposes OpenTelemetry instruments for GLiNER4j.
 *
 * <p>When no OTel SDK is present at runtime, all instruments are automatic no-ops
 * with near-zero overhead.
 */
public class GLiNER4jTelemetry {

  private static final String INSTRUMENTATION_NAME = "gliner4j";

  private final DoubleHistogram extractDuration;
  private final LongCounter textCount;
  private final LongCounter entityCount;

  public GLiNER4jTelemetry() {
    Meter meter = GlobalOpenTelemetry.get().getMeter(INSTRUMENTATION_NAME);
    this.extractDuration =
      meter
        .histogramBuilder("gliner4j.extract.duration")
        .setUnit("ms")
        .setDescription("End-to-end latency per extract/extractBatch call")
        .build();
    this.textCount =
      meter
        .counterBuilder("gliner4j.extract.text.count")
        .setDescription("Number of texts processed")
        .build();
    this.entityCount =
      meter
        .counterBuilder("gliner4j.extract.entity.count")
        .setDescription("Number of entities found")
        .build();
  }

  public void recordExtract(double durationMs, long texts, long entities) {
    extractDuration.record(durationMs);
    textCount.add(texts);
    entityCount.add(entities);
  }
}
