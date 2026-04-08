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
 * <p>Each instance is scoped to a specific operation (e.g. {@code "extract"} for NER,
 * {@code "classify"} for classification), producing distinct metric names such as
 * {@code gliner4j.extract.duration} vs {@code gliner4j.classify.duration}.
 *
 * <p>When no OTel SDK is present at runtime, all instruments are automatic no-ops
 * with near-zero overhead.
 */
public class GLiNER4jTelemetry {

  private static final String INSTRUMENTATION_NAME = "gliner4j";

  private final DoubleHistogram duration;
  private final LongCounter textCount;
  private final LongCounter resultCount;

  /**
   * Creates telemetry instruments scoped to the given operation.
   *
   * @param operation the operation name used as metric prefix (e.g. "extract", "classify")
   */
  public GLiNER4jTelemetry(String operation) {
    Meter meter = GlobalOpenTelemetry.get().getMeter(INSTRUMENTATION_NAME);
    String prefix = "gliner4j." + operation;
    this.duration = meter
      .histogramBuilder(prefix + ".duration")
      .setUnit("ms")
      .setDescription("End-to-end latency per " + operation + " call")
      .build();
    this.textCount = meter
      .counterBuilder(prefix + ".text.count")
      .setDescription("Number of texts processed")
      .build();
    this.resultCount = meter
      .counterBuilder(prefix + ".result.count")
      .setDescription("Number of results produced")
      .build();
  }

  /**
   * Records metrics for a completed operation.
   *
   * @param durationMs wall-clock duration in milliseconds
   * @param texts number of texts processed
   * @param results number of results produced (entities for NER, labels for classification)
   */
  public void record(double durationMs, long texts, long results) {
    duration.record(durationMs);
    textCount.add(texts);
    resultCount.add(results);
  }
}
