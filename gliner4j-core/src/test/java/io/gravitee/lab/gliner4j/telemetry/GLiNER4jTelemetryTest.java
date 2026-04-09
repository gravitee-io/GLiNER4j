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

import static org.assertj.core.api.Assertions.assertThat;

import io.opentelemetry.api.GlobalOpenTelemetry;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.metrics.data.MetricData;
import io.opentelemetry.sdk.testing.exporter.InMemoryMetricReader;
import java.util.Collection;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

class GLiNER4jTelemetryTest {

  private InMemoryMetricReader metricReader;

  @BeforeEach
  void setUp() {
    GlobalOpenTelemetry.resetForTest();
    metricReader = InMemoryMetricReader.create();
    var meterProvider = SdkMeterProvider.builder()
      .registerMetricReader(metricReader)
      .build();
    OpenTelemetrySdk.builder()
      .setMeterProvider(meterProvider)
      .buildAndRegisterGlobal();
  }

  @AfterEach
  void tearDown() {
    GlobalOpenTelemetry.resetForTest();
  }

  @Test
  void extractInstrumentsAreRegistered() {
    var telemetry = new GLiNER4jTelemetry("extract");
    telemetry.record(10.0, 1, 3);

    Collection<MetricData> metrics = metricReader.collectAllMetrics();
    assertThat(metrics)
      .extracting(MetricData::getName)
      .containsExactlyInAnyOrder(
        "gliner4j.extract.duration",
        "gliner4j.extract.text.count",
        "gliner4j.extract.result.count"
      );
  }

  @Test
  void classifyInstrumentsAreRegistered() {
    var telemetry = new GLiNER4jTelemetry("classify");
    telemetry.record(10.0, 1, 2);

    Collection<MetricData> metrics = metricReader.collectAllMetrics();
    assertThat(metrics)
      .extracting(MetricData::getName)
      .containsExactlyInAnyOrder(
        "gliner4j.classify.duration",
        "gliner4j.classify.text.count",
        "gliner4j.classify.result.count"
      );
  }

  @Test
  void extractAndClassifyMetricsAreIndependent() {
    var extract = new GLiNER4jTelemetry("extract");
    var classify = new GLiNER4jTelemetry("classify");
    extract.record(10.0, 1, 3);
    classify.record(20.0, 2, 5);

    Collection<MetricData> metrics = metricReader.collectAllMetrics();
    assertThat(metrics)
      .extracting(MetricData::getName)
      .containsExactlyInAnyOrder(
        "gliner4j.extract.duration",
        "gliner4j.extract.text.count",
        "gliner4j.extract.result.count",
        "gliner4j.classify.duration",
        "gliner4j.classify.text.count",
        "gliner4j.classify.result.count"
      );

    var extractTextCount = findMetric("gliner4j.extract.text.count");
    var classifyTextCount = findMetric("gliner4j.classify.text.count");
    assertThat(
      extractTextCount.getLongSumData().getPoints().iterator().next().getValue()
    ).isEqualTo(1);
    assertThat(
      classifyTextCount
        .getLongSumData()
        .getPoints()
        .iterator()
        .next()
        .getValue()
    ).isEqualTo(2);
  }

  @Test
  void recordsCorrectDuration() {
    var telemetry = new GLiNER4jTelemetry("extract");
    telemetry.record(42.5, 1, 0);

    var duration = findMetric("gliner4j.extract.duration");
    assertThat(duration).isNotNull();
    assertThat(duration.getUnit()).isEqualTo("ms");
    var histogramData = duration.getHistogramData();
    assertThat(histogramData.getPoints()).hasSize(1);
    var point = histogramData.getPoints().iterator().next();
    assertThat(point.getSum()).isEqualTo(42.5);
    assertThat(point.getCount()).isEqualTo(1);
  }

  @Test
  void recordsTextCount() {
    var telemetry = new GLiNER4jTelemetry("extract");
    telemetry.record(10.0, 5, 0);

    var textCount = findMetric("gliner4j.extract.text.count");
    assertThat(textCount).isNotNull();
    var sumData = textCount.getLongSumData();
    assertThat(sumData.getPoints()).hasSize(1);
    var point = sumData.getPoints().iterator().next();
    assertThat(point.getValue()).isEqualTo(5);
  }

  @Test
  void recordsResultCount() {
    var telemetry = new GLiNER4jTelemetry("extract");
    telemetry.record(10.0, 1, 7);

    var resultCount = findMetric("gliner4j.extract.result.count");
    assertThat(resultCount).isNotNull();
    var sumData = resultCount.getLongSumData();
    assertThat(sumData.getPoints()).hasSize(1);
    var point = sumData.getPoints().iterator().next();
    assertThat(point.getValue()).isEqualTo(7);
  }

  @Test
  void accumulatesAcrossMultipleCalls() {
    var telemetry = new GLiNER4jTelemetry("extract");
    telemetry.record(10.0, 1, 3);
    telemetry.record(20.0, 2, 5);

    var textCount = findMetric("gliner4j.extract.text.count");
    var point = textCount.getLongSumData().getPoints().iterator().next();
    assertThat(point.getValue()).isEqualTo(3);

    var resultCount = findMetric("gliner4j.extract.result.count");
    var ePoint = resultCount.getLongSumData().getPoints().iterator().next();
    assertThat(ePoint.getValue()).isEqualTo(8);

    var duration = findMetric("gliner4j.extract.duration");
    var hPoint = duration.getHistogramData().getPoints().iterator().next();
    assertThat(hPoint.getSum()).isEqualTo(30.0);
    assertThat(hPoint.getCount()).isEqualTo(2);
  }

  private MetricData findMetric(String name) {
    return metricReader
      .collectAllMetrics()
      .stream()
      .filter(m -> m.getName().equals(name))
      .findFirst()
      .orElse(null);
  }
}
