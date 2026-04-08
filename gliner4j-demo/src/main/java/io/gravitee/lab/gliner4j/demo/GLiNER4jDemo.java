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
package io.gravitee.lab.gliner4j.demo;

import ai.onnxruntime.OrtSession;
import io.gravitee.lab.gliner4j.GLiNER4jClassifier;
import io.gravitee.lab.gliner4j.GLiNER4jNER;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.metrics.data.HistogramPointData;
import io.opentelemetry.sdk.metrics.data.MetricData;
import io.opentelemetry.sdk.testing.exporter.InMemoryMetricReader;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Demo application for GLiNER4jNER — runs NER on sample sentences.
 */
public class GLiNER4jDemo {

  // ANSI color codes
  private static final String RESET = "\033[0m";
  private static final String BOLD = "\033[1m";
  private static final String DIM = "\033[2m";
  private static final String WHITE = "\033[97m";
  private static final String GRAY = "\033[90m";

  // Entity type colors (background + foreground pairs)
  private static final Map<String, String[]> ENTITY_COLORS =
    new LinkedHashMap<>();

  static {
    ENTITY_COLORS.put(
      "person",
      new String[] { "\033[48;5;63m\033[97m", "\033[38;5;63m" }
    ); // purple
    ENTITY_COLORS.put(
      "organization",
      new String[] { "\033[48;5;37m\033[97m", "\033[38;5;37m" }
    ); // teal
    ENTITY_COLORS.put(
      "location",
      new String[] { "\033[48;5;208m\033[97m", "\033[38;5;208m" }
    ); // orange
    ENTITY_COLORS.put(
      "date",
      new String[] { "\033[48;5;170m\033[97m", "\033[38;5;170m" }
    ); // pink
    ENTITY_COLORS.put(
      "event",
      new String[] { "\033[48;5;196m\033[97m", "\033[38;5;196m" }
    ); // red
    ENTITY_COLORS.put(
      "product",
      new String[] { "\033[48;5;33m\033[97m", "\033[38;5;33m" }
    ); // blue
  }

  private static final Map<String, String[]> LABEL_COLORS =
    new LinkedHashMap<>();

  static {
    LABEL_COLORS.put(
      "positive",
      new String[] { "\033[48;5;35m\033[97m", "\033[38;5;35m" }
    ); // green
    LABEL_COLORS.put(
      "negative",
      new String[] { "\033[48;5;196m\033[97m", "\033[38;5;196m" }
    ); // red
    LABEL_COLORS.put(
      "neutral",
      new String[] { "\033[48;5;245m\033[97m", "\033[38;5;245m" }
    ); // gray
    LABEL_COLORS.put(
      "business",
      new String[] { "\033[48;5;33m\033[97m", "\033[38;5;33m" }
    ); // blue
    LABEL_COLORS.put(
      "health",
      new String[] { "\033[48;5;170m\033[97m", "\033[38;5;170m" }
    ); // pink
    LABEL_COLORS.put(
      "technology",
      new String[] { "\033[48;5;208m\033[97m", "\033[38;5;208m" }
    ); // orange
  }

  public static void main(String[] args) {
    // Initialize OpenTelemetry SDK with an in-memory reader to collect metrics
    var metricReader = InMemoryMetricReader.create();
    var meterProvider = SdkMeterProvider.builder()
      .registerMetricReader(metricReader)
      .build();
    var openTelemetry = OpenTelemetrySdk.builder()
      .setMeterProvider(meterProvider)
      .buildAndRegisterGlobal();

    var modelDir = Path.of("models/gliner2-base-onnx");

    var nerSamples = List.of(
      "Elon Musk unveiled the Tesla Cybertruck at the Los Angeles event in November 2019.",
      "The United Nations Climate Change Conference was held in Paris.",
      "Marie Curie won the Nobel Prize at the University of Paris in 1903.",
      "Apple released the iPhone 15 at their Cupertino headquarters on September 12, 2023."
    );

    var entities = List.of(
      new EntityDefinition("person", "Names of real individuals"),
      new EntityDefinition("organization"),
      new EntityDefinition(
        "location",
        "Cities, countries, geographical places"
      ),
      new EntityDefinition("date"),
      new EntityDefinition(
        "event",
        "Named events, conferences, historical events"
      ),
      new EntityDefinition("product")
    );

    var classifySamples = List.of(
      "This product is absolutely amazing, best purchase I've ever made!",
      "The service was terrible and the staff was rude.",
      "The quarterly earnings report shows a 15% increase in revenue.",
      "New research suggests that regular exercise improves mental health."
    );

    var labels = List.of(
      new ClassificationLabel("positive", "Positive sentiment or opinion"),
      new ClassificationLabel("negative", "Negative sentiment or opinion"),
      new ClassificationLabel("neutral", "Neutral, factual statement"),
      new ClassificationLabel("business", "Business, finance, economics"),
      new ClassificationLabel("health", "Health, medicine, wellness"),
      new ClassificationLabel("technology", "Technology, software, hardware")
    );

    // ── Load model ───────────────────────────────────────────────────

    var runtimeConfig = RuntimeConfig.builder()
      .optimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
      .build();

    System.out.println();
    System.out.println(
      DIM + "  Loading model from " + modelDir + " ..." + RESET
    );

    try (
      var gliner = GLiNER4jNER.load(modelDir, entities, runtimeConfig);
      var classifier = GLiNER4jClassifier.load(modelDir, labels, runtimeConfig)
    ) {
      // ── NER showcase ───────────────────────────────────────────────
      printBanner();
      System.out.println(DIM + "  Model: " + modelDir + RESET);
      System.out.println();
      printEntityConfig(entities);
      printLegend(entities);

      for (int i = 0; i < nerSamples.size(); i++) {
        var text = nerSamples.get(i);
        var results = gliner.extract(text);
        printResult(i + 1, text, results);
      }

      // ── Classification showcase ────────────────────────────────────
      printClassificationBanner();
      System.out.println(DIM + "  Model: " + modelDir + RESET);
      System.out.println();
      printLabelConfig(labels);
      printLabelLegend(labels);

      for (int i = 0; i < classifySamples.size(); i++) {
        var text = classifySamples.get(i);
        var results = classifier.classify(text);
        printClassificationResult(i + 1, text, results);
      }

      // ── Interactive phase ───────────────────────────────────────────

      printInteractiveBanner();
      var counter = new AtomicInteger(1);
      var mode = new String[] { "ner" }; // mutable holder for current mode

      try (var scanner = new Scanner(System.in)) {
        while (true) {
          System.out.print(
            BOLD +
              "  gliner" +
              RESET +
              DIM +
              " [" +
              mode[0] +
              "]" +
              RESET +
              "> "
          );
          if (!scanner.hasNextLine()) break;
          var line = scanner.nextLine().strip();
          if (line.isEmpty()) continue;

          if (line.equalsIgnoreCase("/exit")) {
            break;
          } else if (line.equalsIgnoreCase("/ner")) {
            mode[0] = "ner";
            System.out.println(DIM + "  Switched to NER mode." + RESET);
            continue;
          } else if (line.equalsIgnoreCase("/classify")) {
            mode[0] = "classify";
            System.out.println(
              DIM + "  Switched to classification mode." + RESET
            );
            continue;
          } else if (line.equalsIgnoreCase("/help")) {
            printInteractiveHelp();
            continue;
          }

          if ("ner".equals(mode[0])) {
            var results = gliner.extract(line);
            printResult(counter.getAndIncrement(), line, results);
          } else {
            var results = classifier.classify(line);
            printClassificationResult(counter.getAndIncrement(), line, results);
          }
        }
      }
    }

    // Collect and display metrics
    printMetrics(metricReader.collectAllMetrics());
    openTelemetry.close();
  }

  private static void printBanner() {
    System.out.println();
    System.out.println(
      BOLD +
        "  ┌─────────────────────────────────────────────────────────┐" +
        RESET
    );
    System.out.println(
      BOLD +
        "  │                 GLiNER4jNER — NER Demo                  │" +
        RESET
    );
    System.out.println(
      BOLD +
        "  │       Named Entity Recognition with ONNX Runtime        │" +
        RESET
    );
    System.out.println(
      BOLD +
        "  └─────────────────────────────────────────────────────────┘" +
        RESET
    );
    System.out.println();
  }

  private static void printEntityConfig(List<EntityDefinition> entities) {
    System.out.println(DIM + "  Config:" + RESET);
    for (var entity : entities) {
      var colors = ENTITY_COLORS.getOrDefault(
        entity.name(),
        new String[] { BOLD, WHITE }
      );
      if (entity.description().isBlank()) {
        System.out.println("    " + colors[1] + entity.name() + RESET);
      } else {
        System.out.println(
          "    " +
            colors[1] +
            entity.name() +
            RESET +
            GRAY +
            " — " +
            entity.description() +
            RESET
        );
      }
    }
    System.out.println();
  }

  private static void printLegend(List<EntityDefinition> entities) {
    System.out.print("  ");
    for (var entity : entities) {
      var colors = ENTITY_COLORS.getOrDefault(
        entity.name(),
        new String[] { BOLD, WHITE }
      );
      System.out.print(
        colors[0] + " " + entity.name().toUpperCase() + " " + RESET + "  "
      );
    }
    System.out.println();
    System.out.println(
      DIM +
        "  ─────────────────────────────────────────────────────────────" +
        RESET
    );
  }

  private static void printResult(
    int index,
    String text,
    Map<String, List<EntitySpan>> results
  ) {
    System.out.println();

    // Collect all spans and sort by start position
    var allSpans = new ArrayList<EntitySpan>();
    results.values().forEach(allSpans::addAll);
    allSpans.sort(Comparator.comparingInt(EntitySpan::start));

    // Print annotated text on the same line as the index number
    System.out.print("  " + DIM + index + "." + RESET + " ");
    int pos = 0;
    for (var span : allSpans) {
      if (span.start() > pos) {
        System.out.print(text.substring(pos, span.start()));
      }
      var colors = ENTITY_COLORS.getOrDefault(
        span.type(),
        new String[] { BOLD, WHITE }
      );
      System.out.print(colors[0] + " " + span.text() + " " + RESET);
      pos = span.end();
    }
    if (pos < text.length()) {
      System.out.print(text.substring(pos));
    }
    System.out.println();

    // Print entity details
    if (allSpans.isEmpty()) {
      System.out.println(GRAY + "     No entities found." + RESET);
    } else {
      for (var span : allSpans) {
        var colors = ENTITY_COLORS.getOrDefault(
          span.type(),
          new String[] { BOLD, WHITE }
        );
        var bar = confidenceBar(span.confidence());
        System.out.printf(
          "     %s%-14s%s  %-25s  %s %s%.0f%%%s%n",
          colors[1],
          span.type(),
          RESET,
          "\"" + span.text() + "\"",
          bar,
          DIM,
          span.confidence() * 100,
          RESET
        );
      }
    }
  }

  private static void printClassificationBanner() {
    System.out.println();
    System.out.println(
      BOLD +
        "  ┌─────────────────────────────────────────────────────────┐" +
        RESET
    );
    System.out.println(
      BOLD +
        "  │        GLiNER4jClassifier — Classification Demo         │" +
        RESET
    );
    System.out.println(
      BOLD +
        "  │          Text Classification with ONNX Runtime          │" +
        RESET
    );
    System.out.println(
      BOLD +
        "  └─────────────────────────────────────────────────────────┘" +
        RESET
    );
    System.out.println();
  }

  private static void printLabelConfig(List<ClassificationLabel> labels) {
    System.out.println(DIM + "  Config:" + RESET);
    for (var label : labels) {
      var colors = LABEL_COLORS.getOrDefault(
        label.name(),
        new String[] { BOLD, WHITE }
      );
      if (label.description().isBlank()) {
        System.out.println("    " + colors[1] + label.name() + RESET);
      } else {
        System.out.println(
          "    " +
            colors[1] +
            label.name() +
            RESET +
            GRAY +
            " — " +
            label.description() +
            RESET
        );
      }
    }
    System.out.println();
  }

  private static void printLabelLegend(List<ClassificationLabel> labels) {
    System.out.print("  ");
    for (var label : labels) {
      var colors = LABEL_COLORS.getOrDefault(
        label.name(),
        new String[] { BOLD, WHITE }
      );
      System.out.print(
        colors[0] + " " + label.name().toUpperCase() + " " + RESET + "  "
      );
    }
    System.out.println();
    System.out.println(
      DIM +
        "  ─────────────────────────────────────────────────────────────" +
        RESET
    );
  }

  private static void printClassificationResult(
    int index,
    String text,
    List<ClassificationResult> results
  ) {
    System.out.println();
    System.out.println("  " + DIM + index + "." + RESET + " " + text);

    if (results.isEmpty()) {
      System.out.println(GRAY + "     No labels matched." + RESET);
    } else {
      for (var result : results) {
        var colors = LABEL_COLORS.getOrDefault(
          result.label(),
          new String[] { BOLD, WHITE }
        );
        var bar = confidenceBar(result.confidence());
        System.out.printf(
          "     %s%-14s%s  %s %s%.0f%%%s%n",
          colors[1],
          result.label(),
          RESET,
          bar,
          DIM,
          result.confidence() * 100,
          RESET
        );
      }
    }
  }

  private static void printInteractiveBanner() {
    System.out.println();
    System.out.println(
      DIM +
        "  ─────────────────────────────────────────────────────────────" +
        RESET
    );
    System.out.println();
    System.out.println(
      BOLD +
        "  Interactive mode" +
        RESET +
        DIM +
        " — type a sentence and press Enter" +
        RESET
    );
    System.out.println(
      DIM + "  Commands: /ner  /classify  /help  /exit" + RESET
    );
    System.out.println();
  }

  private static void printInteractiveHelp() {
    System.out.println();
    System.out.println(DIM + "  Available commands:" + RESET);
    System.out.println(
      "    " + BOLD + "/ner" + RESET + DIM + "       Switch to NER mode" + RESET
    );
    System.out.println(
      "    " +
        BOLD +
        "/classify" +
        RESET +
        DIM +
        "  Switch to classification mode" +
        RESET
    );
    System.out.println(
      "    " + BOLD + "/help" + RESET + DIM + "      Show this help" + RESET
    );
    System.out.println(
      "    " +
        BOLD +
        "/exit" +
        RESET +
        DIM +
        "      Quit the interactive session" +
        RESET
    );
    System.out.println();
  }

  private static String confidenceBar(float confidence) {
    int filled = Math.round(confidence * 10);
    var sb = new StringBuilder(GRAY + "│" + RESET);
    for (int i = 0; i < 10; i++) {
      if (i < filled) {
        if (confidence >= 0.8f) {
          sb.append("\033[38;5;35m█" + RESET); // green
        } else if (confidence >= 0.5f) {
          sb.append("\033[38;5;220m█" + RESET); // yellow
        } else {
          sb.append("\033[38;5;196m█" + RESET); // red
        }
      } else {
        sb.append(GRAY + "░" + RESET);
      }
    }
    sb.append(GRAY + "│" + RESET);
    return sb.toString();
  }

  private static void printMetrics(Collection<MetricData> metrics) {
    System.out.println();
    System.out.println(
      BOLD +
        "  ┌─────────────────────────────────────────────────────────┐" +
        RESET
    );
    System.out.println(
      BOLD +
        "  │                    Telemetry Summary                    │" +
        RESET
    );
    System.out.println(
      BOLD +
        "  └─────────────────────────────────────────────────────────┘" +
        RESET
    );
    System.out.println();

    for (var metric : metrics) {
      var name = metric.getName();
      var desc = metric.getDescription();
      var unit = metric.getUnit();

      switch (metric.getType()) {
        case LONG_SUM -> {
          long total = metric
            .getLongSumData()
            .getPoints()
            .stream()
            .mapToLong(p -> p.getValue())
            .sum();
          System.out.printf(
            "  %s%-38s%s  %s%,d%s",
            DIM,
            name,
            RESET,
            BOLD,
            total,
            RESET
          );
          if (!unit.isEmpty()) {
            System.out.print(DIM + " " + unit + RESET);
          }
          System.out.println();
          System.out.println("  " + GRAY + desc + RESET);
          System.out.println();
        }
        case HISTOGRAM -> {
          for (var point : metric.getHistogramData().getPoints()) {
            var hp = (HistogramPointData) point;
            System.out.printf("  %s%-38s%s%n", DIM, name, RESET);
            System.out.println("  " + GRAY + desc + RESET);
            System.out.printf(
              "    count   %s%,d%s%n",
              BOLD,
              hp.getCount(),
              RESET
            );
            System.out.printf(
              "    min     %s%,.1f%s %s%n",
              BOLD,
              hp.getMin(),
              RESET,
              unit
            );
            System.out.printf(
              "    max     %s%,.1f%s %s%n",
              BOLD,
              hp.getMax(),
              RESET,
              unit
            );
            System.out.printf(
              "    avg     %s%,.1f%s %s%n",
              BOLD,
              hp.getCount() > 0 ? hp.getSum() / hp.getCount() : 0.0,
              RESET,
              unit
            );
            System.out.println();
          }
        }
        default -> {}
      }
    }

    System.out.println(
      DIM +
        "  ─────────────────────────────────────────────────────────────" +
        RESET
    );
    System.out.println(BOLD + "  Done." + RESET);
    System.out.println();
  }
}
