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

import static io.gravitee.lab.gliner4j.demo.Mode.NER;
import static io.gravitee.lab.gliner4j.demo.profile.ProfileType.BASE;

import ai.onnxruntime.OrtSession;
import io.gravitee.lab.gliner4j.GLiNER4jClassifier;
import io.gravitee.lab.gliner4j.GLiNER4jNER;
import io.gravitee.lab.gliner4j.demo.profile.Profile;
import io.gravitee.lab.gliner4j.demo.profile.ProfileType;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.testing.exporter.InMemoryMetricReader;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Profile-driven demo for GLiNER4j.
 *
 * <p>Pass a profile name as the first arg (default "base"). The profile JSON lives at
 * /profiles/{name}.json on the classpath and defines:
 *   - modelDir: ONNX model dir on disk
 *   - entities: NER entity definitions
 *   - nerSamples: sample texts for NER
 *   - labels (optional): classification labels — if present, classification demo runs too
 *   - classifySamples (optional): sample texts for classification
 */
public class GLiNER4jDemo {

  // ANSI codes
  private static final String RESET = "\033[0m";
  private static final String BOLD = "\033[1m";
  private static final String DIM = "\033[2m";
  private static final String WHITE = "\033[97m";
  private static final String GRAY = "\033[90m";

  // 12-color palette cycled by hash(label) — scales to any number of entity types.
  private static final Palette PALETTE = new Palette(
    new String[][] {
      { "\033[48;5;63m\033[97m", "\033[38;5;63m" }, // purple
      { "\033[48;5;37m\033[97m", "\033[38;5;37m" }, // teal
      { "\033[48;5;208m\033[97m", "\033[38;5;208m" }, // orange
      { "\033[48;5;170m\033[97m", "\033[38;5;170m" }, // pink
      { "\033[48;5;33m\033[97m", "\033[38;5;33m" }, // blue
      { "\033[48;5;35m\033[97m", "\033[38;5;35m" }, // green
      { "\033[48;5;160m\033[97m", "\033[38;5;160m" }, // red
      { "\033[48;5;142m\033[97m", "\033[38;5;142m" }, // olive
      { "\033[48;5;129m\033[97m", "\033[38;5;129m" }, // violet
      { "\033[48;5;24m\033[97m", "\033[38;5;24m" }, // deep blue
      { "\033[48;5;94m\033[97m", "\033[38;5;94m" }, // brown
      { "\033[48;5;100m\033[97m", "\033[38;5;100m" }, // dark olive
    }
  );

  private static final Printer printer = new Printer(PALETTE, DIM, RESET, BOLD, GRAY);

  public static void main(String[] args) {
    var profileName = args.length > 0 ? ProfileType.valueOf(args[0].toUpperCase()) : BASE;
    var variant = args.length > 1 ? args[1] : "onnx";
    Profile profile = new Profile(profileName);

    var metricReader = InMemoryMetricReader.create();
    var meterProvider = SdkMeterProvider.builder().registerMetricReader(metricReader).build();
    var openTelemetry = OpenTelemetrySdk.builder().setMeterProvider(meterProvider).buildAndRegisterGlobal();

    var modelDir = Path.of(profile.modelDir());
    var entities = profile
      .entities()
      .stream()
      .map(e -> new EntityDefinition(e.name(), e.description()))
      .toList();
    var labels = profile.hasClassification()
      ? profile
        .labels()
        .stream()
        .map(l -> new ClassificationLabel(l.name(), l.description()))
        .toList()
      : List.<ClassificationLabel>of();

    var runtimeConfig = RuntimeConfig.builder()
      .optimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
      .build();

    System.out.println();
    System.out.println(DIM + "  Loading model from " + modelDir + " (variant=" + variant + ") ..." + RESET);

    try (var gliner = GLiNER4jNER.load(modelDir, entities, variant, runtimeConfig)) {
      printer.banner(profile.displayName());
      printLegend(entities, modelDir);

      for (int i = 0; i < profile.nerSamples().size(); i++) {
        var text = profile.nerSamples().get(i);
        var results = gliner.extract(text);
        printer.nerResult(i + 1, text, results);
      }

      // Optional classification demo
      GLiNER4jClassifier classifier = null;
      if (profile.hasClassification()) {
        classifier = GLiNER4jClassifier.load(modelDir, labels, variant, runtimeConfig);
        printer.classificationBanner();
        System.out.println(DIM + "  Model: " + modelDir + RESET);
        System.out.println();
        printLabelLegend(labels);

        var samples = profile.classifySamples() == null ? List.<String>of() : profile.classifySamples();
        for (int i = 0; i < samples.size(); i++) {
          var text = samples.get(i);
          var results = classifier.classify(text);
          printer.classificationResult(i + 1, text, results);
        }
      }

      runInteractive(profile, gliner, classifier);

      if (classifier != null) classifier.close();
    }

    printer.metrics(metricReader.collectAllMetrics());
    openTelemetry.close();
  }

  private static void runInteractive(Profile profile, GLiNER4jNER gliner, GLiNER4jClassifier classifier) {
    printer.interactiveBanner(profile.hasClassification());
    var counter = new AtomicInteger(1);
    var mode = NER;

    try (var scanner = new Scanner(System.in)) {
      while (true) {
        System.out.print(BOLD + "  gliner" + RESET + DIM + " [" + mode.name().toLowerCase() + "]" + RESET + "> ");
        if (!scanner.hasNextLine()) break;
        var line = scanner.nextLine().strip();
        if (line.isEmpty()) continue;

        if (line.equalsIgnoreCase("/exit")) {
          break;
        } else if (line.equalsIgnoreCase("/ner")) {
          mode = NER;
          System.out.println(DIM + "  Switched to NER mode." + RESET);
          continue;
        } else if (line.equalsIgnoreCase("/classify")) {
          if (classifier == null) {
            System.out.println(GRAY + "  This profile has no classification labels — '/classify' unavailable." + RESET);
          } else {
            mode = Mode.CLASSIFY;
            System.out.println(DIM + "  Switched to classification mode." + RESET);
          }
          continue;
        } else if (line.equalsIgnoreCase("/help")) {
          printer.interactiveHelp(profile.hasClassification());
          continue;
        }

        switch (mode) {
          case NER -> printer.nerResult(counter.getAndIncrement(), line, gliner.extract(line));
          default -> printer.classificationResult(counter.getAndIncrement(), line, classifier.classify(line));
        }
      }
    }
  }

  private static void printLegend(List<EntityDefinition> entities, Path modelDir) {
    System.out.println(DIM + "  Model: " + modelDir + RESET);
    System.out.println(DIM + "  Labels: " + entities.size() + " entity types" + RESET);
    System.out.println();
    printer.entity(entities, EntityDefinition::name);
  }

  private static void printLabelLegend(List<ClassificationLabel> labels) {
    printer.entity(labels, ClassificationLabel::name);
  }
}
