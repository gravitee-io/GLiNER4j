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
import io.gravitee.lab.gliner4j.GLiNER4j;
import io.gravitee.lab.gliner4j.GLiNER4jClassifier;
import io.gravitee.lab.gliner4j.GLiNER4jNER;
import io.gravitee.lab.gliner4j.GLiNER4jRelationExtractor;
import io.gravitee.lab.gliner4j.GLiNER4jSchemaExtractor;
import io.gravitee.lab.gliner4j.demo.profile.Profile;
import io.gravitee.lab.gliner4j.demo.profile.ProfileType;
import io.gravitee.lab.gliner4j.runtime.ExecutionProvider;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.Schema;
import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.metrics.SdkMeterProvider;
import io.opentelemetry.sdk.testing.exporter.InMemoryMetricReader;
import java.nio.file.Path;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Profile-driven demo for GLiNER4j.
 *
 * <p>Args: {@code [profile] [variant] [executionProvider]}.
 *   - profile (default "base"): which model + vocabulary to load.
 *   - variant (default "onnx"): ONNX model folder (onnx | onnx_fp16 | onnx_quantized).
 *   - executionProvider (default "cpu"): ORT backend — cpu | cuda | openvino | coreml.
 *
 * <p>The profile JSON lives at
 * /profiles/{name}.json on the classpath and defines:
 *   - modelDir: ONNX model dir on disk
 *   - entities: NER entity definitions
 *   - nerSamples: sample texts for NER
 *   - labels (optional): classification labels — if present, classification demo runs too
 *   - classifySamples (optional): sample texts for classification
 *   - structures (optional): JSON structure definitions — if present, schema demo runs too
 *   - schemaSamples (optional): sample texts for schema extraction
 *   - relations (optional): relation definitions — if present, relation + combined demos run too
 *   - relationSamples (optional): sample texts for relation extraction
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
    // 3rd arg selects the ONNX execution provider: cpu (default) | cuda | openvino | coreml.
    var executionProvider = args.length > 2 ? ExecutionProvider.fromString(args[2]) : ExecutionProvider.CPU;
    Profile profile = new Profile(profileName);

    if (!profile.hasEntities() && !profile.hasLabels()) {
      System.out.println(
        GRAY +
          "  Profile '" +
          profileName.name().toLowerCase() +
          "' has no entities or labels — nothing to run." +
          RESET
      );
      return;
    }

    var metricReader = InMemoryMetricReader.create();
    var meterProvider = SdkMeterProvider.builder().registerMetricReader(metricReader).build();
    var openTelemetry = OpenTelemetrySdk.builder().setMeterProvider(meterProvider).buildAndRegisterGlobal();

    var modelDir = Path.of(profile.modelDir());
    var entities = profile.hasEntities()
      ? profile
        .entities()
        .stream()
        .map(e -> new EntityDefinition(e.name(), e.description()))
        .toList()
      : List.<EntityDefinition>of();
    var labels = profile.hasLabels()
      ? profile
        .labels()
        .stream()
        .map(l -> new ClassificationLabel(l.name(), l.description()))
        .toList()
      : List.<ClassificationLabel>of();
    var structures = profile.hasSchema()
      ? profile
        .structures()
        .stream()
        .map(s -> s.toDefinition())
        .toList()
      : List.<StructureDefinition>of();
    var relations = profile.hasRelations()
      ? profile
        .relations()
        .stream()
        .map(r -> new RelationDefinition(r.name(), r.description()))
        .toList()
      : List.<RelationDefinition>of();

    var runtimeConfig = RuntimeConfig.builder()
      .optimizationLevel(OrtSession.SessionOptions.OptLevel.EXTENDED_OPT)
      .executionProvider(executionProvider)
      .build();

    System.out.println();
    System.out.println(
      DIM +
        "  Loading model from " +
        modelDir +
        " (variant=" +
        variant +
        ", ep=" +
        executionProvider.name().toLowerCase() +
        ") ..." +
        RESET
    );

    GLiNER4jNER gliner = null;
    GLiNER4jClassifier classifier = null;
    GLiNER4jSchemaExtractor schemaExtractor = null;
    GLiNER4jRelationExtractor relationExtractor = null;
    GLiNER4j unified = null;
    try {
      // Optional NER demo (skipped for classification-only profiles)
      if (profile.hasEntities()) {
        gliner = GLiNER4jNER.load(modelDir, entities, variant, runtimeConfig);
        printer.banner(profile.displayName());
        printLegend(entities, modelDir);

        var samples = profile.nerSamples() == null ? List.<String>of() : profile.nerSamples();
        for (int i = 0; i < samples.size(); i++) {
          var text = samples.get(i);
          long t0 = System.nanoTime();
          var results = gliner.extract(text);
          printer.nerResult(i + 1, text, results, elapsedMs(t0));
        }
      }

      // Optional classification demo
      if (profile.hasLabels()) {
        classifier = GLiNER4jClassifier.load(modelDir, labels, variant, runtimeConfig);
        printer.classificationBanner();
        System.out.println(DIM + "  Model: " + modelDir + RESET);
        System.out.println();
        printLabelLegend(labels);

        var samples = profile.classifySamples() == null ? List.<String>of() : profile.classifySamples();
        for (int i = 0; i < samples.size(); i++) {
          var text = samples.get(i);
          long t0 = System.nanoTime();
          var results = classifier.classify(text);
          printer.classificationResult(i + 1, text, results, elapsedMs(t0));
        }
      }

      // Optional schema extraction demo
      if (profile.hasSchema()) {
        schemaExtractor = GLiNER4jSchemaExtractor.load(modelDir, structures, variant, runtimeConfig);
        printer.schemaBanner();
        System.out.println(DIM + "  Model: " + modelDir + RESET);
        System.out.println();
        printer.structureLegend(structures);

        var samples = profile.schemaSamples() == null ? List.<String>of() : profile.schemaSamples();
        for (int i = 0; i < samples.size(); i++) {
          var text = samples.get(i);
          long t0 = System.nanoTime();
          var results = schemaExtractor.extract(text);
          printer.schemaResult(i + 1, text, results, elapsedMs(t0));
        }
      }

      // Optional relation + combined extraction demos
      if (profile.hasRelations()) {
        relationExtractor = GLiNER4jRelationExtractor.load(modelDir, relations, variant, runtimeConfig);
        unified = GLiNER4j.load(modelDir, variant, runtimeConfig);

        printer.relationBanner();
        System.out.println(DIM + "  Model: " + modelDir + RESET);
        System.out.println();
        printRelationLegend(relations);

        var relSamples = profile.relationSamples() == null ? List.<String>of() : profile.relationSamples();
        for (int i = 0; i < relSamples.size(); i++) {
          var text = relSamples.get(i);
          long t0 = System.nanoTime();
          var results = relationExtractor.extract(text);
          printer.relationResult(i + 1, text, results, elapsedMs(t0));
        }

        printer.combinedBanner();
        System.out.println(DIM + "  Model: " + modelDir + RESET);
        System.out.println();
        var schema = Schema.builder().entities(entities).relations(relations).build();
        for (int i = 0; i < relSamples.size(); i++) {
          var text = relSamples.get(i);
          long t0 = System.nanoTime();
          var result = unified.extract(text, schema);
          printer.combinedResult(i + 1, text, result, elapsedMs(t0));
        }
      }

      runInteractive(profile, gliner, classifier, schemaExtractor, relationExtractor, unified, entities, relations);
    } finally {
      if (gliner != null) gliner.close();
      if (classifier != null) classifier.close();
      if (schemaExtractor != null) schemaExtractor.close();
      if (relationExtractor != null) relationExtractor.close();
      if (unified != null) unified.close();
    }

    printer.metrics(metricReader.collectAllMetrics());
    openTelemetry.close();
  }

  private static void runInteractive(
    Profile profile,
    GLiNER4jNER gliner,
    GLiNER4jClassifier classifier,
    GLiNER4jSchemaExtractor schemaExtractor,
    GLiNER4jRelationExtractor relationExtractor,
    GLiNER4j unified,
    List<EntityDefinition> entities,
    List<RelationDefinition> relations
  ) {
    var hasEntities = gliner != null;
    printer.interactiveBanner(hasEntities, profile.hasLabels(), profile.hasSchema(), profile.hasRelations());
    var counter = new AtomicInteger(1);
    var mode = hasEntities ? NER : Mode.CLASSIFY;
    Schema combinedSchema = profile.hasRelations()
      ? Schema.builder().entities(entities).relations(relations).build()
      : null;

    try (var scanner = new Scanner(System.in)) {
      while (true) {
        System.out.print(BOLD + "  gliner" + RESET + DIM + " [" + mode.name().toLowerCase() + "]" + RESET + "> ");
        if (!scanner.hasNextLine()) break;
        var line = scanner.nextLine().strip();
        if (line.isEmpty()) continue;

        if (line.equalsIgnoreCase("/exit")) {
          break;
        } else if (line.equalsIgnoreCase("/ner")) {
          if (!hasEntities) {
            System.out.println(GRAY + "  This profile has no NER entities — '/ner' unavailable." + RESET);
          } else {
            mode = NER;
            System.out.println(DIM + "  Switched to NER mode." + RESET);
          }
          continue;
        } else if (line.equalsIgnoreCase("/classify")) {
          if (classifier == null) {
            System.out.println(GRAY + "  This profile has no classification labels — '/classify' unavailable." + RESET);
          } else {
            mode = Mode.CLASSIFY;
            System.out.println(DIM + "  Switched to classification mode." + RESET);
          }
          continue;
        } else if (line.equalsIgnoreCase("/schema")) {
          if (schemaExtractor == null) {
            System.out.println(GRAY + "  This profile has no structures — '/schema' unavailable." + RESET);
          } else {
            mode = Mode.SCHEMA;
            System.out.println(DIM + "  Switched to schema mode." + RESET);
          }
          continue;
        } else if (line.equalsIgnoreCase("/relations")) {
          if (relationExtractor == null) {
            System.out.println(GRAY + "  This profile has no relations — '/relations' unavailable." + RESET);
          } else {
            mode = Mode.RELATIONS;
            System.out.println(DIM + "  Switched to relation extraction mode." + RESET);
          }
          continue;
        } else if (line.equalsIgnoreCase("/extract")) {
          if (unified == null) {
            System.out.println(GRAY + "  This profile has no relations — '/extract' unavailable." + RESET);
          } else {
            mode = Mode.EXTRACT;
            System.out.println(DIM + "  Switched to combined extraction mode." + RESET);
          }
          continue;
        } else if (line.equalsIgnoreCase("/help")) {
          printer.interactiveHelp(hasEntities, profile.hasLabels(), profile.hasSchema(), profile.hasRelations());
          continue;
        }

        long t0 = System.nanoTime();
        switch (mode) {
          case NER -> printer.nerResult(counter.getAndIncrement(), line, gliner.extract(line), elapsedMs(t0));
          case CLASSIFY -> printer.classificationResult(
            counter.getAndIncrement(),
            line,
            classifier.classify(line),
            elapsedMs(t0)
          );
          case SCHEMA -> printer.schemaResult(
            counter.getAndIncrement(),
            line,
            schemaExtractor.extract(line),
            elapsedMs(t0)
          );
          case RELATIONS -> printer.relationResult(
            counter.getAndIncrement(),
            line,
            relationExtractor.extract(line),
            elapsedMs(t0)
          );
          case EXTRACT -> printer.combinedResult(
            counter.getAndIncrement(),
            line,
            unified.extract(line, combinedSchema),
            elapsedMs(t0)
          );
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

  private static void printRelationLegend(List<RelationDefinition> relations) {
    System.out.println(DIM + "  Relations: " + relations.size() + " relation types" + RESET);
    System.out.println();
    printer.entity(relations, RelationDefinition::name);
  }

  /** Milliseconds elapsed since {@code startNanos}, for per-prediction latency display. */
  private static double elapsedMs(long startNanos) {
    return (System.nanoTime() - startNanos) / 1_000_000.0;
  }
}
