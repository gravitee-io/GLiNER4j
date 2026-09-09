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
import io.gravitee.lab.gliner4j.demo.profile.Profile;
import io.gravitee.lab.gliner4j.demo.profile.ProfileType;
import io.gravitee.lab.gliner4j.demo.profile.RouterDef;
import io.gravitee.lab.gliner4j.demo.profile.RouterFamily;
import io.gravitee.lab.gliner4j.extractor.RelationExtractor;
import io.gravitee.lab.gliner4j.extractor.SchemaExtractor;
import io.gravitee.lab.gliner4j.llamacpp.DecoderKvRouter;
import io.gravitee.lab.gliner4j.llamacpp.DecoderKvSession;
import io.gravitee.lab.gliner4j.llamacpp.StreamingSpanNer;
import io.gravitee.lab.gliner4j.llamacpp.StreamingSpanSession;
import io.gravitee.lab.gliner4j.runtime.ExecutionProvider;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import io.gravitee.lab.gliner4j.schema.RelationDefinition;
import io.gravitee.lab.gliner4j.schema.Schema;
import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import io.gravitee.lab.gliner4j.utils.GlinerNerSupport;
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
 *   - variant (default "onnx"): ONNX model folder (onnx | onnx_fp16 | onnx_quantized); for
 *     llama.cpp bundles the GGUF quantization (f16 | q8_0) where the bundle ships several.
 *   - executionProvider (default "auto"): ORT backend — auto | cpu | cuda | openvino | coreml.
 *     "auto" detects the best provider compiled into the native runtime.
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
    var executionProvider = args.length > 2 ? ExecutionProvider.fromString(args[2]) : ExecutionProvider.AUTO;
    Profile profile = new Profile(profileName);

    if (!profile.hasEntities() && !profile.hasLabels() && !profile.hasRouter()) {
      System.out.println(
        GRAY +
          "  Profile '" +
          profileName.name().toLowerCase() +
          "' has no entities, labels or router — nothing to run." +
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

    // The bundle's engine decides what "variant" and "ep" mean: ONNX Runtime uses the onnx*
    // folder and its provider resolution; the llama.cpp/ggml engine uses the GGUF quantization
    // (f16 / q8_0, else the bundle default) and offloads to Metal / CUDA under auto.
    var bundleConfig = io.gravitee.lab.gliner4j.GLiNER4jConfig.load(modelDir);
    boolean llamacpp = bundleConfig.getEngine() == io.gravitee.lab.gliner4j.arch.Engine.LLAMACPP;
    String device = llamacpp
      ? (switch (executionProvider) {
          case CPU, OPENVINO -> "cpu";
          case AUTO, CUDA -> "gpu (Metal/CUDA/Vulkan if available)";
        })
      : ExecutionProvider.resolve(executionProvider).name().toLowerCase();
    // Only llama.cpp-native backbones (GLiClass ModernBERT, the Qwen3 families) ship several
    // GGUF quantizations; the DeBERTa families are one model.gguf and ignore the variant.
    String defaultGguf = bundleConfig.archString("backbone_gguf", null);
    String variantShown = !llamacpp
      ? variant
      : defaultGguf == null
        ? "gguf/model.gguf"
        : (java.util.Set.of("f16", "q8_0").contains(variant) ? variant : defaultGguf);
    System.out.println();
    System.out.println(
      DIM +
        "  Loading model from " +
        modelDir +
        " (engine=" +
        bundleConfig.getEngine().configValue() +
        ", variant=" +
        variantShown +
        ", ep=" +
        executionProvider.name().toLowerCase() +
        " -> " +
        device +
        ") ..." +
        RESET
    );

    GLiNER4jNER gliner = null;
    GLiNER4jClassifier classifier = null;
    SchemaExtractor schemaExtractor = null;
    RelationExtractor relationExtractor = null;
    GLiNER4j unified = null;
    DecoderKvRouter router = null;
    StreamingSpanNer streamer = null;
    try {
      // Optional LLM-routing demo (GLiClass decoder-kv bundles on llama.cpp)
      if (profile.hasRouter()) {
        router = DecoderKvRouter.load(modelDir, variant, runtimeConfig);
        var def = profile.router();
        printer.routerBanner(profile.displayName());
        System.out.println(DIM + "  Model: " + modelDir + RESET);
        System.out.println(DIM + "  Families: " + def.families().size() + " label families" + RESET);
        System.out.println();
        printer.entity(def.families(), RouterFamily::name);

        var samples = def.samples() == null ? List.<String>of() : def.samples();
        for (int i = 0; i < samples.size(); i++) {
          var text = samples.get(i);
          long t0 = System.nanoTime();
          var results = route(router, def, text);
          printer.routerResult(i + 1, text, results, 3, elapsedMs(t0));
        }

        var turns = def.session() == null ? List.<String>of() : def.session();
        if (!turns.isEmpty()) {
          var family = def.sessionFamilyDef();
          printer.sessionBanner(family.name());
          try (var session = router.openSession("demo")) {
            for (int i = 0; i < turns.size(); i++) {
              long t0 = System.nanoTime();
              session.append(turns.get(i));
              var results = family.singleLabel()
                ? session.classifySingleLabel(family.labels())
                : session.classify(family.labels(), 0.0f);
              printer.sessionTurn(i + 1, turns.get(i), session.cachedTokens(), results, elapsedMs(t0));
            }
          }
        }
      }

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
        schemaExtractor = SchemaExtractor.load(modelDir, structures, variant, runtimeConfig);
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
        relationExtractor = RelationExtractor.load(modelDir, relations, variant, runtimeConfig);
        try {
          unified = GLiNER4j.load(modelDir, variant, runtimeConfig);
        } catch (UnsupportedOperationException e) {
          // Families without the GLiNER2 unified (entities + relations + structures) graph.
          System.out.println(GRAY + "  Combined extraction skipped: " + e.getMessage() + RESET);
        }

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

        if (unified != null) {
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
      }

      // Optional streaming-NER demo (gliner-streaming-span bundles on llama.cpp)
      if (profile.hasStream()) {
        streamer = StreamingSpanNer.load(modelDir, variant, runtimeConfig);
        var labelNames = entities.stream().map(EntityDefinition::name).toList();
        printer.streamBanner();
        int n = 1;
        for (var chunks : profile.stream()) {
          System.out.println();
          System.out.println(DIM + "  Session " + n++ + RESET);
          try (var session = streamer.openSession("demo", labelNames)) {
            int turn = 1;
            for (var chunk : chunks) {
              long t0 = System.nanoTime();
              var snapshot = session.append(chunk, 0.5f);
              printer.streamTurn(
                turn++,
                chunk,
                session.cachedTokens(),
                session.text(),
                GlinerNerSupport.groupByType(snapshot),
                elapsedMs(t0)
              );
            }
          }
        }
      }

      runInteractive(
        profile,
        gliner,
        classifier,
        schemaExtractor,
        relationExtractor,
        unified,
        router,
        streamer,
        entities,
        relations
      );
    } finally {
      if (streamer != null) streamer.close();
      if (router != null) router.close();
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
    SchemaExtractor schemaExtractor,
    RelationExtractor relationExtractor,
    GLiNER4j unified,
    DecoderKvRouter router,
    StreamingSpanNer streamer,
    List<EntityDefinition> entities,
    List<RelationDefinition> relations
  ) {
    var hasEntities = gliner != null;
    var hasRouter = router != null;
    var hasStream = streamer != null;
    var streamLabels = entities.stream().map(EntityDefinition::name).toList();
    StreamingSpanSession stream = null;
    printer.interactiveBanner(
      hasEntities,
      profile.hasLabels(),
      profile.hasSchema(),
      profile.hasRelations(),
      hasRouter,
      hasStream
    );
    var counter = new AtomicInteger(1);
    var mode = hasEntities ? NER : hasRouter ? Mode.ROUTE : Mode.CLASSIFY;
    RouterDef routerDef = hasRouter ? profile.router() : null;
    DecoderKvSession session = null;
    var turn = new AtomicInteger(1);
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
        } else if (line.equalsIgnoreCase("/route")) {
          if (!hasRouter) {
            System.out.println(GRAY + "  This profile has no router — '/route' unavailable." + RESET);
          } else {
            mode = Mode.ROUTE;
            System.out.println(DIM + "  Switched to routing mode (every line is routed on its own)." + RESET);
          }
          continue;
        } else if (line.equalsIgnoreCase("/session")) {
          if (!hasRouter) {
            System.out.println(GRAY + "  This profile has no router — '/session' unavailable." + RESET);
          } else {
            mode = Mode.SESSION;
            if (session == null) session = router.openSession("interactive");
            System.out.println(
              DIM + "  Switched to session mode: each line is appended to the conversation and re-routed." + RESET
            );
          }
          continue;
        } else if (line.equalsIgnoreCase("/reset")) {
          if (session != null) {
            session.close();
            session = null;
            turn.set(1);
          }
          if (mode == Mode.SESSION && router != null) session = router.openSession("interactive");
          System.out.println(DIM + "  Session reset." + RESET);
          continue;
        } else if (line.equalsIgnoreCase("/help")) {
          printer.interactiveHelp(
            hasEntities,
            profile.hasLabels(),
            profile.hasSchema(),
            profile.hasRelations(),
            hasRouter
          );
          continue;
        }

        long t0 = System.nanoTime();
        switch (mode) {
          case ROUTE -> printer.routerResult(
            counter.getAndIncrement(),
            line,
            route(router, routerDef, line),
            3,
            elapsedMs(t0)
          );
          case STREAM -> {
            var chunk = (stream.words() == 0 ? "" : " ") + line;
            var snapshot = stream.append(chunk, 0.5f);
            printer.streamTurn(
              turn.getAndIncrement(),
              line,
              stream.cachedTokens(),
              stream.text(),
              GlinerNerSupport.groupByType(snapshot),
              elapsedMs(t0)
            );
          }
          case SESSION -> {
            var family = routerDef.sessionFamilyDef();
            session.append((session.cachedTokens() == 0 ? "" : " ") + line);
            var results = family.singleLabel()
              ? session.classifySingleLabel(family.labels())
              : session.classify(family.labels(), 0.0f);
            printer.sessionTurn(turn.getAndIncrement(), line, session.cachedTokens(), results, elapsedMs(t0));
          }
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
    } finally {
      if (session != null) session.close();
      if (stream != null) stream.close();
    }
  }

  /** Scores {@code text} against every label family of the router profile, in profile order. */
  private static Map<String, List<ClassificationResult>> route(DecoderKvRouter router, RouterDef def, String text) {
    var out = new LinkedHashMap<String, List<ClassificationResult>>();
    for (var family : def.families()) {
      out.put(
        family.name(),
        family.singleLabel()
          ? router.classifySingleLabel(text, family.labels())
          : router.classify(text, family.labels(), 0.0f)
      );
    }
    return out;
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
