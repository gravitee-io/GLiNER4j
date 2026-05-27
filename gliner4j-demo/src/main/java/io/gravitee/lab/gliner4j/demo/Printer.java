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

import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import io.gravitee.lab.gliner4j.schema.EntitySpan;
import io.gravitee.lab.gliner4j.schema.ExtractionResult;
import io.gravitee.lab.gliner4j.schema.RelationInstance;
import io.gravitee.lab.gliner4j.schema.StructureDefinition;
import io.gravitee.lab.gliner4j.schema.StructureInstance;
import io.gravitee.lab.gliner4j.schema.StructureValue;
import io.gravitee.lab.gliner4j.schema.StructureValue.ListValue;
import io.gravitee.lab.gliner4j.schema.StructureValue.StringValue;
import io.opentelemetry.sdk.metrics.data.LongPointData;
import io.opentelemetry.sdk.metrics.data.MetricData;
import java.util.*;
import java.util.function.Function;

public class Printer {

  private final Palette palette;
  private final String dim;
  private final String reset;
  private final String bold;
  private final String gray;

  public Printer(Palette palette, String dim, String reset, String bold, String gray) {
    this.palette = palette;
    this.dim = dim;
    this.reset = reset;
    this.bold = bold;
    this.gray = gray;
  }

  public <T> void entity(List<T> entities, Function<T, String> toString) {
    int col = 0;
    System.out.print("  ");
    for (var entity : entities) {
      var name = toString.apply(entity);
      var c = palette.colorsFor(name);
      var chip = c[0] + " " + name + " " + reset + "  ";
      System.out.print(chip);
      col += name.length() + 4;
      if (col > 70) {
        System.out.println();
        System.out.print("  ");
        col = 0;
      }
    }
    if (col > 0) System.out.println();
    System.out.println(dim + "  ─────────────────────────────────────────────────────────────" + reset);
  }

  public void interactiveBanner(boolean hasEntities, boolean hasLabels, boolean hasSchema, boolean hasRelations) {
    System.out.println();
    System.out.println(dim + "  ─────────────────────────────────────────────────────────────" + reset);
    System.out.println();
    System.out.println(bold + "  Interactive mode" + reset + dim + " — type a sentence and press Enter" + reset);
    var commands = new StringBuilder("  Commands:");
    if (hasEntities) commands.append("  /ner");
    if (hasLabels) commands.append("  /classify");
    if (hasSchema) commands.append("  /schema");
    if (hasRelations) commands.append("  /relations  /extract");
    commands.append("  /help  /exit");
    System.out.println(dim + commands + reset);
    System.out.println();
  }

  public void banner(String displayName) {
    System.out.println();
    System.out.println(bold + "  ┌─────────────────────────────────────────────────────────┐" + reset);
    System.out.printf((bold + "  │  GLiNER4j — %-44s│%n" + reset), displayName);
    System.out.println(bold + "  │  Named Entity Recognition with ONNX Runtime             │" + reset);
    System.out.println(bold + "  └─────────────────────────────────────────────────────────┘" + reset);
    System.out.println();
  }

  void schemaBanner() {
    System.out.println();
    System.out.println(bold + "  ┌─────────────────────────────────────────────────────────┐" + reset);
    System.out.println(bold + "  │       GLiNER4jSchemaExtractor — Structure Demo          │" + reset);
    System.out.println(bold + "  │         JSON Structure Extraction with ONNX             │" + reset);
    System.out.println(bold + "  └─────────────────────────────────────────────────────────┘" + reset);
    System.out.println();
  }

  public void structureLegend(List<StructureDefinition> structures) {
    for (var structure : structures) {
      var c = palette.colorsFor(structure.name());
      System.out.println("  " + c[0] + " " + structure.name() + " " + reset);
      for (var field : structure.fields()) {
        var typeTag = field.type().name().toLowerCase();
        var line = new StringBuilder()
          .append("      ")
          .append(dim)
          .append("• ")
          .append(reset)
          .append(field.name())
          .append(dim)
          .append(" :")
          .append(typeTag)
          .append(reset);
        if (!field.choices().isEmpty()) {
          line
            .append("  ")
            .append(gray)
            .append("∈ {")
            .append(String.join(" | ", field.choices()))
            .append("}")
            .append(reset);
        }
        if (field.description() != null && !field.description().isBlank()) {
          line.append("  ").append(dim).append("— ").append(field.description()).append(reset);
        }
        System.out.println(line);
      }
    }
    System.out.println(dim + "  ─────────────────────────────────────────────────────────────" + reset);
  }

  public void schemaResult(int index, String text, Map<String, List<StructureInstance>> results, double elapsedMs) {
    System.out.println();
    System.out.println("  " + dim + index + "." + reset + " " + text + elapsed(elapsedMs));

    if (results.isEmpty() || results.values().stream().allMatch(List::isEmpty)) {
      System.out.println(gray + "     No structures found." + reset);
      return;
    }

    for (var entry : results.entrySet()) {
      var structureName = entry.getKey();
      var instances = entry.getValue();
      var c = palette.colorsFor(structureName);
      System.out.println("     " + c[1] + structureName + reset + " [");
      for (int i = 0; i < instances.size(); i++) {
        var instance = instances.get(i);
        System.out.println("       {");
        var fieldEntries = new ArrayList<>(instance.fields().entrySet());
        for (int f = 0; f < fieldEntries.size(); f++) {
          var fieldEntry = fieldEntries.get(f);
          var suffix = (f < fieldEntries.size() - 1) ? "," : "";
          System.out.printf(
            "         %s\"%s\"%s: %s%s%n",
            dim,
            fieldEntry.getKey(),
            reset,
            renderValue(fieldEntry.getValue()),
            suffix
          );
        }
        System.out.println("       " + ((i < instances.size() - 1) ? "}," : "}"));
      }
      System.out.println("     ]");
    }
  }

  private String renderValue(StructureValue value) {
    if (value instanceof StringValue sv) {
      return "\"" + sv.text() + "\"" + dim + "  (" + Math.round(sv.confidence() * 100) + "%)" + reset;
    }
    if (value instanceof ListValue lv) {
      var sb = new StringBuilder("[");
      for (int i = 0; i < lv.items().size(); i++) {
        var item = lv.items().get(i);
        sb
          .append("\"")
          .append(item.text())
          .append("\"")
          .append(dim)
          .append("(")
          .append(Math.round(item.confidence() * 100))
          .append("%)")
          .append(reset);
        if (i < lv.items().size() - 1) sb.append(", ");
      }
      sb.append("]");
      return sb.toString();
    }
    return "null";
  }

  void classificationBanner() {
    System.out.println();
    System.out.println(bold + "  ┌─────────────────────────────────────────────────────────┐" + reset);
    System.out.println(bold + "  │        GLiNER4jClassifier — Classification Demo         │" + reset);
    System.out.println(bold + "  │          Text Classification with ONNX Runtime          │" + reset);
    System.out.println(bold + "  └─────────────────────────────────────────────────────────┘" + reset);
    System.out.println();
  }

  void relationBanner() {
    System.out.println();
    System.out.println(bold + "  ┌─────────────────────────────────────────────────────────┐" + reset);
    System.out.println(bold + "  │      GLiNER4jRelationExtractor — Relations Demo         │" + reset);
    System.out.println(bold + "  │         Relation Extraction with ONNX Runtime           │" + reset);
    System.out.println(bold + "  └─────────────────────────────────────────────────────────┘" + reset);
    System.out.println();
  }

  void combinedBanner() {
    System.out.println();
    System.out.println(bold + "  ┌─────────────────────────────────────────────────────────┐" + reset);
    System.out.println(bold + "  │       GLiNER4j — Combined Extraction Demo               │" + reset);
    System.out.println(bold + "  │       Entities + Relations in a single forward pass     │" + reset);
    System.out.println(bold + "  └─────────────────────────────────────────────────────────┘" + reset);
    System.out.println();
  }

  public void nerResult(int index, String text, Map<String, List<EntitySpan>> results, double elapsedMs) {
    System.out.println();
    var allSpans = new ArrayList<EntitySpan>();
    results.values().forEach(allSpans::addAll);
    allSpans.sort(Comparator.comparingInt(EntitySpan::start));

    System.out.print("  " + dim + index + "." + reset + " ");
    int pos = 0;
    for (var span : allSpans) {
      if (span.start() > pos) {
        System.out.print(text.substring(pos, span.start()));
      }
      var c = palette.colorsFor(span.type());
      System.out.print(c[0] + " " + span.text() + " " + reset);
      pos = span.end();
    }
    if (pos < text.length()) System.out.print(text.substring(pos));
    System.out.println(elapsed(elapsedMs));

    if (allSpans.isEmpty()) {
      System.out.println(gray + "     No entities found." + reset);
      return;
    }
    for (var span : allSpans) {
      var c = palette.colorsFor(span.type());
      System.out.printf(
        "     %s%-22s%s  %-30s  %s %s%.0f%%%s%n",
        c[1],
        span.type(),
        reset,
        "\"" + span.text() + "\"",
        confidenceBar(span.confidence()),
        dim,
        span.confidence() * 100,
        reset
      );
    }
  }

  public void relationResult(int index, String text, Map<String, List<RelationInstance>> results, double elapsedMs) {
    System.out.println();
    System.out.println("  " + dim + index + "." + reset + " " + text + elapsed(elapsedMs));

    boolean any = results
      .values()
      .stream()
      .anyMatch(list -> !list.isEmpty());
    if (!any) {
      System.out.println(gray + "     No relations found." + reset);
      return;
    }
    for (var entry : results.entrySet()) {
      var instances = entry.getValue();
      if (instances.isEmpty()) continue;
      var c = palette.colorsFor(entry.getKey());
      System.out.println("    " + c[1] + entry.getKey() + reset);
      for (var instance : instances) {
        printRelationInstance(instance);
      }
    }
  }

  public void combinedResult(int index, String text, ExtractionResult result, double elapsedMs) {
    System.out.println();
    System.out.println("  " + dim + index + "." + reset + " " + text + elapsed(elapsedMs));

    var entitySpans = new ArrayList<EntitySpan>();
    result.entities().values().forEach(entitySpans::addAll);
    entitySpans.sort(Comparator.comparingInt(EntitySpan::start));

    if (!entitySpans.isEmpty()) {
      System.out.println("    " + dim + "Entities:" + reset);
      for (var span : entitySpans) {
        var c = palette.colorsFor(span.type());
        System.out.printf("      %s%-22s%s  %s%n", c[1], span.type(), reset, "\"" + span.text() + "\"");
      }
    }

    boolean anyRel = result
      .relations()
      .values()
      .stream()
      .anyMatch(list -> !list.isEmpty());
    if (anyRel) {
      System.out.println("    " + dim + "Relations:" + reset);
      for (var entry : result.relations().entrySet()) {
        var instances = entry.getValue();
        if (instances.isEmpty()) continue;
        var c = palette.colorsFor(entry.getKey());
        System.out.println("      " + c[1] + entry.getKey() + reset);
        for (var instance : instances) {
          printRelationInstance(instance);
        }
      }
    }

    if (entitySpans.isEmpty() && !anyRel) {
      System.out.println(gray + "     No entities or relations found." + reset);
    }
  }

  private void printRelationInstance(RelationInstance instance) {
    var head = instance.head();
    var tail = instance.tail();
    var bar = confidenceBar(instance.confidence());
    if (head != null && tail != null) {
      System.out.printf(
        "      %s\"%s\"%s  →  %s\"%s\"%s  %s %s%.0f%%%s%n",
        bold,
        head.text(),
        reset,
        bold,
        tail.text(),
        reset,
        bar,
        dim,
        instance.confidence() * 100,
        reset
      );
    } else {
      var sb = new StringBuilder();
      for (var fieldEntry : instance.fields().entrySet()) {
        sb.append(fieldEntry.getKey()).append("=\"").append(fieldEntry.getValue().text()).append("\"  ");
      }
      System.out.printf("      %s  %s %s%.0f%%%s%n", sb, bar, dim, instance.confidence() * 100, reset);
    }
  }

  public void classificationResult(int index, String text, List<ClassificationResult> results, double elapsedMs) {
    System.out.println();
    System.out.println("  " + dim + index + "." + reset + " " + text + elapsed(elapsedMs));
    if (results.isEmpty()) {
      System.out.println(gray + "     No labels matched." + reset);
      return;
    }
    for (var result : results) {
      var c = palette.colorsFor(result.label());
      System.out.printf(
        "     %s%-14s%s  %s %s%.0f%%%s%n",
        c[1],
        result.label(),
        reset,
        confidenceBar(result.confidence()),
        dim,
        result.confidence() * 100,
        reset
      );
    }
  }

  /** Dim, right-trailing per-prediction latency annotation, e.g. {@code "  (87.4 ms)"}. */
  private String elapsed(double elapsedMs) {
    return String.format("  %s(%.1f ms)%s", dim, elapsedMs, reset);
  }

  private String confidenceBar(float confidence) {
    int filled = Math.round(confidence * 10);
    var sb = new StringBuilder(gray + "│" + reset);
    for (int i = 0; i < 10; i++) {
      if (i < filled) {
        if (confidence >= 0.8f) {
          sb.append("\033[38;5;35m█").append(reset);
        } else if (confidence >= 0.5f) {
          sb.append("\033[38;5;220m█").append(reset);
        } else {
          sb.append("\033[38;5;196m█").append(reset);
        }
      } else {
        sb.append(gray).append("░").append(reset);
      }
    }
    sb.append(gray).append("│").append(reset);
    return sb.toString();
  }

  public void interactiveHelp(boolean hasEntities, boolean hasLabels, boolean hasSchema, boolean hasRelations) {
    System.out.println();
    System.out.println(dim + "  Available commands:" + reset);
    if (hasEntities) {
      System.out.println("    " + bold + "/ner" + reset + dim + "        Switch to NER mode" + reset);
    }
    if (hasLabels) {
      System.out.println("    " + bold + "/classify" + reset + dim + "   Switch to classification mode" + reset);
    }
    if (hasSchema) {
      System.out.println(
        "    " + bold + "/schema" + reset + dim + "     Switch to schema (JSON structure) mode" + reset
      );
    }
    if (hasRelations) {
      System.out.println("    " + bold + "/relations" + reset + dim + "  Switch to relation extraction mode" + reset);
      System.out.println(
        "    " +
          bold +
          "/extract" +
          reset +
          dim +
          "    Switch to combined extraction mode (entities + relations)" +
          reset
      );
    }
    System.out.println("    " + bold + "/help" + reset + dim + "       Show this help" + reset);
    System.out.println("    " + bold + "/exit" + reset + dim + "       Quit the interactive session" + reset);
    System.out.println();
  }

  void metrics(Collection<MetricData> metrics) {
    System.out.println();
    System.out.println(bold + "  ┌─────────────────────────────────────────────────────────┐" + reset);
    System.out.println(bold + "  │                    Telemetry Summary                    │" + reset);
    System.out.println(bold + "  └─────────────────────────────────────────────────────────┘" + reset);
    System.out.println();

    for (var metric : metrics) {
      var name = metric.getName();
      var desc = metric.getDescription();
      var unit = metric.getUnit();

      switch (metric.getType()) {
        case LONG_SUM -> printLongSum(metric, name, unit, desc);
        case HISTOGRAM -> prinHistogram(metric, name, desc, unit);
      }
    }

    System.out.println(dim + "  ─────────────────────────────────────────────────────────────" + reset);
    System.out.println(bold + "  Done." + reset);
    System.out.println();
  }

  private void printLongSum(MetricData metric, String name, String unit, String desc) {
    long total = metric.getLongSumData().getPoints().stream().mapToLong(LongPointData::getValue).sum();
    System.out.printf("  %s%-38s%s  %s%,d%s", dim, name, reset, bold, total, reset);
    if (!unit.isEmpty()) System.out.print(dim + " " + unit + reset);
    System.out.println();
    System.out.println("  " + gray + desc + reset);
    System.out.println();
  }

  private void prinHistogram(MetricData metric, String name, String desc, String unit) {
    for (var hp : metric.getHistogramData().getPoints()) {
      System.out.printf("  %s%-38s%s%n", dim, name, reset);
      System.out.println("  " + gray + desc + reset);
      System.out.printf("    count   %s%,d%s%n", bold, hp.getCount(), reset);
      System.out.printf("    min     %s%,.1f%s %s%n", bold, hp.getMin(), reset, unit);
      System.out.printf("    max     %s%,.1f%s %s%n", bold, hp.getMax(), reset, unit);
      System.out.printf(
        "    avg     %s%,.1f%s %s%n",
        bold,
        hp.getCount() > 0 ? hp.getSum() / hp.getCount() : 0.0,
        reset,
        unit
      );
      System.out.println();
    }
  }
}
