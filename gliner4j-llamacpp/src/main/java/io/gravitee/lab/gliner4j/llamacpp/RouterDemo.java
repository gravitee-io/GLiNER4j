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
package io.gravitee.lab.gliner4j.llamacpp;

import io.gravitee.lab.gliner4j.schema.ClassificationResult;
import java.nio.file.Path;
import java.util.List;

/** Routes a few prompts with {@code models/scx-router-onnx} and re-routes a chat turn by turn. */
public final class RouterDemo {

  static final List<String> MODELS = List.of(
    "coder",
    "DeepSeek-V3.1",
    "gemma-4-31B-it",
    "gpt-oss-120b",
    "Llama-4-Maverick-17B-128E-Instruct",
    "MAGPiE",
    "Meta-Llama-3.3-70B-Instruct",
    "Qwen3-32B"
  );
  static final List<String> TASKS = List.of(
    "analysis",
    "classification",
    "clustering",
    "code",
    "comparison",
    "critique",
    "decision support",
    "evaluation",
    "explanation",
    "fact checking",
    "forecasting",
    "generation",
    "information extraction",
    "information retrieval",
    "instruction following",
    "math & reasoning",
    "multi-turn",
    "planning",
    "problem solving",
    "programming",
    "qa",
    "reasoning",
    "recommendation",
    "rewriting",
    "summarization",
    "translation",
    "verification",
    "visualization"
  );
  static final List<String> REASONING = List.of("reasoning", "nonreasoning");
  static final List<String> DIFFICULTY = List.of(
    "very easy",
    "easy",
    "medium",
    "hard",
    "extra hard"
  );

  private RouterDemo() {}

  public static void main(String[] args) {
    var modelDir = Path.of(
      args.length > 0 ? args[0] : "models/scx-router-onnx"
    );
    var prompts = List.of(
      "Write a Python function that merges two sorted linked lists.",
      "Compare these two vendor contracts and flag the riskier clauses.",
      "What's a good one-line commit message for this typo fix?",
      "Derive the closed-form solution for this second-order linear recurrence and prove convergence."
    );
    try (var router = DecoderKvRouter.load(modelDir)) {
      for (var prompt : prompts) {
        long t0 = System.nanoTime();
        var model = router.classify(prompt, MODELS, 0.0f);
        var task = router.classifySingleLabel(prompt, TASKS).get(0);
        var reasoning = router.classifySingleLabel(prompt, REASONING).get(0);
        var difficulty = router.classifySingleLabel(prompt, DIFFICULTY).get(0);
        double ms = (System.nanoTime() - t0) / 1_000_000.0;
        System.out.printf("%n> %s  (%.0f ms)%n", prompt, ms);
        System.out.printf("  model      %s%n", top(model, 3));
        System.out.printf(
          "  task       %s (%.2f)%n",
          task.label(),
          task.confidence()
        );
        System.out.printf(
          "  reasoning  %s (%.2f)%n",
          reasoning.label(),
          reasoning.confidence()
        );
        System.out.printf(
          "  difficulty %s (%.2f)%n",
          difficulty.label(),
          difficulty.confidence()
        );
      }

      System.out.println(
        "\n--- streaming session: re-routed every turn, only new tokens encoded ---"
      );
      var turns = List.of(
        "I need help refactoring some Rust code.",
        " Specifically the borrow checker keeps rejecting this function.",
        " fn parse(&mut self, buf: &[u8]) -> Result<Token, Error> { ... } -- here's the body."
      );
      try (var session = router.openSession("chat-42")) {
        for (var turn : turns) {
          long t0 = System.nanoTime();
          session.append(turn);
          var model = router == null
            ? List.<ClassificationResult>of()
            : session.classify(MODELS, 0.0f);
          double ms = (System.nanoTime() - t0) / 1_000_000.0;
          System.out.printf(
            "  +%s  [%d cached tokens, %.0f ms] -> %s%n",
            turn.strip(),
            session.cachedTokens(),
            ms,
            top(model, 2)
          );
        }
      }
    }
  }

  private static String top(List<ClassificationResult> results, int n) {
    var sb = new StringBuilder();
    for (int i = 0; i < Math.min(n, results.size()); i++) {
      if (i > 0) sb.append(", ");
      sb
        .append(results.get(i).label())
        .append(String.format(" %.2f", results.get(i).confidence()));
    }
    return sb.toString();
  }
}
