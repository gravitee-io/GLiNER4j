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

import io.gravitee.lab.gliner4j.GLiNER4jDecider;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.DecisionQuestion;
import java.nio.file.Path;
import java.util.List;

/**
 * Small GLiNER2.5-Decide example: four typed questions about a few support tickets on the
 * llama.cpp engine, printed with every option's probability and a risk gauge for the scores.
 * Run with {@code task demo:decide}.
 */
public class DecideExample {

  public static void main(String[] args) {
    var tickets = List.of(
      "I was charged twice for my order and nobody answers my emails. Get me a real person now.",
      "Someone changed the password on my account last night and moved money out of it.",
      "Where is my parcel? The tracking page has not moved since Monday."
    );

    var questions = List.<DecisionQuestion>of(
      new DecisionQuestion.Choice(
        "intent",
        "What does the customer want?",
        List.of(
          new ClassificationLabel("billing_issue", "A wrong, double or unexpected charge"),
          new ClassificationLabel("account_security", "Someone else got into the account"),
          new ClassificationLabel("order_status", "Where an order or a parcel is"),
          new ClassificationLabel("cancel_subscription", "Stop a recurring plan")
        )
      ),
      new DecisionQuestion.Score("urgency", "How urgent is this?", List.of("low", "normal", "high")),
      new DecisionQuestion.Score(
        "risk",
        "How much harm could a wrong answer cause?",
        List.of("none", "low", "medium", "high", "critical")
      ),
      new DecisionQuestion.Noul("handoff", "Does the customer ask for a human agent?")
    );

    var printer = Printer.ansi();
    try (var decider = GLiNER4jDecider.load(Path.of("models/decide-llamacpp"))) {
      for (int i = 0; i < tickets.size(); i++) {
        long t0 = System.nanoTime();
        var answers = decider.decide(tickets.get(i), questions);
        printer.decisionResult(i + 1, tickets.get(i), questions, answers, (System.nanoTime() - t0) / 1e6);
      }
    }
  }
}
