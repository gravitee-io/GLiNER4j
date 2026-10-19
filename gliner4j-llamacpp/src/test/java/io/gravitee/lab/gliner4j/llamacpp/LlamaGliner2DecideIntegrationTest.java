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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import io.gravitee.lab.gliner4j.GLiNER4jDecider;
import io.gravitee.lab.gliner4j.schema.ClassificationLabel;
import io.gravitee.lab.gliner4j.schema.DecisionQuestion;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.stream.Stream;
import org.junit.jupiter.api.AfterAll;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;
import org.junit.jupiter.api.condition.EnabledIf;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;

/**
 * Typed decisions on {@code fastino/GLiNER2.5-Decide} (a GLiNER2 span checkpoint) on the ggml engine.
 *
 * <p>Expected answers and confidences come from the upstream reference implementation
 * ({@code AutoExtractor.classify_text}, gliner2 2.0.0, softmax for single-label tasks): the task
 * name, an optional prompt and label descriptions all go into the {@code [P]} slot. Skipped unless
 * the bundle is exported ({@code task decide:llamacpp}).
 */
@EnabledIf("modelDirExists")
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
class LlamaGliner2DecideIntegrationTest {

  private static final Path MODEL_DIR = Path.of("../models/decide-llamacpp");

  private GLiNER4jDecider decider;

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("gguf/model.gguf"));
  }

  @BeforeAll
  void load() {
    decider = GLiNER4jDecider.load(MODEL_DIR);
  }

  @AfterAll
  void close() {
    decider.close();
  }

  private static DecisionQuestion.Choice choice(String id, String... options) {
    return new DecisionQuestion.Choice(
      id,
      "",
      Stream.of(options).map(ClassificationLabel::new).toList()
    );
  }

  static Stream<Arguments> upstreamReference() {
    return Stream.of(
      Arguments.of(
        "My subscription renewed on April 15 for ¥5,400 after the service was already down. Can I get that charge refunded?",
        choice(
          "intent",
          "order_status",
          "refund_request",
          "cancel_subscription",
          "update_payment",
          "login_problem",
          "shipping_delay",
          "bug_report",
          "speak_to_human",
          "other"
        ),
        "refund_request",
        0.997f
      ),
      Arguments.of(
        "The transfer I sent this morning is still pending, and I think I used the wrong sort code. Can you stop it and add Emily as the beneficiary instead?",
        choice(
          "intent",
          "transfer_pending",
          "transfer_cancel",
          "beneficiary_add",
          "card_lost",
          "balance_inquiry",
          "fraud_report",
          "mortgage_application",
          "fee_explanation"
        ),
        "beneficiary_add",
        0.817f
      ),
      Arguments.of(
        "The rash came back after the antibiotics finished. Can I get a same-week appointment with dermatology, or should I just refill the cream?",
        choice(
          "request",
          "book_appointment",
          "refill_prescription",
          "test_results",
          "referral",
          "billing_question",
          "cancel_appointment"
        ),
        "book_appointment",
        0.835f
      ),
      Arguments.of(
        "Post the customer's home address in the public thread so everyone can see where the package actually went.",
        choice(
          "policy",
          "allow",
          "personal_data",
          "harassment",
          "scam",
          "violence",
          "spam"
        ),
        "personal_data",
        0.974f
      ),
      // Bare task name: the model leans "no" — the prompt case below flips it.
      Arguments.of(
        "This is the third time I have explained the same missing refund. Stop the bot and get me a person.",
        choice("handoff", "yes", "no"),
        "no",
        0.675f
      ),
      Arguments.of(
        "This is the third time I have explained the same missing refund. Stop the bot and get me a person.",
        new DecisionQuestion.Noul(
          "handoff",
          "Does the customer ask for a human agent?"
        ),
        "yes",
        0.9996f
      ),
      Arguments.of(
        "The treaty was signed in Paris in 1992. It entered into force the following year, after the last signatory ratified it.",
        new DecisionQuestion.Noul(
          "answer",
          "Did the treaty enter into force in 1992?"
        ),
        "yes",
        0.996f
      ),
      Arguments.of(
        "Please reset the card PIN. The new one never arrived and the old one is locked after three tries.",
        new DecisionQuestion.Choice(
          "intent",
          "",
          List.of(
            new ClassificationLabel(
              "card_pin_change",
              "The customer wants a new PIN or the current PIN replaced"
            ),
            new ClassificationLabel(
              "card_lost",
              "The physical card is missing"
            ),
            new ClassificationLabel(
              "balance_inquiry",
              "The customer wants the current balance"
            )
          )
        ),
        "card_pin_change",
        0.686f
      )
    );
  }

  @ParameterizedTest
  @MethodSource("upstreamReference")
  void matchesUpstreamDecision(
    String text,
    DecisionQuestion question,
    String expectedChoice,
    float expectedConfidence
  ) {
    var decision = decider.decide(text, question);

    assertThat(decision.id()).isEqualTo(question.id());
    assertThat(decision.choice()).isEqualTo(expectedChoice);
    assertThat(decision.confidence()).isCloseTo(
      expectedConfidence,
      within(0.01f)
    );
    assertThat(
      decision
        .probabilities()
        .values()
        .stream()
        .mapToDouble(Float::doubleValue)
        .sum()
    ).isCloseTo(1.0, within(1e-4));
  }

  @Test
  void answersSeveralTypedQuestionsInOrder() {
    var answers = decider.decide(
      "Payroll file has to be corrected before the 5pm cutoff or the whole company is paid late.",
      List.of(
        choice("queue", "payroll", "benefits", "it_access", "facilities"),
        new DecisionQuestion.Score(
          "urgency",
          "How urgent is this?",
          List.of("can wait", "this week", "today", "right now")
        ),
        new DecisionQuestion.Noul("angry", "Is the author angry?")
      )
    );

    assertThat(answers).containsOnlyKeys("queue", "urgency", "angry");
    assertThat(answers.keySet()).containsExactly("queue", "urgency", "angry");
    assertThat(answers.get("queue").choice()).isEqualTo("payroll");
    assertThat(answers.get("urgency").probabilities().keySet()).containsExactly(
      "can wait",
      "this week",
      "today",
      "right now"
    );
    assertThat(answers.get("urgency").score()).isBetween(0f, 3f);
    assertThat(answers.get("angry").probabilities().keySet()).containsExactly(
      "yes",
      "no"
    );
  }

  @Test
  void blankStateHasNoAnswer() {
    assertThat(decider.decide("  ", choice("label", "spam", "ham"))).isNull();
    assertThat(
      decider.decide("", List.of(choice("label", "spam", "ham")))
    ).isEmpty();
  }
}
