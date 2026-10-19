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
package io.gravitee.lab.gliner4j;

import io.gravitee.lab.gliner4j.arch.LoadContext;
import io.gravitee.lab.gliner4j.arch.ModelArchitectures;
import io.gravitee.lab.gliner4j.runtime.BaseRuntime;
import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.schema.Decision;
import io.gravitee.lab.gliner4j.schema.DecisionQuestion;
import io.gravitee.lab.gliner4j.strategy.gliner2.Gliner2ClassificationStrategy;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import lombok.extern.slf4j.Slf4j;

/**
 * Decision facade: answers typed {@link DecisionQuestion}s (choice, score, yes/no) about a state
 * with one probability per option — the System One shape on a GLiNER2 classifier.
 *
 * <p>Built for decision-tuned GLiNER2 checkpoints such as {@code fastino/GLiNER2.5-Decide}, which
 * are trained with the task name (and an optional prompt) in the {@code [P]} slot: each question
 * is sent as its own task, {@code "id"} or {@code "id: instructions"}, with its options as labels,
 * and the label logits go through a softmax. Any GLiNER2 or GLiNER2.5 classification bundle loads;
 * GLiNER2.5-Decide itself is exported for llama.cpp only ({@code task decide:llamacpp}), with
 * {@code "word_splitter": "whitespace"} so its text matches training. A Decide bundle exported
 * without that key still loads, but with the default text conventions and degraded probabilities.
 *
 * <p>Usage:
 * <pre>{@code
 * try (var decider = GLiNER4jDecider.load(modelDir)) {
 *   Map<String, Decision> answers = decider.decide(text, List.of(
 *     new DecisionQuestion.Choice("intent", "", List.of(
 *       new ClassificationLabel("refund_request"), new ClassificationLabel("order_status"))),
 *     new DecisionQuestion.Noul("handoff", "Does the customer ask for a person?")));
 *   answers.get("intent").choice();
 * }
 * }</pre>
 */
@Slf4j
public final class GLiNER4jDecider implements AutoCloseable {

  private final Gliner2ClassificationStrategy strategy;

  private GLiNER4jDecider(Gliner2ClassificationStrategy strategy) {
    this.strategy = strategy;
  }

  /** Loads a decider using the default "onnx" variant. */
  public static GLiNER4jDecider load(Path modelDir) {
    return load(
      modelDir,
      BaseRuntime.DEFAULT_VARIANT,
      RuntimeConfig.defaults()
    );
  }

  /** Loads a decider with explicit resource control. */
  public static GLiNER4jDecider load(
    Path modelDir,
    RuntimeConfig runtimeConfig
  ) {
    return load(modelDir, BaseRuntime.DEFAULT_VARIANT, runtimeConfig);
  }

  /**
   * Loads a decider with a specific variant and explicit resource control.
   *
   * @param modelDir      path to the root model directory
   * @param variant       ONNX variant folder name (the GGUF quantization on the llama.cpp engine)
   * @param runtimeConfig resource control configuration
   * @return a ready-to-use decider
   * @throws IllegalArgumentException if the bundle's family has no GLiNER2 classifier
   */
  public static GLiNER4jDecider load(
    Path modelDir,
    String variant,
    RuntimeConfig runtimeConfig
  ) {
    var config = GLiNER4jConfig.load(modelDir);
    log.info(
      "Loading GLiNER4jDecider from {} (variant={}, architecture={})",
      modelDir,
      variant,
      config.getArchitecture().configValue()
    );
    var ctx = new LoadContext(
      modelDir,
      variant,
      runtimeConfig,
      config,
      new DjlTokenizerWrapper(modelDir)
    );
    var strategy = ModelArchitectures.forConfig(
      config
    ).newClassificationStrategy(ctx, List.of());
    if (!(strategy instanceof Gliner2ClassificationStrategy gliner2)) {
      strategy.close();
      throw new IllegalArgumentException(
        "Decisions need a GLiNER2 classifier; " +
          modelDir +
          " is a " +
          config.getArchitecture().configValue() +
          " bundle"
      );
    }
    return new GLiNER4jDecider(gliner2);
  }

  /**
   * Answers each question about {@code state}, one forward pass per question.
   *
   * @param state the text the questions are about
   * @param questions the questions
   * @return question id → answer, in question order; empty for a blank state
   */
  public Map<String, Decision> decide(
    String state,
    List<? extends DecisionQuestion> questions
  ) {
    var answers = new LinkedHashMap<String, Decision>();
    for (var question : questions) {
      var answer = decide(state, question);
      if (answer != null) {
        answers.put(question.id(), answer);
      }
    }
    return answers;
  }

  /**
   * Answers one question about {@code state}.
   *
   * @return the answer, or {@code null} for a blank state
   */
  public Decision decide(String state, DecisionQuestion question) {
    var options = question.options();
    var logits = strategy.scoreTask(state, task(question), options);
    if (logits.length == 0) {
      return null;
    }
    var probabilities = softmax(logits);
    var byOption = new LinkedHashMap<String, Float>();
    for (int i = 0; i < options.size(); i++) {
      byOption.put(options.get(i).name(), probabilities[i]);
    }
    return new Decision(question.id(), byOption);
  }

  /** The {@code [P]} prompt text, as upstream builds it: {@code "task"} or {@code "task: prompt"}. */
  static String task(DecisionQuestion question) {
    var instructions = question.instructions();
    return instructions == null || instructions.isBlank()
      ? question.id()
      : question.id() + ": " + instructions;
  }

  static float[] softmax(float[] logits) {
    float max = Float.NEGATIVE_INFINITY;
    for (float l : logits) {
      max = Math.max(max, l);
    }
    var out = new float[logits.length];
    float sum = 0f;
    for (int i = 0; i < logits.length; i++) {
      out[i] = (float) Math.exp(logits[i] - max);
      sum += out[i];
    }
    for (int i = 0; i < out.length; i++) {
      out[i] /= sum;
    }
    return out;
  }

  @Override
  public void close() {
    strategy.close();
  }
}
