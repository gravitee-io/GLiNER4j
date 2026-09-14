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

import io.gravitee.lab.gliner4j.runtime.RuntimeConfig;
import io.gravitee.lab.gliner4j.tokenizer.DjlTokenizerWrapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * The ggml scorer graph must reproduce the ONNX export of the same scorer (which is itself
 * verified against PyTorch at export time) on identical inputs. Skipped unless the bundle exists.
 */
@EnabledIf("bundleExists")
class GgmlScorerParityTest {

  private static final Path MODEL_DIR = Path.of("../models/scx-router-onnx");

  static boolean bundleExists() {
    return (
      Files.exists(MODEL_DIR.resolve("gguf/scorer.gguf")) &&
      Files.exists(MODEL_DIR.resolve("onnx/scorer.onnx"))
    );
  }

  @Test
  void ggmlMatchesOnnxOnGpuAndCpu() {
    try (var tokenizer = new DjlTokenizerWrapper(MODEL_DIR)) {
      var section = LabelSection.build(
        tokenizer,
        RouterDemo.MODELS,
        "<<LABEL>>",
        "<<SEP>>",
        151669,
        151670
      );
      var rng = new Random(42);
      var rows = new ArrayList<float[]>(section.bodyLength());
      for (int i = 0; i < section.bodyLength(); i++) {
        var row = new float[1024];
        for (int j = 0; j < row.length; j++) row[j] =
          (float) rng.nextGaussian();
        rows.add(row);
      }
      float[] onnx;
      try (
        var ort = new DecoderKvScorer(
          MODEL_DIR,
          "onnx",
          RuntimeConfig.defaults()
        )
      ) {
        onnx = ort.score(List.of(rows), section)[0];
      }
      for (boolean gpu : new boolean[] { true, false }) {
        try (
          var ggml = new GgmlScorer(
            MODEL_DIR.resolve("gguf/scorer.gguf"),
            gpu,
            4,
            false,
            1.0f,
            1e-7f
          )
        ) {
          var got = ggml.score(List.of(rows), section)[0];
          assertThat(got).hasSize(onnx.length);
          for (int i = 0; i < onnx.length; i++) {
            assertThat(got[i])
              .as("label %d gpu=%s", i, gpu)
              .isCloseTo(onnx[i], within(2e-3f));
          }
        }
      }
    }
  }
}
