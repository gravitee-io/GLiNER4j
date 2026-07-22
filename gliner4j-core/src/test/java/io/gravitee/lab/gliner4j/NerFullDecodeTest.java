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

import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

/** Single-threaded merged-graph decode over a many-entity schema with length-varied texts. */
@EnabledIf("modelDirExists")
class NerFullDecodeTest {

  private static final Path MODEL_DIR = Path.of("models/gliner2-base-onnx");

  static boolean modelDirExists() {
    return Files.exists(MODEL_DIR.resolve("onnx/ner_full.onnx"));
  }

  @Test
  void mergedExtractBatchDecodes() throws Exception {
    var entities = new ArrayList<EntityDefinition>();
    for (int i = 0; i < 42; i++) {
      entities.add(new EntityDefinition("entity_type_" + i));
    }
    var shortText = "John works at Google in London.";
    var sb = new StringBuilder();
    for (int i = 0; i < 40; i++) {
      sb.append("Alice met Bob at the Paris office of Initech on Tuesday. ");
    }
    var batch = List.of(shortText, sb.toString(), shortText);

    try (
      var gliner = GLiNER4jNER.load(
        MODEL_DIR,
        entities,
        "onnx",
        io.gravitee.lab.gliner4j.runtime.RuntimeConfig.defaults()
      )
    ) {
      var out = gliner.extractBatch(batch, 0.5f);
      org.junit.jupiter.api.Assertions.assertEquals(batch.size(), out.size());
    }
  }
}
