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

import io.gravitee.lab.gliner4j.GLiNER4j;
import io.gravitee.lab.gliner4j.schema.EntityDefinition;
import java.nio.file.Path;
import java.util.List;

/**
 * Demo application for GLiNER4j — runs NER on sample sentences.
 */
public class GLiNER4jDemo {

  public static void main(String[] args) {
    var modelDir = Path.of("models/gliner2-base-onnx");
    var entities = List.of(
      new EntityDefinition("person"),
      new EntityDefinition("organization"),
      new EntityDefinition("location")
    );

    System.out.println("=== GLiNER4j Demo ===\n");
    System.out.println("Loading model from: " + modelDir);

    try (var gliner = GLiNER4j.load(modelDir, entities)) {
      var samples = List.of(
        "John works at Google in Mountain View, California.",
        "Elon Musk founded SpaceX and leads Tesla.",
        "The United Nations headquarters is in New York City.",
        "Marie Curie worked at the University of Paris."
      );

      for (var text : samples) {
        System.out.println("\nInput: " + text);
        var results = gliner.extract(text);

        if (results.isEmpty()) {
          System.out.println("  No entities found.");
        } else {
          results.forEach((type, spans) -> {
            for (var span : spans) {
              System.out.printf(
                "  [%s] \"%s\" (confidence=%.3f, chars=%d-%d)%n",
                type,
                span.text(),
                span.confidence(),
                span.start(),
                span.end()
              );
            }
          });
        }
      }
    }

    System.out.println("\n=== Done ===");
  }
}
