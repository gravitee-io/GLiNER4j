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
package io.gravitee.lab.gliner4j.demo.profile;

import java.util.List;

/**
 * Router section of a demo profile (GLiClass decoder-kv bundles).
 *
 * @param families label families scored for every sample
 * @param samples prompts to route
 * @param session chat turns appended one by one to a streaming session, re-routed after each
 * @param sessionFamily family shown while streaming (defaults to the first family)
 */
public record RouterDef(List<RouterFamily> families, List<String> samples, List<String> session, String sessionFamily) {
  public RouterFamily sessionFamilyDef() {
    if (sessionFamily != null) {
      for (var f : families) if (f.name().equals(sessionFamily)) return f;
    }
    return families.get(0);
  }
}
