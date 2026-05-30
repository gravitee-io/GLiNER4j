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

public enum ProfileType {
  BASE("base.json"),
  PII("pii.json"),
  GLIGUARD("gliguard.json"),
  GLICLASS("gliclass.json"),
  GLICLASS_SENTIMENT("gliclass-sentiment.json"),
  GLICLASS_MULTILANG("gliclass-multilang.json"),
  GLICLASS_EDGE("gliclass-edge.json"),
  GLINER_PII("gliner-pii.json"),
  GLINER_BI("gliner-bi.json");

  public final String filename;

  ProfileType(String filename) {
    this.filename = filename;
  }

  public String getFilename() {
    return filename;
  }
}
