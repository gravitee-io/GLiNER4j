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
  GLINER2DOT5("gliner2dot5.json"),
  SCX_ROUTER("scx-router.json"),
  STREAM_PII("stream-pii.json"),
  GLINER2DOT5_SMALL("gliner2dot5-small.json"),
  GLINER2DOT5_MULTI("gliner2dot5-multi.json"),
  PII("pii.json"),
  GLIGUARD("gliguard.json"),
  GLICLASS("gliclass.json"),
  GLICLASS_SENTIMENT("gliclass-sentiment.json"),
  GLICLASS_MULTILANG("gliclass-multilang.json"),
  GLICLASS_EDGE("gliclass-edge.json"),
  GLICLASS_LARGE("gliclass-large.json"),
  GLICLASS_MODERN_LARGE("gliclass-modern-large.json"),
  GLINER_PII("gliner-pii.json"),
  GLINER_BI("gliner-bi.json"),
  GLINER_X("gliner-x.json"),
  GLINER_MULTITASK("gliner-multitask.json"),
  // llama.cpp / ggml engine bundles (same profiles, engine=llamacpp bundle dirs)
  BASE_LLAMACPP("base-llamacpp.json"),
  PII_LLAMACPP("pii-llamacpp.json"),
  GLIGUARD_LLAMACPP("gliguard-llamacpp.json"),
  GLINER2DOT5_SMALL_LLAMACPP("gliner2dot5-small-llamacpp.json"),
  GLICLASS_EDGE_LLAMACPP("gliclass-edge-llamacpp.json"),
  GLINER_PII_LLAMACPP("gliner-pii-llamacpp.json"),
  GLINER_BI_LLAMACPP("gliner-bi-llamacpp.json"),
  GLINER_MULTITASK_LLAMACPP("gliner-multitask-llamacpp.json");

  public final String filename;

  ProfileType(String filename) {
    this.filename = filename;
  }

  public String getFilename() {
    return filename;
  }
}
