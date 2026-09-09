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
 * One label family of a decoder-kv router profile.
 *
 * @param name display name (e.g. {@code model}, {@code task}, {@code reasoning})
 * @param labels the zero-shot labels
 * @param singleLabel softmax over the labels (single-label) instead of per-label sigmoid
 */
public record RouterFamily(String name, List<String> labels, boolean singleLabel) {}
