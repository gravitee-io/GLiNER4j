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

import static java.lang.Math.floorMod;

public class Palette {

  /** 12-color palette cycled by hash(label) — scales to any number of entity types. */
  public static final Palette DEFAULT = new Palette(
    new String[][] {
      { "\033[48;5;63m\033[97m", "\033[38;5;63m" }, // purple
      { "\033[48;5;37m\033[97m", "\033[38;5;37m" }, // teal
      { "\033[48;5;208m\033[97m", "\033[38;5;208m" }, // orange
      { "\033[48;5;170m\033[97m", "\033[38;5;170m" }, // pink
      { "\033[48;5;33m\033[97m", "\033[38;5;33m" }, // blue
      { "\033[48;5;35m\033[97m", "\033[38;5;35m" }, // green
      { "\033[48;5;160m\033[97m", "\033[38;5;160m" }, // red
      { "\033[48;5;142m\033[97m", "\033[38;5;142m" }, // olive
      { "\033[48;5;129m\033[97m", "\033[38;5;129m" }, // violet
      { "\033[48;5;24m\033[97m", "\033[38;5;24m" }, // deep blue
      { "\033[48;5;94m\033[97m", "\033[38;5;94m" }, // brown
      { "\033[48;5;100m\033[97m", "\033[38;5;100m" }, // dark olive
    }
  );

  private final String[][] palette;

  public Palette(String[][] palette) {
    this.palette = palette;
  }

  public String[] colorsFor(String label) {
    return palette[floorMod(label.hashCode(), palette.length)];
  }
}
