#
# Copyright © 2015 The Gravitee team (http://gravitee.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Render a per-model MODEL_CARD.md from scripts/model_card_template.md.

Usage:
    python scripts/render_model_card.py <variant> <output_path>

    variant: one of {base, pii}
    output_path: where to write the rendered card, e.g.
                 models/gliner2-base-onnx/MODEL_CARD.md
"""
from __future__ import annotations

import sys
from pathlib import Path
from string import Template

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "model_card_template.md"


TASKS_NER_AND_CLASSIFICATION = """\
| Task | Description |
|------|-------------|
| **Named Entity Recognition** | Extract typed entity spans from text with confidence scores |
| **Text Classification** | Assign labels to text with multi-label support and confidence scores |

Both tasks support entity/label descriptions for improved accuracy and per-call overrides without model reloading."""


TASKS_NER_ONLY = """\
| Task | Description |
|------|-------------|
| **Named Entity Recognition** | Extract typed PII entity spans from text with confidence scores |

Supports entity descriptions for improved accuracy and per-call overrides without model reloading."""


USAGE_NER_AND_CLASSIFICATION = """\
### Named Entity Recognition

```java
var entities = List.of(
    new EntityDefinition("person", "Names of individuals"),
    new EntityDefinition("organization", "Company or institution names")
);
var gliner = GLiNER4jNER.load(modelDir, entities);
Map<String, List<EntitySpan>> results = gliner.extract("John works at Google.");
```

### Text Classification

```java
var labels = List.of(
    new ClassificationLabel("positive", "Expresses positive sentiment"),
    new ClassificationLabel("negative", "Expresses negative sentiment")
);
var classifier = GLiNER4jClassifier.load(modelDir, labels);
List<ClassificationResult> results = classifier.classify("Great product!");
```"""


USAGE_PII_NER = """\
### Named Entity Recognition (PII)

```java
var entities = List.of(
    new EntityDefinition("email", "Email address"),
    new EntityDefinition("phone_number", "Phone or mobile number"),
    new EntityDefinition("card_number", "Credit / debit card number"),
    new EntityDefinition("iban", "IBAN"),
    new EntityDefinition("api_key", "API key")
);
var gliner = GLiNER4jNER.load(modelDir, entities);
Map<String, List<EntitySpan>> results = gliner.extract(
    "Charge card 4111-1111-1111-1111 to john.smith@example.com."
);
```

The full label set (42 PII types) is documented in the upstream model card on
[Hugging Face](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi#supported-labels).
See `gliner4j-demo` (run `task demo:pii`) for an interactive example."""


MODELS: dict[str, dict[str, str]] = {
    "base": {
        "display_name": "GLiNER2 Base",
        "base_model": "fastino-ai/gliner2-base",
        "extra_tags": "  - text-classification",
        "tasks_table": TASKS_NER_AND_CLASSIFICATION,
        "usage_section": USAGE_NER_AND_CLASSIFICATION,
        "size_fp32": "~830 MB",
        "size_fp16": "~416 MB, ~50% smaller",
        "size_int8": "~208 MB, ~75% smaller",
    },
    "pii": {
        "display_name": "GLiNER2 PII (42 labels)",
        "base_model": "fastino/gliner2-privacy-filter-PII-multi",
        "extra_tags": "  - pii\n  - privacy",
        "tasks_table": TASKS_NER_ONLY,
        "usage_section": USAGE_PII_NER,
        "size_fp32": "~1.1 GB",
        "size_fp16": "~588 MB, ~50% smaller",
        "size_int8": "~350 MB, ~70% smaller",
    },
}


def render(variant: str, output_path: Path) -> None:
    if variant not in MODELS:
        raise SystemExit(
            f"Unknown variant '{variant}'. Choices: {sorted(MODELS)}"
        )
    template = Template(TEMPLATE.read_text())
    rendered = template.substitute(MODELS[variant])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)
    print(f"Wrote {output_path}")


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    render(sys.argv[1], Path(sys.argv[2]))


if __name__ == "__main__":
    main()
