---
language:
$languages
license: apache-2.0
library_name: onnx
tags:
  - ner
  - named-entity-recognition
  - relation-extraction
  - text-classification
  - gliner
  - gliner2
  - gliner2.5
  - boundary-extraction
  - java
  - onnx-runtime
$extra_tags
pipeline_tag: token-classification
base_model: $base_model
---

# GLiNER4j ONNX — $display_name

ONNX export of [$base_model](https://huggingface.co/$base_model) — a **GLiNER2.5** span-free
(boundary) extractor — for Java inference via [ONNX Runtime](https://onnxruntime.ai/).

Part of the [GLiNER4j](https://github.com/gravitee-io/GLiNER4j) project (`architecture: gliner2dot5`).

## Supported Tasks

| Task | Description |
|------|-------------|
| **Named Entity Recognition** | Boundary-predicted entity spans (any length) with confidence scores; abstention head per label |
| **Relation Extraction** | Typed head/tail relations from the same candidate pool, with fastino's edge deduplication |
| **Text Classification** | Zero-shot labels with multi-label support and confidence scores |

Per-call entity / relation / label overrides need no model reload.

## Repository Structure

```
├── gliner4j_config.json        # architecture = gliner2dot5 + boundary-head parameters
├── tokenizer.json              # Shared HuggingFace tokenizer
├── tokenizer_config.json
├── onnx/                       # Base FP32 ($size_fp32)
│   ├── ner_full.onnx           # encoder + boundary encoder + marginals + shared candidate pool + reranker
│   ├── relation_full.onnx      # the NER pipeline + typed pair generation + relation scorer
│   └── classifier_full.onnx    # encoder + classifier MLP
├── onnx_fp16/                  # FP16 ($size_fp16)
└── onnx_quantized/             # INT8 dynamic quantization ($size_int8)
```

## Model Architecture

GLiNER2.5 replaces span enumeration with boundary prediction: per label the model scores start,
end and inside positions, builds one document-level candidate pool (top-32 starts × top-32 ends plus a
per-label quota, deduplicated to 192 candidates) and reranks it. Every step is exported inside a single
static-shape ONNX graph — top-k, pairing and deduplication included — so the Java side only tokenizes,
thresholds, resolves overlaps (maximum-total-score set) and maps word boundaries to characters.

| Graph | Contents |
|-------|----------|
| `ner_full.onnx` | encoder → word/query gathers → boundary encoder (2 windowed attention layers) → start/end/inside marginals → candidate pool → pooled pair logits, abstention logits |
| `relation_full.onnx` | the above over `[R] head` / `[R] tail` queries → top-32 × top-32 typed pairs (cap 64) → relation MLP + biaffine content term |
| `classifier_full.onnx` | encoder → `[L]` marker gather → classifier MLP |

The Python preprocessing conventions of the fastino processor (lower-cased words, trailing period) are
reproduced by GLiNER4j for this family; both ONNX graphs are verified against PyTorch at export time
(candidate pool identical, logits within 1e-5).

## Variants

| Variant | Folder | Precision | Size | Use case |
|---------|--------|-----------|------|----------|
| Base | `onnx/` | FP32 | $size_fp32 | Maximum accuracy |
| FP16 | `onnx_fp16/` | FP16 | $size_fp16 | Good accuracy/size trade-off |
| Quantized | `onnx_quantized/` | INT8 (QInt8, per-channel) | $size_int8 | Smallest footprint, fastest on CPU |

## Usage

```java
var entities = List.of(new EntityDefinition("person"), new EntityDefinition("organization"));
try (var gliner = GLiNER4jNER.load(modelDir, entities)) {
    Map<String, List<EntitySpan>> results = gliner.extract("Barack Obama visited Berlin with Google executives.");
}

try (var relations = RelationExtractor.load(modelDir, List.of(new RelationDefinition("works_for")))) {
    relations.extract("John works for Apple.");
}

try (var classifier = GLiNER4jClassifier.load(modelDir, List.of(new ClassificationLabel("positive"), new ClassificationLabel("negative")))) {
    classifier.classify("Great product!");
}
```

Select a variant with the third argument, e.g. `GLiNER4jNER.load(modelDir, entities, "onnx_fp16")`.

## License

Apache License 2.0 (model and export).
