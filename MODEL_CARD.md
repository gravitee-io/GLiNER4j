---
language:
  - en
license: apache-2.0
library_name: onnx
tags:
  - ner
  - named-entity-recognition
  - text-classification
  - gliner
  - gliner2
  - java
  - onnx-runtime
pipeline_tag: token-classification
base_model: fastino-ai/gliner2-base
---

# GLiNER4j ONNX

ONNX export of [fastino-ai/gliner2-base](https://huggingface.co/fastino-ai/gliner2-base) for Java inference via [ONNX Runtime](https://onnxruntime.ai/).

Part of the [GLiNER4j](https://github.com/gravitee-io/GLiNER4j) project.

## Supported Tasks

| Task | Description |
|------|-------------|
| **Named Entity Recognition** | Extract typed entity spans from text with confidence scores |
| **Text Classification** | Assign labels to text with multi-label support and confidence scores |

Both tasks support entity/label descriptions for improved accuracy and per-call overrides without model reloading.

## Repository Structure

```
├── gliner4j_config.json        # Shared model configuration
├── tokenizer.json              # Shared HuggingFace tokenizer
├── tokenizer_config.json
├── onnx/                       # Base FP32 (~830 MB)
│   ├── encoder.onnx
│   ├── span_rep.onnx
│   ├── scoring_head.onnx
│   └── classifier_head.onnx
├── onnx_fp16/                  # FP16 (~416 MB, ~50% smaller)
│   ├── encoder.onnx
│   ├── span_rep.onnx
│   ├── scoring_head.onnx
│   └── classifier_head.onnx
└── onnx_quantized/             # INT8 dynamic quantization (~208 MB, ~75% smaller)
    ├── encoder.onnx
    ├── span_rep.onnx
    ├── scoring_head.onnx
    └── classifier_head.onnx
```

## Model Architecture

The model is split into 4 ONNX modules for modular inference:

| Module | Description |
|--------|-------------|
| `encoder.onnx` | DeBERTaV2 transformer encoder (shared) |
| `span_rep.onnx` | Span representation layer (NER) |
| `scoring_head.onnx` | Count-aware scoring head (NER) |
| `classifier_head.onnx` | Classifier head MLP (Classification) |

## Variants

| Variant | Folder | Precision | Size | Use case |
|---------|--------|-----------|------|----------|
| Base | `onnx/` | FP32 | ~830 MB | Maximum accuracy |
| FP16 | `onnx_fp16/` | FP16 | ~416 MB | Good accuracy/size trade-off |
| Quantized | `onnx_quantized/` | INT8 | ~208 MB | Smallest footprint, fastest on CPU |

To download a specific variant only:
```bash
huggingface-cli download <repo> --include "onnx_fp16/*" "*.json"
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Hidden size | 768 |
| Max span width | 8 |
| Max count | 20 |
| Span mode | SpanMarkerV0 |
| Token pooling | first |
| ONNX opset | 17 |

## Usage

Use with [GLiNER4j](https://github.com/gravitee-io/GLiNER4j), a Java library for GLiNER2 inference via ONNX Runtime.

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
```

### Model Variants

```java
// FP16 variant
var gliner = GLiNER4jNER.load(modelDir, entities, "onnx_fp16");
var classifier = GLiNER4jClassifier.load(modelDir, labels, "onnx_fp16");

// Quantized variant
var gliner = GLiNER4jNER.load(modelDir, entities, "onnx_quantized");
var classifier = GLiNER4jClassifier.load(modelDir, labels, "onnx_quantized");
```

## Features

- **Entity/Label Descriptions**: Provide natural language descriptions alongside entity types or classification labels to improve model accuracy
- **Per-call Overrides**: Change entities or labels at inference time without reloading the model
- **Batch Processing**: Batched encoder calls with virtual thread parallelism for scoring
- **OpenTelemetry**: Built-in instrumentation for duration, text count, and result count metrics (zero overhead when no OTel SDK is present)
- **Runtime Configuration**: Control thread pools, graph optimization level, and model caching

## Benchmarks

Measured with JMH (average time, 15 iterations):

#### 4 Entity Types

| Batch Size | Avg Latency (ms/op) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:--------------:|:--------------------:|
| 1 | 26.5 | 26.5 | ~37.7 |
| 4 | 143.5 | 35.9 | ~27.9 |
| 8 | 286.6 | 35.8 | ~27.9 |

#### 8 Entity Types

| Batch Size | Avg Latency (ms/op) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:--------------:|:--------------------:|
| 1 | 34.1 | 34.1 | ~29.3 |
| 4 | 174.6 | 43.7 | ~22.9 |
| 8 | 339.2 | 42.4 | ~23.6 |

## License

Apache License 2.0
