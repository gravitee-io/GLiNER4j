---
language:
  - en
license: apache-2.0
library_name: onnx
tags:
  - ner
  - named-entity-recognition
  - gliner
  - gliner2
  - java
  - onnx-runtime
pipeline_tag: token-classification
base_model: fastino-ai/gliner2-base
---

# GLiNER4j ONNX

ONNX export of [fastino-ai/gliner2-base](https://huggingface.co/fastino-ai/gliner2-base) for Java inference via [ONNX Runtime](https://onnxruntime.ai/).

Part of the [gliner4j](https://github.com/gravitee-io/gliner4j) project.

## Repository Structure

```
├── gliner4j_config.json        # Shared model configuration
├── tokenizer.json              # Shared HuggingFace tokenizer
├── tokenizer_config.json
├── onnx/                       # Base FP32 (~830 MB)
│   ├── encoder.onnx
│   ├── span_rep.onnx
│   └── scoring_head.onnx
├── onnx_fp16/                  # FP16 (~416 MB, ~50% smaller)
│   ├── encoder.onnx
│   ├── span_rep.onnx
│   └── scoring_head.onnx
└── onnx_quantized/             # INT8 dynamic quantization (~208 MB, ~75% smaller)
    ├── encoder.onnx
    ├── span_rep.onnx
    └── scoring_head.onnx
```

## Model Architecture

The model is split into 3 ONNX modules for modular inference:

| Module | Description |
|--------|-------------|
| `encoder.onnx` | DebertaV2 transformer encoder |
| `span_rep.onnx` | Span representation layer |
| `scoring_head.onnx` | Count-aware scoring head |

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

Use with [gliner4j](https://github.com/gravitee-io/gliner4j), a Java library for GLiNER2 inference via ONNX Runtime.

```java
// Base FP32 (default)
var gliner = GLiNER4j.load(modelDir, entities);

// FP16 variant
var gliner = GLiNER4j.load(modelDir, entities, "onnx_fp16");

// Quantized variant
var gliner = GLiNER4j.load(modelDir, entities, "onnx_quantized");
```

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
