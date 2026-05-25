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
$extra_tags
pipeline_tag: token-classification
base_model: $base_model
---

# GLiNER4j ONNX — $display_name

ONNX export of [$base_model](https://huggingface.co/$base_model) for Java inference via [ONNX Runtime](https://onnxruntime.ai/).

Part of the [GLiNER4j](https://github.com/gravitee-io/GLiNER4j) project.

## Supported Tasks

$tasks_table

## Repository Structure

```
├── gliner4j_config.json        # Shared model configuration
├── tokenizer.json              # Shared HuggingFace tokenizer
├── tokenizer_config.json
├── onnx/                       # Base FP32 ($size_fp32)
│   ├── encoder.onnx
│   ├── span_rep.onnx
│   ├── scoring_head.onnx
│   └── classifier_head.onnx
├── onnx_fp16/                  # FP16 ($size_fp16)
│   ├── encoder.onnx
│   ├── span_rep.onnx
│   ├── scoring_head.onnx
│   └── classifier_head.onnx
└── onnx_quantized/             # INT8 dynamic quantization ($size_int8)
    ├── encoder.onnx
    ├── span_rep.onnx
    ├── scoring_head.onnx
    └── classifier_head.onnx
```

## Model Architecture

The model is split into 4 ONNX modules for modular inference:

| Module | Description |
|--------|-------------|
| `encoder.onnx` | Transformer encoder (shared) |
| `span_rep.onnx` | Span representation layer (NER) |
| `scoring_head.onnx` | Count-aware scoring head (NER) |
| `classifier_head.onnx` | Classifier head MLP (Classification) |

## Variants

| Variant | Folder | Precision | Size | Use case |
|---------|--------|-----------|------|----------|
| Base | `onnx/` | FP32 | $size_fp32 | Maximum accuracy |
| FP16 | `onnx_fp16/` | FP16 | $size_fp16 | Good accuracy/size trade-off |
| Quantized | `onnx_quantized/` | INT8 (QUInt8, per-tensor) | $size_int8 | Smallest footprint, fastest on CPU |

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

$usage_section

### Model Variants

```java
// FP16 variant
var gliner = GLiNER4jNER.load(modelDir, entities, "onnx_fp16");

// Quantized variant
var gliner = GLiNER4jNER.load(modelDir, entities, "onnx_quantized");
```

## License

Apache License 2.0
