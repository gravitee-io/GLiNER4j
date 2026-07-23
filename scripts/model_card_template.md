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
│   ├── ner_full.onnx
│   └── classifier_full.onnx
├── onnx_fp16/                  # FP16 ($size_fp16)
│   ├── ner_full.onnx
│   └── classifier_full.onnx
└── onnx_quantized/             # INT8 dynamic quantization ($size_int8)
    ├── ner_full.onnx
    └── classifier_full.onnx
```

An `onnx_optimized_cpu/` folder with the same two files may also be present (ONNX Runtime graph-optimized for CPU).

## Model Architecture

Each variant ships two merged, self-contained ONNX graphs — one per task:

| Graph | Description |
|-------|-------------|
| `ner_full.onnx` | Transformer encoder + span representation + count-aware scoring head (NER) |
| `classifier_full.onnx` | Transformer encoder + classifier head MLP (Classification) |

The graphs are fused at export time from the encoder and task heads; the intermediate split modules are not published.

## Variants

| Variant | Folder | Precision | Size | Use case |
|---------|--------|-----------|------|----------|
| Base | `onnx/` | FP32 | $size_fp32 | Maximum accuracy |
| FP16 | `onnx_fp16/` | FP16 | $size_fp16 | Good accuracy/size trade-off |
| Quantized | `onnx_quantized/` | INT8 (QUInt8, per-channel) | $size_int8 | Smallest footprint, fastest on CPU |

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
