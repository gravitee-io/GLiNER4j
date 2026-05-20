# gliner4j

Java library for running [GLiNER2](https://github.com/fastino-ai/GLiNER2) Named Entity Recognition models using ONNX Runtime.

## Features

- ONNX-based inference with configurable runtime options (thread pools, graph optimization, model caching)
- Entity descriptions for improved extraction accuracy
- Per-call entity override for dynamic use cases
- Batch processing with parallelized scoring
- Support for ONNX model variants (default, fp16, quantized)

## Prerequisites

- Java 21+
- [Task](https://taskfile.dev) - task runner
- [Maven](https://maven.apache.org/install.html) - Java build tool
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package manager (for model download/export/upload)

## Quick Start

```bash
# Download + export the base model (FP32 + FP16 + INT8). `task` alone runs the base chain.
task base

# Build the Java project
task build

# Run the base demo (interactive)
task demo:base

# Run tests
task test
```

For the PII model, swap `:base` → `:pii` in any command (see [PII Detection Model](#pii-detection-model)).

## Available Tasks

Tasks are organized into symmetric `base` / `pii` pairs.

### Shared

| Task | Description |
|------|-------------|
| `task` | Alias for `task base` (default) |
| `task build` | Build the Java project (formats code first) |
| `task test` | Run tests (formats code first) |
| `task format` | Apply license headers and format code |
| `task benchmark` | Run JMH benchmarks |

> Python tooling runs through `uv run` (for scripts) and `uvx` (for the `hf` CLI). No project virtualenv is created — uv manages a cached env automatically. The export script's dependencies (including `gliner2`) are declared inline via PEP 723.

### Base model (`fastino/gliner2-base-v1`)

| Task | Description |
|------|-------------|
| `task base` | Download + export the base model (FP32 + FP16 + INT8) |
| `task model:download:base` | Download base model from HuggingFace |
| `task model:export:base` | Export base PyTorch model to ONNX (FP32 by default, override with `EXPORT_ARGS`) |
| `task model:clean:base` | Remove downloaded and exported base model files |
| `task demo:base` | Run the GLiNER4j demo against the base profile |
| `task hf-upload:base HF_REPO=<user/repo>` | Upload base ONNX model dir to HuggingFace Hub |

### PII model (`fastino/gliner2-privacy-filter-PII-multi`)

| Task | Description |
|------|-------------|
| `task pii` | Download + export the PII model (FP32 + FP16 + INT8) |
| `task model:download:pii` | Download PII model from HuggingFace |
| `task model:export:pii` | Export PII PyTorch model to ONNX (FP32 by default, override with `EXPORT_ARGS`) |
| `task model:clean:pii` | Remove downloaded and exported PII model files |
| `task demo:pii` | Run the GLiNER4j demo against the PII profile (42 PII labels) |
| `task hf-upload:pii HF_REPO=<user/repo>` | Upload PII ONNX model dir to HuggingFace Hub |

## PII Detection Model

Alongside the base GLiNER2 model, gliner4j ships a dedicated chain for
[fastino/gliner2-privacy-filter-PII-multi](https://huggingface.co/fastino/gliner2-privacy-filter-PII-multi),
a PII-tuned fine-tune that recognizes 42 privacy-sensitive entity types
(person, email, phone_number, IBAN, card_number, api_key, password, …).

```bash
# Download + export FP32 / FP16 / INT8 variants into models/gliner2-privacy-onnx
task pii

# Build the Java project
task build

# Run the PII demo: 5 annotated samples, then an interactive prompt
task demo:pii
```

Artifacts land under `models/gliner2-privacy-onnx/`:

- `onnx/` — FP32 (default)
- `onnx_fp16/` — FP16
- `onnx_quantized/` — INT8 dynamic quantization (QUInt8, per-tensor)

## Uploading to HuggingFace

```bash
# Base model
task hf-upload:base HF_REPO=<your-username>/gliner4j-onnx

# PII model
task hf-upload:pii  HF_REPO=<your-username>/gliner4j-privacy-onnx
```

> **Note:** This requires a HuggingFace account and a pre-existing repository. To set up:
>
> ```bash
> # Authenticate (pick one)
> export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"                    # env var (recommended for CI)
> uvx --from huggingface-hub hf auth login                     # interactive (one-time)
>
> # Create a private repo
> uvx --from huggingface-hub hf repo create gliner4j-onnx --type model --private
> ```

## Benchmarks

Measured with JMH (average time, 15 iterations) on the `gliner2-base-onnx` model:

#### 4 Entity Types

| Batch Size | Avg Latency (ms/op) | Error (±ms) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:-----------:|:--------------:|:--------------------:|
| 1          | 43.6                 | 3.8         | 43.6           | ~23.0                |
| 4          | 125.3                | 17.7        | 31.3           | ~31.9                |
| 8          | 220.8                | 17.7        | 27.6           | ~36.2                |

#### 8 Entity Types

| Batch Size | Avg Latency (ms/op) | Error (±ms) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:-----------:|:--------------:|:--------------------:|
| 1          | 62.0                 | 8.5         | 62.0           | ~16.1                |
| 4          | 153.1                | 12.5        | 38.3           | ~26.1                |
| 8          | 274.4                | 27.1        | 34.3           | ~29.2                |
