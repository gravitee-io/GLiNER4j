# gliner4j

Java library for running [GLiNER2](https://github.com/fastino-ai/GLiNER2) Named Entity Recognition models using ONNX Runtime.

## Prerequisites

- [Task](https://taskfile.dev) - task runner
- [Maven](https://maven.apache.org/install.html) - Java build tool
- [uv](https://docs.astral.sh/uv/getting-started/installation/) - Python package manager (for model download/export/upload)

## Quick Start

```bash
# Download the model and export to ONNX
task

# Build the project
task build

# Run tests
task test
```

## Available Tasks

| Task | Description |
|------|-------------|
| `task` | Download model and export to ONNX |
| `task build` | Format code and build the project |
| `task test` | Format code and run tests |
| `task format` | Apply license headers and format code |
| `task benchmark` | Run JMH benchmarks |
| `task model:download` | Download GLiNER2 model from HuggingFace |
| `task model:export` | Export PyTorch model to ONNX format |
| `task model:clean` | Remove downloaded and exported model files |
| `task hf-upload HF_REPO=<user/repo>` | Upload ONNX model to HuggingFace Hub |

## Uploading to HuggingFace

```bash
# Login (one-time)
uv run --with huggingface-hub hf login

# Create a private repo
uv run --with huggingface-hub hf repos create gliner4j-onnx --private

# Upload the ONNX model
task hf-upload HF_REPO=<your-username>/gliner4j-onnx
```

## Benchmarks

Measured with JMH (average time, 15 iterations) on the `gliner2-base-onnx` model:

#### 4 Entity Types

| Batch Size | Avg Latency (ms/op) | Error (±ms) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:-----------:|:--------------:|:--------------------:|
| 1          | 26.5                 | 1.7         | 26.5           | ~37.7                |
| 4          | 143.5                | 12.3        | 35.9           | ~27.9                |
| 8          | 286.6                | 28.8        | 35.8           | ~27.9                |

#### 8 Entity Types

| Batch Size | Avg Latency (ms/op) | Error (±ms) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:-----------:|:--------------:|:--------------------:|
| 1          | 34.1                 | 3.2         | 34.1           | ~29.3                |
| 4          | 174.6                | 9.1         | 43.7           | ~22.9                |
| 8          | 339.2                | 9.3         | 42.4           | ~23.6                |
