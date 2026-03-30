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
task hf-upload HF_REPO=<your-username>/gliner4j-onnx
```

> **Note:** This requires a HuggingFace account and a pre-existing repository. To set up:
>
> ```bash
> # Authenticate (pick one)
> export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"  # env var (recommended for CI)
> uv run --with huggingface-hub hf auth login  # interactive (one-time)
>
> # Create a private repo
> uv run --with huggingface-hub hf repos create gliner4j-onnx --private
> ```

## Benchmarks

Measured with JMH (average time, 15 iterations) on the `gliner2-base-onnx` model:

#### 4 Entity Types

| Batch Size | Avg Latency (ms/op) | Error (±ms) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:-----------:|:--------------:|:--------------------:|
| 1          | 50.4                 | 10.3        | 50.4           | ~19.8                |
| 4          | 122.5                | 17.2        | 30.6           | ~32.7                |
| 8          | 218.9                | 38.8        | 27.4           | ~36.5                |

#### 8 Entity Types

| Batch Size | Avg Latency (ms/op) | Error (±ms) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:-----------:|:--------------:|:--------------------:|
| 1          | 56.9                 | 3.5         | 56.9           | ~17.6                |
| 4          | 159.7                | 21.7        | 39.9           | ~25.0                |
| 8          | 256.1                | 13.8        | 32.0           | ~31.2                |
