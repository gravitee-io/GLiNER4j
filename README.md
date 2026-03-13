# gliner4j

Java library for running [GLiNER2](https://github.com/fastino-ai/GLiNER2) Named Entity Recognition models using ONNX Runtime.

## Benchmarks

Measured with JMH (average time, 15 iterations) on the `gliner2-base-onnx` model:

#### 4 Entity Types

| Batch Size | Avg Latency (ms/op) | Error (±ms) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:-----------:|:--------------:|:--------------------:|
| 1          | 31.1                 | 1.0         | 31.1           | ~32.1                |
| 4          | 167.0                | 7.1         | 41.7           | ~24.0                |
| 8          | 344.5                | 29.2        | 43.1           | ~23.2                |

#### 8 Entity Types

| Batch Size | Avg Latency (ms/op) | Error (±ms) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:-----------:|:--------------:|:--------------------:|
| 1          | 42.0                 | 2.2         | 42.0           | ~23.8                |
| 4          | 204.0                | 4.9         | 51.0           | ~19.6                |
| 8          | 419.1                | 18.7        | 52.4           | ~19.1                |
