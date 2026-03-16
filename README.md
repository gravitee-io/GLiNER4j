# gliner4j

Java library for running [GLiNER2](https://github.com/fastino-ai/GLiNER2) Named Entity Recognition models using ONNX Runtime.

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
