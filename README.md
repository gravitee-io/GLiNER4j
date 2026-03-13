# gliner4j

Java library for running [GLiNER2](https://github.com/fastino-ai/GLiNER2) Named Entity Recognition models using ONNX Runtime.

## Benchmarks

Measured with JMH (average time, 15 iterations) on the `gliner2-base-onnx` model:

#### 4 Entity Types

| Batch Size | Avg Latency (ms/op) | Error (±ms) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:-----------:|:--------------:|:--------------------:|
| 1          | 33.9                 | 2.4         | 33.9           | ~29.5                |
| 4          | 181.4                | 7.5         | 45.3           | ~22.1                |
| 8          | 364.7                | 23.5        | 45.6           | ~21.9                |

#### 8 Entity Types

| Batch Size | Avg Latency (ms/op) | Error (±ms) | Per-Text (ms) | Throughput (texts/s) |
|:----------:|:--------------------:|:-----------:|:--------------:|:--------------------:|
| 1          | 47.2                 | 3.2         | 47.2           | ~21.2                |
| 4          | 234.9                | 14.0        | 58.7           | ~17.0                |
| 8          | 455.5                | 21.1        | 56.9           | ~17.6                |

### Key Observations

- **Single-text latency**: ~34ms for 4 entity types, well within real-time thresholds.
- **Batch scaling**: Latency scales roughly linearly with batch size (no sub-linear speedup from batching).
- **Entity count impact**: Doubling entity types from 4 to 8 adds ~30-39% overhead. Limiting entity types to what you need yields meaningful speedups.
