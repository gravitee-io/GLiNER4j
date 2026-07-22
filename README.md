# gliner4j

Java/JVM runtime for the **GLiNER family** of zero-shot information-extraction models, via ONNX
Runtime. One library hosts multiple model families behind a common API — the family is detected
automatically from the model bundle, so the same Java façades drive every model.

> **Zero-shot:** you pass the entity types / labels you want at inference time — no retraining.

## Supported model families

| Family | `architecture` | Tasks | Topology | Backbones | Example models |
|---|---|---|---|---|---|
| **GLiNER2** (fastino) | `gliner2` *(default)* | NER · classification · relation · structure | uni-encoder + count-aware scoring | DeBERTa-v3 | `fastino/gliner2-base-v1`, PII, GLiGuard |
| **GLiClass** | `gliclass` | zero-shot classification | uni-encoder + score head (`simple`/`mlp`) | ModernBERT, DeBERTa / mDeBERTa, ettin | `knowledgator/gliclass-modern-base-v3.0`, `gliclass-x-base` (multilingual), `gliclass-edge-v3.0` |
| **GLiNER uni-encoder** | `gliner-uni` | NER | single span graph (`markerV0`) | DeBERTa | `knowledgator/gliner-pii-base-v1.0`, `gliner-multitask-large` |
| **GLiNER bi-encoder** | `gliner-bi` | NER | dual text + label encoders | DeBERTa/ModernBERT/ettin + bge/MiniLM | `knowledgator/gliner-bi-small-v1.0`, `modern-gliner-bi-base` |

The family is read from the `architecture` key in `gliner4j_config.json` (absent ⇒ `gliner2`, so
older bundles keep working) and dispatched to the right implementation at load time — no code or
facade change per family.

## Features

- **Multi-family**, auto-dispatched: GLiNER2, GLiClass, original-GLiNER uni- and bi-encoder NER.
- **Tasks**: named-entity recognition, zero-shot text classification, relation extraction,
  structured/JSON extraction (relation & structure are GLiNER2 today).
- **Entity descriptions** and **per-call overrides** for dynamic vocabularies.
- **Batch processing** with a single batched encoder call and parallel per-text scoring.
- **Precision variants**: FP32, FP16, INT8 (signed `QInt8` dynamic quantization).
- **Execution providers**: CPU (default), CUDA, OpenVINO, CoreML — see
  [Execution providers](#execution-providers). DeBERTa bundles are exported `If`-node-free and with
  sanitized tensor names, so they run on the OpenVINO EP too.

## Prerequisites

- Java 21+
- [Task](https://taskfile.dev) — task runner
- [Maven](https://maven.apache.org/install.html) — Java build tool
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — Python package manager (for model
  download/export). No project virtualenv is created; export-script dependencies are declared inline
  via PEP 723, and `uv`/`uvx` manage a cached env automatically.

## Quick start

```bash
# 1. Export a model bundle (downloads from HuggingFace on first run).
task base                 # GLiNER2 base (NER + classification + relation + structure)
#  or: task gliclass      # GLiClass zero-shot classification (ModernBERT)
#  or: task gliner-pii    # GLiNER uni-encoder NER
#  or: task gliner-bi     # GLiNER bi-encoder NER

# 2. Build the Java project.
task build

# 3. Run a demo (interactive after a few annotated samples).
task demo:base            # NER + classification + relations
task demo:gliclass        # zero-shot topic classification
task demo:gliner-pii      # zero-shot PII NER
task demo:gliner-bi       # zero-shot NER (dual encoder)
```

> The demo loads `gliner4j-core` from the local Maven repo, so run `task build` after exporting a
> new model so the demo picks up the latest core.

## Usage

The family is auto-detected; the façade you pick depends on the **task**, not the model family.

### Named entity recognition — `GLiNER4jNER`

Works for `gliner2`, `gliner-uni`, and `gliner-bi` bundles.

```java
var entities = List.of(new EntityDefinition("person"), new EntityDefinition("organization"));
try (var gliner = GLiNER4jNER.load(Path.of("models/gliner-pii-base-onnx"), entities)) {
    Map<String, List<EntitySpan>> results = gliner.extract("John works at Google.");
    // per-call override + custom threshold:
    var override = gliner.extract("Email a@b.com", List.of(new EntityDefinition("email")), 0.4f);
    // batched:
    List<Map<String, List<EntitySpan>>> batch = gliner.extractBatch(List.of("…", "…"));
}
```

### Zero-shot classification — `GLiNER4jClassifier`

Works for `gliner2` and `gliclass` bundles (incl. all GLiClass variants — same code).

```java
var labels = List.of(
    new ClassificationLabel("technology", "Computing, software, AI"),
    new ClassificationLabel("sports", "Athletics and competitions")
);
try (var clf = GLiNER4jClassifier.load(Path.of("models/gliclass-modern-base-onnx"), labels)) {
    List<ClassificationResult> r = clf.classify("Apple unveiled its new M4 chip.", 0.5f);
}
```

### Relation & structured extraction (GLiNER2)

```java
var schema = Schema.builder()
    .entities(List.of(new EntityDefinition("person"), new EntityDefinition("organization")))
    .relations(List.of(new RelationDefinition("works_for", "Employment relationship")))
    .build();
try (var gliner = GLiNER4j.load(Path.of("models/gliner2-base-onnx"))) {
    ExtractionResult result = gliner.extract("John works at Google.", schema);
}
// JSON/structured extraction via SchemaExtractor; see the demo for examples.
```

### Selecting a variant or execution provider

```java
var config = RuntimeConfig.builder()
    .executionProvider(ExecutionProvider.CUDA)
    .gpuDeviceId(0)
    .build();
// 3rd arg is the ONNX variant folder: "onnx" (FP32), "onnx_fp16", "onnx_quantized" (INT8).
try (var gliner = GLiNER4jNER.load(modelDir, entities, "onnx_quantized", config)) { … }
```

## Models & tasks

Tasks are grouped by family. Every export builds **FP32 + FP16 + INT8** variants (override with
`EXPORT_ARGS`), DeBERTa-patched (no dynamic-rank `If` nodes) and name-sanitized for OpenVINO.

### GLiNER2 (fastino) — `task base` / `task pii` / `task gliguard`

| Task | Description |
|------|-------------|
| `task base` | Export `fastino/gliner2-base-v1` (NER + classification + relation + structure) |
| `task pii` | Export `fastino/gliner2-privacy-filter-PII-multi` (42 PII entity types) |
| `task gliguard` | Export `fastino/gliguard-LLMGuardrails-300M` (LLM safety classification) |
| `task demo:base` / `:pii` / `:gliguard` | Run the corresponding demo |

### GLiClass — zero-shot classification

| Task | Model | Notes |
|------|-------|-------|
| `task gliclass` | `gliclass-modern-base-v3.0` | ModernBERT, recommended default |
| `task gliclass:modern-large` | `gliclass-modern-large-v3.0` | highest accuracy |
| `task gliclass:x` | `gliclass-x-base` | **multilingual** (mDeBERTa, 20+ languages) |
| `task gliclass:edge` | `gliclass-edge-v3.0` | tiny/fast (~32M) |
| `task gliclass:large` | `gliclass-large-v3.0` | DeBERTa-large |
| `task demo:gliclass` / `:gliclass-sentiment` / `:gliclass-multilang` / `:gliclass-edge` | — | topic / sentiment / multilingual / edge demos |

GLiClass variants are **config-swaps** — the engine handles them with no code changes.

### GLiNER NER families

| Task | Model | Family |
|------|-------|--------|
| `task gliner-pii` | `gliner-pii-base-v1.0` | uni-encoder (deberta-v3-small) |
| `task gliner-bi` | `gliner-bi-small-v1.0` | bi-encoder (deberta + MiniLM label encoder) |
| `task demo:gliner-pii` / `:gliner-bi` | — | zero-shot NER demos |

### Shared

| Task | Description |
|------|-------------|
| `task build` | Build the Java project (formats first) |
| `task test` | Run tests |
| `task benchmark` | Run JMH benchmarks |
| `task format` | Apply license headers and format code |

## Precision variants

Each bundle lays out `onnx/` (FP32), `onnx_fp16/`, and `onnx_quantized/` (INT8). Select with the
`variant` load argument or the demo's 2nd CLI arg (`onnx` | `onnx_fp16` | `onnx_quantized`).

- **FP16** — ~half the size, near-lossless.
- **INT8** — smallest/fastest (signed `QInt8`, per-channel). Dynamic INT8 can de-calibrate
  confidence scores and is hardware-dependent (best on x86 with VNNI); prefer **FP16** when you
  threshold on scores, INT8 when you only need the argmax / smallest footprint.

## Execution providers

GLiNER4j runs every ONNX session on a selectable execution provider, chosen at run time via
`RuntimeConfig.executionProvider` (the demo takes it as a 3rd arg; the benchmark as a JMH param). A
missing provider logs a warning and falls back to CPU rather than failing the load.

| Provider | `RuntimeConfig` value | Native artifact | Notes |
|----------|-----------------------|-----------------|-------|
| CPU | `ExecutionProvider.CPU` | `com.microsoft.onnxruntime:onnxruntime` | Default. Always available. |
| CUDA | `ExecutionProvider.CUDA` | `com.microsoft.onnxruntime:onnxruntime_gpu` (build `-Pcuda`) | Needs an NVIDIA GPU + matching CUDA/cuDNN. |
| OpenVINO | `ExecutionProvider.OPENVINO` | `onnxruntime_openvino` (`task build:openvino`, then `-Popenvino`) | No official Java artifact; the script builds ORT with the OpenVINO EP. Run-time needs the OpenVINO runtime on the loader path. |
| CoreML | `ExecutionProvider.COREML` | `com.microsoft.onnxruntime:onnxruntime` (macOS) | Apple's accelerator, bundled in the macOS jar. |

```bash
# CPU (default)
task demo:base
# CUDA: build against the GPU runtime, then select the provider at run time
task demo:base EP=cuda MVN_FLAGS=-Pcuda
# CoreML on macOS
task demo:base EP=coreml
# OpenVINO: build the (unofficial) OpenVINO ONNX Runtime jar once, then select it
task build:openvino
task demo:base:openvino
```

## Benchmarks

JMH (`gliner4j-benchmark`) covers NER (`InferenceBenchmark`) and classification
(`ClassificationBenchmark`), parameterized over profile, ONNX variant, execution provider, text
length, batch size, and entity/label count.

### Cross-family comparison

Indicative single-text latency on **one CPU dev machine** (FP32, short text, 4 entity types /
labels). Absolute latency is machine-dependent — read this as a *relative* comparison across
families, **not** as a match to the GLiNER2 table below (measured separately on different hardware).

| Family / model | Task | Backbone | batch=1 (ms) |
|---|---|---|---|
| GLiClass **edge** (`gliclass-edge`) | classification | ettin-32m (~32M) | **36** |
| GLiNER **bi-encoder** (`gliner-bi-small`) | NER | deberta-small + MiniLM | 170 |
| GLiNER **uni-encoder** (`gliner-pii-base`) | NER | deberta-small | 193 |
| GLiClass (`gliclass-modern-base`) | classification | ModernBERT-base | 256 |
| GLiNER2 base (`gliner2-base`) | NER | deberta-base | 321 |

The 32M **edge** classifier is ~7–9× faster than the base-size models; the **bi-encoder** NER (small
DeBERTa + a tiny label encoder) edges out the uni-encoder; and **INT8** (`onnx_quantized`, the
benchmark's default variant) is roughly **~2× faster again** on CPU.

### GLiNER2 reference (upstream measurement)

Measured with JMH (average time, 15 iterations) on the GLiNER2 `gliner2-base-onnx` model, NER:

#### 4 entity types

| Batch | Avg latency (ms/op) | ± (ms) | Per-text (ms) | Throughput (texts/s) |
|:-----:|:-------------------:|:------:|:-------------:|:--------------------:|
| 1 | 43.6 | 3.8 | 43.6 | ~23.0 |
| 4 | 125.3 | 17.7 | 31.3 | ~31.9 |
| 8 | 220.8 | 17.7 | 27.6 | ~36.2 |

#### 8 entity types

| Batch | Avg latency (ms/op) | ± (ms) | Per-text (ms) | Throughput (texts/s) |
|:-----:|:-------------------:|:------:|:-------------:|:--------------------:|
| 1 | 62.0 | 8.5 | 62.0 | ~16.1 |
| 4 | 153.1 | 12.5 | 38.3 | ~26.1 |
| 8 | 274.4 | 27.1 | 34.3 | ~29.2 |

```bash
java -jar gliner4j-benchmark/target/gliner4j-benchmark.jar -p executionProvider=cuda
```

## Adding a model family

A family is `export script + a strategy + register` — the public façades don't change. The export
script (`scripts/export_onnx.py`, with a subcommand per family: `gliner2`, `gliclass`,
`gliner-uni` for the original GLiNER NER families) emits the ONNX graphs plus a
`gliner4j_config.json` carrying the `architecture` discriminator and any family-specific keys under
`architecture_config`.

## Uploading to HuggingFace

```bash
export HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"             # or: uvx --from huggingface-hub hf auth login
task hf-upload:base HF_REPO=<your-username>/gliner4j-onnx
```

> Requires a HuggingFace account and a pre-existing repo
> (`uvx --from huggingface-hub hf repo create gliner4j-onnx --type model --private`). Equivalent
> `hf-upload:*` targets exist per model.
