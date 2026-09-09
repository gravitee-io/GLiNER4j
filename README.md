# gliner4j

[![CircleCI](https://circleci.com/gh/gravitee-io/GLiNER4j/tree/main.svg?style=shield)](https://circleci.com/gh/gravitee-io/GLiNER4j/tree/main)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Community Forum](https://img.shields.io/badge/community-forum-00bfa5.svg)](https://community.gravitee.io)

Java/JVM runtime for the **GLiNER family** of zero-shot information-extraction models, via ONNX
Runtime. One library hosts multiple model families behind a common API — the family is detected
automatically from the model bundle, so the same Java façades drive every model.

> **Zero-shot:** you pass the entity types / labels you want at inference time — no retraining.

## Supported model families

| Family | `architecture` | Tasks | Topology | Backbones | Example models |
|---|---|---|---|---|---|
| **GLiNER2** (fastino) | `gliner2` *(default)* | NER · classification · relation · structure | uni-encoder + count-aware scoring | DeBERTa-v3 | `fastino/gliner2-base-v1`, PII, GLiGuard |
| **GLiNER2.5** (fastino) | `gliner2dot5` | NER · classification · relation | uni-encoder + span-free boundary head (start/end/inside marginals, shared candidate pool, pooled reranker) | DeBERTa-v3, mDeBERTa-v3 | `fastino/gliner2.5-base-v1`, `gliner2.5-small-v1`, `gliner2.5-multi-v1` |
| **GLiClass** | `gliclass` | zero-shot classification | uni-encoder + score head (`simple`/`mlp`) | ModernBERT, DeBERTa / mDeBERTa, ettin | `knowledgator/gliclass-modern-base-v3.0`, `gliclass-x-base` (multilingual), `gliclass-edge-v3.0` |
| **GLiNER streaming-span** (`gliner4j-llamacpp`, Java 25) | `gliner-streaming-span` | NER · streaming NER with rolling revision | causal decoder on **llama.cpp** (KV cache per session) + DeBERTa labels encoder and `markerV2` span head as **ggml graphs** | Qwen3-0.6B | `knowledgator/gliner-stream-pii-v1.0` |
| **GLiClass decoder-kv** (`gliner4j-llamacpp`, Java 25) | `gliclass-decoder-kv` | zero-shot classification · LLM routing · streaming sessions | causal decoder on **llama.cpp** (KV cache per session) + DeBERTa scorer as a **ggml graph** on the same backend | Qwen3-0.6B | `scx-admin/scx-router-v0.1` |
| **GLiNER uni-encoder** | `gliner-uni` | NER | single span graph (`markerV0`) | DeBERTa | `knowledgator/gliner-pii-base-v1.0`, `gliner-multitask-large` |
| **GLiNER bi-encoder** | `gliner-bi` | NER | dual text + label encoders | DeBERTa/ModernBERT/ettin + bge/MiniLM | `knowledgator/gliner-bi-small-v1.0`, `modern-gliner-bi-base` |

The family is read from the `architecture` key in `gliner4j_config.json` (absent ⇒ `gliner2`, so
older bundles keep working) and dispatched to the right implementation at load time — no code or
facade change per family.

## Features

- **Multi-family**, auto-dispatched: GLiNER2, GLiNER2.5, GLiClass, original-GLiNER uni- and bi-encoder NER,
  and the GLiClass decoder-kv router (Qwen3 on llama.cpp via [llamaj.cpp](https://github.com/gravitee-io/llamaj.cpp)).
- **Tasks**: named-entity recognition, zero-shot text classification, relation extraction,
  structured/JSON extraction (relation: GLiNER2 & GLiNER2.5; structure: GLiNER2 today).
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
#  or: task gliner2dot5    # GLiNER2.5 base (span-free boundary NER + classification; -small / -multi too)
#  or: task gliclass      # GLiClass zero-shot classification (ModernBERT)
#  or: task gliner-pii    # GLiNER uni-encoder NER
#  or: task gliner-bi     # GLiNER bi-encoder NER

# 2. Build the Java project.
task build

# 3. Run a demo (interactive after a few annotated samples).
task demo:base            # NER + classification + relations
task demo:gliner2dot5     # GLiNER2.5 span-free NER + classification
task demo:scx-router      # LLM router on llama.cpp (Java 25): routing families + streaming session
task demo:stream-pii      # streaming PII NER on llama.cpp (Java 25): revised snapshots chunk by chunk
task demo:gliclass        # zero-shot topic classification
task demo:gliner-pii      # zero-shot PII NER
task demo:gliner-bi       # zero-shot NER (dual encoder)
```

> The demo loads `gliner4j-core` from the local Maven repo, so run `task build` after exporting a
> new model so the demo picks up the latest core.

## Usage

The family is auto-detected; the façade you pick depends on the **task**, not the model family.

### Named entity recognition — `GLiNER4jNER`

Works for `gliner2`, `gliner2dot5`, `gliner-uni`, and `gliner-bi` bundles.

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

### Streaming NER — `StreamingSpanNer` (gliner4j-llamacpp)

`knowledgator/gliner-stream-pii-v1.0` is a GLiNER `gliner_streaming_span` model: a Qwen3-0.6B decoder
that keeps a conversation in its KV cache, so text can arrive in chunks (ASR, chat, logs). After each
append only the spans ending in the new words, plus those within `right_context_width` of the latest
word, are rescored; the result is always the complete, revised entity snapshot over the text so far.
Backbone on llama.cpp, labels encoder and span head as ggml graphs — no ONNX Runtime involved.

```java
try (var ner = StreamingSpanNer.load(Path.of("models/gliner-stream-pii-onnx"))) {
    var labels = List.of("person", "email address", "phone number");
    ner.extract("Customer Alice Johnson can be reached at alice@example.com.", labels, 0.5f);

    try (var session = ner.openSession("call-42", labels)) {
        session.append("Customer Alice Johnson ", 0.5f);          // [person: Alice Johnson]
        session.append("can be reached at alice@example.com ", 0.5f); // + email address
        session.append("or +1 202-555-0147.", 0.5f);               // + phone number
    }
}
```

The same bundle also loads through `GLiNER4jNER` (stateless). Export with `task stream-pii`, then
`task demo:stream-pii` for the colored demo with chunked sessions and an interactive `/stream` mode.

### LLM routing & streaming classification — `DecoderKvRouter` (gliner4j-llamacpp)

`scx-admin/scx-router-v0.1` is a GLiClass `decoder-kv` model: a Qwen3-0.6B decoder scores which of
several LLMs should serve a request, its task type, difficulty, reasoning need and expected output
length, or any zero-shot label set. Everything runs on llama.cpp/ggml through the `gliner4j-llamacpp`
module (**Java 25**): the backbone as a GGUF (q8_0 by default, Metal/CUDA offload) and the small DeBERTa
scorer as a ggml compute graph built in Java on the same backend, so there is a single GPU runtime in
the process. The bundle also carries the scorer as ONNX (`onnx/scorer.onnx`) — the PyTorch-parity oracle
at export time and an ONNX Runtime fallback pinned to the CPU provider (`scorer_backend: onnx`).

```java
try (var router = DecoderKvRouter.load(Path.of("models/scx-router-onnx"))) {
    var models = List.of("coder", "DeepSeek-V3.1", "gemma-4-31B-it", "gpt-oss-120b", "Qwen3-32B");
    router.classify("Write a Python function that merges two sorted linked lists.", models, 0.5f);
    router.classifySingleLabel(prompt, List.of("reasoning", "nonreasoning")).get(0);

    // Streaming: the conversation stays in the llama.cpp KV cache; every turn re-routes by
    // encoding only the new tokens plus the label section.
    try (var session = router.openSession("chat-42")) {
        session.append("I need help refactoring some Rust code.");
        session.classify(models, 0.5f);
        session.append(" Specifically the borrow checker keeps rejecting this function.");
        session.classify(models, 0.5f);
    }
}
```

**Concurrency.** A llama.cpp context is single-threaded, so the backbone never exposes it: callers
enqueue decode requests and a dispatcher packs whatever is pending — one request per sequence, up to
`n_batch` tokens — into a single `llama_decode`, then hands each caller its rows. Concurrent stateless
calls and appends from different sessions therefore share forward passes; `n_seq_max` (default 8) caps
how many run at once and callers wait for a free slot rather than fail. The same mechanism serves the
streaming-span family below.

The same bundle also loads through `GLiNER4jClassifier` (multi-label) once `gliner4j-llamacpp` is on
the classpath. Export with `task scx-router` (needs a llama.cpp checkout for `convert_hf_to_gguf.py`,
`LLAMA_CPP_DIR`, default `../llama.cpp`), then `task demo:scx-router` for the colored demo: every
sample is routed across the model / task / reasoning / difficulty / output-length families, a chat is
re-routed turn by turn from the KV cache, and the interactive prompt offers `/route`, `/session` and
`/reset`. The demo module itself now targets Java 25 because of this profile.

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

### Engines — ONNX Runtime or llama.cpp

> **Native libraries.** Since llamaj.cpp 2.7 the llama.cpp/ggml dylibs are not inside the jar: the
> loader uses `~/.llama.cpp` (or the directory in `LLAMA_CPP_LIB_PATH`). `task llamacpp:natives`
> downloads the matching llama.cpp release (v0.4.0 for llamaj.cpp 2.8.0) with llamaj.cpp's
> `scripts/download-native-libraries.sh` and installs it there. A stale `~/.llama.cpp` from an older
> llamaj.cpp fails at context creation with `Unsupported ctx type` — rerun the task after a bump.

A bundle declares its engine in `gliner4j_config.json` (`"engine": "onnx"`, the default, or
`"llamacpp"`); the facades dispatch on (family, engine), so the Java code above is identical for
both. llama.cpp bundles (module `gliner4j-llamacpp`, Java 25) keep the encoder as GGUF and the task
heads as small ggml graphs under `gguf/`; the variant argument selects the GGUF quantization
(`"f16"`, `"q8_0"`, or anything else for the bundle default). `ExecutionProvider.AUTO`/`CUDA`
offload to Metal/CUDA/Vulkan, `CPU` stays on the CPU. Every current family runs on both engines:

| Family | llama.cpp bundle | Backbone on | Heads |
|---|---|---|---|
| GLiClass (ModernBERT / BERT) | `task gliclass-edge:llamacpp` | llama.cpp (`llama_encode`) | ggml graph |
| GLiNER uni-encoder (markerV0, token_level) | `task gliner-pii:llamacpp`, `task gliner-multitask:llamacpp` | gliner4j's DeBERTa-v3 ggml graph | ggml graph (biLSTM unrolled) |
| GLiNER bi-encoder | `task gliner-bi:llamacpp` | DeBERTa ggml graph + label encoder on llama.cpp | ggml graph |
| GLiNER2 (base, PII, gliguard) | `task base:llamacpp` / `pii:llamacpp` / `gliguard:llamacpp` | DeBERTa ggml graph | ggml graphs (GRU count head, count transformer) |
| GLiNER2.5 | `task gliner2dot5-small:llamacpp` | DeBERTa ggml graph | ggml graphs + host candidate pool |
| GLiClass decoder-kv, streaming span | `task scx-router` / `task stream-pii` | Qwen3 on llama.cpp | ggml graphs |

llama.cpp has no DeBERTa architecture, so `gliner4j-llamacpp` ships its own DeBERTa-v3 ggml graph
(disentangled attention with log-bucketed relative positions); it matches HF `last_hidden_state`
to about 1e-2 absolute with f16 weights. Each llama.cpp family is verified against the ONNX engine
on the same texts (scores within 0.03) in `gliner4j-llamacpp`'s integration tests.

```bash
task llamacpp:all               # every models/*-llamacpp bundle (needs the *-hf checkouts + a llama.cpp checkout: LLAMA_CPP_DIR)
task benchmark:engines          # ONNX Runtime vs llama.cpp, CPU and Metal/CUDA
task demo:base:llamacpp         # any demo profile has a llama.cpp twin (base, pii, gliguard, gliner2dot5-small, gliclass-edge, gliner-pii, gliner-bi, gliner-multitask)
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

### ONNX Runtime vs llama.cpp (same model)

`task benchmark:engines` — `gliclass-edge` (ettin-32m), 8 labels, INT8 ONNX vs q8_0 GGUF, JMH
average time per `classifyBatch` on an Apple M-series laptop (10 performance cores; ORT has no
GPU provider there, llama.cpp uses Metal under `AUTO`):

| batch × text | ONNX Runtime CPU | llama.cpp CPU | llama.cpp Metal |
|---|---|---|---|
| 1 × short (~64 tok) | 13.0 ms | **4.4 ms** | **2.2 ms** |
| 1 × long (~512 tok) | 50.6 ms | **18.2 ms** | **7.0 ms** |
| 8 × short | 106.8 ms | **25.3 ms** | **9.3 ms** |
| 8 × long | 424.0 ms | **164.1 ms** | **49.9 ms** |

GLiNER2 NER on the same machine (`InferenceBenchmark`, 8 entity types, short text, batch 1):
`gliner2-base` 56 ms ONNX int8 CPU vs 47 ms llama.cpp q8_0 CPU vs 20 ms Metal; `gliner2-privacy`
(mDeBERTa, 250k vocab) 97 ms vs 79 ms vs 41 ms; `gliner-pii-base` (uni-encoder) 34 ms vs 31 ms vs 13 ms.
Numbers re-checked on llamaj.cpp 2.8.0 (llama.cpp v0.4.0): unchanged within noise.

Concurrent calls (the batch fans out on virtual threads) are packed by the encoder dispatcher into
shared `llama_encode` calls, which is where the batch-8 gap comes from. ggml threads default to the
performance cores (`-Dgliner4j.llamacpp.threads=N` to override): one thread per logical CPU makes
llama.cpp several times slower.

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
> `hf-upload:*` targets exist per model: `base`, `pii`, `gliguard`, `gliner2dot5`,
> `gliner2dot5-small`, `gliner2dot5-multi`, `scx-router`, `stream-pii`. Each target re-renders the
> bundle's model card (`scripts/render_model_card.py`, sizes read from the exported files) and mirrors
> the bundle, skipping local `onnx_*optimized*` caches. The llama.cpp bundles (`scx-router`,
> `stream-pii`) upload the GGUF backbones (~1.9 GB) plus `scorer.gguf` and `reference.json`.

```bash
task hf-upload:gliner2dot5-small HF_REPO=gravitee-io/gliner4j-gliner2.5-small-v1
task hf-upload:scx-router        HF_REPO=gravitee-io/gliner4j-scx-router-v0.1
task hf-upload:stream-pii        HF_REPO=gravitee-io/gliner4j-gliner-stream-pii-v1.0
```

The llama.cpp/ggml twins of the ONNX bundles (`models/*-llamacpp`, `"engine": "llamacpp"`) are separate
repos with their own `hf-upload:<name>:llamacpp` targets — `gliclass-edge`, `gliner-pii`, `gliner-bi`,
`gliner-multitask`, `base`, `pii`, `gliguard`, `gliner2dot5-small` — rendered from
`scripts/model_card_template_ggml.md`:

```bash
task hf-upload:base:llamacpp HF_REPO=gravitee-io/gliner4j-gliner2-base-ggml
```
