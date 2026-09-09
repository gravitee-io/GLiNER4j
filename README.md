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

Single-text latency on one CPU dev machine (FP32, short text, 4 entity types / labels), ordered
fastest first. Absolute latency is machine-dependent — run the benchmark on your own hardware.

| Family / model | Task | Backbone |
|---|---|---|
| GLiClass **edge** (`gliclass-edge`) | classification | ettin-32m (~32M) |
| GLiNER **bi-encoder** (`gliner-bi-small`) | NER | deberta-small + MiniLM |
| GLiNER **uni-encoder** (`gliner-pii-base`) | NER | deberta-small |
| GLiClass (`gliclass-modern-base`) | classification | ModernBERT-base |
| GLiNER2 base (`gliner2-base`) | NER | deberta-base |

### ONNX Runtime vs llama.cpp (same model)

`task benchmark:engines` — `gliclass-edge` (ettin-32m), 8 labels, INT8 ONNX vs q8_0 GGUF, JMH
average time per `classifyBatch` on an Apple-silicon laptop (ORT has no GPU provider there,
llama.cpp uses Metal under `AUTO`): llama.cpp CPU beats ONNX Runtime CPU at every batch and text
length, and Metal beats both by a wide margin.

The same holds for GLiNER2 NER on that machine (`InferenceBenchmark`, 8 entity types): for
`gliner2-base`, `gliner2-privacy` (mDeBERTa, 250k vocab) and `gliner-pii-base` (uni-encoder) alike,
llama.cpp q8_0 CPU edges out ONNX int8 CPU and Metal is several times faster than either. Re-checked
on llamaj.cpp 2.8.0 (llama.cpp v0.4.0): unchanged within noise.

On an NVIDIA GPU both engines get a real accelerator (ONNX Runtime with `-Pcuda`, llama.cpp with a
CUDA build via `LLAMA_CPP_LIB_PATH`); the reference runs here used a consumer laptop CUDA card.
Mind the methodology: a throttling card moves every figure substantially between a cool and a hot
run, so only interleaved pairs are comparable (`gliner4j-benchmark/target/ab.sh` pattern).

The ggml DeBERTa encoder builds its disentangled attention as strided views of the position scores
(no `n·n·heads` gather), assembles the position bias in f16 and runs `ggml_flash_attn_ext` with it
as the mask; Q/K/V are one fused projection, batch rows are merged into the sequence axis for the
position products, and the GLiNER2 head matrices are f16 — that is the "composed" graph, an order
of magnitude below the original gather-based one. The fused kernel below replaces that attention
block again. On CUDA rows up to `gliner4j.ggml.batchMaxTokens` (default 384) are scored as one
padded batch, longer ones one by one.

Under the same paired protocol across all families (fused plugin loaded, batched GLiNER2.5 scoring,
fused LSTM + f16 heads for the uni-encoder), llama.cpp f16 CUDA beats ONNX fp16 CUDA on every
single-row call and every long batch of the DeBERTa-backbone families — GLiNER2 base, GLiNER2
privacy (42 labels), GLiGuard classification, GLiNER2.5 small and GLiNER PII base. GLiNER2.5
small's short batches are parity, with a run-to-run spread wider than the gap. That family's
`short b8` improved several-fold in three steps — `scoreEntitiesBatch` pads the rows into one encoder pass, the boundary encoder
and heads run once over the padded `(·, words, B)` stack (per-row `[EOS]` gathered at each
row's length, padding masked out of the boundary attention) and all candidates are scored in
one batched stage-B graph; f16 head matrices; and a host-side candidate pool that sorts packed
primitive keys instead of boxed comparators (it cost more per call than the GPU pass). Two
other gaps closed on the way: the GLiNER2 classifier used to score batch
rows one by one (`classifyBatch` now serves CUDA batches under the same token threshold as NER),
and the GLiNER2.5 and uni-encoder head matrices are exported in
f16 like the GLiNER2 heads (tensor cores instead of `sgemm`; re-export with
`task gliner2dot5-small:llamacpp` / `task gliner-pii:llamacpp`, older bundles keep working).
The uni-encoder's word LSTM is the plugin's second op, one cooperative launch for the whole
recurrence (every block owns a few hidden units, steps separated by a grid-wide barrier; the
composed graph needed ~40 ops per word, and its long-text graph the larger scheduler capacity
`GgmlWeights.SCHED_GRAPH_SIZE` now reserves). Quantized weights do not help on CUDA (q8_0 / q4_0
measure slightly slower than f16: the encoder is not weight-bound and the dequantising matmuls cost
more than they save) and f32 weights are markedly slower (plain `sgemm`, no tensor cores): f16 is
the fast point on a tensor-core GPU.

#### Fused DeBERTa attention plugin (CUDA)

The remaining gap to ONNX Runtime on CUDA was the disentangled attention itself: composed from
ggml ops it materialises two `(2n−1)·n·heads` position-score matrices per layer, reads them back
as strided bands and adds them into an f16 mask for flash attention. `gliner4j-llamacpp/native/ggml-deberta/`
is an out-of-tree **ggml backend plugin** — C++/CUDA living in this repo, built against a stock
llama.cpp checkout at the tag llamaj.cpp pins — that computes
`softmax(scale·q·kᵀ + q·P[i−j] + k·Qp[i−j])·v` in one tensor-core kernel, tile by tile in shared
memory, so nothing n² is ever written. It registers a `DEBERTA` device that shares the CUDA buffer
type; `GgmlWeights` then runs graphs through `ggml_backend_sched` over `[DEBERTA, CUDA, CPU]` with
no copies at the split boundaries. CPU, Metal and plugin-less runs keep the composed graph. The
disentangled attention itself is He et al.'s DeBERTa (see [References](#references)).

The 2-layer DeBERTa-v2 stack of the GLiClass router scorer and the streaming-span labels encoder
(`GgmlDeberta`) can run on the same op (`-Dgliner4j.ggml.fusedScorer=true`, tables built
in-graph, and it is faster on a short label section) but stays on the composed graph by
default: the kernel keeps its `q·kᵀ` tile in f16, which the LayerNorm-bounded backbone tolerates
and the raw Qwen3 hidden states these scorers consume do not (logits drift ~1e-3 at unit scale
and collapse at 50× — enough to flip a span sitting at the 0.5 threshold).

```bash
# needs cmake >= 3.24, nvcc, and ~/dev/llama.cpp built with -DBUILD_SHARED_LIBS=ON -DGGML_BACKEND_DL=ON -DGGML_CUDA=ON
task llamacpp:deberta-plugin            # LLAMA_CPP_DIR, CUDA_ARCHS="80;86;89;90", PLUGIN_DIR overridable
# → ~/.llama.cpp/plugins/libggml-deberta.so, picked up automatically (also $LLAMA_CPP_LIB_PATH/plugins/)
#   -Dgliner4j.ggml.plugin=<file> | none   overrides discovery;  GLINER4J_DEBERTA_REF=1 selects the
#   scalar reference kernel (debugging).  Rebuild the plugin whenever the llama.cpp pin changes: it
#   bakes ggml's backend API version and is rejected at load time otherwise (the run logs it).
```

Two kernel families are compiled in. The original `wmma` kernel (`turing`: 64 queries × 32 keys,
62 KB of shared memory; `ampere`: 64 × 64, 94 KB) stages scores and probabilities through shared
memory as f16. The register-resident kernel (`reg64x32`, `reg64x64`, `reg128x32`, `reg128x64`:
warps × keys, 56 to 147 KB) has the FlashAttention-2 structure on raw `mma.sync` + `ldmatrix`: each
warp owns 16 query rows, the `q·kᵀ` scores stay in f32 registers, the probabilities re-pack in
place as the next tensor-core operand and the accumulator rescale never touches shared memory;
only the two position products `Q·Pᵀ` and `K·Qpᵀ` go through shared memory, because the score
of `(i, j)` reads them on the diagonal `i − j`. On a 64 KB-shared-memory card (`reg64x32`, the one
variant that fits) it beats the wmma kernel, and by more the longer the row.

At first use on a device the plugin **autotunes**: every variant the card's shared memory holds is
timed on a synthetic 1 024-token, 12-head problem and checked against the scalar reference kernel,
the fastest correct one wins, and the whole table is logged (`gliner4j deberta: autotune … →`), so a
deployment on an unfamiliar card shows what it runs. `GLINER4J_DEBERTA_KERNEL=<variant>` forces a
kernel (`auto` re-enables the tuner), `GLINER4J_DEBERTA_AUTOTUNE=0` keeps the static rule (wmma
64 × 64 where it fits, 64 × 32 elsewhere). The standalone harness runs every variant a card holds
against a double-precision reference with timing in one command (`HEADS=12 N=1408 ./test_attn`).
The op also accepts the f32 projections directly and converts them while staging
(`-Dgliner4j.ggml.rawQkv=true` drops the three cast ops per layer); off by default because the
kernel re-reads K / V per key tile and the doubled traffic costs more than the casts it removes.

Requirements: per-head pre-scaled position tables in the bundle (exports from this version; older
bundles silently keep the composed graph), head dim 64 (every DeBERTa-v3 size). The kernel is
validated by `gliner4j-llamacpp/native/ggml-deberta/test_attn.cu` against a double-precision CPU
reference and by `GgmlDebertaV3ParityTest` against HuggingFace hidden states.

**Scheduler ordering.** ggml's scheduler orders consecutive splits on different backends with host
round trips (or not at all when the split has inputs), which costs two per layer here. The build
script applies `native/ggml-deberta/patches/0001-ggml-sched-cross-backend-events.patch` to the
llama.cpp checkout (stream-side event waits between backends that share a device; ~20 lines,
`git checkout ggml/` reverts it) and rebuilds `libggml-base`; the runtime detects it, creates the
scheduler with events and switches the plugin's own host synchronisation off (log line
"event ordering"; `-Dgliner4j.ggml.eventOrdering=false` forces the host-sync path). Without the
patch everything still works, slightly slower on long calls.

The DeBERTa-backbone bundles (`gliner2`, `gliner2dot5`, `gliner-uni`, `gliner-bi`) can also ship
block-quantised encoder matrices: `--quant q8_0 --quant q4_0` on the export writes
`gguf/model-<quant>.gguf` next to the f16 `model.gguf`, selected with `variant=q8_0` / `q4_0`
(`f16` or the ONNX default name means `model.gguf`). GLiNER2 base: 485 MB f16 → 313 MB q8_0
→ 221 MB q4_0. Against ONNX fp32 on five texts q8_0 keeps every span with scores within 0.006,
q4_0 keeps the spans but confidences drift by up to 0.15. Speed is unchanged on both CPU and CUDA
(the encoder is bound by the attention-bias tensors, not by the weight GEMMs), so the quantised
variants are a memory choice, not a latency one.

Concurrent calls (the batch fans out on virtual threads) are packed by the encoder dispatcher into
shared `llama_encode` calls, which is where the batch-8 gap comes from. ggml threads default to the
performance cores (`-Dgliner4j.llamacpp.threads=N` to override): one thread per logical CPU makes
llama.cpp several times slower.

The 32M **edge** classifier is several times faster than the base-size models; the **bi-encoder**
NER (small DeBERTa + a tiny label encoder) edges out the uni-encoder; and **INT8**
(`onnx_quantized`, the benchmark's default variant) is faster again on CPU.

### Running the benchmarks

Batching amortises the encoder: per-text latency drops and throughput rises from batch 1 to 8, and
more entity types cost more per call. Measure on your own hardware:

```bash
java -jar gliner4j-benchmark/target/gliner4j-benchmark.jar -p executionProvider=cuda
```

## References

The disentangled attention that the ggml graphs, the CUDA plugin kernel and the export scripts
implement is the DeBERTa research from Microsoft. This repository reimplements the maths from the
papers and shares no code with the reference implementation; the model weights keep the licences
stated on their HuggingFace model cards.

- Pengcheng He, Xiaodong Liu, Jianfeng Gao, Weizhu Chen. *DeBERTa: Decoding-enhanced BERT with
  Disentangled Attention.* ICLR 2021. [arXiv:2006.03654](https://arxiv.org/abs/2006.03654)
- Pengcheng He, Jianfeng Gao, Weizhu Chen. *DeBERTaV3: Improving DeBERTa using ELECTRA-Style
  Pre-Training with Gradient-Disentangled Embedding Sharing.* ICLR 2023.
  [arXiv:2111.09543](https://arxiv.org/abs/2111.09543)
- Reference implementation: [microsoft/DeBERTa](https://github.com/microsoft/DeBERTa), released
  under the [MIT License](https://github.com/microsoft/DeBERTa/blob/master/LICENSE).

### Third-party licences

`gliner4j-llamacpp/licenses/`, shipped in that module's jar under `licenses/`:

- `DEBERTA-V3-LICENSE.txt` — MIT, Microsoft Corporation
  ([microsoft/DeBERTa](https://github.com/microsoft/DeBERTa)).
- `LLAMA_CPP-LICENSE.txt` — MIT, the ggml authors
  ([ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) `v0.4.0`, build `b10818`).

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
