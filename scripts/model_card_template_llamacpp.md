---
language:
$languages
license: apache-2.0
library_name: gguf
tags:
$extra_tags
  - java
  - llama.cpp
  - ggml
  - gguf
  - streaming
pipeline_tag: $pipeline_tag
base_model: $base_model
---

# GLiNER4j llama.cpp — $display_name

GGUF export of [$base_model](https://huggingface.co/$base_model) for the JVM through
[GLiNER4j](https://github.com/gravitee-io/GLiNER4j)'s `gliner4j-llamacpp` module: the Qwen3 backbone runs
on [llama.cpp](https://github.com/ggml-org/llama.cpp) via [llamaj.cpp](https://github.com/gravitee-io/llamaj.cpp)
(Java 25, FFM), and the task head runs as a ggml compute graph on the same backend (Metal / CUDA / CPU).
No ONNX Runtime is required.

`architecture: $architecture`

## Supported Tasks

$tasks_table

## Repository Structure

```
├── gliner4j_config.json        # architecture = $architecture + head parameters
├── tokenizer.json              # Qwen2 BPE tokenizer with the marker tokens
├── tokenizer_config.json
├── gguf/
│   ├── backbone-q8_0.gguf      # Qwen3-0.6B decoder, Q8_0 ($size_q8) — runtime default
│   ├── backbone-f16.gguf       # Qwen3-0.6B decoder, F16 ($size_f16) — reference precision
│   └── scorer.gguf             # $scorer_desc ($size_scorer)
$extra_tree└── reference.json              # Python-pipeline outputs used by GLiNER4j's parity tests
```

## Model Architecture

$architecture_section

## Usage

$usage_section

Pick the backbone precision with `architecture_config.backbone_gguf` in `gliner4j_config.json`
(`backbone-f16.gguf` for parity checks, `backbone-q8_0.gguf` by default). Requires Java 25 and the
`gliner4j-llamacpp` module and the llama.cpp native libraries in `~/.llama.cpp` (`task llamacpp:natives`; Apple Silicon and Linux x86_64).

## License

Apache License 2.0 (model and export).
