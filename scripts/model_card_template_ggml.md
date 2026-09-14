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
pipeline_tag: $pipeline_tag
base_model: $base_model
---

# GLiNER4j ggml — $display_name

ggml / llama.cpp export of [$base_model](https://huggingface.co/$base_model) for the JVM through
[GLiNER4j](https://github.com/gravitee-io/GLiNER4j)'s `gliner4j-llamacpp` module (Java 25, FFM via
[llamaj.cpp](https://github.com/gravitee-io/llamaj.cpp)). The bundle declares `"engine": "llamacpp"`;
the same GLiNER4j facades (`GLiNER4jNER`, `GLiNER4jClassifier`, `RelationExtractor`, `GLiNER4j`)
load it like an ONNX bundle and run it on Metal, CUDA, Vulkan or the CPU. No ONNX Runtime is required.

`architecture: $architecture` — $family_desc

## Supported Tasks

$tasks_table

## Repository Structure

```
├── gliner4j_config.json        # architecture = $architecture, engine = llamacpp
├── tokenizer.json              # HF tokenizer (GLiNER4j tokenizes; ggml only sees ids)
$gguf_tree└── reference.json              # Python outputs used by GLiNER4j's parity tests
```

## Model Architecture

$architecture_section

## Usage

```java
$usage_snippet
```

Pass `"f16"` or `"q8_0"` as the variant to pick a GGUF quantization where the bundle ships several;
`ExecutionProvider.AUTO` offloads to Metal / CUDA / Vulkan, `ExecutionProvider.CPU` stays on the CPU.
Requires Java 25, the `gliner4j-llamacpp` module and the llama.cpp native libraries in `~/.llama.cpp`
(`task llamacpp:natives` in GLiNER4j installs the release llamaj.cpp pins; Apple Silicon and Linux x86_64).

## License

Apache License 2.0 (model and export).
