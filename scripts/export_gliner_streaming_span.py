#
# Copyright © 2015 The Gravitee team (http://gravitee.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "torch>=2.2",
#     "onnx>=1.15",
#     "onnxruntime>=1.17",
#     "onnxconverter-common>=1.14",
#     "transformers>=5.0",
#     "safetensors>=0.4",
#     "sentencepiece>=0.2",
#     "protobuf>=4",
#     "typer>=0.15",
#     "numpy>=1.26",
#     "rich>=13",
#     "huggingface-hub>=0.20",
#     "gguf>=0.19",
#     "gliner>=0.2.28",
# ]
# ///
"""Export a GLiNER ``gliner_streaming_span`` checkpoint (causal Qwen3 backbone, e.g.
knowledgator/gliner-stream-pii-v1.0) into a gliner4j bundle for the llamaj.cpp runtime.

Bundle layout::

    {output_dir}/gliner4j_config.json      architecture = "gliner-streaming-span"
    {output_dir}/tokenizer.json (+config)  Qwen2 BPE tokenizer with <<LABEL>>/<<SEP>>
    {output_dir}/gguf/backbone-f16.gguf    Qwen3 decoder (reference precision)
    {output_dir}/gguf/backbone-q8_0.gguf   Qwen3 decoder (runtime default)
    {output_dir}/gguf/scorer.gguf          labels encoder (DeBERTa) + prompt/span projection heads (f32)
    {output_dir}/reference.json            Python entities (stateless + streaming snapshots) for parity tests

Usage:
    uv run scripts/export_gliner_streaming_span.py --model-path knowledgator/gliner-stream-pii-v1.0 \\
        --output-dir models/gliner-stream-pii-onnx --llama-cpp-dir ../llama.cpp
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Annotated

import torch
import typer
from rich.console import Console

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_gliclass_decoder_kv as decoder_kv  # noqa: E402

console = Console()
app = typer.Typer(add_completion=False)

PII_LABELS = ["person", "email address", "phone number", "street address", "credit card number", "passport number"]
REFERENCE_TEXTS = [
    "Customer Alice Johnson can be reached at alice@example.com or +1 202-555-0147.",
    "Please ship the order to 42 Maple Street, Springfield and charge card 4111 1111 1111 1111.",
    "Jane Doe (passport X1234567) asked us to call +33 6 12 34 56 78 before Friday.",
    "The meeting is at noon; nothing sensitive here.",
]
REFERENCE_SESSIONS = [
    ["Customer Alice Johnson ", "can be reached at alice@example.com ", "or +1 202-555-0147."],
    ["Jane", " Doe asked us to call", " +1 (415) 555-0132.", " Her card ends in 4242."],
]


def _export_scorer_gguf(model, out_dir: Path) -> Path:
    from gguf import GGUFWriter

    core = model.model
    cfg = model.config
    gguf_dir = out_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    path = gguf_dir / "scorer.gguf"
    w = GGUFWriter(str(path), arch="gliner-streaming-span-scorer")
    enc = core.labels_encoder.encoder.encoder
    attn = enc.layer[0].attention.self
    assert attn.pos_att_type == ["p2c", "c2p"] and not attn.share_att_key and attn.position_buckets <= 0
    assert enc.conv is None and "layer_norm" not in enc.norm_rel_ebd
    assert not hasattr(core, "token_projection"), "backbone hidden size must equal config.hidden_size"
    assert cfg.span_mode == "markerV2" and cfg.span_encoder_config is None
    w.add_uint32("scorer.hidden_size", cfg.hidden_size)
    w.add_uint32("scorer.num_layers", len(enc.layer))
    w.add_uint32("scorer.num_heads", attn.num_attention_heads)
    w.add_uint32("scorer.max_width", cfg.max_width)
    w.add_float32("scorer.layer_norm_eps", float(enc.layer[0].attention.output.LayerNorm.eps))
    for name, tensor in core.state_dict().items():
        if name.startswith("token_rep_layer."):
            continue
        # ggml caps tensor names at 64 chars: drop the redundant module nesting.
        short = name.replace("labels_encoder.encoder.encoder.", "labels_encoder.").replace(
            "span_rep_layer.span_rep_layer.", "span_rep."
        )
        assert len(short) < 64, short
        w.add_tensor(short, tensor.detach().float().cpu().numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    console.print(f"  [green]✓[/green] scorer.gguf ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def _write_config(model, tokenizer, out_dir: Path, default_gguf: str) -> None:
    cfg = model.config
    enc = model.model.token_rep_layer.decoder_layer.model.config
    config = {
        "architecture": "gliner-streaming-span",
        "hidden_size": cfg.hidden_size,
        "max_width": cfg.max_width,
        "architecture_config": {
            "backbone_model_name": cfg.model_name,
            "backbone_gguf": default_gguf,
            "n_ctx_train": int(enc.max_position_embeddings),
            "max_len": int(cfg.max_len),
            "class_token_index": cfg.class_token_index,
            "sep_token_index": cfg.sep_token_index,
            "label_token": cfg.label_token,
            "sep_token": cfg.sep_token,
            "max_width": cfg.max_width,
            "right_context_width": cfg.right_context_width if cfg.right_context_width is not None else cfg.max_width,
            "subtoken_pooling": cfg.subtoken_pooling,
            "layer_norm_eps": 1e-7,
        },
    }
    (out_dir / "gliner4j_config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(str(out_dir))
    console.print("  [green]✓[/green] gliner4j_config.json + tokenizer")


def _write_reference(model, out_dir: Path) -> None:
    stateless = []
    for text in REFERENCE_TEXTS:
        stateless.append({"text": text, "labels": PII_LABELS, "entities": model.predict_entities(text, PII_LABELS, threshold=0.5)})
    sessions = []
    for i, chunks in enumerate(REFERENCE_SESSIONS):
        sid = f"ref-{i}"
        snapshots = []
        for chunk in chunks:
            snapshots.append(model.inference([chunk], PII_LABELS, session_id=[sid], threshold=0.5)[0])
        model.clear_session(sid)
        sessions.append({"chunks": chunks, "labels": PII_LABELS, "snapshots": snapshots})
    (out_dir / "reference.json").write_text(json.dumps({"stateless": stateless, "sessions": sessions}, indent=1) + "\n")
    console.print(f"  [green]✓[/green] reference.json ({len(stateless)} texts, {len(sessions)} sessions)")


@app.command()
def export(
    model_path: Annotated[str, typer.Option("--model-path", help="gliner streaming-span model dir or HF repo ID")],
    output_dir: Annotated[str, typer.Option("--output-dir", help="Output bundle directory")],
    llama_cpp_dir: Annotated[str | None, typer.Option("--llama-cpp-dir", help="llama.cpp checkout (default: $LLAMA_CPP_DIR or ../llama.cpp)")] = None,
    gguf_types: Annotated[list[str] | None, typer.Option("--gguf-type", help="GGUF outtypes (default: f16, q8_0)")] = None,
    default_gguf: Annotated[str, typer.Option("--default-gguf", help="GGUF the runtime loads by default")] = "q8_0",
    skip_backbone: Annotated[bool, typer.Option("--skip-backbone", help="Do not (re)convert the backbone GGUFs")] = False,
) -> None:
    """Export a GLiNER streaming-span checkpoint to a llama.cpp + ggml gliner4j bundle."""
    from gliner import GLiNER

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    llama_dir = Path(llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR") or (Path(__file__).resolve().parents[2] / "llama.cpp"))
    gguf_types = gguf_types or ["f16", "q8_0"]

    console.print(f"\n[bold]Loading model from {model_path}...[/bold]")
    model = GLiNER.from_pretrained(model_path, load_tokenizer=True, map_location="cpu", dtype="fp32").eval()
    cfg = model.config
    if cfg.model_type != "gliner_streaming_span":
        raise typer.BadParameter(f"{model_path} is a {cfg.model_type!r} checkpoint; this command is for gliner_streaming_span")
    tokenizer = model.data_processor.transformer_tokenizer
    console.print(
        f"  backbone={cfg.model_name}, hidden={cfg.hidden_size}, max_width={cfg.max_width}, "
        f"right_context_width={cfg.right_context_width}, class_token={cfg.class_token_index}, sep_token={cfg.sep_token_index}"
    )

    console.print("\n[bold]Exporting scorer (labels encoder + span head) to GGUF...[/bold]")
    _export_scorer_gguf(model, out)

    if skip_backbone:
        console.print("\n[bold]Skipping backbone GGUF conversion (--skip-backbone)[/bold]")
    else:
        console.print(f"\n[bold]Converting backbone to GGUF via {llama_dir}...[/bold]")
        decoder = model.model.token_rep_layer.decoder_layer.model
        ggufs = decoder_kv.export_qwen3_backbone_gguf(decoder, decoder.config, tokenizer, out, llama_dir, gguf_types)
        for path in ggufs:
            decoder_kv._check_gguf_vocab(path, cfg.vocab_size, {cfg.label_token: cfg.class_token_index, cfg.sep_token: cfg.sep_token_index})

    console.print("\n[bold]Writing config, tokenizer and reference outputs...[/bold]")
    _write_config(model, tokenizer, out, f"backbone-{default_gguf}.gguf")
    with torch.no_grad():
        _write_reference(model, out)
    console.print(f"\n[bold]Output structure in {out}:[/bold]")
    decoder_kv.shared._print_tree(out)


if __name__ == "__main__":
    app()
