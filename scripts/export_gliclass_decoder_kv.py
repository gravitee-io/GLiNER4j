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
#     "gliclass>=0.1.20",
# ]
# ///
"""Export a GLiClass ``decoder-kv`` checkpoint (Qwen3 backbone, e.g. scx-admin/scx-router-v0.1)
into a gliner4j bundle whose backbone runs on llama.cpp (via llamaj.cpp) and whose scorer runs on
ONNX Runtime.

Bundle layout::

    {output_dir}/gliner4j_config.json      architecture = "gliclass-decoder-kv"
    {output_dir}/tokenizer.json (+config)  Qwen2 BPE tokenizer with <<LABEL>>/<<SEP>>/<<EXAMPLE>>
    {output_dir}/gguf/backbone-f16.gguf    Qwen3 decoder (reference precision)
    {output_dir}/gguf/backbone-q8_0.gguf   Qwen3 decoder (runtime default)
    {output_dir}/gguf/scorer.gguf          DecoderKVScorer weights (f32) — run as a ggml graph in Java
    {output_dir}/onnx/scorer.onnx          the same scorer as ONNX (PT-parity oracle / ORT fallback)
    {output_dir}/onnx_fp16/, onnx_quantized/   scorer ONNX variants (--variant)
    {output_dir}/reference.json            Python-pipeline scores for the Java parity test

The GGUF conversion shells out to ``convert_hf_to_gguf.py`` from a llama.cpp checkout
(``--llama-cpp-dir`` / ``LLAMA_CPP_DIR``, default ``../llama.cpp``); pin it to the tag llamaj.cpp
ships (see ``<llama.cpp.version>`` in llamaj.cpp's pom).

Usage:
    uv run scripts/export_gliclass_decoder_kv.py --model-path scx-admin/scx-router-v0.1 \\
        --output-dir models/scx-router-onnx --variant fp16 --variant quantized
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
import typer
from rich.console import Console

# Shared helpers (DeBERTa If-node patch, name sanitizing, fp16/int8 variants, tree print).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_onnx as shared  # noqa: E402

console = Console()
app = typer.Typer(add_completion=False)

SCORER_MODEL_FILES = ("scorer.onnx",)

# Label families the router was trained for — used for the parity reference file.
ROUTER_LABELS = [
    "coder", "DeepSeek-V3.1", "gemma-4-31B-it", "gpt-oss-120b",
    "Llama-4-Maverick-17B-128E-Instruct", "MAGPiE",
    "Meta-Llama-3.3-70B-Instruct", "Qwen3-32B",
]
TASK_TYPE_LABELS = [
    "analysis", "classification", "clustering", "code", "comparison", "critique",
    "decision support", "evaluation", "explanation", "fact checking", "forecasting",
    "generation", "information extraction", "information retrieval", "instruction following",
    "math & reasoning", "multi-turn", "planning", "problem solving", "programming", "qa",
    "reasoning", "recommendation", "rewriting", "summarization", "translation",
    "verification", "visualization",
]
REASONING_LABELS = ["reasoning", "nonreasoning"]
DIFFICULTY_LABELS = ["very easy", "easy", "medium", "hard", "extra hard"]
TOPIC_LABELS = ["travel", "finance", "health", "sports", "technology"]

REFERENCE_TEXTS = [
    "Write a Python function that merges two sorted linked lists.",
    "Compare these two vendor contracts and flag the riskier clauses.",
    "What's a good one-line commit message for this typo fix?",
    "Derive the closed-form solution for this second-order linear recurrence and prove convergence.",
    "The new mid-range EV just posted a 480-mile range on a single charge.",
    "I need help refactoring some Rust code. Specifically the borrow checker keeps rejecting "
    "this function. fn parse(&mut self, buf: &[u8]) -> Result<Token, Error> { ... }",
]


# ===========================================================================
# Scorer wrapper
# ===========================================================================


class DecoderKvScorerWrapper(nn.Module):
    """``DecoderKVScorer`` with the label-section layout supplied as indices.

    The Python scorer finds ``<<LABEL>>`` / last-``<<SEP>>`` positions with ``nonzero`` on the
    input ids. The label section (``label1<<LABEL>>…<<SEP>>``) is identical for every row of a
    batch sharing the same labels, so the runtime computes those positions once and passes them
    in, keeping the graph free of data-dependent gathers.

    Inputs: ``label_hidden [B, Ls, H]`` (backbone hidden states of the label section),
    ``label_positions [L]`` (int64 index of each ``<<LABEL>>`` within the section),
    ``sep_position [1]`` (index of the closing ``<<SEP>>``). Output: ``logits [B, L]``.
    """

    def __init__(self, scorer: nn.Module, normalize_features: bool, logit_scale=None) -> None:
        super().__init__()
        self.scorer_encoder = scorer.scorer_encoder
        self.text_projector = scorer.text_projector
        self.label_projector = scorer.label_projector
        self.mlp = scorer.mlp
        self.normalize_features = normalize_features
        self.logit_scale = logit_scale
        self.epsilon = scorer.epsilon

    def forward(
        self,
        label_hidden: torch.Tensor,
        label_positions: torch.Tensor,
        sep_position: torch.Tensor,
    ) -> torch.Tensor:
        b = label_hidden.shape[0]
        attention_mask = torch.ones(
            label_hidden.shape[0], label_hidden.shape[1], dtype=torch.long,
            device=label_hidden.device,
        )
        contextual = self.scorer_encoder(
            label_hidden, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state  # [B, Ls, H]
        text_repr = contextual[:, sep_position, :].reshape(b, -1)  # [B, H]
        label_repr = contextual[:, label_positions, :]  # [B, L, H]

        text_repr = self.text_projector(text_repr)
        label_repr = self.label_projector(label_repr)
        if self.normalize_features:
            text_repr = text_repr / (text_repr.norm(p=2, dim=-1, keepdim=True) + self.epsilon)
            label_repr = label_repr / (label_repr.norm(p=2, dim=-1, keepdim=True) + self.epsilon)
        num_labels = label_repr.shape[1]
        combined = torch.cat(
            (text_repr.unsqueeze(1).expand(b, num_labels, text_repr.shape[-1]), label_repr), dim=-1
        )
        logits = self.mlp(combined).squeeze(-1)  # [B, L]
        if self.normalize_features and self.logit_scale is not None:
            logits = logits * self.logit_scale
        return logits


# ===========================================================================
# Helpers
# ===========================================================================


def _label_section(labels: list[str], label_token: str, sep_token: str) -> str:
    """``<<SEP>>l1<<LABEL>>l2<<LABEL>>…<<SEP>>`` — the transient label part of the sequence."""
    from gliclass.data_processing import format_decoder_kv_labels

    return format_decoder_kv_labels(labels, label_token=label_token, sep_token=sep_token)


def _section_layout(section_ids: list[int], label_id: int, sep_id: int) -> tuple[list[int], int]:
    """Positions of ``<<LABEL>>`` tokens and of the closing ``<<SEP>>`` inside the scorer input.

    The scorer input starts right after the opening ``<<SEP>>`` (``_extract_label_section``), so
    indices are relative to ``section_ids[1:]``.
    """
    body = section_ids[1:]
    label_positions = [i for i, t in enumerate(body) if t == label_id]
    sep_positions = [i for i, t in enumerate(body) if t == sep_id]
    if not label_positions or not sep_positions:
        raise ValueError("label section is missing <<LABEL>> or the closing <<SEP>> token")
    return label_positions, sep_positions[-1]


def _export_scorer(model: nn.Module, out_dir: Path, opset: int) -> Path:
    cfg = model.config
    wrapper = DecoderKvScorerWrapper(
        model.model.scorer,
        bool(cfg.normalize_features),
        getattr(model.model, "logit_scale", None) if cfg.normalize_features else None,
    ).eval()
    shared._patch_deberta_for_onnx()
    hidden = cfg.hidden_size
    dummy = (
        torch.randn(1, 12, hidden),
        torch.tensor([2, 5, 8], dtype=torch.long),
        torch.tensor([11], dtype=torch.long),
    )
    path = out_dir / "scorer.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        opset_version=opset,
        dynamo=False,
        input_names=["label_hidden", "label_positions", "sep_position"],
        output_names=["logits"],
        dynamic_axes={
            "label_hidden": {0: "batch", 1: "label_len"},
            "label_positions": {0: "num_labels"},
            "logits": {0: "batch", 1: "num_labels"},
        },
    )
    if_nodes = shared._count_if_nodes(path)
    note = " [green]0 If nodes[/green]" if if_nodes == 0 else f" [red]{if_nodes} If nodes[/red]"
    console.print(f"  [green]✓[/green] scorer.onnx ({path.stat().st_size / 1e6:.1f} MB,{note})")
    return path


def _export_backbone_gguf(
    model: nn.Module, tokenizer, out_dir: Path, llama_cpp_dir: Path, outtypes: list[str]
) -> list[Path]:
    """Save the Qwen3 decoder as a plain HF causal LM and convert it with llama.cpp's script."""
    return export_qwen3_backbone_gguf(
        model.model.decoder_model, model.config.encoder_config, tokenizer, out_dir, llama_cpp_dir, outtypes
    )


def export_qwen3_backbone_gguf(
    decoder: nn.Module, enc_cfg, tokenizer, out_dir: Path, llama_cpp_dir: Path, outtypes: list[str]
) -> list[Path]:
    """Convert a bare ``Qwen3Model`` (plus its config and tokenizer) to GGUF files under ``out_dir/gguf``."""
    from transformers import Qwen3ForCausalLM

    convert = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise typer.BadParameter(
            f"{convert} not found — point --llama-cpp-dir / LLAMA_CPP_DIR at a llama.cpp checkout"
        )
    enc_cfg.tie_word_embeddings = True
    enc_cfg.use_cache = True
    with tempfile.TemporaryDirectory(prefix="gliclass-qwen3-") as tmp:
        tmp_dir = Path(tmp)
        causal = Qwen3ForCausalLM(enc_cfg)
        missing, unexpected = causal.model.load_state_dict(decoder.state_dict(), strict=False)
        if unexpected or [k for k in missing if "rotary" not in k]:
            raise RuntimeError(f"decoder state mismatch: missing={missing} unexpected={unexpected}")
        causal.tie_weights()
        causal.save_pretrained(str(tmp_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(tmp_dir))
        outputs = []
        gguf_dir = out_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        for outtype in outtypes:
            target = gguf_dir / f"backbone-{outtype}.gguf"
            console.print(f"  converting → {target.name} ...")
            subprocess.run(
                [
                    sys.executable, str(convert), str(tmp_dir),
                    "--outfile", str(target), "--outtype", outtype,
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
            console.print(f"  [green]✓[/green] {target.name} ({target.stat().st_size / 1e9:.2f} GB)")
            outputs.append(target)
    return outputs


def _export_scorer_gguf(model: nn.Module, out_dir: Path) -> Path:
    """Write the scorer weights + hyper-parameters as a small f32 GGUF for the Java ggml graph.

    Tensor names are the ``DecoderKVScorer`` state-dict keys; numpy row-major ``[out, in]``
    matrices become ggml ``ne=(in, out)`` so ``ggml_mul_mat(W, x)`` is ``x @ W.T``.
    """
    from gguf import GGUFWriter

    scorer = model.model.scorer
    enc = scorer.scorer_encoder
    attn = enc.layer[0].attention.self
    cfg = model.config
    gguf_dir = out_dir / "gguf"
    gguf_dir.mkdir(parents=True, exist_ok=True)
    path = gguf_dir / "scorer.gguf"
    w = GGUFWriter(str(path), arch="gliclass-decoder-kv-scorer")
    w.add_uint32("scorer.hidden_size", cfg.hidden_size)
    w.add_uint32("scorer.num_layers", len(enc.layer))
    w.add_uint32("scorer.num_heads", attn.num_attention_heads)
    w.add_uint32("scorer.head_size", attn.attention_head_size)
    w.add_uint32("scorer.intermediate_size", enc.layer[0].intermediate.dense.out_features)
    w.add_uint32("scorer.pos_ebd_size", attn.pos_ebd_size)
    w.add_float32("scorer.layer_norm_eps", float(enc.layer[0].attention.output.LayerNorm.eps))
    w.add_bool("scorer.normalize_features", bool(cfg.normalize_features))
    if cfg.normalize_features:
        w.add_float32("scorer.logit_scale", float(model.model.logit_scale.detach().float()))
    assert attn.pos_att_type == ["p2c", "c2p"] and not attn.share_att_key and attn.position_buckets <= 0, (
        "Java ggml scorer implements pos_att_type=[p2c,c2p], share_att_key=False, no position buckets"
    )
    assert enc.conv is None and "layer_norm" not in enc.norm_rel_ebd
    for name, tensor in scorer.state_dict().items():
        w.add_tensor(name, tensor.detach().float().cpu().numpy())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    console.print(f"  [green]✓[/green] scorer.gguf ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def _check_gguf_vocab(path: Path, expected_vocab: int, specials: dict[str, int]) -> None:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    tokens = reader.fields["tokenizer.ggml.tokens"]
    n = len(tokens.data)
    if n != expected_vocab:
        raise RuntimeError(f"{path.name}: GGUF vocab has {n} tokens, expected {expected_vocab}")
    for token, idx in specials.items():
        piece = bytes(tokens.parts[tokens.data[idx]]).decode("utf-8")
        if piece != token:
            raise RuntimeError(f"{path.name}: token {idx} is {piece!r}, expected {token!r}")
    console.print(f"  [green]✓[/green] {path.name}: vocab {n}, special tokens in place")


def _write_config(model: nn.Module, tokenizer, out_dir: Path, default_gguf: str) -> None:
    cfg = model.config
    enc = cfg.encoder_config
    config = {
        "architecture": "gliclass-decoder-kv",
        "hidden_size": cfg.hidden_size,
        "architecture_config": {
            "backbone_model_name": cfg.encoder_model_name,
            "backbone_gguf": default_gguf,
            "backbone_hidden_size": enc.hidden_size,
            "n_ctx_train": int(enc.max_position_embeddings),
            "class_token_index": cfg.class_token_index,
            "sep_token_index": cfg.sep_token_index,
            "label_token": "<<LABEL>>",
            "sep_token": "<<SEP>>",
            "problem_type": cfg.problem_type,
            "max_num_classes": cfg.max_num_classes,
            "normalize_features": bool(cfg.normalize_features),
        },
    }
    (out_dir / "gliner4j_config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(str(out_dir))
    console.print("  [green]✓[/green] gliner4j_config.json + tokenizer")


def _encode_sequence(tokenizer, text: str, labels: list[str]) -> tuple[torch.Tensor, list[int]]:
    """Tokenize text and label section separately and concatenate — the classic pipeline's
    ``prepare_inputs`` layout, and what the Java runtime does."""
    text_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    section_ids = tokenizer(_label_section(labels, "<<LABEL>>", "<<SEP>>"), add_special_tokens=False)["input_ids"]
    return torch.tensor([text_ids + section_ids], dtype=torch.long), section_ids


def _reference_scores(
    model: nn.Module, tokenizer, text: str, labels: list[str], single_label: bool
) -> list[float]:
    """Full PyTorch model on one text, exactly like the classic pipeline (no padding)."""
    input_ids, _ = _encode_sequence(tokenizer, text, labels)
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids)).logits[0]
    logits = logits[: len(labels)]
    scores = torch.softmax(logits, dim=-1) if single_label else torch.sigmoid(logits)
    return [float(s) for s in scores]


def _write_reference(model: nn.Module, tokenizer, out_dir: Path) -> None:
    families = [
        ("router", ROUTER_LABELS, False),
        ("task_type", TASK_TYPE_LABELS, True),
        ("reasoning", REASONING_LABELS, True),
        ("difficulty", DIFFICULTY_LABELS, True),
        ("topic", TOPIC_LABELS, False),
    ]
    entries = []
    for text in REFERENCE_TEXTS:
        for family, labels, single in families:
            scores = _reference_scores(model, tokenizer, text, labels, single)
            entries.append({
                "text": text,
                "family": family,
                "labels": labels,
                "classification_type": "single-label" if single else "multi-label",
                "scores": scores,
            })
    (out_dir / "reference.json").write_text(json.dumps(entries, indent=1) + "\n")
    console.print(f"  [green]✓[/green] reference.json ({len(entries)} entries)")


def _verify_scorer(model: nn.Module, tokenizer, out_dir: Path) -> bool:
    """Full PyTorch logits vs (PyTorch backbone → ONNX scorer) on unpadded single texts."""
    cfg = model.config
    sess = ort.InferenceSession(str(out_dir / "scorer.onnx"))
    ok = True
    for text in REFERENCE_TEXTS[:3]:
        labels = ROUTER_LABELS
        input_ids, section_ids = _encode_sequence(tokenizer, text, labels)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            ref = model(input_ids=input_ids, attention_mask=attention_mask).logits[0]
            hidden = model.model.decoder_model(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state
        start = input_ids.shape[1] - len(section_ids) + 1  # after the opening <<SEP>>
        label_hidden = hidden[:, start:, :].numpy()
        label_positions, sep_position = _section_layout(
            section_ids, cfg.class_token_index, cfg.sep_token_index
        )
        onnx_logits = sess.run(
            None,
            {
                "label_hidden": label_hidden,
                "label_positions": np.asarray(label_positions, dtype=np.int64),
                "sep_position": np.asarray([sep_position], dtype=np.int64),
            },
        )[0][0]
        max_diff = float(np.max(np.abs(ref[: len(labels)].numpy() - onnx_logits)))
        row_ok = max_diff < 1e-3
        ok &= row_ok
        status = "[green]PASS[/green]" if row_ok else "[red]FAIL[/red]"
        console.print(f"  {status} scorer logits (max_diff={max_diff:.6f}) — {text[:50]!r}")
    return bool(ok)


# ===========================================================================
# CLI
# ===========================================================================


@app.command()
def export(
    model_path: Annotated[
        str, typer.Option("--model-path", help="GLiClass decoder-kv model dir or HF repo ID")
    ],
    output_dir: Annotated[str, typer.Option("--output-dir", help="Output bundle directory")],
    llama_cpp_dir: Annotated[
        str | None,
        typer.Option("--llama-cpp-dir", help="llama.cpp checkout with convert_hf_to_gguf.py "
                     "(default: $LLAMA_CPP_DIR or ../llama.cpp)"),
    ] = None,
    gguf_types: Annotated[
        list[str] | None,
        typer.Option("--gguf-type", help="GGUF outtypes to produce (default: f16, q8_0)"),
    ] = None,
    default_gguf: Annotated[
        str, typer.Option("--default-gguf", help="GGUF the runtime loads by default")
    ] = "q8_0",
    opset: Annotated[int, typer.Option("--opset", help="ONNX opset version")] = 17,
    verify: Annotated[
        bool, typer.Option("--verify/--no-verify", help="Run PT-vs-ONNX scorer verification")
    ] = True,
    skip_backbone: Annotated[
        bool, typer.Option("--skip-backbone", help="Do not (re)convert the backbone GGUFs")
    ] = False,
    variants: Annotated[
        list[shared.Variant] | None,
        typer.Option("--variant", help="Scorer variants to generate (fp16, quantized)"),
    ] = None,
) -> None:
    """Export a GLiClass decoder-kv checkpoint to a llama.cpp + ONNX gliner4j bundle."""
    from gliclass import GLiClassModel
    from transformers import AutoTokenizer

    out = Path(output_dir)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    llama_dir = Path(
        llama_cpp_dir or os.environ.get("LLAMA_CPP_DIR") or (Path(__file__).resolve().parents[2] / "llama.cpp")
    )
    gguf_types = gguf_types or ["f16", "q8_0"]
    if default_gguf not in gguf_types:
        raise typer.BadParameter(f"--default-gguf {default_gguf} is not among --gguf-type {gguf_types}")

    console.print(f"\n[bold]Loading model from {model_path}...[/bold]")
    model = GLiClassModel.from_pretrained(model_path).eval().float()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cfg = model.config
    if cfg.architecture_type != "decoder-kv":
        raise typer.BadParameter(
            f"{model_path} has architecture_type={cfg.architecture_type!r}; this command is for "
            "decoder-kv checkpoints (use `export_onnx.py gliclass` for uni-encoder ones)"
        )
    console.print(
        f"  backbone={cfg.encoder_model_name}, hidden={cfg.hidden_size}, vocab={cfg.vocab_size}, "
        f"class_token={cfg.class_token_index}, sep_token={cfg.sep_token_index}, "
        f"problem_type={cfg.problem_type}"
    )

    console.print(f"\n[bold]Exporting scorer to {onnx_dir} (opset={opset})...[/bold]")
    with torch.no_grad():
        _export_scorer(model, onnx_dir, opset)
    renamed = shared._sanitize_onnx_names(onnx_dir / "scorer.onnx")
    console.print(f"  [green]✓[/green] sanitized scorer.onnx ({renamed} names rewritten)")
    _export_scorer_gguf(model, out)

    if skip_backbone:
        console.print("\n[bold]Skipping backbone GGUF conversion (--skip-backbone)[/bold]")
    else:
        console.print(f"\n[bold]Converting backbone to GGUF via {llama_dir}...[/bold]")
        ggufs = _export_backbone_gguf(model, tokenizer, out, llama_dir, gguf_types)
        specials = {"<<LABEL>>": cfg.class_token_index, "<<SEP>>": cfg.sep_token_index}
        for path in ggufs:
            _check_gguf_vocab(path, cfg.vocab_size, specials)

    console.print("\n[bold]Writing config, tokenizer and reference scores...[/bold]")
    _write_config(model, tokenizer, out, f"backbone-{default_gguf}.gguf")
    _write_reference(model, tokenizer, out)

    if verify:
        console.print("\n[bold]Verifying scorer.onnx against PyTorch...[/bold]")
        if _verify_scorer(model, tokenizer, onnx_dir):
            console.print("\n[bold green]Verification passed![/bold green]")
        else:
            console.print("\n[bold red]Verification failed![/bold red]")
            raise typer.Exit(code=1)

    from onnxruntime.quantization import QuantType

    shared._generate_variants(variants, out, onnx_dir, SCORER_MODEL_FILES, QuantType.QInt8)

    console.print(f"\n[bold]Output structure in {out}:[/bold]")
    shared._print_tree(out)


if __name__ == "__main__":
    app()
