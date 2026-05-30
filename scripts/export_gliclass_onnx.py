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
#     "transformers>=4.48",
#     "numpy>=1.26",
#     "typer>=0.15",
#     "rich>=13",
#     "gliclass",
# ]
# ///
"""GLiClass → ONNX export for gliner4j (spike: gliclass-modern-base-v3.0).

GLiClass is the GLiNER framework adapted for zero-shot sequence classification. For the
`uni-encoder` / `simple` scorer family (e.g. gliclass-modern-base-v3.0) the forward pass is:

    prompt = "<<LABEL>>l1<<LABEL>>l2...<<LABEL>>lN<<SEP>>" + text       (prompt_first=True)
    H      = encoder(input_ids, attention_mask)                        # ModernBERT, [1, seq, hid]
    class_reps = H[:, <<LABEL>> positions, :]                          # one per label
    text_rep   = H[:, 0, :]                                            # CLS (extract_text_features=False, pooling=first)
    logits     = scorer(text_projector(text_rep), classes_projector(class_reps))   # ScorerDot: einsum BD,BCD->BC
    scores     = sigmoid(logits)                                       # multi-label, in Java

This script exports two graphs, mirroring the GLiNER2 exporter's split:
  - encoder.onnx:    ModernBERT (input_ids, attention_mask) → last_hidden_state
  - score_head.onnx: (text_emb[1,hid], class_embs[1,C,hid]) → logits[1,C]  (both projectors + dot product)

Token gathering ([CLS] for text, <<LABEL>> positions for classes) is done by the Java consumer,
exactly as GLiNER4jClassifier gathers [L] markers — so the graphs stay text-length agnostic.

Usage:
    uv run scripts/export_gliclass_onnx.py --model-path knowledgator/gliclass-modern-base-v3.0 \
        --output-dir models/gliclass-modern-base-onnx
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
import typer
from rich.console import Console

console = Console()

# The two graphs that make up a GLiClass bundle (base + every variant).
ONNX_MODEL_FILES = ("encoder.onnx", "score_head.onnx")


class Variant(str, Enum):
    """ONNX model variants generated from the base FP32 export."""

    fp16 = "fp16"
    quantized = "quantized"


# ===========================================================================
# ONNX wrapper modules
# ===========================================================================


class EncoderWrapper(nn.Module):
    """ModernBERT encoder → last_hidden_state, for clean ONNX export."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state


class ScoreHeadWrapper(nn.Module):
    """GLiClass scoring head: both feature projectors + the scorer.

    Inputs are the *raw* encoder hidden states already gathered by the caller:
      - text_emb:   (1, hidden) — the CLS/text-pooled vector
      - class_embs: (1, num_classes, hidden) — one row per label marker

    Output: logits (1, num_classes). Sigmoid is applied downstream (Java), matching
    GLiNER4jClassifier. logit_scale is intentionally NOT applied: the GLiClass model only
    scales logits when normalize_features=True, which is False for this checkpoint.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.text_projector = model.text_projector
        self.classes_projector = model.classes_projector
        self.scorer = model.scorer
        self.normalize_features = bool(model.config.normalize_features)
        self.logit_scale = model.logit_scale if self.normalize_features else None
        self.epsilon = 1e-8

    def forward(
        self, text_emb: torch.Tensor, class_embs: torch.Tensor
    ) -> torch.Tensor:
        pooled = self.text_projector(text_emb)  # (1, hid)
        classes = self.classes_projector(class_embs)  # (1, C, hid)
        if self.normalize_features:
            pooled = pooled / (pooled.norm(p=2, dim=-1, keepdim=True) + self.epsilon)
            classes = classes / (
                classes.norm(p=2, dim=-1, keepdim=True) + self.epsilon
            )
        logits = self.scorer(pooled, classes)  # (1, C)
        if self.normalize_features and self.logit_scale is not None:
            logits = logits * self.logit_scale.to(classes.device)
        return logits


# ===========================================================================
# Encoder patching for ONNX export
# ===========================================================================


def _prepare_modernbert_for_onnx(encoder: nn.Module) -> None:
    """Force ModernBERT onto the eager attention path and disable torch.compile guards.

    ModernBERT's default fast paths (flash-attn unpadding, ``reference_compile``) are not
    traceable by ``torch.onnx.export``. Both are read from the encoder config at forward time
    in recent transformers, so flipping them on the live config is sufficient — no re-init.
    """
    cfg = encoder.config
    if hasattr(cfg, "reference_compile"):
        cfg.reference_compile = False
    cfg._attn_implementation = "eager"


def _patch_deberta_for_onnx() -> int:
    """De-script DeBERTa-v2/v3's relative-position control flow before ONNX export.

    DeBERTa's ``build_rpos`` is ``@torch.jit.script``, so its ``if`` compiles to a real ONNX
    ``If`` node per layer whose then/else branches have different rank (3 vs 4). Such rank-mismatched
    ``If`` nodes break FP16 conversion (the fp16 converter can't reconcile the branch types and ORT
    rejects the model). Our encoder always runs square self-attention, so the ``else`` branch is
    always taken — replacing the scripted helper with a plain Python equivalent bakes in that single
    branch with no ``If`` nodes (identical numerics, kept honest by the PT-vs-ONNX verify). No-op for
    non-DeBERTa backbones (ModernBERT, mT5). Returns the number of helpers patched.
    """
    try:
        import transformers.models.deberta_v2.modeling_deberta_v2 as dv2
    except ImportError:
        return 0

    patched = 0
    for name in dir(dv2):
        obj = getattr(dv2, name)
        original = getattr(obj, "__original_fn", None)
        if original is not None and getattr(obj, "__script_if_tracing_wrapper", False):
            setattr(dv2, name, original)
            patched += 1

    if hasattr(dv2, "build_rpos"):

        def build_rpos(query_layer, key_layer, relative_pos, position_buckets, max_relative_positions):
            if key_layer.size(-2) != query_layer.size(-2):
                return dv2.build_relative_position(
                    key_layer,
                    key_layer,
                    bucket_size=position_buckets,
                    max_position=max_relative_positions,
                )
            return relative_pos

        dv2.build_rpos = build_rpos
        patched += 1

    return patched


def _count_if_nodes(path: Path) -> int:
    model = onnx.load(str(path))
    return sum(1 for n in model.graph.node if n.op_type == "If")


def _sanitize_onnx_names(path: Path) -> int:
    """Replace ``/`` in every tensor/node name so the model runs on the OpenVINO EP.

    PyTorch names intermediate tensors like ``/encoder/Reshape_output_0``. ORT's OpenVINO EP
    truncates such names at the first ``/`` for subgraph-boundary outputs and then fails the lookup
    (``Output names mismatch between OpenVINO and ONNX``). Replacing ``/`` with ``_`` everywhere
    (names are pure identifiers; declared graph I/O has no ``/``) removes the trigger with no change
    to the model's interface or numerics. Returns the number of names rewritten.
    """
    model = onnx.load(str(path))
    count = 0

    def fix(name: str) -> str:
        nonlocal count
        if name and "/" in name:
            count += 1
            return name.replace("/", "_")
        return name

    def sanitize(graph) -> None:
        for init in graph.initializer:
            init.name = fix(init.name)
        for vi in list(graph.value_info) + list(graph.input) + list(graph.output):
            vi.name = fix(vi.name)
        for node in graph.node:
            node.name = fix(node.name)
            node.input[:] = [fix(x) for x in node.input]
            node.output[:] = [fix(x) for x in node.output]
            for attr in node.attribute:
                if attr.type == onnx.AttributeProto.GRAPH:
                    sanitize(attr.g)
                for sg in attr.graphs:
                    sanitize(sg)

    sanitize(model.graph)
    if count:
        onnx.save(model, str(path))
    return count


# ===========================================================================
# Export
# ===========================================================================


def _export_encoder(model: nn.Module, out_dir: Path, opset: int) -> Path:
    encoder = model.encoder_model
    _prepare_modernbert_for_onnx(encoder)
    # DeBERTa backbones emit per-layer dynamic-rank `If` nodes from their scripted relative-position
    # helper; these break FP16 conversion and the OpenVINO CPU plugin. Bake in the single branch.
    patched = _patch_deberta_for_onnx()
    if patched:
        console.print(f"  patched {patched} DeBERTa relative-position helper(s)")
    wrapper = EncoderWrapper(encoder).eval()

    dummy_ids = torch.ones(1, 32, dtype=torch.long)
    dummy_mask = torch.ones(1, 32, dtype=torch.long)

    path = out_dir / "encoder.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_ids, dummy_mask),
        str(path),
        opset_version=opset,
        dynamo=False,
        input_names=["input_ids", "attention_mask"],
        output_names=["last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "last_hidden_state": {0: "batch", 1: "seq_len"},
        },
    )
    if_nodes = _count_if_nodes(path)
    if_note = (
        ", [green]0 If nodes[/green]"
        if if_nodes == 0
        else f", [red]{if_nodes} If nodes remain[/red]"
    )
    console.print(
        f"  [green]✓[/green] encoder.onnx ({path.stat().st_size / 1e6:.1f} MB{if_note})"
    )
    return path


def _export_score_head(
    model: nn.Module, out_dir: Path, opset: int, enc_hidden: int
) -> Path:
    wrapper = ScoreHeadWrapper(model).eval()

    num_classes = 3
    dummy_text = torch.randn(1, enc_hidden)
    dummy_classes = torch.randn(1, num_classes, enc_hidden)

    path = out_dir / "score_head.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_text, dummy_classes),
        str(path),
        opset_version=opset,
        dynamo=False,
        input_names=["text_emb", "class_embs"],
        output_names=["logits"],
        dynamic_axes={
            "class_embs": {1: "num_classes"},
            "logits": {1: "num_classes"},
        },
    )
    console.print(
        f"  [green]✓[/green] score_head.onnx ({path.stat().st_size / 1e6:.1f} MB)"
    )
    return path


# ===========================================================================
# Config / tokenizer
# ===========================================================================


def _write_config(model: nn.Module, tokenizer, out_dir: Path, enc_hidden: int) -> None:
    cfg = model.config
    config = {
        "architecture": "gliclass",
        "hidden_size": enc_hidden,
        "architecture_config": {
            "encoder_model_name": cfg.encoder_model_name,
            "class_token_index": cfg.class_token_index,
            "text_token_index": cfg.text_token_index,
            "label_token": "<<LABEL>>",
            "sep_token": "<<SEP>>",
            "pooling_strategy": cfg.pooling_strategy,
            "scorer_type": cfg.scorer_type,
            "prompt_first": bool(cfg.prompt_first),
            "embed_class_token": bool(cfg.embed_class_token),
            "extract_text_features": bool(cfg.extract_text_features),
            "normalize_features": bool(cfg.normalize_features),
            "max_num_classes": cfg.max_num_classes,
        },
    }
    (out_dir / "gliner4j_config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(str(out_dir))
    console.print("  [green]✓[/green] gliner4j_config.json + tokenizer")


# ===========================================================================
# Verification
# ===========================================================================


def _verify(
    model: nn.Module, tokenizer, out_dir: Path, class_token_index: int
) -> bool:
    """Compare full-model PyTorch logits against ONNX encoder + Java-style gather + score_head."""
    text = "I really loved the cinematography and the soundtrack of this film."
    labels = ["positive", "negative", "neutral"]

    prompt = "".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>" + text
    enc = tokenizer([prompt], return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    with torch.no_grad():
        ref = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_num_classes=len(labels),
        ).logits.numpy()

    enc_sess = ort.InferenceSession(str(out_dir / "encoder.onnx"))
    head_sess = ort.InferenceSession(str(out_dir / "score_head.onnx"))

    hidden = enc_sess.run(
        None,
        {
            "input_ids": input_ids.numpy(),
            "attention_mask": attention_mask.numpy(),
        },
    )[0]  # (1, seq, hid)

    ids = input_ids[0].numpy()
    class_positions = np.where(ids == class_token_index)[0]
    text_emb = hidden[:, 0, :]  # CLS
    class_embs = hidden[:, class_positions, :]  # (1, C, hid)

    onnx_logits = head_sess.run(
        None, {"text_emb": text_emb, "class_embs": class_embs}
    )[0]

    max_diff = float(np.max(np.abs(ref - onnx_logits)))
    ok = bool(np.allclose(ref, onnx_logits, atol=1e-3))
    status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
    console.print(f"  {status} logits (max_diff={max_diff:.6f})")
    console.print(f"    PT  : {np.round(ref[0], 4)}")
    console.print(f"    ONNX: {np.round(onnx_logits[0], 4)}")
    return ok


# ===========================================================================
# Variant conversion (fp16 / INT8)
# ===========================================================================


def _align_fp16_cast_nodes(model: onnx.ModelProto) -> int:
    """Make ``Cast`` ``to`` attributes agree with the fp16 types the converter assigned.

    ``onnxconverter_common.float16`` rewrites intermediate tensors (and their value_info) to
    float16 but leaves the ``to`` attribute of pre-existing ``Cast`` nodes untouched, so a Cast
    can emit float32 into a float16 graph and ORT refuses to load it. For every Cast whose
    declared output type is float16 but whose ``to`` is still FLOAT, set ``to`` to FLOAT16.
    Returns the number of Cast nodes realigned.
    """
    declared: dict[str, int] = {}
    for vi in (
        list(model.graph.value_info)
        + list(model.graph.output)
        + list(model.graph.input)
    ):
        declared[vi.name] = vi.type.tensor_type.elem_type

    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Cast" or not node.output:
            continue
        to_attr = next((a for a in node.attribute if a.name == "to"), None)
        if to_attr is None:
            continue
        if (
            to_attr.i == onnx.TensorProto.FLOAT
            and declared.get(node.output[0]) == onnx.TensorProto.FLOAT16
        ):
            to_attr.i = onnx.TensorProto.FLOAT16
            fixed += 1
    return fixed


def _convert_to_fp16(base_dir: Path, out_dir: Path) -> None:
    """Convert the base FP32 graphs to FP16, validating each loads (else ship FP32)."""
    import shutil

    from onnxconverter_common import float16

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_MODEL_FILES:
        model = onnx.load(str(base_dir / name))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        realigned = _align_fp16_cast_nodes(model_fp16)
        onnx.save(model_fp16, str(out_dir / name))
        _sanitize_onnx_names(out_dir / name)  # float16 pass can mint new '/'-named tensors
        try:
            ort.InferenceSession(str(out_dir / name))
            note = f" (realigned {realigned} casts)" if realigned else ""
            marker = "[green]✓[/green]"
        except Exception as e:  # noqa: BLE001
            shutil.copy(base_dir / name, out_dir / name)
            note = " (fp16 invalid, copied FP32 instead)"
            marker = "[yellow]→[/yellow]"
            console.print(f"  [yellow]![/yellow] {name}: {str(e).splitlines()[0]}")
        size_mb = (out_dir / name).stat().st_size / 1e6
        console.print(f"  {marker} {name} ({size_mb:.1f} MB){note}")


def _convert_to_quantized(base_dir: Path, out_dir: Path) -> None:
    """Convert the base FP32 graphs to INT8 dynamic quantization, validating each loads."""
    import shutil

    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_MODEL_FILES:
        # QInt8 (signed, symmetric) per-channel weights — the accuracy-preserving config for
        # transformer weights. QInt8 (unsigned/asymmetric) is a known de-calibration pitfall on
        # transformers; per_channel + reduce_range alone don't compensate for it.
        quantize_dynamic(
            model_input=base_dir / name,
            model_output=out_dir / name,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
        _sanitize_onnx_names(out_dir / name)  # quantize_dynamic can mint new '/'-named tensors
        try:
            ort.InferenceSession(str(out_dir / name))
            note = ""
            marker = "[green]✓[/green]"
        except Exception as e:  # noqa: BLE001
            (out_dir / name).unlink(missing_ok=True)
            shutil.copy(base_dir / name, out_dir / name)
            note = " (quantization invalid, copied FP32 instead)"
            marker = "[yellow]→[/yellow]"
            console.print(f"  [yellow]![/yellow] {name}: {str(e).splitlines()[0]}")
        size_mb = (out_dir / name).stat().st_size / 1e6
        console.print(f"  {marker} {name} ({size_mb:.1f} MB){note}")


# ===========================================================================
# CLI
# ===========================================================================

app = typer.Typer(add_completion=False)


@app.command()
def export(
    model_path: Annotated[
        str, typer.Option("--model-path", help="GLiClass model dir or HF repo ID")
    ],
    output_dir: Annotated[
        str, typer.Option("--output-dir", help="Output directory for ONNX models")
    ],
    opset: Annotated[int, typer.Option("--opset", help="ONNX opset version")] = 17,
    verify: Annotated[
        bool, typer.Option("--verify/--no-verify", help="Run PT-vs-ONNX verification")
    ] = True,
    variants: Annotated[
        list[Variant] | None,
        typer.Option("--variant", help="Additional variants to generate (fp16, quantized)"),
    ] = None,
) -> None:
    """Export a GLiClass uni-encoder model to ONNX (encoder + score_head) for gliner4j.

    Base FP32 graphs go to {output_dir}/onnx/. Each requested variant (fp16, quantized) is
    generated from the base into {output_dir}/onnx_{variant}/.
    """
    from gliclass import GLiClassModel
    from transformers import AutoTokenizer

    out = Path(output_dir)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Loading GLiClass model from {model_path}...[/bold]")
    model = GLiClassModel.from_pretrained(model_path)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if model.config.architecture_type != "uni-encoder":
        raise typer.BadParameter(
            f"This script handles uni-encoder GLiClass only; got "
            f"architecture_type={model.config.architecture_type!r}"
        )
    if model.config.scorer_type not in ("simple",):
        console.print(
            f"  [yellow]![/yellow] scorer_type={model.config.scorer_type!r} — "
            f"exporting via the model's own scorer module (verified below)."
        )

    enc_hidden = model.config.encoder_config.hidden_size
    console.print(
        f"  encoder={model.config.encoder_model_name}, enc_hidden={enc_hidden}, "
        f"scorer={model.config.scorer_type}, pooling={model.config.pooling_strategy}, "
        f"prompt_first={model.config.prompt_first}"
    )

    # GLiClassModel is a thin dispatcher; the real uni-encoder module (encoder_model,
    # text_projector, classes_projector, scorer, logit_scale) lives under `.model`.
    uni = model.model

    console.print(f"\n[bold]Exporting ONNX models to {onnx_dir} (opset={opset})...[/bold]")
    with torch.no_grad():
        _export_encoder(uni, onnx_dir, opset)
        _export_score_head(uni, onnx_dir, opset, enc_hidden)

    # Strip '/' from tensor names for the OpenVINO EP — before variants so they inherit clean names.
    for fname in ONNX_MODEL_FILES:
        renamed = _sanitize_onnx_names(onnx_dir / fname)
        console.print(f"  [green]✓[/green] sanitized {fname} ({renamed} names)")

    console.print("\n[bold]Writing config and tokenizer...[/bold]")
    _write_config(model, tokenizer, out, enc_hidden)

    if verify:
        console.print("\n[bold]Verifying (PyTorch vs ONNX)...[/bold]")
        ok = _verify(model, tokenizer, onnx_dir, model.config.class_token_index)
        if not ok:
            console.print("\n[bold red]Verification FAILED[/bold red]")
            raise typer.Exit(code=1)
        console.print("\n[bold green]Verification passed![/bold green]")

    if variants:
        for variant in variants:
            variant_dir = out / f"onnx_{variant.value}"
            console.print(
                f"\n[bold]Generating {variant.value} variant → {variant_dir}...[/bold]"
            )
            if variant == Variant.fp16:
                _convert_to_fp16(onnx_dir, variant_dir)
            elif variant == Variant.quantized:
                _convert_to_quantized(onnx_dir, variant_dir)


if __name__ == "__main__":
    app()
