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
#     "transformers>=4.40",
#     "numpy>=1.26",
#     "typer>=0.15",
#     "rich>=13",
#     "gliner",
# ]
# ///
"""Original-GLiNER (uni-encoder) → ONNX export for gliner4j (spike: gliner-multitask-large-v0.5).

Unlike the GLiNER2/GLiClass exporters (which split encoder + heads and need custom wrappers), the
`gliner` library ships its own monolithic ONNX export. For a uni-encoder token-level model the
graph is:

    inputs : input_ids, attention_mask, words_mask, text_lengths   (token_level: no span_idx)
    output : logits

This script just drives the library's `export_to_onnx`, verifies the exported ONNX reproduces the
PyTorch model's entities, lays the bundle out the gliner4j way (onnx/ + gliner4j_config.json at
root), and prints the exact ONNX I/O so the Java side can replicate the processor + decoder.

Usage:
    uv run scripts/export_gliner_uni_onnx.py --model-path knowledgator/gliner-multitask-large-v0.5 \
        --output-dir models/gliner-multitask-large-onnx
"""

from __future__ import annotations

import json
import shutil
from enum import Enum
from pathlib import Path
from typing import Annotated

import numpy as np
import onnx
import onnxruntime as ort
import torch
import typer
from rich.console import Console

console = Console()

# The gliner uni/bi span model is a single monolithic graph.
ONNX_MODEL_FILES = ("model.onnx",)


class Variant(str, Enum):
    fp16 = "fp16"
    quantized = "quantized"


def _patch_deberta_for_onnx() -> int:
    """De-script DeBERTa-v2/v3's relative-position control flow before ONNX export.

    DeBERTa's ``build_rpos`` is scripted, so its ``if`` becomes a per-layer dynamic-rank ONNX ``If``
    node. Those break FP16 conversion and the OpenVINO CPU plugin (which can't compile dynamic-rank
    ``If``). Our encoder always runs square self-attention, so the ``else`` branch is always taken —
    replacing the scripted helper bakes in that single branch (no ``If`` nodes, identical numerics).
    Global monkeypatch on the transformers module, so it takes effect inside gliner's own
    ``export_to_onnx``. No-op for non-DeBERTa backbones. Returns the number of helpers patched.
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
    """Replace ``/`` in tensor/node names so the model runs on the OpenVINO EP (which truncates
    ``/``-prefixed subgraph-boundary output names and then fails the lookup). Names are pure
    identifiers and the declared I/O has no ``/``, so interface + numerics are unchanged.
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


def _align_fp16_cast_nodes(model: onnx.ModelProto) -> int:
    """Make pre-existing ``Cast`` ``to`` attributes agree with the fp16 types the converter assigned,
    so a float32-emitting Cast doesn't break loading of an otherwise-fp16 graph."""
    declared: dict[str, int] = {}
    for vi in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input):
        declared[vi.name] = vi.type.tensor_type.elem_type
    fixed = 0
    for node in model.graph.node:
        if node.op_type != "Cast" or not node.output:
            continue
        to_attr = next((a for a in node.attribute if a.name == "to"), None)
        if to_attr is None:
            continue
        if to_attr.i == onnx.TensorProto.FLOAT and declared.get(node.output[0]) == onnx.TensorProto.FLOAT16:
            to_attr.i = onnx.TensorProto.FLOAT16
            fixed += 1
    return fixed


def _convert_to_fp16(base_dir: Path, out_dir: Path) -> None:
    from onnxconverter_common import float16

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_MODEL_FILES:
        model = onnx.load(str(base_dir / name))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        realigned = _align_fp16_cast_nodes(model_fp16)
        onnx.save(model_fp16, str(out_dir / name))
        _sanitize_onnx_names(out_dir / name)
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
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_MODEL_FILES:
        # QInt8 (signed) per-channel — accuracy-preserving for transformer weights (QUInt8 is the
        # de-calibration pitfall; gliner's own quantize=True uses QUInt8, which is why we do our own).
        quantize_dynamic(
            model_input=base_dir / name,
            model_output=out_dir / name,
            weight_type=QuantType.QInt8,
            per_channel=True,
            reduce_range=True,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
        _sanitize_onnx_names(out_dir / name)
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


def _write_gliner4j_config(gliner_config: dict, out_dir: Path) -> None:
    """Map the library's gliner_config.json into a gliner4j_config.json with our discriminator.

    A non-null ``labels_encoder`` ⇒ bi-encoder (separate text + label encoders); else uni-encoder.
    """
    labels_encoder = gliner_config.get("labels_encoder")
    architecture = "gliner-bi" if labels_encoder else "gliner-uni"
    cfg = {
        "architecture": architecture,
        "hidden_size": gliner_config.get("hidden_size"),
        "max_width": gliner_config.get("max_width"),
        "architecture_config": {
            "model_name": gliner_config.get("model_name"),
            "labels_encoder": labels_encoder,
            "span_mode": gliner_config.get("span_mode"),
            "class_token_index": gliner_config.get("class_token_index"),
            "ent_token": gliner_config.get("ent_token", "<<ENT>>"),
            "sep_token": gliner_config.get("sep_token", "<<SEP>>"),
            "embed_ent_token": gliner_config.get("embed_ent_token", True),
            "subtoken_pooling": gliner_config.get("subtoken_pooling", "first"),
            "words_splitter_type": gliner_config.get("words_splitter_type", "whitespace"),
            "max_len": gliner_config.get("max_len"),
            "max_types": gliner_config.get("max_types"),
        },
    }
    (out_dir / "gliner4j_config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    console.print(f"  [green]✓[/green] gliner4j_config.json (architecture={architecture})")


app = typer.Typer(add_completion=False)


@app.command()
def export(
    model_path: Annotated[
        str, typer.Option("--model-path", help="GLiNER model dir or HF repo ID")
    ],
    output_dir: Annotated[
        str, typer.Option("--output-dir", help="Output directory for the ONNX bundle")
    ],
    opset: Annotated[int, typer.Option("--opset", help="ONNX opset version")] = 19,
    verify: Annotated[
        bool, typer.Option("--verify/--no-verify", help="Verify ONNX entities vs PyTorch")
    ] = True,
    variants: Annotated[
        list[Variant] | None,
        typer.Option("--variant", help="Additional variants to generate (fp16, quantized)"),
    ] = None,
) -> None:
    """Export an original-GLiNER uni/bi-encoder model to ONNX via the gliner library.

    Base FP32 → onnx/model.onnx. Each requested variant (fp16, quantized=INT8/QInt8) is generated
    into onnx_{variant}/. DeBERTa backbones are patched (no dynamic-rank If nodes) and tensor names
    sanitized, so all variants load on CPU/CUDA and the OpenVINO EP.
    """
    from gliner import GLiNER

    # TODO(gliner-x): GLiNER-X (mT5) trains with words_splitter_type=stanza (language-aware; needs
    # stanza+langdetect + per-language models). We don't ship Stanza and can't replicate it in the
    # JVM, so force whitespace splitting by monkeypatching WordsSplitter before load — the exported
    # graph is splitter-independent (only the Java-side word boundaries change) and the Java consumer
    # (GlinerUniNerStrategy) also splits on whitespace. APPROXIMATION valid only for space-separated
    # languages: CJK (Chinese/Japanese/Thai) is NOT supported and punctuation boundaries may drift.
    # Revisit with a real word splitter (stanza port / ICU) on both export and Java sides. No-op for
    # models already on whitespace (gliner-pii / gliner-bi / gliner-multitask).
    import gliner.data_processing.tokenizer as _gtok

    _orig_ws_init = _gtok.WordsSplitter.__init__

    def _force_whitespace_init(self, splitter_type="whitespace", *a, **k):
        _orig_ws_init(self, "whitespace", *a, **k)

    _gtok.WordsSplitter.__init__ = _force_whitespace_init

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    console.print(f"\n[bold]Loading GLiNER model from {model_path}...[/bold]")
    model = GLiNER.from_pretrained(model_path)
    model.eval()

    cfg = model.config.to_dict() if hasattr(model.config, "to_dict") else dict(model.config.__dict__)
    console.print(
        f"  model_name={cfg.get('model_name')}, span_mode={cfg.get('span_mode')}, "
        f"ent_token={cfg.get('ent_token')!r}, class_token_index={cfg.get('class_token_index')}, "
        f"subtoken_pooling={cfg.get('subtoken_pooling')}"
    )

    onnx_file = out / "model.onnx"
    if onnx_file.exists():
        console.print(f"\n[bold]Reusing existing {onnx_file} (delete to re-export).[/bold]")
    else:
        # Bake out DeBERTa's dynamic-rank If nodes (global monkeypatch — takes effect inside
        # gliner's own export_to_onnx) so FP16 conversion and the OpenVINO EP work. No-op otherwise.
        patched = _patch_deberta_for_onnx()
        if patched:
            console.print(f"  patched {patched} DeBERTa relative-position helper(s)")
        console.print(f"\n[bold]Exporting ONNX (opset={opset}) to {out}...[/bold]")
        model.export_to_onnx(out, onnx_filename="model.onnx", opset=opset)
    # export_to_onnx does not reliably drop the config/tokenizer in this gliner build, so save
    # them explicitly — the gliner ORT loader (used for verification below) needs both at root.
    model.config.to_json_file(str(out / "gliner_config.json"))
    model.data_processor.transformer_tokenizer.save_pretrained(str(out))
    console.print("  [green]✓[/green] model.onnx + gliner_config.json + tokenizer")

    # Introspect the exact ONNX I/O contract the Java side must satisfy.
    sess = ort.InferenceSession(str(out / "model.onnx"))
    console.print("\n[bold]ONNX I/O contract:[/bold]")
    for i in sess.get_inputs():
        console.print(f"  in  {i.name}: {i.shape} {i.type}")
    for o in sess.get_outputs():
        console.print(f"  out {o.name}: {o.shape} {o.type}")

    if verify:
        # Logits-compare: drive the library's own preprocessing + ONNX wrapper on one batch, then
        # run the same inputs through PyTorch (the wrapper) and ONNX Runtime and compare logits.
        # Version-robust — avoids the gliner ORT loader, which expects the torch weights present.
        console.print("\n[bold]Verifying ONNX vs PyTorch (logits)...[/bold]")
        labels = ["person", "organization", "location", "date"]
        text = "Barack Obama visited Berlin in July 2015 with Angela Merkel."
        batch = model._build_dummy_batch(labels=labels, text=text)
        all_inputs, spec = model._prepare_onnx_batch(batch)
        wrapper = model._create_onnx_wrapper(model.model.to("cpu").eval()).eval()
        with torch.no_grad():
            pt_out = wrapper(*all_inputs)
        pt_logits = (
            pt_out[0] if isinstance(pt_out, (tuple, list)) else getattr(pt_out, "logits", pt_out)
        )
        pt_logits = pt_logits.detach().cpu().numpy()
        feeds = {
            name: all_inputs[i].cpu().numpy()
            for i, name in enumerate(spec["input_names"])
        }
        onnx_logits = sess.run(None, feeds)[0]
        max_diff = float(np.max(np.abs(pt_logits - onnx_logits)))
        ok = bool(np.allclose(pt_logits, onnx_logits, atol=1e-3))
        status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        console.print(f"  {status} logits {pt_logits.shape} (max_diff={max_diff:.6f})")
        if not ok:
            console.print("\n[bold red]Verification FAILED[/bold red]")
            raise typer.Exit(code=1)
        console.print("\n[bold green]Verification passed![/bold green]")

    # Lay the bundle out the gliner4j way: onnx/model.onnx + gliner4j_config.json at root,
    # tokenizer.json already at root from export_to_onnx.
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(exist_ok=True)
    shutil.move(str(out / "model.onnx"), str(onnx_dir / "model.onnx"))
    renamed = _sanitize_onnx_names(onnx_dir / "model.onnx")  # OpenVINO '/'-name fix
    if_nodes = _count_if_nodes(onnx_dir / "model.onnx")
    console.print(
        f"  [green]✓[/green] sanitized model.onnx ({renamed} names, "
        + ("[green]0 If nodes[/green]" if if_nodes == 0 else f"[red]{if_nodes} If nodes[/red]")
        + ")"
    )
    gliner_cfg = json.loads((out / "gliner_config.json").read_text())
    _write_gliner4j_config(gliner_cfg, out)

    # Bi-encoder: export_to_onnx only saves the text tokenizer. The label encoder uses its own
    # tokenizer (e.g. bge-small / MiniLM) for labels_input_ids — save it under labels_tokenizer/.
    labels_encoder = gliner_cfg.get("labels_encoder")
    if labels_encoder:
        from transformers import AutoTokenizer

        label_tok_dir = out / "labels_tokenizer"
        AutoTokenizer.from_pretrained(labels_encoder).save_pretrained(
            str(label_tok_dir)
        )
        console.print(
            f"  [green]✓[/green] labels_tokenizer/ ({labels_encoder})"
        )

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

    console.print(f"\n[bold]Bundle ready at {out} (onnx/model.onnx)[/bold]")


if __name__ == "__main__":
    app()
