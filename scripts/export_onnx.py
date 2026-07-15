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
#     "safetensors>=0.4",
#     "typer>=0.15",
#     "numpy>=1.26",
#     "rich>=13",
#     "huggingface-hub>=0.20",
#     "pydantic>=2.0",
#     "urllib3>=2.0",
#     "requests>=2.31",
#     "onnxscript>=0.1",
#     "gliner2",
#     "gliclass",
#     "gliner",
# ]
# ///
"""ONNX export for gliner4j — all supported model families in one script.

Subcommands (one per family):
  gliner2     GLiNER2 (Extractor): encoder / span_rep / scoring_head / classifier_head split
              graphs plus the single-graph deployment artifacts ner_full / classifier_full.
  gliclass    GLiClass uni-encoder: encoder + score_head graphs.
  gliner-uni  Original GLiNER uni/bi-encoder: monolithic model.onnx via the gliner library.

Every family lays the bundle out the gliner4j way:
  {output_dir}/onnx/            Base FP32 graphs
  {output_dir}/onnx_fp16/       FP16 variant (--variant fp16)
  {output_dir}/onnx_quantized/  INT8 dynamically-quantized variant (--variant quantized)
  {output_dir}/                 gliner4j_config.json + tokenizer files (shared across variants)

Usage:
    uv run scripts/export_onnx.py gliner2 --model-path fastino-ai/gliner2-base --output-dir models/out
    uv run scripts/export_onnx.py gliclass --model-path knowledgator/gliclass-modern-base-v3.0 \
        --output-dir models/gliclass-modern-base-onnx
    uv run scripts/export_onnx.py gliner-uni --model-path knowledgator/gliner-multitask-large-v0.5 \
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
import torch.nn as nn
import typer
from rich.console import Console

console = Console()

app = typer.Typer(add_completion=False)


class Variant(str, Enum):
    """ONNX model variants that can be generated from the base FP32 export."""

    fp16 = "fp16"
    quantized = "quantized"


# ===========================================================================
# Shared helpers (all families)
# ===========================================================================


def _patch_deberta_for_onnx() -> int:
    """De-script DeBERTa-v2/v3's relative-position control flow before ONNX export.

    ``transformers.models.deberta_v2.modeling_deberta_v2`` defines ``build_rpos``
    with ``@torch.jit.script``::

        @torch.jit.script
        def build_rpos(query_layer, key_layer, relative_pos, position_buckets, max_relative_positions):
            if key_layer.size(-2) != query_layer.size(-2):
                return build_relative_position(key_layer, key_layer, ...)  # rank 3
            else:
                return relative_pos                                        # rank 4

    Because it is *scripted*, that ``if`` compiles to a real ONNX ``If`` node —
    one per layer — whose then/else branches have different rank (3 vs 4). The
    node therefore has dynamic rank, which breaks FP16 conversion and which
    OpenVINO's CPU plugin cannot compile::

        CPU plug-in doesn't support If operation with dynamic rank

    Our encoder always runs plain square self-attention, so ``query_size ==
    key_size`` and the ``else`` branch (``return relative_pos``) is always
    taken. Replacing the scripted ``build_rpos`` with a plain Python equivalent
    lets the tracer evaluate the size comparison at export time and bake in that
    single branch — no ``If`` nodes — with identical numerics (kept honest by
    the PT-vs-ONNX verification, atol 1e-4).

    Also unwraps any ``@torch.jit.script_if_tracing`` helpers, which older
    transformers releases used for the same relative-position code. The patch is
    a global monkeypatch on the transformers module, so it also takes effect
    inside third-party export paths (e.g. gliner's own ``export_to_onnx``).
    No-op for non-DeBERTa backbones. Returns the number of helpers patched.
    """
    try:
        import transformers.models.deberta_v2.modeling_deberta_v2 as dv2
    except ImportError:
        return 0

    patched = 0

    # Older transformers: relative-position helpers used @torch.jit.script_if_tracing.
    for name in dir(dv2):
        obj = getattr(dv2, name)
        original = getattr(obj, "__original_fn", None)
        if original is not None and getattr(obj, "__script_if_tracing_wrapper", False):
            setattr(dv2, name, original)
            console.print(f"  [dim]unwrapped script_if_tracing: {name}[/dim]")
            patched += 1

    # Current transformers (>=5.x): build_rpos is @torch.jit.script. Replace it
    # with a traceable Python version that calls the module's (unscripted)
    # build_relative_position, so the size comparison folds at trace time.
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
        console.print("  [dim]de-scripted build_rpos[/dim]")
        patched += 1

    return patched


def _count_if_nodes(path: Path) -> int:
    """Count ONNX ``If`` nodes in a model (used to confirm the DeBERTa unwrap worked)."""
    model = onnx.load(str(path))
    return sum(1 for n in model.graph.node if n.op_type == "If")


def _sanitize_onnx_names(path: Path) -> int:
    """Replace ``/`` in every tensor/node name so the model runs on OpenVINO.

    PyTorch's ONNX exporter names intermediate tensors like
    ``/encoder/Reshape_output_0`` — leading slash, multiple slashes. ONNX
    Runtime's OpenVINO execution provider, when a constant-folded tensor becomes
    a subgraph-boundary output, truncates the output name at the first ``/``
    (``backend_utils.cc`` ``GetOutputTensor``) and looks the result up in the
    subgraph's output map. A name starting with ``/`` truncates to ``""`` and the
    lookup throws::

        [OpenVINO-EP] Output names mismatch between OpenVINO and ONNX

    Replacing ``/`` with ``_`` everywhere — node names, tensor names,
    initializers, all references, recursing into subgraphs — removes the
    trigger. Names are pure identifiers and the declared graph I/O contains no
    ``/``, so the model's interface and numerics are unchanged (and CPU/CUDA
    ignore names entirely). Run before the fp16/quantized conversions so every
    variant inherits the clean names.

    Returns the number of names rewritten.
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


def _dedupe_constant_nodes(model: onnx.ModelProto) -> int:
    """Collapse duplicate Constant nodes into shared initializers.

    The ``onnxconverter_common.float16`` pass materializes shared FP32
    initializers as separate Constant nodes (one per consumer). When the
    GRU subgraph is unrolled across max_count timesteps, this multiplies
    a small set of weight tensors by ~20×, bloating the FP16 file from
    ~30 MB to ~150 MB. This pass folds them back into shared initializers.

    Returns the number of Constant nodes removed.
    """
    from collections import defaultdict

    # Group Constants with raw_data by (dtype, shape, bytes).
    groups: dict = defaultdict(list)
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.t.raw_data and len(attr.t.raw_data) > 1024:
                key = (attr.t.data_type, tuple(attr.t.dims), bytes(attr.t.raw_data))
                groups[key].append(node)
                break

    if not any(len(v) >= 2 for v in groups.values()):
        return 0

    rewrites: dict[str, str] = {}
    nodes_to_remove = []
    existing_names = {init.name for init in model.graph.initializer}
    counter = 0
    for (dtype, dims, data), nodes in groups.items():
        if len(nodes) < 2:
            continue
        # Build a single shared initializer from the first node's tensor.
        ref_node = nodes[0]
        ref_attr = next(a for a in ref_node.attribute if a.name == "value")
        shared = onnx.TensorProto()
        shared.CopyFrom(ref_attr.t)
        # Ensure unique name
        base_name = f"_dedup_const_{counter}"
        while base_name in existing_names:
            counter += 1
            base_name = f"_dedup_const_{counter}"
        shared.name = base_name
        existing_names.add(base_name)
        counter += 1
        model.graph.initializer.append(shared)
        for n in nodes:
            rewrites[n.output[0]] = shared.name
            nodes_to_remove.append(n)

    # Rewrite consumer inputs.
    for node in model.graph.node:
        for i, inp in enumerate(node.input):
            if inp in rewrites:
                node.input[i] = rewrites[inp]

    # Remove the now-redundant Constant nodes.
    for n in nodes_to_remove:
        model.graph.node.remove(n)
    return len(nodes_to_remove)


def _align_fp16_cast_nodes(model: onnx.ModelProto) -> int:
    """Make ``Cast`` ``to`` attributes agree with the fp16 types the converter assigned.

    ``onnxconverter_common.float16.convert_float_to_float16`` rewrites
    intermediate tensors (and their ``value_info``) to float16 but leaves the
    ``to`` attribute of *pre-existing* ``Cast`` nodes untouched. DeBERTa's
    embeddings have ``embeddings * mask.to(dtype)`` → a ``Cast(to=FLOAT)`` whose
    output feeds a ``Mul``. After conversion the converter marks that Cast's
    output ``value_info`` as float16, but the node still casts to float32 — so it
    emits float32 into a float16 graph and ORT refuses to load it::

        Type (tensor(float16)) of output arg (...Cast_output_0) ... does not
        match expected type (tensor(float))

    For every ``Cast`` whose declared output type is float16 but whose ``to`` is
    still ``FLOAT``, set ``to`` to ``FLOAT16`` — aligning the node with the
    converter's own type decision (and keeping the downstream ``Mul`` all-fp16).

    Returns the number of Cast nodes realigned.
    """
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


def _convert_to_fp16(base_dir: Path, output_dir: Path, model_files: tuple[str, ...]) -> None:
    """Convert base FP32 ONNX models to FP16, validating each loads (else ship FP32).

    Args:
        base_dir: Directory containing base FP32 ONNX models.
        output_dir: Directory to write FP16 models.
        model_files: The ONNX file names that make up this family's bundle.
    """
    from onnxconverter_common import float16

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in model_files:
        model = onnx.load(str(base_dir / name))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        # Fold duplicate Constants back into shared initializers
        # (works around the unrolled-loop bloat introduced by the converter).
        removed = _dedupe_constant_nodes(model_fp16)
        # Align pre-existing Cast nodes with the converter's fp16 value_info so
        # the model actually loads in ONNX Runtime (see _align_fp16_cast_nodes).
        realigned = _align_fp16_cast_nodes(model_fp16)
        onnx.save(model_fp16, str(output_dir / name))
        # Re-sanitize: the float16 converter / dedupe pass can mint new names.
        _sanitize_onnx_names(output_dir / name)

        # Validate the fp16 model actually loads, and ship the FP32 base if it
        # doesn't, so the bundle is always usable (mirrors the quantized
        # fallback). This guard matters because the export's verification step
        # only checks the *base* FP32 models numerically — the fp16/quantized
        # variants are never otherwise load-tested, so without this check a
        # non-loadable variant ships silently. That is exactly how the
        # Cast/value_info mismatch above went unnoticed until a consumer first
        # tried to load an fp16 model.
        try:
            ort.InferenceSession(str(output_dir / name))
            notes = []
            if removed:
                notes.append(f"deduped {removed}")
            if realigned:
                notes.append(f"realigned {realigned} casts")
            note = f" ({', '.join(notes)})" if notes else ""
        except Exception as e:  # noqa: BLE001
            shutil.copy(base_dir / name, output_dir / name)
            note = " (fp16 invalid, copied FP32 instead)"
            console.print(f"  [yellow]![/yellow] {name}: {str(e).splitlines()[0]}")

        size_mb = (output_dir / name).stat().st_size / 1e6
        marker = "[yellow]→[/yellow]" if "invalid" in note else "[green]✓[/green]"
        console.print(f"  {marker} {name} ({size_mb:.1f} MB){note}")


def _convert_to_quantized(
    base_dir: Path,
    output_dir: Path,
    model_files: tuple[str, ...],
    weight_type,
) -> None:
    """Convert base FP32 ONNX models to INT8 dynamic quantization, validating each loads.

    Args:
        base_dir: Directory containing base FP32 ONNX models.
        output_dir: Directory to write quantized models.
        model_files: The ONNX file names that make up this family's bundle.
        weight_type: ``QuantType`` for the weights. GLiNER2 uses QUInt8 (its
            unrolled-GRU subgraph is the historical constraint); GLiClass and
            original GLiNER use QInt8 (signed, symmetric) — the
            accuracy-preserving config for transformer weights, since
            unsigned/asymmetric is a known de-calibration pitfall there.
    """
    from onnxruntime.quantization import quantize_dynamic

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in model_files:
        # Per-channel + reduce_range; DefaultTensorType handles nodes whose type
        # shape-inference can't determine — e.g. matmul outputs inside an
        # unrolled GRU subgraph.
        quantize_dynamic(
            model_input=base_dir / name,
            model_output=output_dir / name,
            weight_type=weight_type,
            per_channel=True,
            reduce_range=True,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )
        # Re-sanitize: quantize_dynamic can mint new '/'-named tensors.
        _sanitize_onnx_names(output_dir / name)

        # Validate: some upstream subgraphs (notably nn.GRU's unrolled form in
        # scoring_head for the GLiNER2 base model) produce MatMulInteger nodes
        # with shapes ORT cannot reconcile at load time. If the quantized file
        # fails to instantiate, fall back to shipping the FP32 file instead.
        try:
            ort.InferenceSession(str(output_dir / name))
            note = ""
        except Exception as e:  # noqa: BLE001
            (output_dir / name).unlink(missing_ok=True)
            shutil.copy(base_dir / name, output_dir / name)
            note = " (quantization invalid, copied FP32 instead)"
            console.print(f"  [yellow]![/yellow] {name}: {str(e).splitlines()[0]}")

        size_mb = (output_dir / name).stat().st_size / 1e6
        marker = "[yellow]→[/yellow]" if note else "[green]✓[/green]"
        console.print(f"  {marker} {name} ({size_mb:.1f} MB){note}")


def _generate_variants(
    variants: list[Variant] | None,
    out: Path,
    base_dir: Path,
    model_files: tuple[str, ...],
    quant_weight_type,
) -> None:
    """Generate the requested fp16/quantized variants from the base FP32 graphs."""
    if not variants:
        return
    for variant in variants:
        variant_dir = out / f"onnx_{variant.value}"
        console.print(f"\n[bold]Generating {variant.value} variant → {variant_dir}...[/bold]")
        if variant == Variant.fp16:
            _convert_to_fp16(base_dir, variant_dir, model_files)
        elif variant == Variant.quantized:
            _convert_to_quantized(base_dir, variant_dir, model_files, quant_weight_type)


def _print_verify(name: str, ok: bool, pt: np.ndarray, onnx_val: np.ndarray) -> None:
    """Print a PT-vs-ONNX verification result."""
    max_diff = float(np.max(np.abs(pt - onnx_val)))
    status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
    console.print(f"  {status} {name} (max_diff={max_diff:.6f})")


def _print_tree(root: Path, prefix: str = "  ") -> None:
    """Print a simple directory tree."""
    entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    for entry in entries:
        if entry.is_dir():
            console.print(f"{prefix}[bold]{entry.name}/[/bold]")
            _print_tree(entry, prefix + "  ")
        else:
            size = entry.stat().st_size
            label = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB"
            console.print(f"{prefix}{entry.name} ({label})")


class EncoderWrapper(nn.Module):
    """Wraps a HuggingFace encoder (input_ids, attention_mask → last_hidden_state)."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Run encoder and return last_hidden_state.

        Args:
            input_ids: Token IDs of shape (batch, seq_len).
            attention_mask: Attention mask of shape (batch, seq_len).

        Returns:
            Last hidden state of shape (batch, seq_len, hidden_size).
        """
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


def _export_encoder_graph(wrapper: nn.Module, path: Path, opset: int) -> Path:
    """torch.onnx.export an EncoderWrapper with the shared I/O contract."""
    dummy_ids = torch.ones(1, 32, dtype=torch.long)
    dummy_mask = torch.ones(1, 32, dtype=torch.long)
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
        else f", [red]{if_nodes} If nodes remain (OpenVINO will reject)[/red]"
    )
    console.print(f"  [green]✓[/green] encoder.onnx ({path.stat().st_size / 1e6:.1f} MB{if_note})")
    return path


# ===========================================================================
# GLiNER2 (Extractor) — wrappers
# ===========================================================================

# The graphs that make up a GLiNER2 bundle (base + every variant).
GLINER2_MODEL_FILES = (
    "encoder.onnx",
    "span_rep.onnx",
    "scoring_head.onnx",
    "classifier_head.onnx",
    "ner_full.onnx",
    "classifier_full.onnx",
)

# The deployment artifacts. The split graphs above them are intermediates: they
# exist to be verified head-by-head and to feed build_merged_graphs.py's
# encoder-fusion pipeline, and are deleted after export unless --keep-split.
GLINER2_FINAL_FILES = ("ner_full.onnx", "classifier_full.onnx")


class SpanRepWrapper(nn.Module):
    """Wraps SpanRepLayer for ONNX export.

    Handles both marker-based (require span_idx) and non-marker-based modes.
    """

    def __init__(self, span_rep: nn.Module, *, uses_span_idx: bool) -> None:
        super().__init__()
        self.span_rep = span_rep
        self.uses_span_idx = uses_span_idx

    def forward(
        self, token_embeddings: torch.Tensor, span_idx: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute span representations.

        Args:
            token_embeddings: Shape (1, text_len, hidden_size).
            span_idx: Shape (1, text_len*max_width, 2) — only for marker modes.

        Returns:
            Span representations of shape (1, text_len, max_width, hidden_size).
        """
        if self.uses_span_idx:
            return self.span_rep(token_embeddings, span_idx)
        return self.span_rep(token_embeddings)


class ScoringHeadWrapper(nn.Module):
    """Wraps the count-prediction + count-embed + scoring pipeline for ONNX.

    Two design constraints drive how this wrapper differs from a faithful
    reproduction of the original PyTorch forward pass:

    1. Static unrolling for dynamic counts
    ---------------------------------------
    CountLSTM/v2/MoE use a Python int ``count`` to control ``torch.arange``,
    which produces data-dependent shapes that ONNX export cannot trace. We
    always unroll the GRU for ``max_count`` steps and slice to ``count`` at
    the end, so the graph has fully static shapes.

    2. MatMul-only gate projections (for clean INT8 quantization)
    -------------------------------------------------------------
    ``onnxruntime.quantization.quantize_dynamic`` rewrites ``Gemm`` nodes by
    decomposing them into a synthesised ``MatMul + Add`` pair and quantizing
    the new ``MatMul``. During that decomposition the quantizer loses the
    ``transA`` / ``transB`` / α / β bookkeeping that ``Gemm`` carries; the
    resulting ``MatMulInteger`` ends up with operand shapes that look
    pairwise sensible to ONNX's static checker but don't actually match for
    matrix multiplication when ORT instantiates the session — producing the
    runtime error ::

        [ShapeInferenceError] Incompatible dimensions for matrix multiplication

    To avoid emitting any ``Gemm`` in the GRU subgraph, this wrapper:

    * **Pre-transposes the weights once** in ``__init__`` and stores them as
      tensor attributes (``self._weight_ih_t`` / ``self._weight_hh_t``).
      ``F.linear(x, W, b)`` exports as a single ``Gemm`` (with ``transB=1``);
      ``x @ W_t`` with an already-transposed constant exports as ``MatMul``
      (with a ``Transpose`` of a constant feeding in, which ONNX shape-folds
      away). ``quantize_dynamic`` converts ``MatMul`` to ``MatMulInteger``
      directly — no decomposition, no lost bookkeeping.
    * **Unrolls the GRU loop in our own forward()** instead of delegating to
      ``self.gru(...)``. When ``self.gru`` is ``nn.GRU`` (base model),
      PyTorch's exporter traces through C++ kernels and emits a
      ``Gemm``-heavy subgraph we can't influence from outside. When it's
      ``gliner2.layers.CompileSafeGRU`` (PII fine-tunes), the Python loop
      traces to ops named ``/gru/...`` but those ops are still
      ``F.linear``-based (i.e. ``Gemm``). Rewriting the loop here covers
      both model types in one path.
    * **Keeps the bias as a standalone ``Add``** rather than fusing it into
      the matmul. ``MatMul`` + standalone ``Add`` is exactly the pattern
      ``quantize_dynamic`` knows how to handle (it quantizes the ``MatMul``
      and leaves the ``Add`` alone, since biases aren't quantized).

    The unroll is mathematically identical to both ``nn.GRU`` (single layer,
    default activations) and ``CompileSafeGRU.forward``: same gate formulas,
    same ``[W_ir; W_iz; W_in]`` weight stacking, same ``r * (W_hn·h + b_hn)``
    placement of the reset gate. Because ``nn.GRU`` and ``CompileSafeGRU``
    share the same ``weight_*_l0`` / ``bias_*_l0`` parameter names, one
    forward path drives both with no model-specific branching.

    No weights change. No retraining or calibration. The existing
    ``_verify_scoring_head`` check (PT vs ONNX, ``atol=1e-4``) keeps both
    legs honest: both legs now run this manual unroll, and the ONNX graph
    expresses the same math as ``MatMul`` + ``Add`` + ``Sigmoid`` + ``Tanh``
    instead of fused ``Gemm`` + ``GRU``.

    The validate-and-fallback step in ``_convert_to_quantized`` stays in
    place as a belt-and-suspenders safety net: if some other GRU variant
    (e.g. ``count_lstm_v2`` / ``count_lstm_moe``) still produces a
    non-loadable quantized file, the FP32 source is copied into
    ``onnx_quantized/`` instead and the bundle remains usable.
    """

    def __init__(
        self,
        count_pred: nn.Module,
        count_embed: nn.Module,
        *,
        max_count: int = 20,
        counting_layer: str = "count_lstm",
    ) -> None:
        super().__init__()
        self.count_pred = count_pred
        self.max_count = max_count
        self.counting_layer = counting_layer

        # Extract sub-modules from the count_embed layer
        self.pos_embedding = count_embed.pos_embedding
        self.gru = count_embed.gru
        # Pre-transpose the GRU weights so the manual unroll in forward()
        # emits MatMul (via `x @ W_t`) rather than Gemm (via `F.linear`).
        # quantize_dynamic handles MatMul → MatMulInteger directly, while its
        # Gemm → MatMul + Add decomposition loses the transA/transB/α/β
        # bookkeeping and produces MatMulInteger nodes whose operand shapes
        # ORT rejects at load time. nn.GRU (base) and CompileSafeGRU (pii)
        # share the same weight_*_l0 / bias_*_l0 names so one path covers both.
        # Register as buffers (not plain tensor attributes) so PyTorch's ONNX
        # tracer emits a single shared initializer per weight, instead of one
        # embedded Constant copy per unrolled timestep. Plain attributes get
        # duplicated 20× and balloon scoring_head.onnx by ~10×.
        self.register_buffer("_weight_ih_t", self.gru.weight_ih_l0.detach().t().contiguous())
        self.register_buffer("_weight_hh_t", self.gru.weight_hh_l0.detach().t().contiguous())
        self.register_buffer("_bias_ih", self.gru.bias_ih_l0.detach().clone())
        self.register_buffer("_bias_hh", self.gru.bias_hh_l0.detach().clone())

        if counting_layer == "count_lstm":
            self.projector = count_embed.projector
        elif counting_layer == "count_lstm_v2":
            # Rebuild DownscaledTransformer with ONNX-compatible manual layer calls
            # instead of nn.TransformerEncoder (which uses a fused kernel)
            orig_tf = count_embed.transformer
            self.tf_in_projector = orig_tf.in_projector
            self.tf_layers = orig_tf.transformer.layers  # nn.ModuleList
            self.tf_norm = orig_tf.transformer.norm  # final LayerNorm (may be None)
            self.tf_out_projector = orig_tf.out_projector
        elif counting_layer == "count_lstm_moe":
            self.w1 = count_embed.w1
            self.b1 = count_embed.b1
            self.w2 = count_embed.w2
            self.b2 = count_embed.b2
            self.dropout = count_embed.dropout
            self.router = count_embed.router

    def forward(
        self,
        span_rep: torch.Tensor,
        schema_emb_p: torch.Tensor,
        schema_emb_fields: torch.Tensor,
        count: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the scoring head.

        Args:
            span_rep: Span representations, shape (batch, text_len, max_width, hidden).
                The leading batch axis lets the runtime score several texts of one
                batch in a single call — the schema/count leg is text-independent, so
                only the final einsum broadcasts over it.
            schema_emb_p: The [P] token embedding, shape (hidden,).
            schema_emb_fields: Field embeddings, shape (num_fields, hidden).
            count: Scalar int64 tensor — predicted count.

        Returns:
            count_logits: Shape (1, 20).
            span_scores: Shape (batch, count, num_fields, text_len, max_width).
        """
        # Count prediction
        count_logits = self.count_pred(schema_emb_p.unsqueeze(0))  # (1, 20)

        # Full GRU unroll to max_count
        M, D = schema_emb_fields.shape
        full_idx = torch.arange(self.max_count, device=schema_emb_fields.device)
        pos_seq = self.pos_embedding(full_idx)  # (max_count, D)
        pos_seq = pos_seq.unsqueeze(1).expand(self.max_count, M, D)  # (max_count, M, D)

        # Manual GRU unroll over max_count steps. Mathematically identical to
        # both PyTorch's nn.GRU (single layer) and gliner2's CompileSafeGRU;
        # we reuse the same weights via self._weight_*. The "@ + Add" pattern
        # exports as ONNX MatMul (+ Add), which quantize_dynamic converts
        # cleanly to MatMulInteger — unlike Gemm, whose decomposition during
        # quantization breaks operand shapes (see __init__ for the why).
        h = schema_emb_fields  # (M, D)
        outputs = []
        for t in range(self.max_count):
            gi = pos_seq[t] @ self._weight_ih_t + self._bias_ih  # (M, 3D)
            gh = h @ self._weight_hh_t + self._bias_hh
            i_r, i_z, i_n = gi.chunk(3, dim=-1)
            h_r, h_z, h_n = gh.chunk(3, dim=-1)
            r = torch.sigmoid(i_r + h_r)
            z = torch.sigmoid(i_z + h_z)
            n = torch.tanh(i_n + r * h_n)
            h = (1 - z) * n + z * h
            outputs.append(h)
        output = torch.stack(outputs, dim=0)  # (max_count, M, D)

        # Apply variant-specific projection
        if self.counting_layer == "count_lstm":
            pc_broadcast = schema_emb_fields.unsqueeze(0).expand_as(output)
            projected = self.projector(
                torch.cat([output, pc_broadcast], dim=-1)
            )  # (max_count, M, D)
        elif self.counting_layer == "count_lstm_v2":
            pc_broadcast = schema_emb_fields.unsqueeze(0).expand_as(output)
            projected = self._downscaled_transformer(output + pc_broadcast)  # (max_count, M, D)
        elif self.counting_layer == "count_lstm_moe":
            projected = self._moe_project(output)  # (max_count, M, D)
        else:
            msg = f"Unsupported counting_layer: {self.counting_layer}"
            raise ValueError(msg)

        # Slice to actual count (dynamic tensor slice — ONNX supports this)
        struct_proj = projected[:count]  # (count, M, D)

        # Score via einsum + sigmoid, broadcasting over the text batch axis
        span_scores = torch.sigmoid(
            torch.einsum("blkd,cpd->bcplk", span_rep, struct_proj)
        )

        return count_logits, span_scores

    def _downscaled_transformer(self, x: torch.Tensor) -> torch.Tensor:
        """ONNX-compatible DownscaledTransformer forward pass.

        Manually implements TransformerEncoderLayer forward to avoid the fused
        aten::_transformer_encoder_layer_fwd kernel that ONNX cannot export.

        Args:
            x: Input tensor of shape (L, M, input_size).

        Returns:
            Output tensor of shape (L, M, input_size).
        """
        import torch.nn.functional as F

        original_x = x
        x = self.tf_in_projector(x)

        for layer in self.tf_layers:
            # Manual multi-head self-attention (avoids fused kernel entirely)
            residual = x
            L, M, D = x.shape
            num_heads = layer.self_attn.num_heads
            head_dim = D // num_heads

            # QKV projection using the packed in_proj weights
            qkv = F.linear(x, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)

            # Reshape: (L, M, D) -> (M*num_heads, L, head_dim)
            q = q.reshape(L, M * num_heads, head_dim).transpose(0, 1)
            k = k.reshape(L, M * num_heads, head_dim).transpose(0, 1)
            v = v.reshape(L, M * num_heads, head_dim).transpose(0, 1)

            # Scaled dot-product attention
            scale = float(head_dim) ** -0.5
            attn_weights = torch.bmm(q * scale, k.transpose(1, 2))
            attn_weights = F.softmax(attn_weights, dim=-1)
            attn_out = torch.bmm(attn_weights, v)

            # Reshape back: (M*num_heads, L, head_dim) -> (L, M, D)
            attn_out = attn_out.transpose(0, 1).reshape(L, M, D)
            attn_out = F.linear(attn_out, layer.self_attn.out_proj.weight, layer.self_attn.out_proj.bias)

            x = layer.norm1(residual + attn_out)

            # Feed-forward block (no dropout in eval mode)
            residual = x
            x2 = layer.linear2(F.relu(layer.linear1(x)))
            x = layer.norm2(residual + x2)

        if self.tf_norm is not None:
            x = self.tf_norm(x)
        x = torch.cat([x, original_x], dim=-1)
        return self.tf_out_projector(x)

    def _moe_project(self, h: torch.Tensor) -> torch.Tensor:
        """Mixture-of-Experts projection.

        Args:
            h: GRU output of shape (L, M, D).

        Returns:
            Projected output of shape (L, M, D).
        """
        import torch.nn.functional as F

        gates = self.router(h)  # (L, M, E)
        x = torch.einsum("lmd,edh->lmeh", h, self.w1) + self.b1  # (L, M, E, inner)
        x = F.gelu(x)
        # Note: dropout is no-op in eval mode, but keep for structural fidelity
        x = self.dropout(x)
        x = torch.einsum("lmeh,ehd->lmed", x, self.w2) + self.b2  # (L, M, E, D)
        return (gates.unsqueeze(-1) * x).sum(dim=2)  # (L, M, D)


class ClassifierWrapper(nn.Module):
    """Wraps the classifier MLP (2-layer: hidden→hidden*2→1, ReLU) for ONNX export.

    The classifier operates on label embeddings extracted at [L] token positions
    from the encoder's hidden states. It produces a single logit per label.
    """

    def __init__(self, classifier: nn.Module) -> None:
        super().__init__()
        self.classifier = classifier

    def forward(self, label_embeddings: torch.Tensor) -> torch.Tensor:
        """Run classifier MLP on label embeddings.

        Args:
            label_embeddings: Shape (num_labels, hidden_size).

        Returns:
            Logits of shape (num_labels, 1).
        """
        return self.classifier(label_embeddings)


class NerFullWrapper(nn.Module):
    """Full single-graph NER: encoder → word/schema gather → span_rep → scoring head.

    Composes the individually-verified EncoderWrapper / SpanRepWrapper / ScoringHeadWrapper and
    does the word/schema gather (previously done Java-side between sessions) in the forward, so a
    single ``torch.onnx.export`` produces ``ner_full.onnx`` straight from the safetensors — no
    post-export merge, no intermediate on disk. The gather semantics mirror the runtime exactly:
    ``word_positions``/``p_position`` of -1 map to zero rows (padding / absent [P]).
    """

    def __init__(
        self,
        encoder: nn.Module,
        span_rep: nn.Module,
        scoring: nn.Module,
        hidden_size: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.span_rep = span_rep
        self.scoring = scoring
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_positions: torch.Tensor,  # int64 [batch, text_len], -1 = padded word
        p_position: torch.Tensor,  # int64 [1], -1 = absent
        field_positions: torch.Tensor,  # int64 [num_fields]
        span_idx: torch.Tensor,  # int64 [batch, text_len*max_width, 2]
        count: torch.Tensor,  # int64 scalar
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(input_ids, attention_mask)  # [b, seq, h]

        # per-word first-subword rows, padded (-1) rows zeroed
        wp = word_positions.clamp(min=0)
        gather_idx = wp.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        tok = torch.gather(hidden, 1, gather_idx)  # [b, text_len, h]
        tok = tok * (word_positions >= 0).unsqueeze(-1).to(tok.dtype)

        # schema markers from row 0 (schema prefix is identical across the batch)
        row0 = hidden[0]  # [seq, h]
        p_emb = row0[p_position.clamp(min=0)].squeeze(0)  # [h]
        p_emb = p_emb * (p_position >= 0).to(p_emb.dtype).reshape(())
        field_emb = row0[field_positions]  # [num_fields, h]

        span_rep = self.span_rep(tok, span_idx)  # [b, text_len, max_width, h]
        return self.scoring(span_rep, p_emb, field_emb, count)


class ClassifierFullWrapper(nn.Module):
    """Full single-graph classification: encoder → label gather → classifier MLP.

    Gathers the [L] label-marker rows for every batch row (positions are shared across the batch)
    and flattens to keep the rank-2 classifier head batched. Output logits are [batch*num_labels, 1]
    (the runtime reshapes to [batch][num_labels]).
    """

    def __init__(self, encoder: nn.Module, classifier: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        label_positions: torch.Tensor,  # int64 [num_labels]
    ) -> torch.Tensor:
        hidden = self.encoder(input_ids, attention_mask)  # [b, seq, h]
        label_emb = hidden[:, label_positions]  # [b, num_labels, h]
        flat = label_emb.reshape(-1, label_emb.shape[-1])  # [b*num_labels, h]
        return self.classifier(flat)  # [b*num_labels, 1]


# ===========================================================================
# GLiNER2 — export functions
# ===========================================================================


def _disable_transformer_fast_path(module: nn.Module) -> None:
    """Recursively disable the fused fast-path on TransformerEncoder layers.

    The fused kernel (aten::_transformer_encoder_layer_fwd) is not exportable
    to ONNX. Setting enable_nested_tensor=False forces the non-fused path.

    Args:
        module: Root module to walk.
    """
    for m in module.modules():
        if isinstance(m, nn.TransformerEncoder):
            m.enable_nested_tensor = False
        if isinstance(m, nn.TransformerEncoderLayer):
            # Ensure the layer doesn't use the fused path
            if hasattr(m, "_is_full_qkv_same_device"):
                m._is_full_qkv_same_device = False


def _g2_export_encoder(model: nn.Module, output_dir: Path, opset: int) -> Path:
    """Export encoder.onnx."""
    wrapper = EncoderWrapper(model.encoder)
    wrapper.eval()

    # DeBERTa backbones emit per-layer `If` nodes (dynamic rank) from their
    # @torch.jit.script_if_tracing relative-position helper, which OpenVINO's
    # CPU plugin cannot compile. Unwrap them so the export bakes in the single
    # static branch our usage always takes. No-op for non-DeBERTa backbones.
    patched = _patch_deberta_for_onnx()
    if patched:
        console.print(f"  patched {patched} DeBERTa relative-position helper(s) for ONNX")

    return _export_encoder_graph(wrapper, output_dir / "encoder.onnx", opset)


def _detect_uses_span_idx(span_rep: nn.Module) -> bool:
    """Detect whether the span_rep layer requires span_idx.

    Args:
        span_rep: The SpanRepLayer module.

    Returns:
        True if the underlying layer is marker-based and needs span_idx.
    """
    class_name = span_rep.span_rep_layer.__class__.__name__
    return class_name in ("SpanMarker", "SpanMarkerV0", "SpanMarkerV1")


def _g2_export_span_rep(
    model: nn.Module,
    output_dir: Path,
    opset: int,
    hidden_size: int,
    max_width: int,
) -> Path:
    """Export span_rep.onnx."""
    uses_span_idx = _detect_uses_span_idx(model.span_rep)
    wrapper = SpanRepWrapper(model.span_rep, uses_span_idx=uses_span_idx)
    wrapper.eval()

    text_len = 16
    dummy_emb = torch.randn(1, text_len, hidden_size)

    path = output_dir / "span_rep.onnx"

    if uses_span_idx:
        num_spans = text_len * max_width
        dummy_span_idx = torch.zeros(1, num_spans, 2, dtype=torch.long)
        for i in range(text_len):
            for j in range(max_width):
                idx = i * max_width + j
                if i + j < text_len:
                    dummy_span_idx[0, idx, 0] = i
                    dummy_span_idx[0, idx, 1] = i + j
                # else stays (0, 0) — safe default

        torch.onnx.export(
            wrapper,
            (dummy_emb, dummy_span_idx),
            str(path),
            opset_version=opset,
            dynamo=False,
            input_names=["token_embeddings", "span_idx"],
            output_names=["span_rep"],
            dynamic_axes={
                "token_embeddings": {0: "batch", 1: "text_len"},
                "span_idx": {0: "batch", 1: "num_spans"},
                "span_rep": {0: "batch", 1: "text_len"},
            },
        )
    else:
        torch.onnx.export(
            wrapper,
            (dummy_emb,),
            str(path),
            opset_version=opset,
            dynamo=False,
            input_names=["token_embeddings"],
            output_names=["span_rep"],
            dynamic_axes={
                "token_embeddings": {0: "batch", 1: "text_len"},
                "span_rep": {0: "batch", 1: "text_len"},
            },
        )

    console.print(f"  [green]✓[/green] span_rep.onnx ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def _g2_export_scoring_head(
    model: nn.Module,
    output_dir: Path,
    opset: int,
    hidden_size: int,
    max_width: int,
    counting_layer: str,
) -> Path:
    """Export scoring_head.onnx."""
    wrapper = ScoringHeadWrapper(
        count_pred=model.count_pred,
        count_embed=model.count_embed,
        max_count=20,
        counting_layer=counting_layer,
    )
    wrapper.eval()

    # Disable fused TransformerEncoder fast-path so TorchScript can trace it
    _disable_transformer_fast_path(wrapper)

    batch = 2
    text_len = 16
    num_fields = 3
    dummy_span_rep = torch.randn(batch, text_len, max_width, hidden_size)
    dummy_p_emb = torch.randn(hidden_size)
    dummy_fields = torch.randn(num_fields, hidden_size)
    dummy_count = torch.tensor(2, dtype=torch.long)

    path = output_dir / "scoring_head.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_span_rep, dummy_p_emb, dummy_fields, dummy_count),
        str(path),
        opset_version=opset,
        dynamo=False,
        input_names=["span_rep", "schema_emb_p", "schema_emb_fields", "count"],
        output_names=["count_logits", "span_scores"],
        dynamic_axes={
            "span_rep": {0: "batch", 1: "text_len"},
            "schema_emb_fields": {0: "num_fields"},
            "count_logits": {},
            "span_scores": {0: "batch", 1: "count", 2: "num_fields", 3: "text_len"},
        },
    )
    console.print(
        f"  [green]✓[/green] scoring_head.onnx ({path.stat().st_size / 1e6:.1f} MB)"
    )
    return path


def _g2_export_classifier_head(
    model: nn.Module,
    output_dir: Path,
    opset: int,
    hidden_size: int,
) -> Path:
    """Export classifier_head.onnx.

    The classifier is a 2-layer MLP (hidden_size → hidden_size*2 → 1, ReLU)
    that operates on label embeddings extracted at [L] token positions.
    """
    wrapper = ClassifierWrapper(model.classifier)
    wrapper.eval()

    num_labels = 3
    dummy_label_embs = torch.randn(num_labels, hidden_size)

    path = output_dir / "classifier_head.onnx"
    torch.onnx.export(
        wrapper,
        (dummy_label_embs,),
        str(path),
        opset_version=opset,
        dynamo=False,
        input_names=["label_embeddings"],
        output_names=["logits"],
        dynamic_axes={
            "label_embeddings": {0: "num_labels"},
            "logits": {0: "num_labels"},
        },
    )
    console.print(
        f"  [green]✓[/green] classifier_head.onnx ({path.stat().st_size / 1e6:.1f} MB)"
    )
    return path


def _g2_export_ner_full(
    model: nn.Module,
    output_dir: Path,
    opset: int,
    hidden_size: int,
    max_width: int,
    counting_layer: str,
) -> Path:
    """Export ner_full.onnx — encoder + gather + span_rep + scoring in one traced graph."""
    uses_span_idx = _detect_uses_span_idx(model.span_rep)
    wrapper = NerFullWrapper(
        EncoderWrapper(model.encoder),
        SpanRepWrapper(model.span_rep, uses_span_idx=uses_span_idx),
        ScoringHeadWrapper(
            count_pred=model.count_pred,
            count_embed=model.count_embed,
            max_count=20,
            counting_layer=counting_layer,
        ),
        hidden_size,
    )
    wrapper.eval()
    _patch_deberta_for_onnx()
    _disable_transformer_fast_path(wrapper)

    seq_len, text_len, num_fields = 40, 12, 3
    dummy = (
        torch.ones(1, seq_len, dtype=torch.long),  # input_ids
        torch.ones(1, seq_len, dtype=torch.long),  # attention_mask
        torch.arange(text_len, dtype=torch.long).unsqueeze(0),  # word_positions
        torch.tensor([1], dtype=torch.long),  # p_position
        torch.arange(2, 2 + num_fields, dtype=torch.long),  # field_positions
        torch.zeros(1, text_len * max_width, 2, dtype=torch.long),  # span_idx
        torch.tensor(1, dtype=torch.long),  # count
    )
    path = output_dir / "ner_full.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        opset_version=opset,
        dynamo=False,
        input_names=[
            "input_ids", "attention_mask", "word_positions",
            "p_position", "field_positions", "span_idx", "count",
        ],
        output_names=["count_logits", "span_scores"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "word_positions": {0: "batch", 1: "text_len"},
            "field_positions": {0: "num_fields"},
            "span_idx": {0: "batch", 1: "num_spans"},
            "span_scores": {0: "batch", 1: "count", 2: "num_fields", 3: "text_len"},
        },
    )
    console.print(f"  [green]✓[/green] ner_full.onnx ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def _g2_export_classifier_full(
    model: nn.Module,
    output_dir: Path,
    opset: int,
) -> Path:
    """Export classifier_full.onnx — encoder + label gather + classifier in one traced graph."""
    wrapper = ClassifierFullWrapper(EncoderWrapper(model.encoder), model.classifier)
    wrapper.eval()
    _patch_deberta_for_onnx()

    seq_len, num_labels = 40, 3
    dummy = (
        torch.ones(1, seq_len, dtype=torch.long),
        torch.ones(1, seq_len, dtype=torch.long),
        torch.arange(2, 2 + num_labels, dtype=torch.long),
    )
    path = output_dir / "classifier_full.onnx"
    torch.onnx.export(
        wrapper,
        dummy,
        str(path),
        opset_version=opset,
        dynamo=False,
        input_names=["input_ids", "attention_mask", "label_positions"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq_len"},
            "attention_mask": {0: "batch", 1: "seq_len"},
            "label_positions": {0: "num_labels"},
            "logits": {0: "batch_labels"},
        },
    )
    console.print(f"  [green]✓[/green] classifier_full.onnx ({path.stat().st_size / 1e6:.1f} MB)")
    return path


def _g2_write_config(
    model: nn.Module,
    output_dir: Path,
    counting_layer: str,
    token_pooling: str,
    hidden_size: int,
    max_width: int,
) -> None:
    """Write gliner4j_config.json with model metadata for the Java consumer."""
    uses_span_idx = _detect_uses_span_idx(model.span_rep)
    span_mode = model.span_rep.span_rep_layer.__class__.__name__

    # Map special tokens to their IDs
    tokenizer = model.processor.tokenizer
    special_tokens = {"P": "[P]", "C": "[C]", "E": "[E]", "R": "[R]", "L": "[L]"}
    special_token_ids = {
        k: tokenizer.convert_tokens_to_ids(v) for k, v in special_tokens.items()
    }

    config = {
        "hidden_size": hidden_size,
        "max_width": max_width,
        "max_count": 20,
        "span_mode": span_mode,
        "counting_layer": counting_layer,
        "token_pooling": token_pooling,
        "uses_span_idx": uses_span_idx,
        "special_tokens": special_tokens,
        "special_token_ids": special_token_ids,
    }

    config_path = output_dir / "gliner4j_config.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    console.print("  [green]✓[/green] gliner4j_config.json")


# ===========================================================================
# GLiNER2 — verification
# ===========================================================================


def _g2_verify_encoder(model: nn.Module, output_dir: Path) -> bool:
    """Verify encoder.onnx against PyTorch."""
    wrapper = EncoderWrapper(model.encoder)
    wrapper.eval()

    batch, seq_len = 1, 24
    dummy_ids = torch.ones(batch, seq_len, dtype=torch.long)
    dummy_mask = torch.ones(batch, seq_len, dtype=torch.long)

    with torch.no_grad():
        pt_out = wrapper(dummy_ids, dummy_mask).numpy()

    session = ort.InferenceSession(str(output_dir / "encoder.onnx"))
    onnx_out = session.run(
        None,
        {"input_ids": dummy_ids.numpy(), "attention_mask": dummy_mask.numpy()},
    )[0]

    ok = np.allclose(pt_out, onnx_out, atol=1e-4)
    _print_verify("encoder.onnx", ok, pt_out, onnx_out)
    return ok


def _g2_verify_span_rep(
    model: nn.Module, output_dir: Path, hidden_size: int, max_width: int
) -> bool:
    """Verify span_rep.onnx against PyTorch."""
    uses_span_idx = _detect_uses_span_idx(model.span_rep)
    wrapper = SpanRepWrapper(model.span_rep, uses_span_idx=uses_span_idx)
    wrapper.eval()

    text_len = 10
    dummy_emb = torch.randn(1, text_len, hidden_size)

    feeds: dict[str, np.ndarray] = {"token_embeddings": dummy_emb.numpy()}

    if uses_span_idx:
        num_spans = text_len * max_width
        dummy_span_idx = torch.zeros(1, num_spans, 2, dtype=torch.long)
        for i in range(text_len):
            for j in range(max_width):
                idx = i * max_width + j
                if i + j < text_len:
                    dummy_span_idx[0, idx, 0] = i
                    dummy_span_idx[0, idx, 1] = i + j
        feeds["span_idx"] = dummy_span_idx.numpy()
        with torch.no_grad():
            pt_out = wrapper(dummy_emb, dummy_span_idx).numpy()
    else:
        with torch.no_grad():
            pt_out = wrapper(dummy_emb).numpy()

    session = ort.InferenceSession(str(output_dir / "span_rep.onnx"))
    onnx_out = session.run(None, feeds)[0]

    # Looser atol than encoder/scoring_head: span_rep cascades multiple ops
    # over random inputs and accumulates fp32 drift of ~1e-3 (still 1000x
    # below any meaningful behavioral difference).
    ok = np.allclose(pt_out, onnx_out, atol=1e-3)
    _print_verify("span_rep.onnx", ok, pt_out, onnx_out)
    return ok


def _g2_verify_scoring_head(
    model: nn.Module,
    output_dir: Path,
    hidden_size: int,
    max_width: int,
    counting_layer: str,
) -> bool:
    """Verify scoring_head.onnx against PyTorch."""
    wrapper = ScoringHeadWrapper(
        count_pred=model.count_pred,
        count_embed=model.count_embed,
        max_count=20,
        counting_layer=counting_layer,
    )
    wrapper.eval()

    batch = 3
    text_len = 10
    num_fields = 2
    dummy_span_rep = torch.randn(batch, text_len, max_width, hidden_size)
    dummy_p_emb = torch.randn(hidden_size)
    dummy_fields = torch.randn(num_fields, hidden_size)
    dummy_count = torch.tensor(3, dtype=torch.long)

    with torch.no_grad():
        pt_logits, pt_scores = wrapper(
            dummy_span_rep, dummy_p_emb, dummy_fields, dummy_count
        )
        pt_logits = pt_logits.numpy()
        pt_scores = pt_scores.numpy()

    session = ort.InferenceSession(str(output_dir / "scoring_head.onnx"))
    onnx_logits, onnx_scores = session.run(
        None,
        {
            "span_rep": dummy_span_rep.numpy(),
            "schema_emb_p": dummy_p_emb.numpy(),
            "schema_emb_fields": dummy_fields.numpy(),
            "count": dummy_count.numpy(),
        },
    )

    ok_logits = np.allclose(pt_logits, onnx_logits, atol=1e-4)
    ok_scores = np.allclose(pt_scores, onnx_scores, atol=1e-4)
    ok = ok_logits and ok_scores

    _print_verify("scoring_head.onnx (logits)", ok_logits, pt_logits, onnx_logits)
    _print_verify("scoring_head.onnx (scores)", ok_scores, pt_scores, onnx_scores)
    return ok


def _g2_verify_classifier_head(
    model: nn.Module, output_dir: Path, hidden_size: int
) -> bool:
    """Verify classifier_head.onnx against PyTorch."""
    wrapper = ClassifierWrapper(model.classifier)
    wrapper.eval()

    num_labels = 4
    dummy_label_embs = torch.randn(num_labels, hidden_size)

    with torch.no_grad():
        pt_out = wrapper(dummy_label_embs).numpy()

    session = ort.InferenceSession(str(output_dir / "classifier_head.onnx"))
    onnx_out = session.run(
        None,
        {"label_embeddings": dummy_label_embs.numpy()},
    )[0]

    ok = np.allclose(pt_out, onnx_out, atol=1e-4)
    _print_verify("classifier_head.onnx", ok, pt_out, onnx_out)
    return ok


# ===========================================================================
# GLiNER2 — CLI command
# ===========================================================================


@app.command("gliner2")
def export_gliner2(
    model_path: Annotated[
        str, typer.Option("--model-path", help="GLiNER2 model dir or HuggingFace repo ID")
    ],
    output_dir: Annotated[
        str, typer.Option("--output-dir", help="Output directory for ONNX models")
    ],
    opset: Annotated[int, typer.Option("--opset", help="ONNX opset version")] = 17,
    verify: Annotated[
        bool, typer.Option("--verify/--no-verify", help="Run verification after export")
    ] = True,
    variants: Annotated[
        list[Variant] | None,
        typer.Option("--variant", help="Additional variants to generate (fp16, quantized)"),
    ] = None,
    keep_split: Annotated[
        bool,
        typer.Option(
            "--keep-split/--no-keep-split",
            help="Keep the intermediate split graphs (encoder/span_rep/scoring_head/"
            "classifier_head) after export — needed if build_merged_graphs.py will run next",
        ),
    ] = False,
) -> None:
    """Export a GLiNER2 PyTorch model to ONNX models for gliner4j.

    Base FP32 models are always exported to {output_dir}/onnx/.
    Additional variants (fp16, quantized) are placed in {output_dir}/onnx_{variant}/.
    Config and tokenizer are written to {output_dir}/ (shared across variants).
    The final deployment artifacts are ner_full.onnx + classifier_full.onnx; the split
    graphs are intermediates and are removed after export unless --keep-split.
    """
    # Import directly from gliner2.model to avoid __init__ pulling in api_client
    from gliner2.model import Extractor  # noqa: E402

    out = Path(output_dir)
    base_dir = out / "onnx"
    base_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Load model
    console.print(f"\n[bold]Loading model from {model_path}...[/bold]")
    model = Extractor.from_pretrained(model_path)
    model.eval()

    hidden_size: int = model.hidden_size
    max_width: int = model.max_width
    counting_layer: str = model.config.counting_layer
    token_pooling: str = model.config.token_pooling

    console.print(
        f"  hidden_size={hidden_size}, max_width={max_width}, "
        f"counting_layer={counting_layer}, token_pooling={token_pooling}"
    )

    # Step 2–4: Export base ONNX models to onnx/ subfolder
    console.print(f"\n[bold]Exporting ONNX models to {base_dir} (opset={opset})...[/bold]")

    with torch.no_grad():
        _g2_export_encoder(model, base_dir, opset)
        _g2_export_span_rep(model, base_dir, opset, hidden_size, max_width)
        _g2_export_scoring_head(model, base_dir, opset, hidden_size, max_width, counting_layer)
        _g2_export_classifier_head(model, base_dir, opset, hidden_size)
        # Single-graph exports (encoder + gather + heads) — the deployment artifacts.
        _g2_export_ner_full(model, base_dir, opset, hidden_size, max_width, counting_layer)
        _g2_export_classifier_full(model, base_dir, opset)

    # Step 4b: Strip '/' from tensor names so the models run on the OpenVINO EP.
    # Done before the fp16/quantized conversions so every variant inherits the
    # clean names (see _sanitize_onnx_names for the OpenVINO EP details).
    console.print("\n[bold]Sanitizing tensor names for OpenVINO...[/bold]")
    for name in GLINER2_MODEL_FILES:
        renamed = _sanitize_onnx_names(base_dir / name)
        console.print(f"  [green]✓[/green] {name} ({renamed} names rewritten)")

    # Step 5: Write config to root
    console.print("\n[bold]Writing config and tokenizer...[/bold]")
    _g2_write_config(model, out, counting_layer, token_pooling, hidden_size, max_width)

    # Step 6: Copy tokenizer to root
    model.processor.tokenizer.save_pretrained(str(out))
    console.print("  [green]✓[/green] tokenizer files")

    # Step 7: Verification of base models
    if verify:
        console.print("\n[bold]Verifying base ONNX models...[/bold]")
        all_ok = True
        all_ok &= _g2_verify_encoder(model, base_dir)
        all_ok &= _g2_verify_span_rep(model, base_dir, hidden_size, max_width)
        all_ok &= _g2_verify_scoring_head(model, base_dir, hidden_size, max_width, counting_layer)
        all_ok &= _g2_verify_classifier_head(model, base_dir, hidden_size)

        if all_ok:
            console.print("\n[bold green]Base model verifications passed![/bold green]")
        else:
            console.print("\n[bold red]Some verifications failed![/bold red]")
            raise typer.Exit(code=1)

    # Step 8: Generate additional variants. GLiNER2 keeps QUInt8 weights — the
    # historical choice validated against its unrolled-GRU scoring head.
    from onnxruntime.quantization import QuantType

    _generate_variants(variants, out, base_dir, GLINER2_MODEL_FILES, QuantType.QUInt8)

    # Step 9: Drop the intermediate split graphs — verification (step 7) and the
    # variant conversions (step 8) have already consumed them, and the runtime
    # only loads the merged deployment artifacts.
    if not keep_split:
        console.print("\n[bold]Removing intermediate split graphs (--keep-split to retain)...[/bold]")
        intermediates = tuple(f for f in GLINER2_MODEL_FILES if f not in GLINER2_FINAL_FILES)
        for variant_dir in (base_dir, *(out / f"onnx_{v.value}" for v in variants or [])):
            for name in intermediates:
                target = variant_dir / name
                if target.exists():
                    target.unlink()
                    console.print(f"  [green]✓[/green] removed {target.relative_to(out)}")

    # Summary
    console.print(f"\n[bold]Output structure in {out}:[/bold]")
    _print_tree(out)


# ===========================================================================
# GLiClass (uni-encoder) — wrappers & export
# ===========================================================================

# The two graphs that make up a GLiClass bundle (base + every variant).
GLICLASS_MODEL_FILES = ("encoder.onnx", "score_head.onnx")


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


def _gc_export_encoder(model: nn.Module, out_dir: Path, opset: int) -> Path:
    encoder = model.encoder_model
    _prepare_modernbert_for_onnx(encoder)
    # DeBERTa backbones emit per-layer dynamic-rank `If` nodes from their scripted relative-position
    # helper; these break FP16 conversion and the OpenVINO CPU plugin. Bake in the single branch.
    patched = _patch_deberta_for_onnx()
    if patched:
        console.print(f"  patched {patched} DeBERTa relative-position helper(s)")
    wrapper = EncoderWrapper(encoder).eval()
    return _export_encoder_graph(wrapper, out_dir / "encoder.onnx", opset)


def _gc_export_score_head(
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


def _gc_write_config(model: nn.Module, tokenizer, out_dir: Path, enc_hidden: int) -> None:
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


def _gc_verify(
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


@app.command("gliclass")
def export_gliclass(
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
            f"This command handles uni-encoder GLiClass only; got "
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
        _gc_export_encoder(uni, onnx_dir, opset)
        _gc_export_score_head(uni, onnx_dir, opset, enc_hidden)

    # Strip '/' from tensor names for the OpenVINO EP — before variants so they inherit clean names.
    for fname in GLICLASS_MODEL_FILES:
        renamed = _sanitize_onnx_names(onnx_dir / fname)
        console.print(f"  [green]✓[/green] sanitized {fname} ({renamed} names)")

    console.print("\n[bold]Writing config and tokenizer...[/bold]")
    _gc_write_config(model, tokenizer, out, enc_hidden)

    if verify:
        console.print("\n[bold]Verifying (PyTorch vs ONNX)...[/bold]")
        ok = _gc_verify(model, tokenizer, onnx_dir, model.config.class_token_index)
        if not ok:
            console.print("\n[bold red]Verification FAILED[/bold red]")
            raise typer.Exit(code=1)
        console.print("\n[bold green]Verification passed![/bold green]")

    # QInt8 (signed, symmetric) per-channel weights — the accuracy-preserving config for
    # transformer weights. QUInt8 (unsigned/asymmetric) is a known de-calibration pitfall on
    # transformers; per_channel + reduce_range alone don't compensate for it.
    from onnxruntime.quantization import QuantType

    _generate_variants(variants, out, onnx_dir, GLICLASS_MODEL_FILES, QuantType.QInt8)


# ===========================================================================
# Original GLiNER (uni/bi-encoder) — export
# ===========================================================================

# The gliner uni/bi span model is a single monolithic graph.
GLINER_UNI_MODEL_FILES = ("model.onnx",)


def _gu_write_config(gliner_config: dict, out_dir: Path) -> None:
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


@app.command("gliner-uni")
def export_gliner_uni(
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
    _gu_write_config(gliner_cfg, out)

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

    # QInt8 (signed) per-channel — accuracy-preserving for transformer weights (QUInt8 is the
    # de-calibration pitfall; gliner's own quantize=True uses QUInt8, which is why we do our own).
    from onnxruntime.quantization import QuantType

    _generate_variants(variants, out, onnx_dir, GLINER_UNI_MODEL_FILES, QuantType.QInt8)

    console.print(f"\n[bold]Bundle ready at {out} (onnx/model.onnx)[/bold]")


if __name__ == "__main__":
    app()
