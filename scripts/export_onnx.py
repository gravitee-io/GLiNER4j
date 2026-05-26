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
#     "transformers>=4.40",
#     "safetensors>=0.4",
#     "typer>=0.15",
#     "numpy>=1.26",
#     "rich>=13",
#     "huggingface-hub>=0.20",
#     "pydantic>=2.0",
#     "urllib3>=2.0",
#     "requests>=2.31",
#     "onnxscript>=0.1",
#     "onnxconverter-common>=1.14",
#     "gliner2",
# ]
# ///
"""GLiNER2 → ONNX export script for gliner4j.

Exports a GLiNER2 PyTorch model into 3 ONNX variants:
  - onnx/              Base FP32 models
  - onnx_fp16/         FP16 converted models (optional)
  - onnx_quantized/    INT8 dynamically-quantized models (optional, most compact)

Each variant folder contains:
  - encoder.onnx:          Transformer encoder (input_ids → last_hidden_state)
  - span_rep.onnx:         Span representation layer
  - scoring_head.onnx:     Count-aware scoring head
  - classifier_head.onnx:  Classifier MLP for text classification

Shared files (config, tokenizer) live at the output root.

Usage:
    uv run scripts/export_onnx.py --model-path fastino-ai/gliner2-base --output-dir models/out
    uv run scripts/export_onnx.py --model-path fastino-ai/gliner2-base --output-dir models/out --variants fp16 quantized
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


# ===========================================================================
# ONNX Wrapper Modules
# ===========================================================================


class EncoderWrapper(nn.Module):
    """Wraps the HuggingFace encoder for clean ONNX export."""

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
            span_rep: Span representations, shape (text_len, max_width, hidden).
            schema_emb_p: The [P] token embedding, shape (hidden,).
            schema_emb_fields: Field embeddings, shape (num_fields, hidden).
            count: Scalar int64 tensor — predicted count.

        Returns:
            count_logits: Shape (1, 20).
            span_scores: Shape (count, num_fields, text_len, max_width).
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

        # Score via einsum + sigmoid
        span_scores = torch.sigmoid(
            torch.einsum("lkd,bpd->bplk", span_rep, struct_proj)
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


# ===========================================================================
# Helpers
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
    node therefore has dynamic rank, which OpenVINO's CPU plugin cannot
    compile::

        CPU plug-in doesn't support If operation with dynamic rank

    Our encoder always runs plain square self-attention, so ``query_size ==
    key_size`` and the ``else`` branch (``return relative_pos``) is always
    taken. Replacing the scripted ``build_rpos`` with a plain Python equivalent
    lets the tracer evaluate the size comparison at export time and bake in that
    single branch — no ``If`` nodes — with identical numerics (kept honest by
    the ``_verify_encoder`` PT-vs-ONNX check, atol 1e-4).

    Also unwraps any ``@torch.jit.script_if_tracing`` helpers, which older
    transformers releases used for the same relative-position code.

    Returns the number of helpers patched.
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


# ===========================================================================
# Export Functions
# ===========================================================================


def _export_encoder(
    model: nn.Module,
    output_dir: Path,
    opset: int,
    hidden_size: int,
) -> Path:
    """Export encoder.onnx.

    Args:
        model: The full Extractor model.
        output_dir: Directory to write the ONNX file.
        opset: ONNX opset version.
        hidden_size: Hidden size for verification.

    Returns:
        Path to the exported ONNX file.
    """
    wrapper = EncoderWrapper(model.encoder)
    wrapper.eval()

    # DeBERTa backbones emit per-layer `If` nodes (dynamic rank) from their
    # @torch.jit.script_if_tracing relative-position helper, which OpenVINO's
    # CPU plugin cannot compile. Unwrap them so the export bakes in the single
    # static branch our usage always takes. No-op for non-DeBERTa backbones.
    patched = _patch_deberta_for_onnx()
    if patched:
        console.print(f"  patched {patched} DeBERTa relative-position helper(s) for ONNX")

    batch, seq_len = 1, 32
    dummy_ids = torch.ones(batch, seq_len, dtype=torch.long)
    dummy_mask = torch.ones(batch, seq_len, dtype=torch.long)

    path = output_dir / "encoder.onnx"
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
        f", [green]0 If nodes[/green]"
        if if_nodes == 0
        else f", [red]{if_nodes} If nodes remain (OpenVINO will reject)[/red]"
    )
    console.print(f"  [green]✓[/green] encoder.onnx ({path.stat().st_size / 1e6:.1f} MB{if_note})")
    return path


def _detect_uses_span_idx(span_rep: nn.Module) -> bool:
    """Detect whether the span_rep layer requires span_idx.

    Args:
        span_rep: The SpanRepLayer module.

    Returns:
        True if the underlying layer is marker-based and needs span_idx.
    """
    class_name = span_rep.span_rep_layer.__class__.__name__
    return class_name in ("SpanMarker", "SpanMarkerV0", "SpanMarkerV1")


def _export_span_rep(
    model: nn.Module,
    output_dir: Path,
    opset: int,
    hidden_size: int,
    max_width: int,
) -> Path:
    """Export span_rep.onnx.

    Args:
        model: The full Extractor model.
        output_dir: Directory to write the ONNX file.
        opset: ONNX opset version.
        hidden_size: Model hidden size.
        max_width: Maximum span width.

    Returns:
        Path to the exported ONNX file.
    """
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


def _export_scoring_head(
    model: nn.Module,
    output_dir: Path,
    opset: int,
    hidden_size: int,
    max_width: int,
    counting_layer: str,
) -> Path:
    """Export scoring_head.onnx.

    Args:
        model: The full Extractor model.
        output_dir: Directory to write the ONNX file.
        opset: ONNX opset version.
        hidden_size: Model hidden size.
        max_width: Maximum span width.
        counting_layer: Type of counting layer (count_lstm, count_lstm_v2, count_lstm_moe).

    Returns:
        Path to the exported ONNX file.
    """
    wrapper = ScoringHeadWrapper(
        count_pred=model.count_pred,
        count_embed=model.count_embed,
        max_count=20,
        counting_layer=counting_layer,
    )
    wrapper.eval()

    # Disable fused TransformerEncoder fast-path so TorchScript can trace it
    _disable_transformer_fast_path(wrapper)

    text_len = 16
    num_fields = 3
    dummy_span_rep = torch.randn(text_len, max_width, hidden_size)
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
            "span_rep": {0: "text_len"},
            "schema_emb_fields": {0: "num_fields"},
            "count_logits": {},
            "span_scores": {0: "count", 1: "num_fields", 2: "text_len"},
        },
    )
    console.print(
        f"  [green]✓[/green] scoring_head.onnx ({path.stat().st_size / 1e6:.1f} MB)"
    )
    return path


def _export_classifier_head(
    model: nn.Module,
    output_dir: Path,
    opset: int,
    hidden_size: int,
) -> Path:
    """Export classifier_head.onnx.

    The classifier is a 2-layer MLP (hidden_size → hidden_size*2 → 1, ReLU)
    that operates on label embeddings extracted at [L] token positions.

    Args:
        model: The full Extractor model.
        output_dir: Directory to write the ONNX file.
        opset: ONNX opset version.
        hidden_size: Model hidden size.

    Returns:
        Path to the exported ONNX file.
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


# ===========================================================================
# Config & Tokenizer
# ===========================================================================


def _write_config(
    model: nn.Module,
    output_dir: Path,
    counting_layer: str,
    token_pooling: str,
    hidden_size: int,
    max_width: int,
) -> None:
    """Write gliner4j_config.json with model metadata for the Java consumer.

    Args:
        model: The full Extractor model.
        output_dir: Directory to write the config file.
        counting_layer: Type of counting layer.
        token_pooling: Token pooling strategy.
        hidden_size: Model hidden size.
        max_width: Maximum span width.
    """
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
    console.print(f"  [green]✓[/green] gliner4j_config.json")


def _copy_tokenizer(model: nn.Module, output_dir: Path) -> None:
    """Save HuggingFace tokenizer files to the output directory.

    Args:
        model: The full Extractor model.
        output_dir: Directory to save tokenizer files.
    """
    model.processor.tokenizer.save_pretrained(str(output_dir))
    console.print(f"  [green]✓[/green] tokenizer files")


# ===========================================================================
# Verification
# ===========================================================================


def _verify_encoder(model: nn.Module, output_dir: Path, hidden_size: int) -> bool:
    """Verify encoder.onnx against PyTorch.

    Args:
        model: The full Extractor model.
        output_dir: Directory containing the ONNX file.
        hidden_size: Model hidden size.

    Returns:
        True if verification passes.
    """
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


def _verify_span_rep(
    model: nn.Module, output_dir: Path, hidden_size: int, max_width: int
) -> bool:
    """Verify span_rep.onnx against PyTorch.

    Args:
        model: The full Extractor model.
        output_dir: Directory containing the ONNX file.
        hidden_size: Model hidden size.
        max_width: Maximum span width.

    Returns:
        True if verification passes.
    """
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


def _verify_scoring_head(
    model: nn.Module,
    output_dir: Path,
    hidden_size: int,
    max_width: int,
    counting_layer: str,
) -> bool:
    """Verify scoring_head.onnx against PyTorch.

    Args:
        model: The full Extractor model.
        output_dir: Directory containing the ONNX file.
        hidden_size: Model hidden size.
        max_width: Maximum span width.
        counting_layer: Type of counting layer.

    Returns:
        True if verification passes.
    """
    wrapper = ScoringHeadWrapper(
        count_pred=model.count_pred,
        count_embed=model.count_embed,
        max_count=20,
        counting_layer=counting_layer,
    )
    wrapper.eval()

    text_len = 10
    num_fields = 2
    dummy_span_rep = torch.randn(text_len, max_width, hidden_size)
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


def _verify_classifier_head(
    model: nn.Module, output_dir: Path, hidden_size: int
) -> bool:
    """Verify classifier_head.onnx against PyTorch.

    Args:
        model: The full Extractor model.
        output_dir: Directory containing the ONNX file.
        hidden_size: Model hidden size.

    Returns:
        True if verification passes.
    """
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


def _print_verify(
    name: str, ok: bool, pt: np.ndarray, onnx_val: np.ndarray
) -> None:
    """Print verification result.

    Args:
        name: Model name for display.
        ok: Whether verification passed.
        pt: PyTorch output array.
        onnx_val: ONNX output array.
    """
    max_diff = float(np.max(np.abs(pt - onnx_val)))
    status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
    console.print(f"  {status} {name} (max_diff={max_diff:.6f})")


# ===========================================================================
# Variant Conversion
# ===========================================================================

ONNX_MODEL_FILES = ("encoder.onnx", "span_rep.onnx", "scoring_head.onnx", "classifier_head.onnx")


class Variant(str, Enum):
    """ONNX model variants that can be generated from the base export."""

    fp16 = "fp16"
    quantized = "quantized"


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


def _convert_to_fp16(base_dir: Path, output_dir: Path) -> None:
    """Convert base ONNX models to FP16.

    Args:
        base_dir: Directory containing base FP32 ONNX models.
        output_dir: Directory to write FP16 models.
    """
    import shutil

    from onnxconverter_common import float16

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_MODEL_FILES:
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
        # (_verify_*) only checks the *base* FP32 models numerically — the
        # fp16/quantized variants are never otherwise load-tested, so without
        # this check a non-loadable variant ships silently. That is exactly how
        # the Cast/value_info mismatch above went unnoticed until a consumer
        # first tried to load an fp16 model.
        try:
            ort.InferenceSession(str(output_dir / name))
            note = f" (deduped {removed}, realigned {realigned} casts)" if removed or realigned else ""
        except Exception as e:
            shutil.copy(base_dir / name, output_dir / name)
            note = " (fp16 invalid, copied FP32 instead)"
            console.print(f"  [yellow]![/yellow] {name}: {str(e).splitlines()[0]}")

        size_mb = (output_dir / name).stat().st_size / 1e6
        marker = "[yellow]→[/yellow]" if "invalid" in note else "[green]✓[/green]"
        console.print(f"  {marker} {name} ({size_mb:.1f} MB){note}")


def _convert_to_quantized(base_dir: Path, output_dir: Path) -> None:
    """Convert base ONNX models to INT8 dynamic quantization.

    Args:
        base_dir: Directory containing base FP32 ONNX models.
        output_dir: Directory to write quantized models.
    """
    import shutil

    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_MODEL_FILES:
        # QUInt8 (unsigned) weights, per-tensor quantization (per_channel=False).
        # DefaultTensorType handles nodes whose type shape-inference can't
        # determine — e.g. matmul outputs inside an unrolled GRU subgraph.
        quantize_dynamic(
            model_input=base_dir / name,
            model_output=output_dir / name,
            weight_type=QuantType.QUInt8,
            per_channel=True,
            reduce_range=True,
            extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT},
        )

        # Validate: some upstream subgraphs (notably nn.GRU's unrolled form in
        # scoring_head for the base model) produce MatMulInteger nodes with
        # shapes ORT cannot reconcile at load time. If the quantized file
        # fails to instantiate, fall back to shipping the FP32 file instead.
        try:
            ort.InferenceSession(str(output_dir / name))
            note = ""
        except Exception as e:
            (output_dir / name).unlink(missing_ok=True)
            shutil.copy(base_dir / name, output_dir / name)
            note = " (quantization invalid, copied FP32 instead)"
            console.print(f"  [yellow]![/yellow] {name}: {e}")

        # Re-sanitize: quantize_dynamic can mint new '/'-named tensors.
        _sanitize_onnx_names(output_dir / name)

        size_mb = (output_dir / name).stat().st_size / 1e6
        marker = "[yellow]→[/yellow]" if note else "[green]✓[/green]"
        console.print(f"  {marker} {name} ({size_mb:.1f} MB){note}")


# ===========================================================================
# CLI
# ===========================================================================

app = typer.Typer(add_completion=False)


@app.command()
def export(
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
) -> None:
    """Export a GLiNER2 PyTorch model to ONNX models for gliner4j.

    Base FP32 models are always exported to {output_dir}/onnx/.
    Additional variants (fp16, quantized) are placed in {output_dir}/onnx_{variant}/.
    Config and tokenizer are written to {output_dir}/ (shared across variants).
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
        _export_encoder(model, base_dir, opset, hidden_size)
        _export_span_rep(model, base_dir, opset, hidden_size, max_width)
        _export_scoring_head(model, base_dir, opset, hidden_size, max_width, counting_layer)
        _export_classifier_head(model, base_dir, opset, hidden_size)

    # Step 4b: Strip '/' from tensor names so the models run on the OpenVINO EP.
    # Done before the fp16/quantized conversions so every variant inherits the
    # clean names (see _sanitize_onnx_names for the OpenVINO EP details).
    console.print(f"\n[bold]Sanitizing tensor names for OpenVINO...[/bold]")
    for name in ONNX_MODEL_FILES:
        renamed = _sanitize_onnx_names(base_dir / name)
        console.print(f"  [green]✓[/green] {name} ({renamed} names rewritten)")

    # Step 5: Write config to root
    console.print(f"\n[bold]Writing config and tokenizer...[/bold]")
    _write_config(model, out, counting_layer, token_pooling, hidden_size, max_width)

    # Step 6: Copy tokenizer to root
    _copy_tokenizer(model, out)

    # Step 7: Verification of base models
    if verify:
        console.print(f"\n[bold]Verifying base ONNX models...[/bold]")
        all_ok = True
        all_ok &= _verify_encoder(model, base_dir, hidden_size)
        all_ok &= _verify_span_rep(model, base_dir, hidden_size, max_width)
        all_ok &= _verify_scoring_head(model, base_dir, hidden_size, max_width, counting_layer)
        all_ok &= _verify_classifier_head(model, base_dir, hidden_size)

        if all_ok:
            console.print("\n[bold green]Base model verifications passed![/bold green]")
        else:
            console.print("\n[bold red]Some verifications failed![/bold red]")
            raise typer.Exit(code=1)

    # Step 8: Generate additional variants
    if variants:
        for variant in variants:
            variant_dir = out / f"onnx_{variant.value}"
            console.print(f"\n[bold]Generating {variant.value} variant → {variant_dir}...[/bold]")
            if variant == Variant.fp16:
                _convert_to_fp16(base_dir, variant_dir)
            elif variant == Variant.quantized:
                _convert_to_quantized(base_dir, variant_dir)

    # Summary
    console.print(f"\n[bold]Output structure in {out}:[/bold]")
    _print_tree(out)


def _print_tree(root: Path, prefix: str = "  ") -> None:
    """Print a simple directory tree.

    Args:
        root: Root directory to display.
        prefix: Indentation prefix.
    """
    entries = sorted(root.iterdir(), key=lambda p: (not p.is_dir(), p.name))
    for entry in entries:
        if entry.is_dir():
            console.print(f"{prefix}[bold]{entry.name}/[/bold]")
            _print_tree(entry, prefix + "  ")
        else:
            size = entry.stat().st_size
            label = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB"
            console.print(f"{prefix}{entry.name} ({label})")


if __name__ == "__main__":
    app()
