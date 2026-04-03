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
#     "gliner>=0.2",
#     "typer>=0.15",
#     "numpy>=1.26",
#     "rich>=13",
#     "huggingface-hub>=0.20",
#     "pydantic>=2.0",
#     "urllib3>=2.0",
#     "requests>=2.31",
#     "onnxscript>=0.1",
#     "onnxconverter-common>=1.14",
# ]
# ///
"""GLiNER2 → ONNX export script for gliner4j.

Exports a GLiNER2 PyTorch model into 4 ONNX models organized by variant:
  - onnx/              Base FP32 models
  - onnx_fp16/         FP16 converted models (optional)
  - onnx_quantized/    INT8 dynamically-quantized models (optional)

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
import sys
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

# ---------------------------------------------------------------------------
# Ensure GLiNER2 repo is importable (it lives next to gliner4j)
# ---------------------------------------------------------------------------
_GLINER2_ROOT = Path(__file__).resolve().parent.parent.parent / "GLiNER2"
if _GLINER2_ROOT.exists():
    sys.path.insert(0, str(_GLINER2_ROOT))

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

    The key challenge is that CountLSTM/v2/MoE use a Python int `count` to
    control torch.arange, producing data-dependent shapes. We solve this by
    always unrolling the GRU for max_count steps, then slicing to `count`.
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

        h0 = schema_emb_fields.unsqueeze(0)  # (1, M, D)
        output, _ = self.gru(pos_seq, h0)  # (max_count, M, D)

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
    console.print(f"  [green]✓[/green] encoder.onnx ({path.stat().st_size / 1e6:.1f} MB)")
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

    ok = np.allclose(pt_out, onnx_out, atol=1e-4)
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


def _convert_to_fp16(base_dir: Path, output_dir: Path) -> None:
    """Convert base ONNX models to FP16.

    Args:
        base_dir: Directory containing base FP32 ONNX models.
        output_dir: Directory to write FP16 models.
    """
    from onnxconverter_common import float16

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_MODEL_FILES:
        model = onnx.load(str(base_dir / name))
        model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
        onnx.save(model_fp16, str(output_dir / name))
        size_mb = (output_dir / name).stat().st_size / 1e6
        console.print(f"  [green]✓[/green] {name} ({size_mb:.1f} MB)")


def _convert_to_quantized(base_dir: Path, output_dir: Path) -> None:
    """Convert base ONNX models to INT8 dynamic quantization.

    Args:
        base_dir: Directory containing base FP32 ONNX models.
        output_dir: Directory to write quantized models.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in ONNX_MODEL_FILES:
        quantize_dynamic(
            model_input=base_dir / name,
            model_output=output_dir / name,
            weight_type=QuantType.QInt8,
        )
        size_mb = (output_dir / name).stat().st_size / 1e6
        console.print(f"  [green]✓[/green] {name} ({size_mb:.1f} MB)")


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
