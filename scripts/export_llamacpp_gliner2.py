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
#     "transformers>=4.48,<5",
#     "safetensors>=0.4",
#     "sentencepiece>=0.2",
#     "protobuf>=4",
#     "typer>=0.15",
#     "numpy>=1.26",
#     "rich>=13",
#     "huggingface-hub>=0.20",
#     "gguf>=0.19",
#     "gliner2[local]>=2.0.0",
# ]
# ///
"""GLiNER2 and GLiNER2.5 (fastino) → llama.cpp/ggml engine bundles (``"engine": "llamacpp"``).

Separate from ``export_llamacpp.py`` only because the ``gliner2`` package pins ``transformers<5``
while gliclass / gliner need ``>=5``; the GGUF helpers are imported from there.

    uv run scripts/export_llamacpp_gliner2.py gliner2 --model-path models/gliner2-base-hf --output-dir models/gliner2-base-llamacpp
    uv run scripts/export_llamacpp_gliner2.py gliner2dot5 --model-path models/gliner2dot5-small-hf --output-dir models/gliner2dot5-small-llamacpp
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import typer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_llamacpp import HeadsWriter, _print_tree, _short_name, add_deberta_encoder, console  # noqa: E402

app = typer.Typer(add_completion=False, help=__doc__)


@app.callback()
def _main() -> None:
    """fastino families for the llama.cpp / ggml engine."""


# ===========================================================================
# gliner2 (fastino span model)
# ===========================================================================

GLINER2_REFERENCE = [
    ("John Smith works at Acme Corp in Berlin since 2019.", ["person", "organization", "location", "date"]),
    (
        "Contact Dr. Maria Lopez (maria.lopez@clinic.org, +34 600 123 456) at Hospital del Mar before 12 March 2025.",
        ["person", "email", "phone number", "organization", "date"],
    ),
    (
        "Apple unveiled the iPhone 16 in Cupertino, and Tim Cook said sales in China rose 8% last quarter.",
        ["company", "product", "location", "person", "percentage"],
    ),
    ("ok", ["person"]),
]

_GLINER2_RENAMES = {
    "span_rep.span_rep_layer.": "span_rep.",
    "count_embed.transformer.transformer.layers.": "count_tf.layers.",
    "count_embed.transformer.": "count_tf.",
}


@app.command("gliner2")
def gliner2(
    model_path: str = typer.Option(..., help="HF id or dir of a GLiNER2 checkpoint (fastino/gliner2-*)"),
    output_dir: Path = typer.Option(..., help="Bundle directory to write"),
    dtype: str = typer.Option("f16", help="f16 or f32 for the encoder / head matrices"),
    quant: list[str] = typer.Option(
        [], "--quant", help="also write gguf/model-<quant>.gguf with block-quantised encoder matrices (q8_0, q5_0, q4_0)"
    ),
):
    """GLiNER2 → one model.gguf (DeBERTa encoder + span/count/classifier heads) bundle."""
    from gliner2 import GLiNER2

    console.print(f"[bold]GLiNER2 → ggml[/bold]  {model_path}")
    model = GLiNER2.from_pretrained(model_path).eval()
    core = getattr(model, "model", model)
    cfg = core.config
    d = cfg.to_dict() if hasattr(cfg, "to_dict") else dict(vars(cfg))
    counting = d.get("counting_layer", "count_lstm")
    if d.get("token_pooling", "first") != "first":
        raise typer.BadParameter("only token_pooling=first")
    if type(core.span_rep.span_rep_layer).__name__ != "SpanMarkerV0":
        raise typer.BadParameter(f"span_rep {type(core.span_rep.span_rep_layer).__name__}: only SpanMarkerV0")
    if counting not in ("count_lstm", "count_lstm_v2"):
        raise typer.BadParameter(f"counting_layer={counting}: only count_lstm / count_lstm_v2")
    encoder = core.encoder
    # The processor's (converted, fast) tokenizer is what the gliner2 runtime uses; its normalizer
    # differs from the hub's tokenizer.json, so save it — not AutoTokenizer — like export_onnx.py.
    tokenizer = model.processor.tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)

    max_count = int(core.count_embed.pos_embedding.weight.shape[0])
    hidden = int(encoder.config.hidden_size)
    head_state = {
        _short_name(k, _GLINER2_RENAMES): v
        for k, v in core.state_dict().items()
        if not k.startswith("encoder.")
    }
    with torch.no_grad():
        gru = core.count_embed.gru
        # input side of the GRU for all count steps at once: pos_emb @ W_ihᵀ + b_ih  → (max_count, 3·hidden)
        head_state["count_embed.gi"] = core.count_embed.pos_embedding.weight @ gru.weight_ih_l0.T + gru.bias_ih_l0

    # Head matrices the Java graph only multiplies with (GgmlWeights.linear / projection, the count
    # GRU's W_hh, the count transformer's in_proj): f16 → tensor-core GEMMs on CUDA. count_embed.gi,
    # pos_embedding, LayerNorm params and biases are read element-wise and stay f32.
    def head_f16(name: str, arr) -> bool:
        if dtype != "f16":
            return False
        return (
            name.startswith(("span_rep.", "classifier.", "count_pred.", "count_tf.")) and name.endswith(".weight")
        ) or name.endswith("in_proj_weight") or name == "count_embed.gru.weight_hh_l0"

    def write_gguf(name: str, q: str | None) -> None:
        heads = HeadsWriter(output_dir / "gguf" / name, "gliner4j-gliner2")
        add_deberta_encoder(heads, encoder, dtype, prefix="enc.", quant=q)
        heads.kv("gliner2.hidden_size", hidden)
        heads.kv("gliner2.max_width", int(d["max_width"]))
        heads.kv("gliner2.max_count", max_count)
        heads.kv("gliner2.counting_layer", counting)
        heads.tensors(head_state, f16_matrix=head_f16)
        heads.close()

    console.print(f"[cyan]1/3[/cyan] model.gguf (encoder + heads){' + ' + ', '.join(quant) if quant else ''}")
    write_gguf("model.gguf", None)
    for q in quant:
        write_gguf(f"model-{q}.gguf", q)  # same heads, block-quantised encoder matrices (variant=<q>)

    console.print("[cyan]2/3[/cyan] config + tokenizer")
    added = tokenizer.get_added_vocab()
    specials = {k: f"[{k}]" for k in ("P", "C", "E", "R", "L")}
    special_ids = {k: added[v] for k, v in specials.items()}
    config = {
        "architecture": "gliner2",
        "engine": "llamacpp",
        "hidden_size": hidden,
        "max_width": d["max_width"],
        "max_count": max_count,
        "span_mode": "SpanMarkerV0",
        "counting_layer": counting,
        "token_pooling": "first",
        "uses_span_idx": True,
        "special_tokens": specials,
        "special_token_ids": special_ids,
        "architecture_config": {
            "encoder_model_name": d.get("model_name"),
            "backbone_kind": "deberta",
            "max_len": int(encoder.config.max_position_embeddings),
        },
    }
    (output_dir / "gliner4j_config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(str(output_dir))

    console.print("[cyan]3/3[/cyan] reference.json")
    entries = []
    for text, labels in GLINER2_REFERENCE:
        r = model.extract_entities(text, labels, threshold=0.3, include_confidence=True)
        entries.append({"text": text, "labels": labels, "threshold": 0.3, "entities": r.get("entities", r)})
    (output_dir / "reference.json").write_text(json.dumps(entries, indent=1, default=str) + "\n")
    console.print(f"  [green]✓[/green] reference.json ({len(entries)} texts)")
    _print_tree(output_dir)




# ===========================================================================
# gliner2dot5 (boundary architecture)
# ===========================================================================

# Head tensors the Java graph uses (everything else — proposer, pair_scorer, candidate_encoder,
# count_head, record_decoder — is training-time or unserved and skipped).
_G25_PREFIXES = ("boundary_head.boundary_encoder.", "boundary_head.boundary_query_head.",
                 "boundary_head.shared_pool_builder.", "boundary_head.shared_pool_scorer.",
                 "boundary_head.null_projection.", "relation_scorer.", "classifier.")
_G25_RENAMES = {
    "boundary_head.boundary_encoder.": "benc.",
    "boundary_head.boundary_query_head.": "bqh.",
    "boundary_head.shared_pool_builder.": "pool.",
    "boundary_head.shared_pool_scorer.": "scorer.",
    "boundary_head.null_projection.": "null_projection.",
    "relation_scorer.": "rel.",
}

G25_REFERENCE = [
    ("John Smith works at Acme Corp in Berlin since 2019.", ["person", "organization", "location", "date"]),
    ("Contact Dr. Maria Lopez (maria.lopez@clinic.org) at Hospital del Mar before 12 March 2025.", ["person", "email", "organization", "date"]),
    ("ok", ["person"]),
]


@app.command("gliner2dot5")
def gliner2dot5(
    model_path: str = typer.Option(..., help="HF id or dir of a GLiNER2.5 checkpoint (fastino/gliner2.5-*)"),
    output_dir: Path = typer.Option(..., help="Bundle directory to write"),
    dtype: str = typer.Option("f16", help="f16 or f32 for the encoder / head matrices"),
    quant: list[str] = typer.Option(
        [], "--quant", help="also write gguf/model-<quant>.gguf with block-quantised encoder matrices (q8_0, q5_0, q4_0)"
    ),
):
    """GLiNER2.5 (boundary) → one model.gguf (DeBERTa encoder + boundary/pool/scorer/relation/classifier heads)."""
    from gliner2 import AutoExtractor

    console.print(f"[bold]GLiNER2.5 → ggml[/bold]  {model_path}")
    model = AutoExtractor.from_pretrained(model_path).eval()
    settings = model.boundary_settings
    head = model.boundary_head
    if settings.candidate_pool != "shared":
        raise typer.BadParameter(f"candidate_pool={settings.candidate_pool!r}: only 'shared'")
    if settings.candidate_attention_layers or settings.query_attention_layers:
        raise typer.BadParameter("candidate/query attention layers are not supported")
    pooler = head.shared_pool_scorer.content_pooler
    if pooler is None or pooler.use_soft_max_pool:
        raise typer.BadParameter("expected a mean content pooler")
    if not head.use_inside_evidence:
        raise typer.BadParameter("expected use_inside_evidence=True")
    encoder = model.encoder
    tokenizer = model.processor.tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    gen = model.relation_pair_generator.settings if getattr(model, "relation_pair_generator", None) else None

    attn = head.boundary_encoder.attention_blocks[0]
    head_state = {
        _short_name(k, _G25_RENAMES): v
        for k, v in model.state_dict().items()
        if k.startswith(_G25_PREFIXES)
    }

    def head_f16(name: str, arr) -> bool:
        # 2-D head matrices that only feed mul_mat (boundary encoder, boundary / pool / scorer
        # projections, classifier): f16 puts them on the tensor cores. Everything read
        # element-wise (states, biases, norms, relation tables) stays f32.
        return dtype == "f16" and name.endswith(".weight") and name.startswith(
            ("benc.", "bqh.", "pool.", "scorer.", "classifier.", "null_projection.")
        )

    def write_gguf(name: str, q: str | None) -> None:
        heads = HeadsWriter(output_dir / "gguf" / name, "gliner4j-gliner2dot5")
        add_deberta_encoder(heads, encoder, dtype, prefix="enc.", quant=q)
        heads.kv("g25.hidden_size", int(model.hidden_size))
        heads.kv("g25.boundary_dim", int(settings.boundary_dim))
        heads.kv("g25.attention_heads", int(attn.num_heads))
        heads.kv("g25.attention_window", int(attn.window))
        heads.kv("g25.pool_top_k", int(head.shared_pool_builder.pool_boundary_top_k))
        heads.kv("g25.pool_size", int(head.shared_pool_builder.pool_size))
        heads.kv("g25.min_pool_per_query", int(head.shared_pool_builder.min_pool_per_query))
        heads.kv("g25.content_dim", int(pooler.value_projection.out_features))
        heads.kv("g25.layer_norm_eps", float(head.boundary_encoder.layer_norm.eps))
        if gen is not None:
            heads.kv("g25.relation_heads", int(gen.heads_per_relation))
            heads.kv("g25.relation_tails", int(gen.tails_per_relation))
            heads.kv("g25.relation_pair_cap", int(gen.pair_cap))
            heads.kv("g25.relation_argument_threshold", float(gen.argument_threshold))
            heads.kv("g25.relation_biaffine", bool(model.relation_scorer.use_biaffine_content))
        heads.tensors(head_state, f16_matrix=head_f16)
        heads.close()

    console.print(f"[cyan]1/3[/cyan] model.gguf (encoder + heads){' + ' + ', '.join(quant) if quant else ''}")
    write_gguf("model.gguf", None)
    for q in quant:
        write_gguf(f"model-{q}.gguf", q)

    console.print("[cyan]2/3[/cyan] config + tokenizer")
    added = tokenizer.get_added_vocab()
    specials = {k: f"[{k}]" for k in ("P", "C", "E", "R", "L")}
    src_cfg = json.loads((Path(model_path) / "gliner4j_config.json").read_text()) if (Path(model_path) / "gliner4j_config.json").exists() else None
    config = {
        "architecture": "gliner2dot5",
        "engine": "llamacpp",
        "hidden_size": int(model.hidden_size),
        "token_pooling": "first",
        "special_tokens": specials,
        "special_token_ids": {k: added[v] for k, v in specials.items()},
        "architecture_config": {
            "backbone": getattr(model.config, "model_name", None) or getattr(encoder.config, "_name_or_path", None),
            "backbone_kind": "deberta",
            "max_len": int(getattr(model.config, "max_len", 4096) or 4096),
            "boundary_dim": int(settings.boundary_dim),
            "pool_size": int(head.shared_pool_builder.pool_size),
            "pair_temperature": float(getattr(settings, "pair_temperature", 1.0)),
            "enable_abstention": bool(head.null_projection is not None),
            "abstention_threshold": float(getattr(settings, "abstention_threshold", 0.5)),
            "overlap_policy": getattr(settings, "overlap_policy", "flat"),
            "classification_temperature": float(getattr(settings, "classification_temperature", 1.0)),
            "relation_temperature": float(getattr(settings, "relation_temperature", 1.0)),
            "relation_pair_cap": int(gen.pair_cap) if gen else 0,
            "relation_argument_proposal_threshold": float(gen.argument_threshold) if gen else 0.0,
            "lowercase_words": bool(getattr(model.processor, "lowercase_words", True)),
            "append_period": bool(getattr(model.processor, "append_period", True)),
        },
    }
    (output_dir / "gliner4j_config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(str(output_dir))

    console.print("[cyan]3/3[/cyan] reference.json")
    entries = []
    for text, labels in G25_REFERENCE:
        r = model.extract_entities(text, labels, threshold=0.3, include_confidence=True)
        entries.append({"text": text, "labels": labels, "threshold": 0.3, "entities": r.get("entities", r)})
    (output_dir / "reference.json").write_text(json.dumps(entries, indent=1, default=str) + "\n")
    console.print(f"  [green]✓[/green] reference.json ({len(entries)} texts)")
    _print_tree(output_dir)


if __name__ == "__main__":
    app()
