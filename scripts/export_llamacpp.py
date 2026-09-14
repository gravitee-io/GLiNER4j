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
#     "gliner>=0.2.28",
# ]
# ///
"""Export GLiNER4j model families for the llama.cpp / ggml engine (``"engine": "llamacpp"``).

One sub-command per family. Each writes a bundle::

    {output_dir}/gguf/backbone-{f16,q8_0}.gguf   encoder converted by llama.cpp's convert_hf_to_gguf.py
    {output_dir}/gguf/heads.gguf                 task heads (f32) for the Java ggml graphs
    {output_dir}/gliner4j_config.json            engine=llamacpp + architecture_config
    {output_dir}/tokenizer.json …                HF tokenizer files (Java tokenizes; llama.cpp only sees ids)
    {output_dir}/reference.json                  PyTorch scores for the Java parity tests

Families:
    gliclass         GLiClass uni-encoder on a llama.cpp-native encoder (ModernBERT / BERT / XLM-R).
    gliner-uni       Original GLiNER uni-encoder (markerV0 or token_level, DeBERTa backbone): the
                     encoder (+ projection, LSTM) + span / token heads in one model.gguf.
    gliner-bi        Original GLiNER bi-encoder: DeBERTa text model in model.gguf plus the label
                     encoder (BERT/MiniLM) as a llama.cpp-native GGUF under gguf/labels-*.gguf.
    gliner2          → scripts/export_llamacpp_gliner2.py (separate script: the gliner2 package
                     pins transformers<5 while gliclass/gliner need >=5).
    deberta-encoder  A bare DeBERTa-v2/v3 encoder as gliner4j's own ggml graph (llama.cpp has no
                     DeBERTa) — used standalone for the encoder parity test, and by the GLiNER
                     families whose backbone is DeBERTa.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import typer
from rich.console import Console
from torch import nn

app = typer.Typer(add_completion=False, help=__doc__)
console = Console()


@app.callback()
def _main() -> None:
    """Export GLiNER4j families for the llama.cpp / ggml engine."""

# Encoder architectures llama.cpp can convert and run with llama_encode (per-token embeddings).
LLAMA_NATIVE_ENCODERS = {
    "ModernBertModel",
    "ModernBertForMaskedLM",
    "BertModel",
    "BertForMaskedLM",
    "XLMRobertaModel",
    "NomicBertModel",
    "NeoBERT",
    "EuroBertModel",
}

GLICLASS_REFERENCE = [
    (
        "The new graphics card doubles ray-tracing performance while drawing less power.",
        ["technology", "sports", "politics", "business", "health", "travel", "science", "entertainment"],
    ),
    (
        "The striker scored twice in extra time to send her team to the final.",
        ["technology", "sports", "politics", "business", "health", "travel", "science", "entertainment"],
    ),
    (
        "Parliament passed the budget after a week of tense negotiations between the coalition partners.",
        ["technology", "sports", "politics", "business", "health", "travel", "science", "entertainment"],
    ),
    (
        "Quarterly revenue rose 12% on strong cloud demand, and the board raised the dividend.",
        ["technology", "sports", "politics", "business", "health", "travel", "science", "entertainment"],
    ),
    (
        "Researchers sequenced the genome of a deep-sea microbe that survives without sunlight.",
        ["science", "health", "environment"],
    ),
    (
        "I loved the movie but the ending felt rushed and the soundtrack was forgettable.",
        ["positive", "negative", "neutral"],
    ),
    (
        "Our flight to Lisbon was cancelled twice and the hotel lost our reservation.",
        ["travel", "complaint", "praise", "question"],
    ),
    ("ok", ["yes", "no"]),
]


# ===========================================================================
# GGUF helpers
# ===========================================================================


_CONVERT_WRAPPER = """
import runpy, sys
sys.path.insert(0, {llama_dir!r})
import conversion.base as base
# GLiNER-family checkpoints add marker tokens, which changes the BPE pre-tokenizer fingerprint the
# converter hashes; the pre-tokenizer itself is unchanged, and Java tokenizes anyway (llama.cpp only
# ever sees token ids). Pin the name instead of failing on the unknown hash.
for cls in list(vars(base).values()):
    if isinstance(cls, type) and "get_vocab_base_pre" in vars(cls):
        cls.get_vocab_base_pre = lambda self, tokenizer, _p={pre!r}: _p
sys.argv = ["convert_hf_to_gguf.py", {src!r}, "--outfile", {target!r}, "--outtype", {outtype!r}]
runpy.run_path({convert!r}, run_name="__main__")
"""


def _llama_cpp_convert(llama_cpp_dir: Path, src_dir: Path, target: Path, outtype: str, pre: str) -> None:
    convert = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert.exists():
        raise typer.BadParameter(
            f"{convert} not found — point --llama-cpp-dir / LLAMA_CPP_DIR at a llama.cpp checkout"
        )
    console.print(f"  converting → {target.name} ...")
    code = _CONVERT_WRAPPER.format(
        llama_dir=str(llama_cpp_dir.resolve()),
        pre=pre,
        src=str(src_dir),
        target=str(target),
        outtype=outtype,
        convert=str(convert.resolve()),
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if proc.returncode != 0:
        console.print(proc.stdout[-4000:])
        console.print(proc.stderr[-4000:])
        raise RuntimeError(f"convert_hf_to_gguf.py failed for {target.name}")
    console.print(f"  [green]✓[/green] {target.name} ({target.stat().st_size / 1e6:.1f} MB)")


# tokenizer.ggml.pre per HF encoder class (only matters to llama.cpp's own tokenizer, unused here)
_TOKENIZER_PRE = {"ModernBertModel": "modern-bert", "ModernBertForMaskedLM": "modern-bert"}


def export_encoder_gguf(
    encoder: nn.Module, tokenizer, out_dir: Path, llama_cpp_dir: Path, outtypes: list[str]
) -> list[Path]:
    """Convert a bare HF encoder (``AutoModel``) + tokenizer to ``out_dir/gguf/backbone-{outtype}.gguf``."""
    arch = type(encoder).__name__
    pre = _TOKENIZER_PRE.get(arch, "default")
    if arch not in LLAMA_NATIVE_ENCODERS:
        raise typer.BadParameter(
            f"{arch} is not a llama.cpp-native encoder ({sorted(LLAMA_NATIVE_ENCODERS)}); "
            "DeBERTa backbones use the gliner4j ggml encoder path instead"
        )
    with tempfile.TemporaryDirectory(prefix="gliner4j-enc-") as tmp:
        tmp_dir = Path(tmp)
        encoder.config.architectures = [arch]
        encoder.save_pretrained(str(tmp_dir), safe_serialization=True)
        tokenizer.save_pretrained(str(tmp_dir))
        gguf_dir = out_dir / "gguf"
        gguf_dir.mkdir(parents=True, exist_ok=True)
        outputs = []
        for outtype in outtypes:
            target = gguf_dir / f"backbone-{outtype}.gguf"
            _llama_cpp_convert(llama_cpp_dir, tmp_dir, target, outtype, pre)
            outputs.append(target)
    return outputs


def check_gguf_vocab(path: Path, expected_vocab: int, specials: dict[str, int]) -> None:
    from gguf import GGUFReader

    reader = GGUFReader(str(path))
    field = reader.fields.get("tokenizer.ggml.tokens")
    if field is None:
        raise RuntimeError(f"{path.name}: no tokenizer.ggml.tokens")
    n = len(field.data)
    if n != expected_vocab:
        raise RuntimeError(f"{path.name}: GGUF vocab has {n} tokens, expected {expected_vocab}")
    for tok, idx in specials.items():
        got = bytes(field.parts[field.data[idx]]).decode("utf-8", errors="replace")
        if got != tok:
            raise RuntimeError(f"{path.name}: token {idx} is {got!r}, expected {tok!r}")
    console.print(f"  [green]✓[/green] {path.name}: vocab {n}, specials {specials}")


class HeadsWriter:
    """Small f32 GGUF with the task-head tensors, named by their state-dict keys."""

    def __init__(self, path: Path, arch: str):
        from gguf import GGUFWriter

        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.w = GGUFWriter(str(path), arch=arch)
        self.n = 0

    def kv(self, key: str, value) -> None:
        if isinstance(value, bool):
            self.w.add_bool(key, value)
        elif isinstance(value, int):
            self.w.add_uint32(key, value)
        elif isinstance(value, float):
            self.w.add_float32(key, value)
        else:
            self.w.add_string(key, str(value))

    def tensors(self, state: dict[str, torch.Tensor], rename=lambda k: k, f16_matrix=None) -> None:
        """Write ``state``; ``f16_matrix(name, arr)`` selects 2-D matrices stored as f16 (only ones the
        graph uses as ``mul_mat`` weights — anything read element-wise must stay f32)."""
        import numpy as np

        for name, tensor in state.items():
            short = rename(name)
            if len(short.encode()) >= 64:
                raise RuntimeError(f"ggml tensor name too long ({len(short)}): {short}")
            arr = tensor.detach().float().cpu().numpy()
            if f16_matrix is not None and arr.ndim == 2 and f16_matrix(short, arr):
                arr = arr.astype(np.float16)
            self.w.add_tensor(short, arr)
            self.n += 1

    def close(self) -> None:
        self.w.write_header_to_file()
        self.w.write_kv_data_to_file()
        self.w.write_tensors_to_file()
        self.w.close()
        console.print(
            f"  [green]✓[/green] {self.path.name} ({self.n} tensors, {self.path.stat().st_size / 1e6:.1f} MB)"
        )


def _tokenizer_specials(tokenizer, names: list[str]) -> dict[str, int]:
    return {name: tokenizer.convert_tokens_to_ids(name) for name in names}


def _print_tree(out_dir: Path) -> None:
    console.print(f"\n[bold]{out_dir}[/bold]")
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            console.print(f"  {p.relative_to(out_dir)}  ({p.stat().st_size / 1e6:.1f} MB)")


# ===========================================================================
# DeBERTa-v3 encoder → encoder.gguf (gliner4j ggml graph)
# ===========================================================================

DEBERTA_REFERENCE_TEXTS = {
    "tiny": "Hello world.",
    "medium": " ".join(
        [
            "The committee met on Tuesday to review the quarterly figures, which had come in well",
            "below the forecast published in March; several members argued for a revised outlook",
            "while others preferred to wait for the audited statements due next month.",
        ]
        * 3
    ),
    "long": " ".join(
        [
            "In the northern valleys the snow rarely melts before May, and the roads that link the",
            "villages stay closed for most of the winter, so supplies are brought in by sled or,",
            "when the weather allows, by a small aircraft that lands on the frozen lake.",
        ]
        * 12
    ),
}


def add_deberta_encoder(
    heads: "HeadsWriter", encoder: nn.Module, dtype: str = "f16", prefix: str = "", quant: str | None = None
) -> None:
    """Write a HF ``DebertaV2Model`` (embeddings + encoder) as ``encoder.gguf`` for ``GgmlDebertaV3``.

    Input-independent work is folded in: the rel-embedding LayerNorm is applied to the table, and
    with ``share_att_key`` the per-layer position keys/queries (``key_proj``/``query_proj`` of that
    table) are stored as ``layer.N.pos_key`` / ``layer.N.pos_query`` so the Java graph only gathers.

    ``quant`` (``q8_0`` / ``q5_0`` / ``q4_0``) block-quantises the big matrices the graph only ever
    uses as ``mul_mat`` / ``get_rows`` sources — the six linear weights per layer and the word
    embeddings (row length must be a multiple of the 32-element block). Position tables, LayerNorm
    params and biases stay f16 / f32.
    """
    import numpy as np

    cfg = encoder.config
    assert cfg.model_type == "deberta-v2", cfg.model_type
    assert cfg.relative_attention and set(cfg.pos_att_type) == {"p2c", "c2p"}, cfg.pos_att_type
    assert cfg.share_att_key, "GgmlDebertaV3 implements share_att_key=True"
    assert not cfg.position_biased_input and cfg.type_vocab_size == 0, "absolute/token-type embeddings unsupported"
    assert not getattr(cfg, "conv_kernel_size", 0), "ConvLayer unsupported"
    assert "layer_norm" in (cfg.norm_rel_ebd or ""), cfg.norm_rel_ebd
    emb_size = getattr(cfg, "embedding_size", cfg.hidden_size) or cfg.hidden_size
    assert emb_size == cfg.hidden_size, "embed_proj unsupported"
    buckets = cfg.position_buckets
    max_rel = cfg.max_relative_positions if cfg.max_relative_positions > 0 else cfg.max_position_embeddings
    att_span = buckets if buckets > 0 else max_rel

    enc = encoder.encoder
    heads.kv("deberta.hidden_size", int(cfg.hidden_size))
    heads.kv("deberta.num_layers", int(cfg.num_hidden_layers))
    heads.kv("deberta.num_heads", int(cfg.num_attention_heads))
    heads.kv("deberta.intermediate_size", int(cfg.intermediate_size))
    heads.kv("deberta.position_buckets", int(buckets))
    heads.kv("deberta.max_relative_positions", int(max_rel))
    heads.kv("deberta.att_span", int(att_span))
    heads.kv("deberta.layer_norm_eps", float(cfg.layer_norm_eps))
    heads.kv("deberta.vocab_size", int(cfg.vocab_size))
    heads.kv("deberta.pos_per_head", 1)  # pos_key / pos_query: (head_dim, 2·att_span, heads), pre-scaled

    np_dtype = np.float16 if dtype == "f16" else np.float32
    with torch.no_grad():
        rel = enc.get_rel_embedding()  # LayerNorm already applied
        state: dict[str, torch.Tensor] = {
            "embeddings.word_embeddings.weight": encoder.embeddings.word_embeddings.weight,
            "embeddings.LayerNorm.weight": encoder.embeddings.LayerNorm.weight,
            "embeddings.LayerNorm.bias": encoder.embeddings.LayerNorm.bias,
        }
        for i, layer in enumerate(enc.layer):
            a = layer.attention.self
            p = f"layer.{i}."
            # Stored per head and pre-scaled by 1/sqrt(3·d): (heads, 2·att_span, head_dim) → ggml
            # ne = (head_dim, 2·att_span, heads), so the Java graph gathers rows straight into the
            # attention layout without a permute/copy or a scale pass (see GgmlDebertaV3).
            head_dim = cfg.hidden_size // cfg.num_attention_heads
            attn_scale = 1.0 / math.sqrt(3.0 * head_dim)
            for name, proj in (("pos_key", a.key_proj), ("pos_query", a.query_proj)):
                table = proj(rel) * attn_scale  # (2·att_span, hidden)
                state[p + name] = table.view(table.shape[0], cfg.num_attention_heads, head_dim).permute(1, 0, 2).contiguous()
            # q / k / v as one (3·hidden, hidden) projection: one GEMM (and one f32↔f16 round trip on
            # CUDA) per layer instead of three; the Java graph slices the result.
            state[p + "qkv.weight"] = torch.cat([a.query_proj.weight, a.key_proj.weight, a.value_proj.weight], 0)
            state[p + "qkv.bias"] = torch.cat([a.query_proj.bias, a.key_proj.bias, a.value_proj.bias], 0)
            for name, mod in [
                ("attn_out", layer.attention.output.dense), ("attn_ln", layer.attention.output.LayerNorm),
                ("ffn_up", layer.intermediate.dense), ("ffn_down", layer.output.dense), ("ffn_ln", layer.output.LayerNorm),
            ]:
                state[p + name + ".weight"] = mod.weight
                state[p + name + ".bias"] = mod.bias
    # large matrices in the requested dtype, vectors/LN params in f32
    qtype = _quant_type(quant)
    for name, t in state.items():
        arr = t.detach().float().cpu().numpy()
        if qtype is not None and _quantizable_encoder_matrix(name, arr):
            from gguf import quants

            heads.w.add_tensor(prefix + name, quants.quantize(arr, qtype), raw_dtype=qtype)
        else:
            if arr.ndim >= 2 and dtype == "f16" and not name.endswith("_ln.weight"):
                arr = arr.astype(np_dtype)
            heads.w.add_tensor(prefix + name, arr)
        heads.n += 1


_QUANTIZABLE_SUFFIXES = (".qkv.weight", ".q.weight", ".k.weight", ".v.weight", ".attn_out.weight", ".ffn_up.weight", ".ffn_down.weight")


def _quant_type(quant: str | None):
    if not quant:
        return None
    from gguf import GGMLQuantizationType

    try:
        return GGMLQuantizationType[quant.upper()]
    except KeyError as e:
        raise typer.BadParameter(f"unknown quantization {quant!r} (q8_0, q5_0, q4_0, …)") from e


def _quantizable_encoder_matrix(name: str, arr) -> bool:
    """Matrices used only as mul_mat / get_rows sources (row length a multiple of the 32-wide block)."""
    if arr.ndim != 2 or arr.shape[1] % 32 != 0:
        return False
    return name == "embeddings.word_embeddings.weight" or name.endswith(_QUANTIZABLE_SUFFIXES)


def export_deberta_encoder_gguf(encoder: nn.Module, out_path: Path, dtype: str = "f16") -> Path:
    heads = HeadsWriter(out_path, "gliner4j-deberta-v3")
    add_deberta_encoder(heads, encoder, dtype)
    heads.close()
    return out_path


def deberta_reference(encoder: nn.Module, tokenizer, max_len: int = 512) -> list[dict]:
    """``last_hidden_state`` rows (first, middle, last token) + global max|h| per reference text."""
    entries = []
    for name, text in DEBERTA_REFERENCE_TEXTS.items():
        ids = tokenizer(text, truncation=True, max_length=max_len)["input_ids"]
        with torch.no_grad():
            h = encoder(input_ids=torch.tensor([ids]), attention_mask=torch.ones(1, len(ids), dtype=torch.long)).last_hidden_state[0]
        n = len(ids)
        rows = {str(i): [float(x) for x in h[i]] for i in {0, n // 2, n - 1}}
        entries.append({"name": name, "input_ids": ids, "rows": rows, "max_abs": float(h.abs().max())})
    return entries


@app.command("deberta-encoder")
def deberta_encoder(
    model_path: str = typer.Option(..., help="HF id or dir of a DeBERTa-v2/v3 encoder (e.g. microsoft/deberta-v3-small)"),
    output_dir: Path = typer.Option(..., help="Directory to write encoder.gguf + reference.json"),
    dtype: str = typer.Option("f16", help="f16 or f32 for the large matrices"),
):
    """Standalone DeBERTa encoder export for the ggml parity test."""
    from transformers import AutoModel, AutoTokenizer

    console.print(f"[bold]DeBERTa encoder → ggml[/bold]  {model_path}")
    encoder = AutoModel.from_pretrained(model_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    export_deberta_encoder_gguf(encoder, output_dir / "gguf" / "encoder.gguf", dtype)
    ref = deberta_reference(encoder, tokenizer, encoder.config.max_position_embeddings)
    (output_dir / "reference.json").write_text(json.dumps(ref) + "\n")
    tokenizer.save_pretrained(str(output_dir))
    console.print(f"  [green]✓[/green] reference.json ({[len(e['input_ids']) for e in ref]} tokens)")
    _print_tree(output_dir)


# ===========================================================================
# gliner-uni (original GLiNER, markerV0 span model)
# ===========================================================================

GLINER_UNI_REFERENCE = [
    (
        "John Smith lives at 221B Baker Street, London and his email is john.smith@example.com; call +44 20 7946 0958.",
        ["person", "address", "email", "phone number", "city"],
    ),
    (
        "Patient Maria Gonzalez (DOB 1985-03-12, SSN 123-45-6789) was admitted to St. Mary's Hospital on 4 June.",
        ["person", "date of birth", "social security number", "organization", "date"],
    ),
    (
        "Transfer 2,500 EUR from IBAN DE89 3704 0044 0532 0130 00 to the account of Acme GmbH before Friday.",
        ["amount", "iban", "organization", "date"],
    ),
    (
        "Contact our support team at support@acme.io or visit https://acme.io/help; our office is in Berlin.",
        ["email", "url", "city", "organization"],
    ),
    ("ok", ["person"]),
]

_GLINER_UNI_RENAMES = {
    "span_rep_layer.span_rep_layer.": "span_rep.",
    "prompt_rep_layer.": "prompt_rep.",
    "rnn.lstm.": "lstm.",
}


def _short_name(name: str, renames: dict[str, str]) -> str:
    for old, new in renames.items():
        if name.startswith(old):
            return new + name[len(old):]
    return name


def _force_whitespace_splitter() -> None:
    """gliner4j splits words on whitespace; make the Python reference do the same (see export_onnx.py)."""
    import gliner.data_processing.tokenizer as _gtok

    orig = _gtok.WordsSplitter.__init__

    def _init(self, splitter_type="whitespace", *a, **k):
        orig(self, "whitespace", *a, **k)

    _gtok.WordsSplitter.__init__ = _init


def _gliner_bundle(
    model, gcfg: dict, output_dir: Path, dtype: str, architecture: str, labels_gguf: str | None, extra_cfg: dict,
    quant: list[str] | None = None,
) -> None:
    """Common part of the gliner-uni / gliner-bi exports: model.gguf, config, tokenizer, reference."""
    core = model.model
    encoder = core.token_rep_layer.bert_layer.model
    has_rnn = hasattr(core, "rnn")
    has_projection = hasattr(core.token_rep_layer, "projection")
    span_mode = gcfg.get("span_mode")
    if gcfg.get("post_fusion_schema") or gcfg.get("fuse_layers"):
        raise typer.BadParameter("post_fusion / fuse_layers checkpoints are not supported here")
    if gcfg.get("subtoken_pooling", "first") != "first" or not gcfg.get("embed_ent_token", True):
        raise typer.BadParameter("only subtoken_pooling=first, embed_ent_token=true")
    if has_rnn:
        assert core.rnn.lstm.num_layers == 1 and core.rnn.lstm.bidirectional, "expected 1-layer biLSTM"

    renames = dict(_GLINER_UNI_RENAMES)
    renames["token_rep_layer.projection."] = "proj."
    renames["token_rep_layer.labels_projection."] = "labels_projection."
    head_state = {
        _short_name(k, renames): v
        for k, v in core.state_dict().items()
        if not k.startswith("token_rep_layer.bert_layer.") and not k.startswith("token_rep_layer.labels_encoder.")
    }

    def head_f16(name: str, arr) -> bool:
        # Every 2-D head matrix is a mul_mat weight (span / prompt / scorer projections, the
        # encoder projection, the LSTM input and recurrent weights) — f16 puts them on the tensor
        # cores like the GLiNER2 heads; biases stay f32. The plugin's fused LSTM reads W_hh in
        # either type.
        return dtype == "f16" and (name.endswith(".weight") or name.startswith("lstm.weight_"))

    def write_gguf(name: str, q: str | None) -> None:
        heads = HeadsWriter(output_dir / "gguf" / name, "gliner4j-" + architecture)
        add_deberta_encoder(heads, encoder, dtype, prefix="enc.", quant=q)
        heads.kv("gliner.hidden_size", int(gcfg["hidden_size"]))
        heads.kv("gliner.max_width", int(gcfg["max_width"]))
        heads.kv("gliner.has_rnn", bool(has_rnn))
        heads.kv("gliner.span_mode", str(span_mode))
        heads.tensors(head_state, f16_matrix=head_f16)
        heads.close()

    quant = quant or []
    console.print(f"[cyan]1/3[/cyan] model.gguf (encoder + heads){' + ' + ', '.join(quant) if quant else ''}")
    write_gguf("model.gguf", None)
    for q in quant:
        write_gguf(f"model-{q}.gguf", q)  # variant=<q> in GgmlWeights.resolveModelGguf

    console.print("[cyan]2/3[/cyan] config + tokenizer")
    config = {
        "architecture": architecture,
        "engine": "llamacpp",
        "hidden_size": gcfg.get("hidden_size"),
        "max_width": gcfg.get("max_width"),
        "architecture_config": {
            "model_name": gcfg.get("model_name"),
            "backbone_kind": "deberta",
            "labels_encoder": gcfg.get("labels_encoder"),
            "span_mode": span_mode,
            "class_token_index": gcfg.get("class_token_index"),
            "ent_token": gcfg.get("ent_token", "<<ENT>>"),
            "sep_token": gcfg.get("sep_token", "<<SEP>>"),
            "embed_ent_token": gcfg.get("embed_ent_token", True),
            "subtoken_pooling": gcfg.get("subtoken_pooling", "first"),
            "words_splitter_type": "whitespace",
            "max_len": gcfg.get("max_len"),
            "max_types": gcfg.get("max_types"),
            "has_rnn": bool(has_rnn),
            "has_projection": bool(has_projection),
            **({"labels_gguf": labels_gguf} if labels_gguf else {}),
            **extra_cfg,
        },
    }
    (output_dir / "gliner4j_config.json").write_text(json.dumps(config, indent=2) + "\n")
    model.data_processor.transformer_tokenizer.save_pretrained(str(output_dir))

    console.print("[cyan]3/3[/cyan] reference.json")
    entries = []
    for text, labels in GLINER_UNI_REFERENCE:
        ents = model.predict_entities(text, labels, threshold=0.3, flat_ner=True)
        entries.append({
            "text": text,
            "labels": labels,
            "threshold": 0.3,
            "entities": [{"start": e["start"], "end": e["end"], "text": e["text"], "label": e["label"], "score": float(e["score"])} for e in ents],
        })
    (output_dir / "reference.json").write_text(json.dumps(entries, indent=1) + "\n")
    console.print(f"  [green]✓[/green] reference.json ({len(entries)} texts)")
    _print_tree(output_dir)


def _load_gliner(model_path: str):
    _force_whitespace_splitter()
    from gliner import GLiNER

    model = GLiNER.from_pretrained(model_path).eval()
    gcfg = model.config.to_dict() if hasattr(model.config, "to_dict") else dict(model.config.__dict__)
    return model, gcfg


@app.command("gliner-uni")
def gliner_uni(
    model_path: str = typer.Option(..., help="HF id or dir of an original-GLiNER uni-encoder checkpoint"),
    output_dir: Path = typer.Option(..., help="Bundle directory to write"),
    dtype: str = typer.Option("f16", help="f16 or f32 for the encoder / head matrices"),
    quant: list[str] = typer.Option(
        [], "--quant", help="also write gguf/model-<quant>.gguf with block-quantised encoder matrices (q8_0, q5_0, q4_0)"
    ),
):
    """Original GLiNER uni-encoder (markerV0 span model or token_level) → one model.gguf bundle."""
    console.print(f"[bold]GLiNER uni-encoder → ggml[/bold]  {model_path}")
    model, gcfg = _load_gliner(model_path)
    kind = type(model.model).__name__
    if kind not in ("UniEncoderSpanModel", "UniEncoderTokenModel"):
        raise typer.BadParameter(f"{kind}: only UniEncoderSpanModel / UniEncoderTokenModel")
    if kind == "UniEncoderSpanModel" and gcfg.get("span_mode") != "markerV0":
        raise typer.BadParameter(f"span_mode={gcfg.get('span_mode')}: only markerV0 for span models")
    if gcfg.get("labels_encoder"):
        raise typer.BadParameter("this is a bi-encoder checkpoint — use the gliner-bi command")
    output_dir.mkdir(parents=True, exist_ok=True)
    _gliner_bundle(model, gcfg, output_dir, dtype, "gliner-uni", None, {}, quant=quant)


@app.command("gliner-bi")
def gliner_bi(
    model_path: str = typer.Option(..., help="HF id or dir of an original-GLiNER bi-encoder checkpoint"),
    output_dir: Path = typer.Option(..., help="Bundle directory to write"),
    llama_cpp_dir: Path = typer.Option(
        Path(os.environ.get("LLAMA_CPP_DIR", "../llama.cpp")), help="llama.cpp checkout (convert_hf_to_gguf.py)"
    ),
    dtype: str = typer.Option("f16", help="f16 or f32 for the text encoder / head matrices"),
    quant: list[str] = typer.Option(
        [], "--quant", help="also write gguf/model-<quant>.gguf with block-quantised encoder matrices (q8_0, q5_0, q4_0)"
    ),
    labels_gguf_type: str = typer.Option("f16", help="GGUF outtype for the label encoder"),
):
    """Original GLiNER bi-encoder → model.gguf (text model) + labels-*.gguf (llama.cpp-native label encoder)."""
    console.print(f"[bold]GLiNER bi-encoder → ggml[/bold]  {model_path}")
    model, gcfg = _load_gliner(model_path)
    if type(model.model).__name__ != "BiEncoderSpanModel" or gcfg.get("span_mode") != "markerV0":
        raise typer.BadParameter(f"{type(model.model).__name__} / span_mode={gcfg.get('span_mode')}: only BiEncoderSpanModel + markerV0")
    labels_encoder = model.model.token_rep_layer.labels_encoder.model
    labels_tokenizer = model.data_processor.labels_tokenizer
    output_dir.mkdir(parents=True, exist_ok=True)
    console.print("[cyan]0/3[/cyan] label encoder → GGUF")
    outputs = export_encoder_gguf(labels_encoder, labels_tokenizer, output_dir, llama_cpp_dir, [labels_gguf_type])
    labels_gguf = None
    for p in outputs:
        target = p.with_name(p.name.replace("backbone-", "labels-"))
        p.rename(target)
        labels_gguf = target.name
    labels_tokenizer.save_pretrained(str(output_dir / "labels_tokenizer"))
    extra = {
        "labels_hidden_size": int(labels_encoder.config.hidden_size),
        "labels_n_ubatch": 512,
    }
    _gliner_bundle(model, gcfg, output_dir, dtype, "gliner-bi", labels_gguf, extra, quant=quant)


# ===========================================================================
# gliclass (uni-encoder)
# ===========================================================================


def _gc_reference(model: nn.Module, tokenizer, cfg) -> list[dict]:
    entries = []
    for text, labels in GLICLASS_REFERENCE:
        prompt = "".join(f"<<LABEL>>{l}" for l in labels) + "<<SEP>>"
        prompt = prompt + text if cfg.prompt_first else text + prompt
        enc = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits[0]
        scores = torch.sigmoid(logits[: len(labels)])
        entries.append({"text": text, "labels": labels, "scores": [float(s) for s in scores]})
    return entries


@app.command()
def gliclass(
    model_path: str = typer.Option(..., help="HF id or local dir of a GLiClass uni-encoder checkpoint"),
    output_dir: Path = typer.Option(..., help="Bundle directory to write"),
    llama_cpp_dir: Path = typer.Option(
        Path(os.environ.get("LLAMA_CPP_DIR", "../llama.cpp")), help="llama.cpp checkout (convert_hf_to_gguf.py)"
    ),
    gguf_type: list[str] = typer.Option(["f16", "q8_0"], help="GGUF outtypes for the encoder"),
    default_gguf: str = typer.Option("backbone-q8_0.gguf", help="Encoder GGUF the Java side loads by default"),
    n_ubatch: int = typer.Option(2048, help="Max tokens per llama_encode call (whole packed batch)"),
):
    """GLiClass uni-encoder (ModernBERT/BERT backbone) → llama.cpp encoder + ggml head bundle."""
    from gliclass import GLiClassModel
    from transformers import AutoTokenizer

    console.print(f"[bold]GLiClass → llama.cpp[/bold]  {model_path}")
    model = GLiClassModel.from_pretrained(model_path).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    cfg = model.config
    inner = model.model
    if cfg.architecture_type != "uni-encoder":
        raise typer.BadParameter(f"architecture_type={cfg.architecture_type}; only uni-encoder is supported here")
    if cfg.pooling_strategy != "first" or cfg.extract_text_features or not cfg.embed_class_token:
        raise typer.BadParameter("only pooling_strategy=first, extract_text_features=false, embed_class_token=true")
    if getattr(cfg, "use_lstm", False):
        raise typer.BadParameter("use_lstm=true checkpoints are not supported")
    if cfg.scorer_type not in ("simple", "mlp"):
        raise typer.BadParameter(f"scorer_type={cfg.scorer_type}; only simple / mlp")

    encoder = inner.encoder_model
    enc_hidden = encoder.config.hidden_size
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]1/4[/cyan] encoder → GGUF")
    outputs = export_encoder_gguf(encoder, tokenizer, output_dir, llama_cpp_dir, gguf_type)
    specials = _tokenizer_specials(tokenizer, ["<<LABEL>>", "<<SEP>>"])
    for p in outputs:
        check_gguf_vocab(p, encoder.config.vocab_size, specials)

    console.print("[cyan]2/4[/cyan] heads → heads.gguf")
    heads = HeadsWriter(output_dir / "gguf" / "heads.gguf", "gliclass-heads")
    heads.kv("gliclass.hidden_size", int(enc_hidden))
    heads.kv("gliclass.scorer_type", cfg.scorer_type)
    heads.kv("gliclass.projector_hidden_act", cfg.projector_hidden_act)
    heads.kv("gliclass.normalize_features", bool(cfg.normalize_features))
    heads.kv("gliclass.logit_scale", float(inner.logit_scale.detach().float()))
    head_state = {k: v for k, v in inner.state_dict().items() if not k.startswith("encoder_model.")}
    head_state["logit_scale"] = inner.logit_scale.detach().float().reshape(1)
    heads.tensors(head_state)
    heads.close()

    console.print("[cyan]3/4[/cyan] config + tokenizer + reference")
    config = {
        "architecture": "gliclass",
        "engine": "llamacpp",
        "hidden_size": enc_hidden,
        "architecture_config": {
            "encoder_model_name": cfg.encoder_model_name,
            "backbone_kind": "llama",
            "backbone_gguf": default_gguf,
            "n_ubatch": n_ubatch,
            "n_ctx_train": int(getattr(encoder.config, "max_position_embeddings", 8192)),
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
    (output_dir / "gliner4j_config.json").write_text(json.dumps(config, indent=2) + "\n")
    tokenizer.save_pretrained(str(output_dir))
    ref = _gc_reference(model, tokenizer, cfg)
    (output_dir / "reference.json").write_text(json.dumps(ref, indent=1) + "\n")
    console.print(f"  [green]✓[/green] gliner4j_config.json, tokenizer, reference.json ({len(ref)} entries)")

    console.print("[cyan]4/4[/cyan] done")
    _print_tree(output_dir)


if __name__ == "__main__":
    app()
