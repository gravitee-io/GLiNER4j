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
#     "onnx>=1.15",
#     "onnxruntime>=1.17",
#     "numpy>=1.26",
#     "typer>=0.15",
#     "coloredlogs",
#     "sympy",
# ]
# ///
"""Post-export graph surgery for GLiNER2 artifacts (merged graphs + encoder fusion).

The gliner4j runtime auto-detects merged artifacts by file name and picks the fastest
available tier: ``ner_full.onnx`` > ``span_scoring.onnx`` > split sessions. This script
produces those artifacts from the split per-variant exports of ``export_onnx.py``:

* ``optimize-encoder`` — onnxruntime.transformers BERT fusion pass on ``encoder.onnx``
  (SkipLayerNormalization/BiasGelu fusion + shape-glue cleanup; ~-20% nodes, was worth
  ~+50% serving capacity on CUDA). Rewrites the encoder in place, per variant.
* ``span-scoring`` — composes ``span_rep.onnx`` + ``scoring_head.onnx`` into
  ``span_scoring.onnx`` (io_map: span_rep output → scoring input). The span-representation
  intermediate — up to ~150MB per bucket — then never leaves the GPU.
* ``ner-full`` — composes ``encoder.onnx`` + a generated gather adapter + a span_scoring half
  RECOMPOSED IN MEMORY from ``span_rep.onnx`` + ``scoring_head.onnx``, into ``ner_full.onnx``.
  It never reads a ``span_scoring.onnx`` off disk, so a stale intermediate cannot poison it. The
  adapter replicates the Java-side per-word first-subword gather (``word_positions`` int64 input,
  -1 = padded word → zero row) and the schema-marker gathers (``p_position`` may be -1 → zero
  vector, matching ``Gliner2NerStrategy``). One session run per bucket; only int64 tensors cross.
* ``classifier-full`` — composes ``encoder.onnx`` + a label-gather adapter + ``classifier_head``
  into ``classifier_full.onnx`` (also head-based, no intermediate).
* ``rebuild`` — ONE command: regenerate every merged graph for a model, all present variants,
  from the split heads. This is the command to use.
* ``verify`` — parity checks. IMPORTANT: span_scores are gated internally, so random inputs
  yield all-zero scores and a vacuous check — verification drives the real encoder and compares
  merged vs split on encoder-derived activations (expects fp32 bit-exact, fp16 within ~1e-3).

Typical regeneration for a model directory (both variants, everything, from heads):

    uv run scripts/build_merged_graphs.py rebuild models/gliner2-privacy-onnx
    uv run scripts/build_merged_graphs.py verify  models/gliner2-privacy-onnx onnx

To go truly from scratch when the ONNX heads are gone, first re-export them from the PyTorch
source (``export_onnx.py``), optionally ``optimize-encoder`` (fp32 first, ``--fp16-from-fp32``
to derive the fp16 encoder — fusion is unreliable on fp16 graphs), then ``rebuild``.

Pure fp16 is correct with the single-instance scoring the Java side requests (count=1); an
earlier "broadcast" failure was a STALE fp16 span_scoring, now impossible since ner-full
recomposes it. The ``--scoring-variant onnx`` escape hatch on ``ner-full`` remains only as a
fallback if a future fp16 scoring export ever misbehaves.
"""

from pathlib import Path

import numpy as np
import onnx
import typer
from onnx import TensorProto as TP
from onnx import compose
from onnx import helper as h

app = typer.Typer(add_completion=False)

HIDDEN = 768


# ── encoder fusion ───────────────────────────────────────────────────────────


@app.command("optimize-encoder")
def optimize_encoder(
    model_dir: Path,
    variant: str = typer.Argument("onnx"),
    fp16_from_fp32: bool = typer.Option(
        False,
        help="Also convert the optimized fp32 encoder to onnx_fp16/encoder.onnx (keep_io_types)",
    ),
):
    """BERT fusion pass on encoder.onnx, rewritten in place."""
    from onnxruntime.transformers.optimizer import optimize_model

    path = model_dir / variant / "encoder.onnx"
    # opt_level=0 skips ORT's optimize_by_onnxruntime pre-pass (which imports torch) but still
    # runs the fusion transforms — verified to fuse SkipLayerNormalization/BiasGelu identically,
    # so this stays a lightweight graph-surgery env with no torch dependency.
    m = optimize_model(
        str(path), model_type="bert", num_heads=12, hidden_size=HIDDEN, use_gpu=True, opt_level=0
    )
    m.save_model_to_file(str(path), use_external_data_format=False)
    g = onnx.load(str(path), load_external_data=False).graph
    fused = sum(1 for n in g.node if n.op_type in ("SkipLayerNormalization", "BiasGelu"))
    typer.echo(f"{path}: {len(g.node)} nodes, {fused} fused ops")
    if fused == 0:
        typer.echo("WARNING: no fusions applied — wrong variant (fp16?) or already optimized")

    if fp16_from_fp32:
        from onnxruntime.transformers.onnx_model import OnnxModel

        m16 = OnnxModel(onnx.load(str(path)))
        m16.convert_float_to_float16(keep_io_types=True)
        out16 = model_dir / "onnx_fp16" / "encoder.onnx"
        m16.save_model_to_file(str(out16), use_external_data_format=False)
        typer.echo(f"{out16}: fp16 (keep_io_types) derived from optimized fp32")


# ── span_scoring = span_rep + scoring_head ───────────────────────────────────


def _unprefix(model, renames):
    for n in model.graph.node:
        n.input[:] = [renames.get(x, x) for x in n.input]
        n.output[:] = [renames.get(x, x) for x in n.output]
    for vi in list(model.graph.input) + list(model.graph.output):
        if vi.name in renames:
            vi.name = renames[vi.name]


def _compose_span_scoring(d: Path) -> onnx.ModelProto:
    """Compose span_rep + scoring_head in memory from the authoritative heads. Never reads a
    span_scoring.onnx off disk — that is what let a stale intermediate poison ner_full."""
    sr = onnx.load(str(d / "span_rep.onnx"))
    sc = onnx.load(str(d / "scoring_head.onnx"))
    ir = min(sr.ir_version, sc.ir_version)
    sr.ir_version = sc.ir_version = ir
    merged = compose.merge_models(sr, sc, io_map=[("span_rep", "span_rep")], prefix1="sr_", prefix2="sc_")
    _unprefix(
        merged,
        {
            "sr_token_embeddings": "token_embeddings",
            "sr_span_idx": "span_idx",
            "sc_schema_emb_p": "schema_emb_p",
            "sc_schema_emb_fields": "schema_emb_fields",
            "sc_count": "count",
            "sc_count_logits": "count_logits",
            "sc_span_scores": "span_scores",
        },
    )
    onnx.checker.check_model(merged)
    return merged


@app.command("span-scoring")
def span_scoring(model_dir: Path, variant: str = typer.Argument("onnx")):
    """Compose span_rep + scoring_head into span_scoring.onnx (intermediate; ner-full no longer
    reads it — it recomposes from the heads)."""
    d = model_dir / variant
    merged = _compose_span_scoring(d)
    onnx.save(merged, str(d / "span_scoring.onnx"))
    typer.echo(f"{d / 'span_scoring.onnx'}: {[i.name for i in merged.graph.input]} -> "
               f"{[o.name for o in merged.graph.output]}")


# ── ner_full = encoder + gather adapter + span_scoring ───────────────────────


def _build_adapter(hidden_elem: int) -> onnx.ModelProto:
    """Gather adapter replicating the Java-side embedding extraction.

    * token_embeddings[b, w] = last_hidden_state[b, word_positions[b, w]]; rows with
      position -1 (padded words) are zeroed — mirrors extractTextEmbeddingsFlat.
    * schema_emb_p / schema_emb_fields gathered from batch row 0 (the schema prefix is
      identical across rows); p_position may be -1 → zero vector — mirrors
      extractSchemaEmbeddings.
    """
    inputs = [
        h.make_tensor_value_info("last_hidden_state", hidden_elem, ["b", "seq", HIDDEN]),
        h.make_tensor_value_info("word_positions", TP.INT64, ["b", "text_len"]),
        h.make_tensor_value_info("p_position", TP.INT64, [1]),
        h.make_tensor_value_info("field_positions", TP.INT64, ["num_fields"]),
    ]
    outputs = [
        h.make_tensor_value_info("token_embeddings", hidden_elem, ["b", "text_len", HIDDEN]),
        h.make_tensor_value_info("schema_emb_p", hidden_elem, [HIDDEN]),
        h.make_tensor_value_info("schema_emb_fields", hidden_elem, ["num_fields", HIDDEN]),
    ]
    zero_scalar = h.make_tensor("zero_i64", TP.INT64, [], [0])
    h_const = h.make_tensor("h_dim", TP.INT64, [1], [HIDDEN])
    nodes = [
        h.make_node("Max", ["word_positions", "zero_i64"], ["wp_clip"]),
        h.make_node("Constant", [], ["axes2"], value=h.make_tensor("axes2_t", TP.INT64, [1], [2])),
        h.make_node("Unsqueeze", ["wp_clip", "axes2"], ["wp_unsq"]),
        h.make_node("Shape", ["word_positions"], ["wp_shape"]),
        h.make_node("Concat", ["wp_shape", "h_dim"], ["expand_shape"], axis=0),
        h.make_node("Expand", ["wp_unsq", "expand_shape"], ["wp_idx"]),
        h.make_node("GatherElements", ["last_hidden_state", "wp_idx"], ["gathered"], axis=1),
        h.make_node("Constant", [], ["zero_i64_c"], value=h.make_tensor("z2", TP.INT64, [], [0])),
        h.make_node("GreaterOrEqual", ["word_positions", "zero_i64_c"], ["wp_valid"]),
        h.make_node("Cast", ["wp_valid"], ["wp_mask"], to=hidden_elem),
        h.make_node("Unsqueeze", ["wp_mask", "axes2"], ["wp_mask3"]),
        h.make_node("Mul", ["gathered", "wp_mask3"], ["token_embeddings"]),
        h.make_node("Gather", ["last_hidden_state", "zero_i64"], ["row0"], axis=0),
        h.make_node("Max", ["p_position", "zero_i64"], ["p_clip"]),
        h.make_node("Gather", ["row0", "p_clip"], ["p_raw"], axis=0),
        h.make_node("GreaterOrEqual", ["p_position", "zero_i64_c"], ["p_valid"]),
        h.make_node("Cast", ["p_valid"], ["p_maskf"], to=hidden_elem),
        h.make_node("Constant", [], ["axes1"], value=h.make_tensor("axes1_t", TP.INT64, [1], [1])),
        h.make_node("Unsqueeze", ["p_maskf", "axes1"], ["p_mask2"]),
        h.make_node("Mul", ["p_raw", "p_mask2"], ["p_masked"]),
        h.make_node("Constant", [], ["axes0"], value=h.make_tensor("axes0_t", TP.INT64, [1], [0])),
        h.make_node("Squeeze", ["p_masked", "axes0"], ["schema_emb_p"]),
        h.make_node("Gather", ["row0", "field_positions"], ["schema_emb_fields"], axis=0),
    ]
    graph = h.make_graph(nodes, "gather_adapter", inputs, outputs, [zero_scalar, h_const])
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)])
    onnx.checker.check_model(model)
    return model


@app.command("ner-full")
def ner_full(
    model_dir: Path,
    variant: str = typer.Argument("onnx"),
    scoring_variant: str = typer.Option(
        "",
        help="Recompose the scoring half from a different variant's heads (e.g. 'onnx' to score "
        "in fp32 under an fp16 encoder). Empty = same variant.",
    ),
):
    """Compose encoder + gather adapter + (freshly recomposed) span_scoring into ner_full.onnx.

    The span_scoring half is rebuilt IN MEMORY from span_rep + scoring_head every time — ner_full
    never reads a span_scoring.onnx off disk, so a stale intermediate can't poison it.
    """
    d = model_dir / variant
    scoring_dir = model_dir / (scoring_variant or variant)
    enc = onnx.load(str(d / "encoder.onnx"))
    ss = _compose_span_scoring(scoring_dir)
    ir = min(enc.ir_version, ss.ir_version)
    enc.ir_version = ss.ir_version = ir
    # the fp16 export keeps fp32 IO, so the adapter dtype follows the encoder output
    out_elem = enc.graph.output[0].type.tensor_type.elem_type
    adapter = _build_adapter(out_elem)
    adapter.ir_version = ir
    m1 = compose.merge_models(
        enc, adapter, io_map=[(enc.graph.output[0].name, "last_hidden_state")],
        prefix1="e_", prefix2="a_",
    )
    m1.ir_version = ir
    m2 = compose.merge_models(
        m1, ss,
        io_map=[
            ("a_token_embeddings", "token_embeddings"),
            ("a_schema_emb_p", "schema_emb_p"),
            ("a_schema_emb_fields", "schema_emb_fields"),
        ],
        prefix2="s_",
    )
    _unprefix(
        m2,
        {
            "e_input_ids": "input_ids",
            "e_attention_mask": "attention_mask",
            "a_word_positions": "word_positions",
            "a_p_position": "p_position",
            "a_field_positions": "field_positions",
            "s_span_idx": "span_idx",
            "s_count": "count",
            "s_count_logits": "count_logits",
            "s_span_scores": "span_scores",
        },
    )
    onnx.checker.check_model(m2, full_check=False)
    out = d / "ner_full.onnx"
    onnx.save(m2, str(out), save_as_external_data=m2.ByteSize() > 2**31 - 100)
    typer.echo(f"{out}: {[i.name for i in m2.graph.input]} -> {[o.name for o in m2.graph.output]}")


@app.command("classifier-full")
def classifier_full(
    model_dir: Path,
    variant: str = typer.Argument("onnx"),
    head_variant: str = typer.Option(
        "",
        help="Pull classifier_head.onnx from this variant instead of `variant` (fp16 parity of "
        "the head, if it ever proves shaky, can be sidestepped the way ner-full does for scoring).",
    ),
):
    """Compose encoder + in-graph label gather + classifier_head into classifier_full.onnx.

    classifier_head is rank-2 ([num_labels,768] -> [num_labels,1]); the adapter gathers the label
    marker rows for every batch row (Gather axis=1) and flattens [b,L,768] -> [b*L,768] so the head
    stays batched. The output logits are [b*L,1]; the Java side reshapes back to [b][L].
    """
    d = model_dir / variant
    head_dir = model_dir / (head_variant or variant)
    enc = onnx.load(str(d / "encoder.onnx"))
    ch = onnx.load(str(head_dir / "classifier_head.onnx"))
    ir = min(enc.ir_version, ch.ir_version)
    enc.ir_version = ch.ir_version = ir
    elem = enc.graph.output[0].type.tensor_type.elem_type

    ins = [
        h.make_tensor_value_info("last_hidden_state", elem, ["b", "seq", HIDDEN]),
        h.make_tensor_value_info("label_positions", TP.INT64, ["num_labels"]),
    ]
    outs = [
        h.make_tensor_value_info("label_embeddings_flat", elem, ["bl", HIDDEN])
    ]
    nodes = [
        h.make_node("Gather", ["last_hidden_state", "label_positions"], ["gathered"], axis=1),
        h.make_node("Constant", [], ["flat_shape"], value=h.make_tensor("fs", TP.INT64, [2], [-1, HIDDEN])),
        h.make_node("Reshape", ["gathered", "flat_shape"], ["label_embeddings_flat"]),
    ]
    adapter = h.make_model(
        h.make_graph(nodes, "label_adapter", ins, outs), opset_imports=[h.make_opsetid("", 17)]
    )
    adapter.ir_version = ir
    onnx.checker.check_model(adapter)

    m1 = compose.merge_models(
        enc, adapter, io_map=[(enc.graph.output[0].name, "last_hidden_state")],
        prefix1="e_", prefix2="a_",
    )
    m1.ir_version = ir
    m2 = compose.merge_models(
        m1, ch, io_map=[("a_label_embeddings_flat", "label_embeddings")], prefix2="c_"
    )
    _unprefix(
        m2,
        {
            "e_input_ids": "input_ids",
            "e_attention_mask": "attention_mask",
            "a_label_positions": "label_positions",
            "c_logits": "logits",
        },
    )
    onnx.checker.check_model(m2, full_check=False)
    out = d / "classifier_full.onnx"
    onnx.save(m2, str(out), save_as_external_data=m2.ByteSize() > 2**31 - 100)
    typer.echo(f"{out}: {[i.name for i in m2.graph.input]} -> {[o.name for o in m2.graph.output]}")


@app.command("all")
def build_all(model_dir: Path, variant: str = typer.Argument("onnx")):
    """ner-full + classifier-full for one variant (span-scoring is recomposed in-memory)."""
    ner_full(model_dir, variant, scoring_variant="")
    classifier_full(model_dir, variant, head_variant="")


@app.command("rebuild")
def rebuild(model_dir: Path):
    """ONE command: regenerate every merged graph for a model, for all present variants, from the
    authoritative split heads.

    For each of onnx / onnx_fp16 that exists, rebuilds ner_full + classifier_full by composing the
    encoder / span_rep / scoring_head / classifier_head in memory — no intermediate is ever read
    off disk, so there is no stale-artifact class of bug. (Run `optimize-encoder` first if you want
    the fused encoder; this command uses whatever encoder.onnx is on disk.)
    """
    built = []
    for variant in ("onnx", "onnx_fp16"):
        d = model_dir / variant
        if not (d / "encoder.onnx").exists():
            continue
        if (d / "span_rep.onnx").exists() and (d / "scoring_head.onnx").exists():
            ner_full(model_dir, variant, scoring_variant="")
            built.append(f"{variant}/ner_full.onnx")
        if (d / "classifier_head.onnx").exists():
            classifier_full(model_dir, variant, head_variant="")
            built.append(f"{variant}/classifier_full.onnx")
    if not built:
        raise typer.Exit(
            typer.echo(
                f"nothing to build under {model_dir} — no variant has the split heads "
                "(encoder + span_rep/scoring_head or classifier_head)"
            )
            or 1
        )
    typer.echo("rebuilt: " + ", ".join(built))


# ── parity verification ──────────────────────────────────────────────────────


@app.command()
def verify(model_dir: Path, variant: str = typer.Argument("onnx"), seed: int = 3):
    """Parity of span_scoring and ner_full vs the split pipeline, on real encoder activations."""
    import onnxruntime as ort

    d = model_dir / variant
    rng = np.random.default_rng(seed)
    B, T, W, F, SEQ = 2, 30, 8, 5, 90
    ids = rng.integers(1, 5000, size=(B, SEQ)).astype(np.int64)
    mask = np.ones((B, SEQ), dtype=np.int64)
    wp = np.stack([np.sort(rng.choice(np.arange(40, SEQ), size=T, replace=False)) for _ in range(B)])
    wp[:, -3:] = -1  # padded words must map to zero rows
    p_pos = np.array([5], dtype=np.int64)
    f_pos = rng.choice(np.arange(7, 35), size=F, replace=False).astype(np.int64)
    sidx = np.zeros((B, T * W, 2), dtype=np.int64)
    for s in range(T):
        for w in range(W):
            if s + w < T:
                sidx[:, s * W + w] = [s, s + w]
    count = np.array(20, dtype=np.int64)

    cpu = ["CPUExecutionProvider"]
    hid = ort.InferenceSession(str(d / "encoder.onnx"), providers=cpu).run(
        None, {"input_ids": ids, "attention_mask": mask}
    )[0]
    tok = np.zeros((B, T, HIDDEN), dtype=np.float32)
    for b in range(B):
        for t in range(T):
            if wp[b, t] >= 0:
                tok[b, t] = hid[b, wp[b, t]]

    sr = ort.InferenceSession(str(d / "span_rep.onnx"), providers=cpu).run(
        None, {"token_embeddings": tok, "span_idx": sidx}
    )[0]
    ref = ort.InferenceSession(str(d / "scoring_head.onnx"), providers=cpu).run(
        None,
        {"span_rep": sr.astype(np.float32), "schema_emb_p": hid[0, p_pos[0]],
         "schema_emb_fields": hid[0][f_pos], "count": count},
    )

    # Recompose span_scoring in-memory (rebuild no longer writes it to disk) and run from bytes.
    ss_model = _compose_span_scoring(d)
    ss = ort.InferenceSession(
        ss_model.SerializeToString(), providers=cpu
    ).run(
        ["count_logits", "span_scores"],
        {"token_embeddings": tok, "span_idx": sidx, "schema_emb_p": hid[0, p_pos[0]],
         "schema_emb_fields": hid[0][f_pos], "count": count},
    )
    nf = ort.InferenceSession(str(d / "ner_full.onnx"), providers=cpu).run(
        ["count_logits", "span_scores"],
        {"input_ids": ids, "attention_mask": mask, "word_positions": wp, "p_position": p_pos,
         "field_positions": f_pos, "span_idx": sidx, "count": count},
    )

    nonzero = (nf[1] != 0).mean()
    if nonzero < 0.05:
        typer.echo("WARNING: span_scores nearly all zero — parity check is vacuous")
    tol = 0.0 if variant == "onnx" else 5e-3
    checks = [
        ("span_scoring count_logits", np.abs(ref[0] - ss[0]).max()),
        ("span_scoring span_scores", np.abs(ref[1] - ss[1]).max()),
        ("ner_full count_logits", np.abs(ref[0] - nf[0]).max()),
        ("ner_full span_scores", np.abs(ref[1] - nf[1]).max()),
    ]
    ok = True
    for name, diff in checks:
        status = "OK" if diff <= tol else "FAIL"
        ok &= diff <= tol
        typer.echo(f"  {name}: max |diff| = {diff:.2e} [{status}]")
    typer.echo(f"  span_scores nonzero: {nonzero:.0%}")
    raise typer.Exit(0 if ok else 1)


if __name__ == "__main__":
    app()
