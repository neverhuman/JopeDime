#!/usr/bin/env python3
"""Source-bound manifests: bind every published context's outer split and GPU inputs/outputs to its build (red-team 2026-10-07).

The gap-publish red-teams (receipts redteam-rawgroup-v2g-publish 80641c69, redteam-g3append-publish 10b9d8a8; both FAIL, clean
controls and population check clean) found 20 faults that pass strip, scan, key-scheme, the v2_publish dry run and the frozen
tools/v2_rowsel_raw_manifest.py (sha256 16b7f263...) with --require-cut:
  - a re-labelled outer split: a corrupted or other-dataset cut_id, and on the raw route (its context schema is outside
    v2_publish.CUT_SCHEMAS) also a v2/v2g prefix swap or another split schema; the frozen tool compares only context_sha256 and
    rights with the source build, so nothing binds the cut or split block;
  - a GPU output paired with the wrong context (another fold's .npy with its files entry updated): nothing binds GPU files to the
    runner's own record;
  - a forged gpu.input.binding (for example one that claims a target-copy feature was kept): only the encoder sha256 is compared.

This adapter refuses all three before any manifest file or client exists, then produces the manifest exactly as the #239 adapter
gpu_manifest_cut.py (pinned below; it runs the frozen tool unchanged except its source-bound cut_of) and adds a `binding_adapter`
block. For each context of the tree and its --source build context (same view_id; the frozen tool still refuses a missing source,
identity, rights or GPU-input mismatch):
  1 split   the cut block, the cut_id (the #239 source-bound cut_of on both sides, so a raw GPU context that inherits its cut is
            compared on the inherited value), the split block, rowsel.split_schema, split_version, fold and dataset_hash equal the
            source's;
  2 input   a GPU context's gpu.input.binding equals the runner's binding recomputed from the source encoder JTT1: train_rows and
            features_in from its header, selection_fit "outer_train_only" and train_block_sha256 = sha256 of the train range
            (dltrain.jtt1.train_range; the outer-test range is never decoded);
  3 output  with GPU contexts, --gpu-out DIR (tools/v2_gpu_encode.py OUT, repeatable) is required, and each context's GPU files
            (name, sha256, bytes) equal the runner's own record OUT/<view_id>/context.json (with .done).
Values compared are identities only: no label, prediction or outer-test value is read or copied into the manifest.
Usage (lane slice, from deploy/dl-trainonly/lane/src/dl_training_v1 like the frozen tool):
  python ../../../manifest_binding.py --build TREE --name NAME --source BUILD_TREE --summary PATH#SHA --require-cut
         [--gpu-out OUT ...] [--rights-research] [--publish]
  Optional --local-dir DIR (tests only): write the local manifest there; refused together with --publish.
"""

import contextlib
import glob
import gzip
import hashlib
import importlib.util
import io
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).absolute().parent
LANE = HERE / "lane/src/dl_training_v1"
sys.path.insert(0, str(LANE))  # the frozen tools import dltrain from the lane root, as in gpu_manifest_cut.py
CUT_ADAPTER = (
    HERE / "gpu_manifest_cut.py",
    "0d0037db8cd040fe0508661d86fa27b5888d6d1a49d7538b16ce9b7616e126a7",
)
BINDING_SCHEMA = "jope.dl-manifest-binding-adapter/v1"
SELECTION_FIT = "outer_train_only"
RULES = {
    "split": "cut block, source-bound cut_id, split block, rowsel.split_schema, split_version, fold and dataset_hash equal the "
    "--source build context with the same view_id",
    "input": "gpu.input.binding equals train_rows/features_in from the source encoder JTT1 header, selection_fit outer_train_only "
    "and sha256 of its train range",
    "output": "every GPU file (name, sha256, bytes) equals the runner record --gpu-out/<view_id>/context.json; required with GPU contexts",
}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load_cut_adapter():
    b = CUT_ADAPTER[0].read_bytes()
    if sha(b) != CUT_ADAPTER[1]:
        raise SystemExit(f"{CUT_ADAPTER[0]} is not the pinned #239 adapter {CUT_ADAPTER[1]}")
    spec = importlib.util.spec_from_file_location("gpu_manifest_cut", CUT_ADAPTER[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def adapter_identity(path=Path(__file__)):
    return {
        "schema": BINDING_SCHEMA,
        "path": "deploy/dl-trainonly/manifest_binding.py",
        "sha256": sha(Path(path).read_bytes()),
        "cut_adapter_sha256": CUT_ADAPTER[1],
        "rules": RULES,
    }


def source_index(trees):
    """view_id -> (context, dir) for the build trees, with the frozen tool's rule (shard-*/<view>/context.json with .done)."""
    out = {}
    for t in trees:
        for p in glob.glob(f"{t}/shard-*/*/context.json"):
            d = Path(p).parent
            if (d / ".done").exists():
                c = json.loads(Path(p).read_text())
                out[c["view_id"]] = (c, d)
    return out


def split_of(ctx):
    return ctx.get("split") or (ctx.get("rowsel") or {}).get("split")


def split_reason(ctx, sc, cut_of):
    if (ctx.get("cut") or {}) != (sc.get("cut") or {}):
        return "cut block differs from the source build"
    if cut_of(ctx) != cut_of(sc):
        return "cut_id differs from the source build"
    sp = split_of(ctx)
    if sp is not None and sp != split_of(sc):
        return "split block differs from the source build"
    schema = (ctx.get("rowsel") or {}).get("split_schema")
    if schema is not None and schema != (split_of(sc) or {}).get("schema"):
        return "rowsel.split_schema differs from the source build"
    for k in ("split_version", "fold", "dataset_hash"):
        if ctx.get(k) != sc.get(k):
            return f"{k} differs from the source build"
    return None


def runner_binding(src_dir):
    """The runner's input binding (tools/v2_gpu_encode.py Runner.encode) recomputed from the source encoder's train range only."""
    from dltrain import jtt1

    raw = gzip.decompress((Path(src_dir) / "encoder.jtt1.gz").read_bytes())
    h = jtt1.parse_header(raw)
    start, stop = jtt1.train_range(h)
    return {
        "train_rows": h["n_train"],
        "features_in": h["p"],
        "selection_fit": SELECTION_FIT,
        "train_block_sha256": sha(bytes(memoryview(raw)[start:stop])),
    }


def gpu_reason(ctx, src_dir, gpu_out):
    binding = (ctx["gpu"].get("input") or {}).get("binding") or {}
    want = runner_binding(src_dir)
    if {k: binding.get(k) for k in want} != want:
        return "gpu.input.binding does not match the source encoder's train block"
    rec = next(
        (Path(o) / ctx["view_id"] for o in gpu_out if (Path(o) / ctx["view_id"] / ".done").exists()),
        None,
    )
    if rec is None:
        return "no runner output record under --gpu-out"
    rf = json.loads((rec / "context.json").read_bytes()).get("files") or {}
    mine = {n: m for n, m in ctx["files"].items() if n.endswith(".npy")}
    if not mine or any({k: rf.get(n, {}).get(k) for k in ("sha256", "bytes")} != {k: m.get(k) for k in ("sha256", "bytes")}
                       for n, m in mine.items()):
        return "GPU files differ from the runner's output record"
    return None


def bind_tree(build, sources, gpu_out, cut_of, vp):
    """Refuse the first unbound context; returns (contexts bound, GPU contexts bound)."""
    plans, refused = vp.scan_build(Path(build))
    if refused:
        raise SystemExit(f"{len(refused)} contexts refused by the publisher scan; first {refused[0]}")
    src = source_index(sources)
    n = g = 0
    if any("gpu" in p["ctx"] for p in plans) and not gpu_out:
        raise SystemExit("GPU contexts need --gpu-out (the runner's output records)")
    for p in plans:
        ctx = p["ctx"]
        hit = src.get(p["view_id"])
        if hit is None:
            raise SystemExit(f"{p['view_id']}: not found in the source build tree")
        sc, sdir = hit
        why = split_reason(ctx, sc, cut_of)
        if why is None and "gpu" in ctx:
            why = gpu_reason(ctx, sdir, gpu_out)
            g += 1
        if why:
            raise SystemExit(f"{p['view_id']}: {why}; refused before any manifest")
        n += 1
    return n, g


def main(argv=None, clients=None, out_root=None):
    """clients/out_root are test seams (fake transport, local manifest directory); the CLI never sets them."""
    argv = list(sys.argv[1:] if argv is None else argv)
    gpu_out = []
    while "--gpu-out" in argv:
        i = argv.index("--gpu-out")
        gpu_out.append(argv[i + 1])
        del argv[i : i + 2]
    local = None
    if "--local-dir" in argv:
        i = argv.index("--local-dir")
        local = Path(argv[i + 1])
        del argv[i : i + 2]
    publish = "--publish" in argv
    if local is not None and publish:
        raise SystemExit("--local-dir is for tests only; never with --publish")
    argv = [x for x in argv if x != "--publish"]
    gmc = load_cut_adapter()
    m, vp = gmc.load(gmc.TOOL, "v2_rowsel_raw_manifest"), gmc.load(gmc.TRANSPORT, "v2_publish")
    name = argv[argv.index("--name") + 1]
    rights_research = "--rights-research" in argv
    if "--ledger" not in argv:
        argv += ["--ledger", str(m.TV2 / "publish" / "v2-publish-ledger.jsonl")]
    ledger_path = argv[argv.index("--ledger") + 1]
    sources = [argv[i + 1] for i, x in enumerate(argv) if x == "--source"]
    cut_of = gmc.source_bound_cut_of(m.cut_of, {v: c for v, (c, _) in source_index(sources).items()})
    bound, gpu_bound = bind_tree(argv[argv.index("--build") + 1], sources, gpu_out, cut_of, vp)
    with tempfile.TemporaryDirectory(prefix="manifest-binding-") as tmp:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = gmc.main(argv + ["--local-dir", tmp])
        summary = json.loads(buf.getvalue().strip().splitlines()[-1])
        draft = json.loads(Path(summary["local"]).read_bytes())
    doc = dict(
        draft,
        binding_adapter=dict(
            adapter_identity(), contexts_bound=bound, gpu_contexts_bound=gpu_bound, gpu_out=sorted(gpu_out)
        ),
    )
    body = gmc.canonical(doc)
    digest = sha(body)
    root = local if local is not None else (Path(out_root) if out_root is not None else m.TV2)
    out = root / "publish" / f"v2-publish-manifest-{name}-{digest[:12]}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        if out.read_bytes() != body:
            raise SystemExit(f"{out} exists with different bytes")
    else:
        with open(out, "xb") as f:
            f.write(body)
    summary.update(
        local=str(out),
        sha256=digest,
        binding_adapter_sha256=doc["binding_adapter"]["sha256"],
        contexts_bound=bound,
        gpu_contexts_bound=gpu_bound,
        cut_adapter_draft_sha256=summary["sha256"],
    )
    if publish:
        if draft.get("missing_in_ledger"):
            raise SystemExit(f"refusing to publish: {draft['missing_in_ledger']} objects lack a verified ledger record")
        writer, reader = clients if clients is not None else vp.real_clients()
        summary["published"] = gmc.publish_manifest(vp, writer, reader, ledger_path, name, rights_research, body)
    print(json.dumps(summary, sort_keys=True))
    return rc


if __name__ == "__main__":
    sys.exit(main())
