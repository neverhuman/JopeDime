#!/usr/bin/env python3
"""Manifest for raw-route GPU encoding sets whose context.json carries no rowsel.split (owner ENG:watcher 16:28 MT item 3).

The GPU encoder copied a slim parent context: rowsel.identity and split_schema, but no rowsel.split block, so the frozen
tools/v2_rowsel_raw_manifest.py (sha256 16b7f263...) finds no cut and --require-cut refuses the set (publish-batch-v3 exit 7,
gpu-rowsel-raw-clean-v2). This adapter runs that tool UNCHANGED except cut_of(): for a context with a `gpu` block and no cut
of its own, the cut is the tool's own cut_of() applied to the --source build-tree context with the SAME view_id and the SAME
context_sha256 (the tool separately refuses any identity, rights or GPU-input mismatch with that source). The inherited value
must be the target-free outer-split identity: the source split schema must be rowsel-raw-split-v2 and its sha256 a lowercase
64-hex digest, and the result must read 'trainonly-split-v2:' + that digest; anything else stops the run before a manifest
or client exists. No rows, labels, predictions or outer-test values are read or copied. A GPU context whose source is absent
or has a different identity keeps cut None, so --require-cut still refuses it.

Provenance: the frozen tool only drafts the manifest into a temporary directory (it never sees --publish). The adapter adds a
`cut_adapter` block (its own sha256, the frozen tool and transport sha256, the rule and the views whose cut was inherited),
recomputes the canonical digest, writes the local manifest create-only and, with --publish, PUTs it through the frozen #224
transport exactly as the tool does (same key scheme, create-only, exact-VersionId readback).
Usage (lane slice, from deploy/dl-trainonly/lane/src/dl_training_v1 like the frozen tool):
  python ../../../gpu_manifest_cut.py --build TREE --name NAME --source BUILD_TREE --summary PATH#SHA --require-cut [--rights-research] [--publish]
  Optional --local-dir DIR (tests only): write the local manifest there; refused together with --publish.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).absolute().parent
LANE = HERE / "lane/src/dl_training_v1"
sys.path.insert(
    0, str(LANE)
)  # the frozen tool and transport import dltrain from the lane root, as in hwm_publish.py
TOOL = (
    LANE / "tools/v2_rowsel_raw_manifest.py",
    "16b7f2631dd57f15b5ecf84c603ef6e10add7ff49fc7a789213c885c43931748",
)
TRANSPORT = (
    LANE / "tools/v2_publish.py",
    "f175f9a48f52155450a92837e0b461aec6cf7ee03afd11732c563c77fa973c7a",
)
ADAPTER_SCHEMA = "jope.dl-gpu-manifest-cut-adapter/v1"
RAW_SPLIT_SCHEMA = "rowsel-raw-split-v2"
HEX64 = re.compile(r"[0-9a-f]{64}")
CUT = re.compile(r"trainonly-split-v2:[0-9a-f]{64}")
RULE = (
    "GPU context without its own cut: cut = frozen cut_of(source) for the --source context with the same view_id and "
    "context_sha256; source split schema rowsel-raw-split-v2 with a lowercase 64-hex sha256; nothing else is read"
)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def load(pin, name):
    b = pin[0].read_bytes()
    if sha(b) != pin[1]:
        raise SystemExit(f"{pin[0]} is not the pinned {name} {pin[1]}")
    spec = importlib.util.spec_from_file_location(name, pin[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def adapter_identity(path=Path(__file__)):
    return {
        "schema": ADAPTER_SCHEMA,
        "path": "deploy/dl-trainonly/gpu_manifest_cut.py",
        "sha256": sha(Path(path).read_bytes()),
        "frozen_tool_sha256": TOOL[1],
        "transport_sha256": TRANSPORT[1],
        "rule": RULE,
    }


def source_bound_cut_of(orig, src_index, inherited=None):
    def cut_of(ctx):
        c = orig(ctx)
        if c is not None or "gpu" not in ctx:
            return c
        sc = src_index.get(ctx.get("view_id"))
        if sc is None or sc.get("context_sha256") != ctx.get("context_sha256"):
            return None
        sp = (sc.get("rowsel") or {}).get("split") or {}
        digest = sp.get("sha256")
        if (
            sp.get("schema") != RAW_SPLIT_SCHEMA
            or not isinstance(digest, str)
            or not HEX64.fullmatch(digest)
        ):
            raise SystemExit(
                f"{ctx.get('view_id')}: source split is not a {RAW_SPLIT_SCHEMA} lowercase hex digest; refused before any manifest"
            )
        cut = orig(sc)
        if (
            not isinstance(cut, str)
            or not CUT.fullmatch(cut)
            or cut != "trainonly-split-v2:" + digest
        ):
            raise SystemExit(
                f"{ctx.get('view_id')}: inherited cut {str(cut)[:40]!r} is malformed; refused before any manifest"
            )
        if inherited is not None:
            inherited.add(ctx["view_id"])
        return cut

    return cut_of


def canonical(doc):
    return (json.dumps(doc, sort_keys=True, separators=(",", ":")) + "\n").encode()


def publish_manifest(vp, writer, reader, ledger_path, name, rights_research, body):
    """The frozen tool's own publish step: same key scheme, create-only PUT, exact-VersionId readback."""
    key = f"{vp.s3io.V2_PREFIX_KEY}{'research-only/' if rights_research else ''}manifests/{name}/{sha(body)[:12]}.json"
    ledger = vp.PublishLedger(Path(ledger_path))
    try:
        return vp.Publisher(writer, reader, ledger).publish_bytes(
            key, body, "application/json", "manifest", f"manifest:{name}"
        )
    finally:
        ledger.close()


def main(argv=None, clients=None, out_root=None):
    """clients/out_root are test seams (fake transport, local manifest directory); the CLI never sets them."""
    argv = list(sys.argv[1:] if argv is None else argv)
    local = None
    if "--local-dir" in argv:
        i = argv.index("--local-dir")
        local = Path(argv[i + 1])
        del argv[i : i + 2]
    publish = "--publish" in argv
    if local is not None and publish:
        raise SystemExit("--local-dir is for tests only; never with --publish")
    argv = [x for x in argv if x != "--publish"]
    m, vp = load(TOOL, "v2_rowsel_raw_manifest"), load(TRANSPORT, "v2_publish")
    name = argv[argv.index("--name") + 1]
    rights_research = "--rights-research" in argv
    if "--ledger" not in argv:
        argv += ["--ledger", str(m.TV2 / "publish" / "v2-publish-ledger.jsonl")]
    ledger_path = argv[argv.index("--ledger") + 1]
    out_root = (
        local
        if local is not None
        else (Path(out_root) if out_root is not None else m.TV2)
    )
    src_index, inherited = {}, set()
    for i, x in enumerate(argv):
        if x == "--source":
            src_index.update(m.source_ctx(Path(argv[i + 1])))
    m.cut_of = source_bound_cut_of(m.cut_of, src_index, inherited)
    with tempfile.TemporaryDirectory(prefix="gpu-manifest-cut-") as tmp:
        m.TV2 = Path(tmp)
        (m.TV2 / "publish").mkdir()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = m.main(argv)
        summary = json.loads(buf.getvalue().strip().splitlines()[-1])
        draft = json.loads(Path(summary["local"]).read_bytes())
    doc = dict(
        draft,
        cut_adapter=dict(adapter_identity(), inherited_cut_views=sorted(inherited)),
    )
    body = canonical(doc)
    digest = sha(body)
    out = out_root / "publish" / f"v2-publish-manifest-{name}-{digest[:12]}.json"
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
        cut_adapter_sha256=doc["cut_adapter"]["sha256"],
        inherited_cut_views=len(inherited),
        frozen_draft_sha256=summary["sha256"],
    )
    if publish:
        if draft.get("missing_in_ledger"):
            raise SystemExit(
                f"refusing to publish: {draft['missing_in_ledger']} objects lack a verified ledger record"
            )
        writer, reader = clients if clients is not None else vp.real_clients()
        summary["published"] = publish_manifest(
            vp, writer, reader, ledger_path, name, rights_research, body
        )
    print(json.dumps(summary, sort_keys=True))
    return rc


if __name__ == "__main__":
    sys.exit(main())
