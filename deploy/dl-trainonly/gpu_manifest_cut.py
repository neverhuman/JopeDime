#!/usr/bin/env python3
"""Manifest for raw-route GPU encoding sets whose context.json carries no rowsel.split (owner ENG:watcher 16:28 MT item 3).

The GPU encoder copied a slim parent context: rowsel.identity and split_schema, but no rowsel.split block, so the frozen
tools/v2_rowsel_raw_manifest.py (sha256 16b7f263...) finds no cut and --require-cut refuses the set (publish-batch-v3 exit 7,
gpu-rowsel-raw-clean-v2). This wrapper runs that tool UNCHANGED except cut_of(): for a context with a `gpu` block and no cut of
its own, the cut is the tool's own cut_of() applied to the --source build-tree context with the SAME view_id and the SAME
context_sha256 (the tool separately refuses any identity, rights or GPU-input mismatch with that source). The value is
'trainonly-split-v2:' + rowsel.split.sha256, the target-free outer-split identity; no rows, labels, predictions or outer-test
values are read or copied. A GPU context whose source is absent or differs keeps cut None, so --require-cut still refuses it.
Usage (lane slice, from deploy/dl-trainonly/lane/src/dl_training_v1 like the frozen tool):
  python ../../../gpu_manifest_cut.py --build TREE --name NAME --source BUILD_TREE --summary PATH#SHA --require-cut [--rights-research] [--publish]
  Optional --local-dir DIR (tests only): write the local manifest there; refused together with --publish.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).absolute().parent
TOOL = (HERE / "lane/src/dl_training_v1/tools/v2_rowsel_raw_manifest.py",
        "16b7f2631dd57f15b5ecf84c603ef6e10add7ff49fc7a789213c885c43931748")   # frozen #224 custody copy, never modified


def load(tool=TOOL):
    b = tool[0].read_bytes()
    if hashlib.sha256(b).hexdigest() != tool[1]:
        raise SystemExit(f"{tool[0]} is not the pinned manifest tool {tool[1]}")
    spec = importlib.util.spec_from_file_location("v2_rowsel_raw_manifest", tool[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def source_bound_cut_of(orig, src_index):
    def cut_of(ctx):
        c = orig(ctx)
        if c is not None or "gpu" not in ctx:
            return c
        sc = src_index.get(ctx.get("view_id"))
        if sc is None or sc.get("context_sha256") != ctx.get("context_sha256"):
            return None
        return orig(sc)
    return cut_of


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    local = None
    if "--local-dir" in argv:
        i = argv.index("--local-dir")
        local = Path(argv[i + 1])
        del argv[i:i + 2]
    m = load()
    sources = [argv[i + 1] for i, x in enumerate(argv) if x == "--source"]
    src_index = {}
    for s in sources:
        src_index.update(m.source_ctx(Path(s)))
    m.cut_of = source_bound_cut_of(m.cut_of, src_index)
    if local is not None:
        if "--publish" in argv:
            raise SystemExit("--local-dir is for tests only; never with --publish")
        (local / "publish").mkdir(parents=True, exist_ok=True)
        if "--ledger" not in argv:
            argv += ["--ledger", str(m.TV2 / "publish" / "v2-publish-ledger.jsonl")]
        m.TV2 = local
    return m.main(argv)


if __name__ == "__main__":
    sys.exit(main())
