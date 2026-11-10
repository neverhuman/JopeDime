#!/usr/bin/env python3
"""H317 GP staging manifest + DL-Val exclusion check (run inside sandbox_h317.sh on xbabe2, used by stage_xbabe0.sh).

Groups (source root on xbabe2 -> target root on xbabe0):
  code        H317 source dir (top-level *.py *.sh *.md *.json)              -> $STAGE/code
  train_view  res01-train-view (1,219 TRAIN tables + pools, champions, winners, source_manifest, redacted selected.json)
                                                                              -> $STAGE/res01-train-view
  micro_v     Stage V only: the 5 LOCKed DOPE/encoder files (dope_mlc_v3_multitask_best.pt a36510b9...,
              frozen_stage1_stage2_inference_params_v3.pt 136b2381..., python/model.py 0aa76360...,
              python/hpc/landmark_workspace.py, bin/libjope_hpc.so) -> $STAGE/micro (hash-only, never executed)
  micro       JopeDimeMicro subset: bin/bootstrap.s1main, python/{rust_evaluator,rank_prepare,phase12_multimodal_model}.py,
              assets/projections/program-feature-names.json (no assets/benchmark_data, no results)  -> $STAGE/micro
  views       $H317_DATA/neural/views_c32_train1219.npz + $H317_DATA/views-search/<cid>.npz (SEARCH-row view sets of
              the pilot) + views-search/SHA256SUMS (5bf608e5...)               -> $H317_DATA on xbabe0 (mounted ro)
  screener    $H317_DATA/neural/runs/H0.1-A/seed{317101,317102}/fold{0..4}/{model.pt,config.json,receipt.json}
              (model sha == receipt.model_sha256; training cid lists TRAIN only)  -> $H317_DATA on xbabe0 (ro)
Asserts (any failure exits 1, nothing is written): no staged path or npz cid list names a DL-Val context id (12-hex
prefix), dataset id or family id; tables/pools are exactly the 1,219 TRAIN cids; selected.json is the redacted
7d64774a...; worker sha e3be9386...; views_c32 cids are TRAIN; views-search files are kind=search, recipe
h317-views-v1, one per pilot context. Writes <out>.json (sha256 + bytes per file) and one <out>.<group>.sha256 list per
group in `sha256sum -c` format, paths relative to the group root.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

C.install_guard()

import numpy as np  # noqa: E402

REDACTED_SELECTED_SHA256 = (
    "7d64774ad9dacb5f"  # prefix of the redacted selected.json (parent, 13:16Z)
)
VIEWS_SEARCH_SUMS_SHA256 = (
    "5bf608e5"  # prefix of views-search/SHA256SUMS (parent, P205)
)
SCREENER_RUN = "neural/runs/H0.1-A"
WORKER_SHA256 = "e3be938644722d14a1c1a0e82168518838c4d174cfe031076a339745c48870a8"
# Stage V only (stage_v_mounts.md §1): the LOCKed DOPE / encoder files, staged so resume_stage_v can hash every LOCK
# file in the xbabe0 container (the audit never runs the encoder). Known pins: H58 cache manifest values.
MICRO_V_FILES = {
    "assets/models/dope_mlc_v3_multitask_best.pt": "a36510b9",
    "assets/projections/frozen_stage1_stage2_inference_params_v3.pt": "136b2381",
    "python/model.py": "0aa76360",
    "python/hpc/landmark_workspace.py": None,
    "bin/libjope_hpc.so": None,
}
MICRO_FILES = (
    "bin/bootstrap.s1main",
    "python/rust_evaluator.py",
    "python/rank_prepare.py",
    "python/phase12_multimodal_model.py",
    "assets/projections/program-feature-names.json",
)


def sha(p: Path) -> str:
    return C.sha256_file(p)


def main() -> None:
    out = Path(sys.argv[1])
    train_view = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else C.DATA_ROOT / "res01-train-view"
    )
    sel = json.loads((train_view / "selected.json").read_text())
    if not sha(train_view / "selected.json").startswith(REDACTED_SELECTED_SHA256):
        raise SystemExit("selected.json is not the redacted TRAIN-view copy")
    dev = sel["dev"]
    bad_tokens = (
        {r["context_sha256"][:12] for r in dev}
        | {r["dataset_id"] for r in dev}
        | {r["family_id"].split("-")[-1][:12] for r in dev}
    )
    rows = C.train_rows()
    train = {r["context_sha256"] for r in rows}
    pilot = set(C.pilot_shard(rows, C.family_folds(rows)))

    def clean(rel: str) -> None:
        low = rel.lower()
        hit = [t for t in bad_tokens if t in low]
        if hit:
            raise SystemExit(f"DL-Val identifier {hit[0]} in staged path {rel}")

    groups = {}
    files = sorted(
        p
        for p in HERE.iterdir()
        if p.is_file() and p.suffix in (".py", ".sh", ".md", ".json")
    )
    groups["code"] = (HERE, [p.name for p in files])
    tv = []
    for sub in ("tables", "pools"):
        names = sorted(p.name for p in (train_view / sub).iterdir())
        cids = {n.split(".")[0] for n in names}
        if cids != train or len(names) != len(train):
            raise SystemExit(f"{sub}: staged set != the 1,219 TRAIN cids")
        tv += [f"{sub}/{n}" for n in names]
    tv += [
        n
        for n in (
            "champions.json",
            "winners.json",
            "source_manifest.json",
            "selected.json",
        )
    ]
    groups["train_view"] = (train_view, tv)
    if sha(C.MICRO / "bin" / "bootstrap.s1main") != WORKER_SHA256:
        raise SystemExit("worker sha mismatch")
    groups["micro"] = (C.MICRO, list(MICRO_FILES))
    for rel, pin in MICRO_V_FILES.items():
        if pin is not None and not sha(C.MICRO / rel).startswith(pin):
            raise SystemExit(f"{rel}: sha does not match the pinned {pin}...")
    groups["micro_v"] = (C.MICRO, list(MICRO_V_FILES))
    merged = C.DATA_ROOT / "neural" / "views_c32_train1219.npz"
    with np.load(merged, allow_pickle=False) as z:
        mc = {str(c) for c in z["cids"]}
    if not mc <= train or any(c[:12] in bad_tokens for c in mc):
        raise SystemExit("views_c32 contains a non-TRAIN cid")
    vs = []
    for p in sorted((C.DATA_ROOT / "views-search").glob("*.npz")):
        cid = p.stem
        if cid not in pilot:
            raise SystemExit(f"views-search/{p.name} is not a pilot context")
        with np.load(p, allow_pickle=False) as z:
            if str(z["kind"]) != "search" or str(z["recipe"]) != "h317-views-v1":
                raise SystemExit(
                    f"{p.name}: not a SEARCH view set of recipe h317-views-v1"
                )
        vs.append(f"views-search/{p.name}")
    if len(vs) != len(pilot):
        raise SystemExit(
            f"views-search has {len(vs)} pilot view sets, need {len(pilot)}"
        )
    if not sha(C.DATA_ROOT / "views-search" / "SHA256SUMS").startswith(
        VIEWS_SEARCH_SUMS_SHA256
    ):
        raise SystemExit(
            "views-search/SHA256SUMS is not the announced 5bf608e5... list"
        )
    groups["views"] = (
        C.DATA_ROOT,
        ["neural/views_c32_train1219.npz"] + vs + ["views-search/SHA256SUMS"],
    )
    import torch

    scr = []
    for d in sorted((C.DATA_ROOT / SCREENER_RUN).glob("seed*/fold*")):
        rec = json.loads((d / "receipt.json").read_text())
        if sha(d / "model.pt") != rec["model_sha256"]:
            raise SystemExit(f"{d}: model.pt sha != receipt")
        ck = torch.load(d / "model.pt", map_location="cpu", weights_only=False)
        if not set(ck["train_cids"]) <= train or any(
            c[:12] in bad_tokens for c in ck["train_cids"]
        ):
            raise SystemExit(f"{d}: training cid list is not TRAIN-only")
        rel = d.relative_to(C.DATA_ROOT)
        scr += [f"{rel}/model.pt", f"{rel}/config.json", f"{rel}/receipt.json"]
    if len(scr) != 3 * 2 * C.N_FOLDS:
        raise SystemExit(
            f"screener run has {len(scr) // 3} fold models, need {2 * C.N_FOLDS}"
        )
    groups["screener"] = (C.DATA_ROOT, scr)
    manifest = {"pilot_n": len(pilot), "groups": {}}
    for g, (root, rels) in groups.items():
        ent = {}
        for rel in rels:
            clean(rel)
            p = root / rel
            if g == "code" or (
                g == "train_view" and rel.endswith(".json") and rel != "selected.json"
            ):
                low = p.read_text(
                    errors="replace"
                ).lower()  # text files: no DL-Val identifier in the content
                hit = [t for t in bad_tokens if t in low]
                if hit:
                    raise SystemExit(
                        f"DL-Val identifier {hit[0]} inside staged file {rel}"
                    )
            ent[rel] = {"sha256": sha(p), "bytes": p.stat().st_size}
        manifest["groups"][g] = {
            "source_root": str(root),
            "files": ent,
            "n_files": len(ent),
            "bytes": int(sum(v["bytes"] for v in ent.values())),
        }
        with open(out.with_suffix(f".{g}.sha256"), "w") as f:
            for rel, v in ent.items():
                f.write(f"{v['sha256']}  {rel}\n")
        with open(out.with_suffix(f".{g}.files"), "w") as f:
            f.write("\n".join(rels) + "\n")
    manifest["manifest_sha256"] = hashlib.sha256(
        json.dumps(manifest["groups"], sort_keys=True).encode()
    ).hexdigest()
    manifest["guard"] = C.guard_report()
    C.write_json(out.with_suffix(".json"), manifest)
    print(
        json.dumps(
            {g: (v["n_files"], v["bytes"]) for g, v in manifest["groups"].items()}
            | {"manifest_sha256": manifest["manifest_sha256"]}
        )
    )


if __name__ == "__main__":
    main()
