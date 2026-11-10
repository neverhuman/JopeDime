#!/usr/bin/env python3
"""H417 APEX core: fresh-family contexts through the frozen H317 / H335 / H410 / H415 / H416 GP stack, unmodified.

The frozen modules know only the pinned TRAIN roster (1,219 contexts / 205 families). APEX adds the 1,153 fresh families
of H286 / H288 / H292 / H294 (one train-only JTT1 context each; manifest APEX_COHORTS.json, pinned below) without
changing any frozen byte:
  * `C.table_path` is wrapped: an APEX context resolves to $H317_DATA/apex/tables/<cid>.jtt1; every other context goes
    to the original function (the pinned TRAIN view);
  * `build_rt` is `h335_lab.build_rt`'s Phase V construction (the one used for new families on dev91) without the
    Stage V unlock: bank = TRAIN champions / winners only, family and dataset ids of the APEX rows added to the
    exclusion maps, and the fold-0 out-of-fold screener pair for every APEX target (no fold model saw an APEX family).
    A TRAIN context keeps its own fold's screener (`FoldScreeners.for_context`), so stored TRAIN results replay.
The read guard stays installed from the pinned TRAIN roster view; `selfcheck` asserts that no APEX identifier collides
with a DL-Val identifier and that the guard's matcher still holds the 91 DL-Val ids.

Manifest, cohorts and tiers: apex_core_meta.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import deploy_h416 as DH  # noqa: E402  (FIRST: deploy_h415 -> deploy_h410 -> gp_h381 -> h335_core pins + guard)

import crop_shuffle_synthetic_views as V  # noqa: E402  (pinned H317 module: the DOPE view recipe)
import screener_h317 as S  # noqa: E402  (pinned H317 module, already imported by the stack)

DP15, DP, GPH, K, C, LAB, L, D, ST, G, E = (DH.DP15, DH.DP, DH.GPH, DH.K, DH.C, DH.LAB, DH.L, DH.D, DH.ST, DH.G,
                                            DH.E)

import apex_core_meta as AM  # noqa: E402  (manifest, cohorts, tiers; stdlib only)

WAVE = "H417"
MANIFEST_SHA256 = AM.MANIFEST_SHA256
TABLES = Path(C.DATA_ROOT) / "apex" / "tables"
COHORTS = AM.COHORTS
SEARCH_SEED = 417101

if AM.data_root() != Path(C.DATA_ROOT):
    raise RuntimeError("apex_core_meta data root differs from H317_DATA")
rows, is_core, cohort, panel, family_of = AM.rows, AM.is_core, AM.cohort, AM.panel, AM.family_of


# ----------------------------------------------------------------------------------------------------------------------
# Table path + guard self-check
# ----------------------------------------------------------------------------------------------------------------------
_ORIG_TABLE_PATH = C.table_path
_ORIG_CANON_SHA = V.canonical_table_sha256


def _table_path(cid: str) -> Path:
    if cid in rows():
        C._require_guard()
        return TABLES / f"{cid}.jtt1"
    return _ORIG_TABLE_PATH(cid)


def _canonical_table_sha256(cid: str):
    r = rows().get(cid)
    return r["train_only_jtt1_sha256"] if r is not None else _ORIG_CANON_SHA(cid)


def install() -> None:
    """Wrap C.table_path and the view recipe's canonical-sha lookup (idempotent) and run the guard self-check."""
    selfcheck()
    C.table_path = _table_path
    V.canonical_table_sha256 = _canonical_table_sha256  # the view recipe checks the table bytes against this sha


def selfcheck() -> dict:
    if not C._STATE["installed"] or C._STATE["stage_v"]:
        raise RuntimeError("read guard must be installed and Stage V locked")
    train = C.train_rows()
    t_cids = {r["context_sha256"] for r in train}
    t_ds = {r["dataset_id"] for r in train}
    n = 0
    for cid, r in rows().items():
        if cid in t_cids or r["dataset_id"] in t_ds:
            raise RuntimeError(f"APEX context {cid[:12]} is in the TRAIN roster")
        for ident in (cid, r["dataset_id"].lower(), r["family_key"]):
            if C._dev_named(ident):
                raise RuntimeError(f"APEX identifier collides with a DL-Val identifier: {ident}")
        n += 1
    # The matcher must still know the 91 DL-Val ids (no refusal is provoked here: a probe would write to the shared
    # cross-process refusal log).
    if len(C._STATE["dev_cids"]) != 91 or not all(C._dev_named(c) for c in C._STATE["dev_cids"]):
        raise RuntimeError("guard state does not hold the 91 DL-Val ids")
    return {"apex_contexts": n, "guard_knows_dlval_ids": 91}


# ----------------------------------------------------------------------------------------------------------------------
# Runtime
# ----------------------------------------------------------------------------------------------------------------------
def _screener_for(fs):
    f0 = fs.by_fold[0]
    apex = rows()
    return lambda cid: f0 if cid in apex else fs.for_context(cid)


def build_rt(run_dir: Path, cores: Sequence[int], worker: Path, timeout_s: float, seeds: Sequence[int],
             neural_dirs: Dict[str, Path], torch_threads: int = 1) -> Tuple[object, Dict[str, object], dict]:
    """(rt, {source: screener_for}, screener manifest). rt.screener_for = the first source."""
    import torch

    install()
    torch.set_num_threads(max(1, int(torch_threads)))
    G.MAX_EXPANDED_WIDTH = L.RUN_MAX_EXPANDED_WIDTH
    runner = K.rust_runner_cpu(Path(worker), timeout_s)
    gp_root = Path(run_dir) / f"gproot_{K.GROUP}"
    train = C.train_rows()
    fam_of = {r["context_sha256"]: r["family_id"] for r in train}
    ds_of = {r["context_sha256"]: r["dataset_id"] for r in train}
    for cid, r in rows().items():
        fam_of[cid] = r["family_id"]
        ds_of[cid] = r["dataset_id"]
    folds = C.family_folds(train)
    bank, _ = D.build_bank(train)
    rg = E.ResourceGuard(log_path=Path(run_dir) / "resource_pauses.jsonl")
    engine = E.EvalEngine(cores, root=gp_root, timeout_s=timeout_s, guard=rg, runner=runner)
    views = S.SearchViewProvider()
    scr, man = {}, {}
    for name, nd in neural_dirs.items():
        fs = S.load_neural_fold_screeners(Path(nd), folds, views, family_of=fam_of)
        man[name] = L.screener_manifest(Path(nd), fs)
        scr[name] = _screener_for(fs)
    first = next(iter(neural_dirs))
    rt = L.Runtime(engine, E.AuditLedger(engine), scr[first], D.NeuralContextCoder(views), bank, fam_of, ds_of,
                   L.Store(Path(run_dir)), tuple(seeds))
    return rt, scr, man


def code_files() -> Tuple[str, ...]:
    return ("apex_core.py", "apex_core_meta.py") + DH.code_files()


if __name__ == "__main__":
    install()
    r = rows()
    print(json.dumps({"wave": WAVE, "manifest_sha256": MANIFEST_SHA256, "selfcheck": selfcheck(),
                      "cohorts": {c: {"all": len(cohort(c, "all")), "core": len(cohort(c))} for c in COHORTS},
                      "guard": C.guard_report()}))
