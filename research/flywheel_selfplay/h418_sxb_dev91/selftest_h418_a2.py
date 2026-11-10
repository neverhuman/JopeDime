#!/usr/bin/env python3
"""H418 AMENDMENT_2 self-test on SYNTHETIC fixtures only: no dev91 byte, no board file, no worker, no GP.

selftest_h418.py's fixture builder and checks, at their published bytes, are run against analyze_h418_a2 (the
module `selftest_h418.A` is rebound to it). Every fixture gets an A2 cells file (--a2-cells). A GE / GL replay
failure is made by editing the fixture's H415 pool record after build(): the `<source>.stock` value (+3.4e-5) and/or
its last member.

Cases:
  * every selftest_h418 case with an empty A2 list: the clean and amendment-1 cases score exactly as there, create-once
    holds, the real tools/ladder.py reproduces the Elo, and each single harness fault is INCOMPLETE on its own gate;
  * listed GE / GL failures score X0b on that (context, seed) and report the deploy's own value as descriptive only
    (its Elo equals an independent elo_panel fit); also with an amendment-1 exemption in the same fixture;
  * an unlisted failure, a listed cell that does not fail, a second unlisted failure, the wrong source, the wrong seed
    and a malformed A2 file are each INCOMPLETE on their own gates, with no SXB value in the output.

  selftest_h418_a2.py --dir NEW_DIR [--ladder PATH/TO/ladder.py]
Exit 0 = every case passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_h418_a2 as A2  # noqa: E402
import selftest_h418 as S  # noqa: E402  (imports analyze_h418 as S.A; rebound below)

S.A = A2
np, H = S.np, S.H
SEEDS = A2.SEEDS
GE_CELL = (2, 335102)  # the fixture cell standing in for the real (43022ae6, 335102) GE failure
DELTA = 3.4e-5


def _perturb(fx: dict, i: int, s: int, src: str, how: str) -> None:
    """Make the deploy's replay of `src` differ from the arm record on (context i, seed s)."""
    c, t = fx["cids"][i], A2.TAGS[s]
    st = f"{t}r" if (i, s) == S.RECOMP_UNIT else t
    p = fx["root"] / "pulled" / t / "stages" / f"H415_{st}" / f"{c}__pool.jsonl"
    recs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    o = recs[-1]["outputs"][f"{src}.stock"]
    if how in ("value", "both"):
        o["value"] = o["value"] + DELTA
    if how in ("members", "both"):
        o["spec"]["members"] = o["spec"]["members"][:-1] + ["a2-swapped-member"]
    S._wl(p, recs)


def case(name: str, root: Path, m: dict, check, perturb=(), cells=(), raw_file=None) -> tuple:
    fx = S.build(root / name, m)
    for i, s, src, how in perturb:
        _perturb(fx, i, s, src, how)
    doc = raw_file if raw_file is not None else {
        "schema": A2.A2_SCHEMA, "cells": [{"cid": fx["cids"][i] if isinstance(i, int) else i, "seed": s,
                                           "source": src} for i, s, src in cells]}
    p = S._w(fx["root"] / "a2_cells.json", doc)
    fx["argv"] = fx["argv"] + ["--a2-cells", str(p)]
    rc, so = S.run(fx)
    try:
        ok, why = check(fx, rc, so)
    except Exception as e:  # noqa: BLE001  (a crashed check is a failed case)
        ok, why = False, f"check raised {type(e).__name__}: {e}"
    return name, ok, why, fx, rc


def chk_a2(fx, rc, so, *, a2cells, a1=None):
    """A2 cells (index, seed) score X0b; the deploy's own values appear only under descriptive.a2_deploy_values."""
    root, cids, x0, sxb = fx["root"], fx["cids"], fx["x0"], fx["sxb"]
    if rc != 0:
        return False, f"exit {rc!r}: {so[-400:]}"
    o = json.loads((root / "out.json").read_text())
    rows = [json.loads(x) for x in (root / "rows.jsonl").read_text().splitlines()]
    if [r["cid"] for r in rows] != sorted(cids) or any(sorted(r) != S.BOARD_KEYS for r in rows):
        return False, "rows: contexts or keys differ from the board record format"
    if o["rows"]["sha256"] != S._sha(root / "rows.jsonl"):
        return False, "rows sha"
    x0_cells = {S.ERR_UNIT, *a2cells} | ({a1} if a1 else set())
    for r in rows:
        i = cids.index(r["cid"])
        want = [x0[s][r["cid"]] if (i, s) in x0_cells else sxb[s][r["cid"]] for s in SEEDS]
        if r["per_seed"] != want or r["utility"] != sum(want) / 2:
            return False, f"row value {r['cid'][:8]}"
        if r["harness_version"] != A2.HARNESS or "amendment_2" not in r["source"]:
            return False, "row harness / source"
    vs = o["value_source"]
    for s in SEEDS:
        n_err = int(S.ERR_UNIT[1] == s)
        n_a1 = int(bool(a1) and a1[1] == s)
        n_a2 = sum(1 for _, ss in a2cells if ss == s)
        want = {"no_usable_H_x0b": n_err, "a1_exempt_x0b": n_a1, "a2_exempt_x0b": n_a2, "sxb": S.N - n_err - n_a1 - n_a2}
        if {k: vs[str(s)].get(k, 0) for k in want} != want:
            return False, f"value_source {vs}"
        if sorted(o["a2_exempt"][str(s)]) != sorted(cids[i][:12] for i, ss in a2cells if ss == s):
            return False, f"a2_exempt {o['a2_exempt']}"
    vals = {**fx["field"], A2.ENTRANT: {r["cid"]: r["utility"] for r in rows}}
    el = H.elo_panel(S._ladder_order(vals), fx["fam"], A2.ANCHOR, seed=A2.LADDER_SEED, n_boot=A2.N_BOOT)["elo"]
    ro = o["readout"]
    d = max(abs(el[e]["elo"] - ro["board_elo"][e]["elo"]) for e in el)
    if d > 1e-9:
        return False, f"board Elo differs from an independent elo_panel by {d:.2e}"
    rank = 1 + sum(1 for e in el if e != A2.ENTRANT and el[e]["elo"] > el[A2.ENTRANT]["elo"])
    if ro["rank"] != rank:
        return False, f"rank {ro['rank']} (want {rank})"
    de = ro["descriptive"].get("a2_deploy_values")
    if not de or de["units"] != len(a2cells):
        return False, "a2_deploy_values missing or wrong unit count"
    got = {(c["cid"], c["seed"]): (c["deploy_sxb_value"], c["x0b_value_scored"]) for c in de["cells"]}
    if got != {(cids[i][:12], s): (sxb[s][cids[i]], x0[s][cids[i]]) for i, s in a2cells}:
        return False, "a2 descriptive cell values"
    u3 = {}
    for r in rows:
        i = cids.index(r["cid"])
        ps = [sxb[s][r["cid"]] if (i, s) in a2cells else r["per_seed"][k] for k, s in enumerate(SEEDS)]
        u3[r["cid"]] = sum(ps) / 2
    e3 = H.elo_panel(S._ladder_order({**fx["field"], A2.ENTRANT: u3}), fx["fam"], A2.ANCHOR, seed=A2.LADDER_SEED,
                     n_boot=A2.N_BOOT)["elo"][A2.ENTRANT]["elo"]
    if abs(e3 - de["elo"]["elo"]) > 1e-9:
        return False, f"a2 descriptive Elo differs from an independent fit by {abs(e3 - de['elo']['elo']):.2e}"
    if bool(a1) != ("a1_deploy_values" in ro["descriptive"]):
        return False, "a1 sensitivity presence"
    if any(repr(sxb[s][cids[i]]) in so.split("descriptive", 1)[0] for i, s in a2cells):
        return False, "an a2 deploy value printed outside the descriptive line"
    return True, (f"SCORED rank {ro['rank']}/{ro['n_entrants']} elo {ro['board_elo'][A2.ENTRANT]['elo']:.2f}; "
                  f"a2 descriptive elo {de['elo']['elo']:.2f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ladder", default="")
    a = ap.parse_args()
    root = Path(a.dir)
    if root.exists():
        raise SystemExit(f"{root} exists (a new directory)")
    root.mkdir(parents=True)
    A2.N_CONTEXTS, A2.N_FAMILIES, A2.N_STORED = S.N, S.NF, len(S.STORED)
    A2.EXPOSED = (hashlib.sha256(f"h418-selftest-{S.EXPOSED_IX}".encode()).hexdigest()[:8],)
    res = []

    def ladder(name, r):
        if a.ladder and r[1]:
            ok, why = S.chk_ladder(r[3], Path(a.ladder), json.loads((r[3]["root"] / "out.json").read_text())["readout"])
            res.append((name, ok, why))

    def create_once(name, fx):
        shas = [S._sha(fx["root"] / f) for f in ("out.json", "rows.jsonl")]
        rc, _ = S.run(fx)
        ok = isinstance(rc, SystemExit) and "create-once" in str(rc) and \
            shas == [S._sha(fx["root"] / f) for f in ("out.json", "rows.jsonl")]
        res.append((name, ok, "refused, files unchanged" if ok else f"rc {rc!r}"))

    # 1. the selftest_h418 cases, empty A2 list
    r = case("clean", root, {}, S.chk_clean)
    res.append(r[:3])
    create_once("create_once", r[3])
    ladder("ladder_py", r)
    res.append(case("a1_exempt", root, {"gs_perturb": (1, 335101), "conflict": 1},
                    lambda fx, rc, so: S.chk_clean(fx, rc, so, exempt=(1, 335101)))[:3])
    for name, m, gates, exact in (
            ("gs_replay_no_conflict", {"gs_perturb": (1, 335101)}, ["gs_replay_335101"], True),
            ("gs_replay_other_conflict", {"gs_perturb": (1, 335101), "conflict": 2}, ["gs_replay_335101"], True),
            ("h416_on_board", {"h416_on_board": True}, ["board_entrants"], False),
            ("sxb_offline_mismatch", {"sxb_perturb": (2, 335102)}, ["sxb_offline_335102"], True),
            ("board_tampered", {"tamper_board": True}, ["board_files"], True),
            ("lock_board_manifest", {"lock_wrong": True}, ["lock"], True),
            ("ladder_not_reproduced", {"ladder_perturb": True}, ["board_ladder"], True),
            ("x0_mean_wrong", {"x0_mean_wrong": 3}, ["x0_board"], True),
            ("config_seed", {"config_seed": True}, ["configs"], True),
            ("missing_arm_record", {"missing_ge": (3, 335102)}, ["harness_inputs"], True),
            ("audit_envelope", {"envelope_wrong": (6, 335101)}, ["audit_envelope"], True),
            ("bad_error", {"bad_error": (7, 335102)}, ["records"], True)):
        res.append(case(name, root, m, S.expect_incomplete(gates, exact))[:3])

    # 2. listed failures score X0b on that unit; the deploy's value is descriptive only
    r = case("a2_ge_listed", root, {}, lambda fx, rc, so: chk_a2(fx, rc, so, a2cells=[GE_CELL]),
             perturb=[(*GE_CELL, "GE", "both")], cells=[(*GE_CELL, "GE")])
    res.append(r[:3])
    create_once("a2_create_once", r[3])
    ladder("a2_ladder_py", r)
    res.append(case("a2_gl_members_only", root, {}, lambda fx, rc, so: chk_a2(fx, rc, so, a2cells=[(3, 335101)]),
                    perturb=[(3, 335101, "GL", "members")], cells=[(3, 335101, "GL")])[:3])
    res.append(case("a2_ge_value_only", root, {}, lambda fx, rc, so: chk_a2(fx, rc, so, a2cells=[(8, 335101)]),
                    perturb=[(8, 335101, "GE", "value")], cells=[(8, 335101, "GE")])[:3])
    res.append(case("a2_with_a1", root, {"gs_perturb": (1, 335101), "conflict": 1},
                    lambda fx, rc, so: chk_a2(fx, rc, so, a2cells=[GE_CELL], a1=(1, 335101)),
                    perturb=[(*GE_CELL, "GE", "both")], cells=[(*GE_CELL, "GE")])[:3])

    # 3. anything else stays INCOMPLETE on its own gate
    bad_file = {"schema": "wrong", "cells": []}
    for name, perturb, cells, raw, gates in (
            ("a2_unlisted", [(*GE_CELL, "GE", "both")], [], None, ["ge_replay_335102"]),
            ("a2_listed_not_failing", [], [(*GE_CELL, "GE")], None, ["ge_replay_335102"]),
            ("a2_second_unlisted", [(*GE_CELL, "GE", "both"), (4, 335102, "GE", "value")], [(*GE_CELL, "GE")], None,
             ["ge_replay_335102"]),
            ("a2_wrong_source", [(*GE_CELL, "GE", "both")], [(*GE_CELL, "GL")], None,
             ["ge_replay_335102", "gl_replay_335102"]),
            ("a2_wrong_seed", [(*GE_CELL, "GE", "both")], [(2, 335101, "GE")], None,
             ["ge_replay_335101", "ge_replay_335102"]),
            ("a2_cid_not_on_board", [], [("f" * 64, 335102, "GE")], None, ["a2_cells_file"]),
            ("a2_bad_schema", [(*GE_CELL, "GE", "both")], [], bad_file, ["a2_cells_file", "ge_replay_335102"])):
        res.append(case(name, root, {}, S.expect_incomplete(gates, True), perturb=perturb, cells=cells,
                        raw_file=raw)[:3])
    w = max(len(x[0]) for x in res)
    for name, ok, why in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name:{w}s}  {why}")
    bad = [x[0] for x in res if not x[1]]
    print(f"selftest_h418_a2: {len(res) - len(bad)}/{len(res)} cases passed" + (f"; FAILED {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
