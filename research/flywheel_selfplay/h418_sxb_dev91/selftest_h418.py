#!/usr/bin/env python3
"""H418 analysis self-test on SYNTHETIC fixtures only: no dev91 byte, no board file, no worker, no GP. Each case
builds a fresh synthetic pulled tree + board + LOCK under --dir (a new directory) and runs analyze_h418.main on it
in-process (module constants set to the fixture's size). The fixture's SXB values are computed by meta_h416's own
BCAR.mix design on synthetic npz matrices; its committed ladder is h335_stats.elo_panel's (= tools/ladder.py's).

  selftest_h418.py --dir NEW_DIR [--ladder PATH/TO/ladder.py]

--ladder also runs the real tools/ladder.py on the clean case's board + rows (as `run_h418.sh board` does) and
requires its h416.SXB Elo to equal analyze_h418's (<= 1e-9), and the fixture's committed ladder to equal ladder.py's.
Exit 0 = every case passed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyze_h418 as A  # noqa: E402
import train_h416 as T16  # noqa: E402

np, H = A._stats()
import meta_h416 as M  # noqa: E402

SEEDS, TAGS = A.SEEDS, A.TAGS
N, NF = 12, 4
STORED = {"h335.H257_REPLICA": 0.0, "h335.H335_AGG": None, "h364.XSTACK_2L": 0.03, "kumo_like.top": 0.2,
          "aa.weak": -0.1}  # offset vs X0b (None = X0b itself, with per_seed)
BOARD_KEYS = ["board", "board_sha256", "cid", "envelope_sha256", "harness_version", "model_id", "per_seed", "source",
              "status", "utility"]  # = the keys of the stored dev91-audit-v1 records (h364.XSTACK_2L, h335.*)
ERR_UNIT = (0, 335102)  # every fixture: one "no usable H program" unit (scores X0b)
RECOMP_UNIT = (5, 335101)  # every fixture: one unit whose H415 record is under H415_D1r (recomputed_as)
EXPOSED_IX = 4  # the clean case marks this context exposed (descriptive sensitivity)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _w(p: Path, obj) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, sort_keys=True) + "\n")
    return p


def _wl(p: Path, recs) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in recs))
    return p


def _members(tag: str, c: str):
    return [hashlib.sha256(f"{tag}-{c}-{j}".encode()).hexdigest()[:16] for j in range(3)]


def _ladder_order(values: dict) -> dict:  # written independently of analyze_h418._ladder_order
    return {e: values[e] for e in sorted(values, key=lambda e: e + ".jsonl")}


def build(root: Path, m: dict) -> dict:
    """A synthetic pulled tree / board / LOCK under root. m = the mutations of this case."""
    rng = np.random.default_rng(418)
    cids = [hashlib.sha256(f"h418-selftest-{i}".encode()).hexdigest() for i in range(N)]
    fam = {c: f"F{i % NF}" for i, c in enumerate(cids)}
    env = {c: hashlib.sha256(f"env-{c}".encode()).hexdigest() for c in cids}
    x0 = {s: {c: float(rng.normal(0.3, 0.1)) for c in cids} for s in SEEDS}
    arm = {n: {s: {c: float(rng.normal(0.3, 0.1)) for c in cids} for s in SEEDS} for n in A.ARMS}
    P, B = root / "pulled", root / "board" / "dev91-audit-v1"

    # board: BOARD.json, scores, the committed ladder, the sha manifest
    board = {"schema": "jope.reference-board.board/v1", "board": "dev91-audit-v1", "split": "selftest",
             "description": "H418 selftest synthetic board", "n_contexts": N, "n_families": NF,
             "board_sha256": hashlib.sha256(b"h418-selftest-board").hexdigest(),
             "contexts": [{"cid": c, "envelope_sha256": env[c], "family_id": fam[c]} for c in cids]}
    _w(B / "BOARD.json", board)
    field = {}
    for e, off in STORED.items():
        recs = []
        for i, c in enumerate(cids):
            r = {"board": board["board"], "board_sha256": board["board_sha256"], "cid": c,
                 "envelope_sha256": env[c], "model_id": e, "status": "completed", "harness_version": "selftest",
                 "source": {"selftest": True}}
            if off is None:
                ps = [x0[s][c] for s in SEEDS]
                r["per_seed"], r["utility"] = ps, sum(ps) / 2 + (1e-6 if m.get("x0_mean_wrong") == i else 0.0)
            else:
                r["utility"] = sum(x0[s][c] for s in SEEDS) / 2 + off + float(rng.normal(0, 0.05))
            recs.append(r)
        _wl(B / "scores" / f"{e}.jsonl", recs)
        field[e] = {r["cid"]: r["utility"] for r in recs}
    el = H.elo_panel(_ladder_order(field), fam, A.ANCHOR, seed=A.LADDER_SEED, n_boot=A.N_BOOT)
    if m.get("ladder_perturb"):
        el["elo"]["aa.weak"]["elo"] += 1e-3
    _w(B / "ladder.json", {"schema": "jope.reference-board.ladder/v1", "board": board["board"], "elo": el})
    if m.get("h416_on_board"):
        _wl(B / "scores" / "h416.SXB.jsonl", [{"board": board["board"], "cid": c, "model_id": "h416.SXB",
                                               "utility": 0.3, "status": "completed"} for c in cids])
    man = root / "inputs" / "board_audit.sha256"
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text("".join(f"{_sha(p)}  {p.relative_to(B)}\n" for p in sorted(B.rglob("*")) if p.is_file()))
    if m.get("tamper_board"):  # changed bytes after the manifest (same parsed records)
        p = B / "scores" / "aa.weak.jsonl"
        p.write_text(p.read_text() + "\n")

    # LOCK, selection, configs
    design_p = HERE / "DESIGN.json"
    design = json.loads(design_p.read_text())
    lock = {"schema": "jope.h418.lock/v1", "wave": "H418", "entrant": A.ENTRANT, "seeds": list(SEEDS),
            "design": A.DESIGN_NAME, "design_sha256": _sha(design_p), "prereg_commit": "0" * 40,
            "inputs_sha256": {"/selftest/inputs/board_audit.sha256": ("f" * 64) if m.get("lock_wrong") else _sha(man),
                              "/selftest/h335-prep/x0_prep.tables.sha256": "e" * 64}}
    lock_p = _w(P / "LOCK.json", lock)
    ls = _sha(lock_p)
    _w(P / "cids91.json", {"schema": "jope.h418.cids91/v1", "cids": cids, "family_of": fam, "lock_sha256": ls})
    for s, t in TAGS.items():
        _w(P / t / f"config_{t}.json", {"seed": (s + 1) if (m.get("config_seed") and t == "D2") else s, "tag": t,
                                        "mode": "confirm", "design": design, "lock_sha256": ls,
                                        "entrant": A.ENTRANT, "cids": cids})
        _w(P / t / f"status_{t}_1.json", {"wall_s": 100.0, "physical_by_kind": {"audit": 7, "search": 11}})
    for n in A.ARMS:
        _w(P / n / "config.json", {"arm": n, "seeds": list(SEEDS), "stage": T16.STAGE_GP[n], "lock_sha256": ls,
                                   "cids": cids})
        _w(P / n / "status_gp_1.json", {"wall_s": 50.0, "physical_by_kind": {"search": 13}})

    # stored GS (V_AGG) and arm (E416_AGG) records; H415 pool records; H416 confirm records + npz
    sxb = {s: {} for s in SEEDS}
    for s, t in TAGS.items():
        for i, c in enumerate(cids):
            meta = {"type": "meta", "cid": c}
            _wl(P / "GS" / "stages" / A.STAGE_GS / f"{c}__{s}.jsonl",
                [meta, {"type": "result", "cid": c, "outputs": {A.KEY: {"value": x0[s][c],
                                                                        "spec": {"members": _members("GS", c)}}}}])
            for n in A.ARMS:
                if m.get("missing_ge") == (i, s) and n == "GE":
                    continue
                _wl(P / n / "stages" / T16.STAGE_GP[n] / f"{c}__{s}.jsonl",
                    [{"type": "result", "cid": c, "outputs": {A.KEY: {"value": arm[n][s][c],
                                                                      "spec": {"members": _members(n, c)}}}}])
            gs_v = x0[s][c] + (1e-7 if m.get("gs_perturb") == (i, s) else 0.0)
            o15 = {"GS.stock": {"value": gs_v, "spec": {"members": _members("GS", c)}},
                   **{f"{n}.stock": {"value": arm[n][s][c], "spec": {"members": _members(n, c)}} for n in A.ARMS}}
            rec = {"type": "result", "schema": "selftest", "wave": "H416", "tag": t, "mode": "confirm", "cid": c,
                   "recomputed_as": f"{t}r" if (i, s) == RECOMP_UNIT else None, "outputs": {},
                   "audit_board_sha256": ("0" * 64) if m.get("envelope_wrong") == (i, s) else env[c]}
            if (i, s) == ERR_UNIT or m.get("bad_error") == (i, s):
                rec["error"] = "no usable H program" if (i, s) == ERR_UNIT else "worker crashed"
            else:
                na, ns, k = 20, 30, 6
                y = rng.normal(size=ns)
                Q = y[:, None] + rng.normal(scale=np.linspace(0.4, 1.2, k), size=(ns, k))
                ya = rng.normal(size=na)
                AKF = ya[:, None] + rng.normal(scale=np.linspace(1.5, 3.0, k), size=(na, k))  # SXB near the field
                AKF[:, k - 1] = np.nan  # one failed fold column (None in the offline design)
                PA = ya[:, None] + rng.normal(scale=np.linspace(1.6, 3.1, k), size=(na, k))
                ym = float(y.mean())
                ba = float(np.sqrt(np.mean((ya - ym) ** 2)))
                REF = (ya + rng.normal(scale=0.6, size=na))[:, None]
                buf = io.BytesIO()
                np.savez_compressed(buf, Q=Q, y=y, AKF=AKF, PA=PA, akf_nfold=np.full(k, 5), pa_ok=np.ones(k, bool),
                                    ya=ya, ba=np.float64(ba), ym=np.float64(ym), shas=np.asarray(_members("H", c) * 2),
                                    ref_keys=np.asarray(["P3+LIB3.KFB"]), REF=REF, ref_none=np.asarray([False]))
                raw = buf.getvalue()
                npz = P / t / f"h416_{t}_npz" / f"{c}.npz"
                npz.parent.mkdir(parents=True, exist_ok=True)
                npz.write_bytes(raw)
                rec["npz"] = {"path": str(npz), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
                z = T16.load_npz(np, rec, P / t)
                pred, info = M.design_pred(A.DESIGN_NAME, z["Q"], z["y"], c, z["bag"], z["refit"], z["refs"],
                                           z["ym"], z["na"])
                u = M.utility(pred, z["ya"], z["ba"])[0]
                kfb = M.utility(z["refs"]["P3+LIB3.KFB"], z["ya"], z["ba"])[0]
                o15["P3+LIB3.KFB"] = {"value": kfb, "invalid": False, "raw": {"status": "completed"}}
                rec["kcar100_bag_u"], rec["kfb_u"] = kfb, kfb
                sxb[s][c] = u
                rec["outputs"][A.SXB_KEY] = {"value": u + (1e-5 if m.get("sxb_perturb") == (i, s) else 0.0),
                                             "invalid": False, "raw": {"status": "completed", "utility": u},
                                             "spec": {"design": A.DESIGN_NAME, **info}}
            st15 = rec["recomputed_as"] or t
            _wl(P / t / "stages" / f"H415_{st15}" / f"{c}__pool.jsonl", [{"type": "result", "cid": c,
                                                                          "outputs": o15}])
            _wl(P / t / "stages" / f"H416_{t}" / f"{c}__confirm.jsonl", [meta, rec])
    conf = _wl(root / "memo_conflicts.jsonl",
               [{"cid": c, "gs_conflicts": 2 if m.get("conflict") == i else 0, "arm_conflicts": 0} for i, c in
                enumerate(cids)])
    argv = ["--pulled", str(P), "--board", str(B), "--board-manifest", str(man), "--lock", str(lock_p),
            "--design", str(design_p), "--conflicts", str(conf), "--out", str(root / "out.json"),
            "--rows", str(root / "rows.jsonl")]
    return {"cids": cids, "fam": fam, "x0": x0, "sxb": sxb, "field": field, "argv": argv, "root": root, "B": B}


def run(fx: dict):
    so = io.StringIO()
    with contextlib.redirect_stdout(so):
        try:
            rc = A.main(fx["argv"])
        except SystemExit as e:
            rc = e
    return rc, so.getvalue()


def _no_value_leak(fx: dict, text: str) -> bool:
    """No fixture SXB value (as json / repr writes it) appears in an INCOMPLETE output."""
    return not any(repr(v) in text for d in fx["sxb"].values() for v in d.values())


def case(name: str, root: Path, m: dict, check) -> tuple:
    fx = build(root / name, m)
    rc, so = run(fx)
    try:
        ok, why = check(fx, rc, so)
    except Exception as e:  # noqa: BLE001  (a crashed check is a failed case)
        ok, why = False, f"check raised {type(e).__name__}: {e}"
    return name, ok, why, fx, rc


def expect_incomplete(gates, exact=True):
    def chk(fx, rc, so):
        root = fx["root"]
        if rc != 3:
            return False, f"exit {rc!r}"
        if (root / "rows.jsonl").exists():
            return False, "rows written"
        o = json.loads((root / "out.json").read_text())
        txt = (root / "out.json").read_text() + so
        if o.get("status") != "INCOMPLETE" or "readout" in o:
            return False, "out not INCOMPLETE"
        if not set(gates) <= set(o["gate_failed"]) or (exact and o["gate_failed"] != sorted(gates)):
            return False, f"gate_failed {o['gate_failed']} (want {gates})"
        if not _no_value_leak(fx, txt):
            return False, "an SXB value leaked into an INCOMPLETE output"
        return True, f"INCOMPLETE {o['gate_failed']}"
    return chk


def chk_clean(fx, rc, so, *, exempt=None):
    root, cids, x0, sxb = fx["root"], fx["cids"], fx["x0"], fx["sxb"]
    if rc != 0:
        return False, f"exit {rc!r}: {so[-400:]}"
    o = json.loads((root / "out.json").read_text())
    rows = [json.loads(x) for x in (root / "rows.jsonl").read_text().splitlines()]
    if [r["cid"] for r in rows] != sorted(cids) or any(sorted(r) != BOARD_KEYS for r in rows):
        return False, "rows: contexts or keys differ from the board record format"
    if o["rows"]["sha256"] != _sha(root / "rows.jsonl"):
        return False, "rows sha"
    want = {}
    for c in cids:
        ps = []
        for s in SEEDS:
            i = cids.index(c)
            if (i, s) == ERR_UNIT or (exempt and (i, s) == exempt):
                ps.append(x0[s][c])
            else:
                ps.append(sxb[s][c])
        want[c] = ps
    for r in rows:
        if r["per_seed"] != want[r["cid"]] or r["utility"] != sum(r["per_seed"]) / 2:
            return False, f"row value {r['cid'][:8]}"
        if r["model_id"] != A.ENTRANT or r["harness_version"] != A.HARNESS or r["status"] != "completed":
            return False, "row identity"
    vals = {**fx["field"], A.ENTRANT: {r["cid"]: r["utility"] for r in rows}}
    el = H.elo_panel(_ladder_order(vals), fx["fam"], A.ANCHOR, seed=A.LADDER_SEED, n_boot=A.N_BOOT)["elo"]
    ro = o["readout"]
    d = max(abs(el[e]["elo"] - ro["board_elo"][e]["elo"]) for e in el)
    if d > 1e-9:
        return False, f"board Elo differs from an independent elo_panel by {d:.2e}"
    rank = 1 + sum(1 for e in el if e != A.ENTRANT and el[e]["elo"] > el[A.ENTRANT]["elo"])
    if ro["rank"] != rank or ro["n_entrants"] != len(STORED) + 1:
        return False, f"rank {ro['rank']} of {ro['n_entrants']} (want {rank} of {len(STORED) + 1})"
    if abs(ro["gap_elo"]["vs_" + A.XSTACK] - (el[A.ENTRANT]["elo"] - el[A.XSTACK]["elo"])) > 1e-9:
        return False, "gap vs XSTACK"
    top = max((e for e in el if e != A.ENTRANT), key=lambda e: el[e]["elo"])
    if ro["top_other"] != top:
        return False, f"top_other {ro['top_other']} (want {top})"
    vs = o["value_source"]
    n_err = {str(s): int(ERR_UNIT[1] == s) for s in SEEDS}
    n_ex = {str(s): int(bool(exempt) and exempt[1] == s) for s in SEEDS}
    for s in map(str, SEEDS):
        if vs[s].get("no_usable_H_x0b", 0) != n_err[s] or vs[s].get("a1_exempt_x0b", 0) != n_ex[s] or \
                vs[s].get("sxb", 0) != N - n_err[s] - n_ex[s]:
            return False, f"value_source {vs}"
    de = ro["descriptive"]
    if de["without_exposed_contexts"]["contexts"] != N - 1:
        return False, "exposed-context sensitivity did not drop exactly the exposed context"
    if bool(exempt) != ("a1_deploy_values" in de):
        return False, "a1 sensitivity presence"
    if o["checks"]["records"]["recomputed_as"] != 1:
        return False, "recomputed_as unit not read from H415_<tag>r"
    return True, f"SCORED rank {ro['rank']}/{ro['n_entrants']} elo {ro['board_elo'][A.ENTRANT]['elo']:.2f}"


def chk_ladder(fx, ladder: Path, ro: dict) -> tuple:
    """The real tools/ladder.py on (a) the fixture board alone and (b) the board + the rows."""
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    pre = fx["root"] / "board-pre"
    shutil.copytree(fx["B"], pre)
    subprocess.run([sys.executable, "-B", str(ladder), str(pre), "--write"], env=env, check=True,
                   stdout=subprocess.DEVNULL)
    a = json.loads((fx["B"] / "ladder.json").read_text())["elo"]["elo"]
    b = json.loads((pre / "ladder.json").read_text())["elo"]["elo"]
    if a != b:
        return False, "fixture ladder != ladder.py's"
    new = fx["root"] / "board-new"
    shutil.copytree(fx["B"], new)
    shutil.copyfile(fx["root"] / "rows.jsonl", new / "scores" / "h416.SXB.jsonl")
    subprocess.run([sys.executable, "-B", str(ladder), str(new), "--write"], env=env, check=True,
                   stdout=subprocess.DEVNULL)
    lj = json.loads((new / "ladder.json").read_text())
    if lj.get("incomplete"):
        return False, f"ladder.py lists incomplete {lj['incomplete']}"
    d = max(abs(lj["elo"]["elo"][e]["elo"] - ro["board_elo"][e]["elo"]) for e in ro["board_elo"])
    if set(lj["elo"]["elo"]) != set(ro["board_elo"]) or d > 1e-9:
        return False, f"ladder.py Elo differs from analyze_h418's by {d:.2e}"
    return True, f"ladder.py reproduces analyze_h418 (max |dElo| {d:.1e}); fixture ladder = ladder.py's"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--ladder", default="")
    a = ap.parse_args()
    root = Path(a.dir)
    if root.exists():
        raise SystemExit(f"{root} exists (a new directory)")
    root.mkdir(parents=True)
    A.N_CONTEXTS, A.N_FAMILIES, A.N_STORED = N, NF, len(STORED)
    cids = [hashlib.sha256(f"h418-selftest-{i}".encode()).hexdigest() for i in range(N)]
    A.EXPOSED = (cids[EXPOSED_IX][:8],)
    res = []
    r = case("clean", root, {}, chk_clean)
    res.append(r[:3])
    clean_fx = r[3]
    # create-once: a second analyze on the same out / rows refuses and changes nothing
    shas = [_sha(clean_fx["root"] / f) for f in ("out.json", "rows.jsonl")]
    rc, _ = run(clean_fx)
    ok = isinstance(rc, SystemExit) and "create-once" in str(rc) and \
        shas == [_sha(clean_fx["root"] / f) for f in ("out.json", "rows.jsonl")]
    res.append(("create_once", ok, "refused, files unchanged" if ok else f"rc {rc!r}"))
    if a.ladder and r[1]:
        ok, why = chk_ladder(clean_fx, Path(a.ladder), json.loads((clean_fx["root"] / "out.json").read_text())
                             ["readout"])
        res.append(("ladder_py", ok, why))
    res.append(case("a1_exempt", root, {"gs_perturb": (1, 335101), "conflict": 1},
                    lambda fx, rc, so: chk_clean(fx, rc, so, exempt=(1, 335101)))[:3])
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
        res.append(case(name, root, m, expect_incomplete(gates, exact))[:3])
    w = max(len(x[0]) for x in res)
    for name, ok, why in res:
        print(f"{'PASS' if ok else 'FAIL'}  {name:{w}s}  {why}")
    bad = [x[0] for x in res if not x[1]]
    print(f"selftest_h418: {len(res) - len(bad)}/{len(res)} cases passed" + (f"; FAILED {bad}" if bad else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
