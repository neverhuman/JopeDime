#!/usr/bin/env python3
"""H418 readout (PREREG §4-§5), on xbabe2 from the sha-checked pulled copies. numpy + h335_stats (sha-pinned) +
meta_h416 + train_h416's helpers only: no GP stack, no worker, no dev91 table.

  analyze_h418.py --pulled P --board B --board-manifest F --lock F --design F --conflicts F --out OUT.json
                  --rows ROWS.jsonl

  P   D1/ D2/ (deploy run dirs: config_D<n>.json, stages/H416_D<n>, stages/H415_D<n>[r], h416_D<n>_npz, status_*),
      GE/ GL/ (config.json, stages/E416_AGG, status_*), GS/stages/V_AGG (the stored H335 Stage V records),
      cids91.json (the select output)
  B   the dev91-audit-v1 board at the LOCKed bytes: BOARD.json, ladder.json, scores/*.jsonl

Every harness gate (PREREG §4) is checked first. If one fails, OUT.json holds the failed gate names and harness
numbers only (no SXB value is computed into it or printed), ROWS is not written, and the exit code is 3
(INCOMPLETE). Otherwise ROWS (create-once) holds the 91 board records of h416.SXB, and OUT.json the readout.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_h416 as T16  # noqa: E402  (stdlib at import; its helpers take np / H)

WAVE = "H418"
ENTRANT = "h416.SXB"
X0B = T16.X0B  # h335.H335_AGG: the locked recipe, the per-seed fallback
XSTACK = "h364.XSTACK_2L"
ANCHOR = T16.ANCHOR  # h335.H257_REPLICA = 1500
LADDER_SEED, N_BOOT = T16.LADDER_SEED, T16.N_BOOT  # 335160, 10,000 = tools/ladder.py (h335_stats.elo_panel)
SEEDS = (335101, 335102)
TAGS = {335101: "D1", 335102: "D2"}
ARMS = ("GE", "GL")
KEY = T16.KEY_GP  # AGG.g50.CV3@12
STAGE_GS = "V_AGG"
SXB_KEY = "P3+LIB3.SXB"
DESIGN_NAME = "BCAR.mix"
DESIGN_COMMIT = "25d8c2d329"
OK_ERRORS = ("no usable H program",)
TOL_REPLAY = 1e-9  # = H416 CHECK_TOL
TOL_DERIV = 1e-6  # PREREG §4.6 (H416 used 1e-9; its P201b maximum was 8.4e-10)
TOL_LADDER = 1e-6
TOL_MEAN = 1e-12
N_CONTEXTS, N_FAMILIES, N_STORED = 91, 19, 31
# the 5 contexts exposed by H381 / H410 (board README rule 4), by cid prefix: descriptive sensitivity only
EXPOSED = ("ac07466e", "8eeca816", "016648dc", "6ecbb7d5", "b1a5c1a1")
H335_STATS_SHA256 = "ee3b45439a08bc395d53079570ed8965b61aad8c2d2e4e89fd1b32c490bbccb1"
HARNESS = "h418-sxb-v1"
BOARD_MANIFEST_NAME = "board_audit.sha256"


class GateFail(Exception):
    """A harness input is missing or malformed: INCOMPLETE, no value read."""


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _json(p) -> dict:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError) as e:
        raise GateFail(f"{Path(p).name}: {type(e).__name__}")


def _result(p: Path) -> dict:
    if not p.is_file():
        raise GateFail(f"missing record {p.parent.name}/{p.name[:12]}")
    rs = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    rs = [r for r in rs if r.get("type") == "result"]
    if len(rs) != 1:
        raise GateFail(f"{p.parent.name}/{p.name[:12]}: {len(rs)} result records")
    return rs[0]


def _out(rec: dict, key: str) -> dict:
    o = (rec.get("outputs") or {}).get(key)
    if not isinstance(o, dict) or "value" not in o:
        raise GateFail(f"{str(rec.get('cid', '?'))[:12]}: no output {key}")
    return o


def _members(o: dict):
    return (o.get("spec") or {}).get("members")


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _ladder_order(values: dict) -> dict:
    """Entrants in tools/ladder.py's order (sorted scores/<id>.jsonl paths), so the fit is ladder.py's to the bit."""
    return dict(sorted(values.items(), key=lambda kv: kv[0] + ".jsonl"))


def _jd(o):
    """json default: numpy scalars as Python numbers."""
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(type(o).__name__)


def _stats():
    np, H = T16._stats_mod()
    if _sha(H.__file__) != H335_STATS_SHA256:
        raise SystemExit(f"h335_stats at {H.__file__} is not the pinned file")
    return np, H


# ----------------------------------------------------------------------------------------------------------------------
# board (tools/ladder.py's load rules) + its LOCKed manifest
# ----------------------------------------------------------------------------------------------------------------------
def load_board(bdir: Path, manifest: Path) -> dict:
    man = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            h, name = line.split(None, 1)
            man[name.strip()] = h
    have = sorted(str(p.relative_to(bdir)) for p in bdir.rglob("*") if p.is_file())
    files_ok = have == sorted(man) and all(_sha(bdir / n) == man[n] for n in have)
    board = _json(bdir / "BOARD.json")
    env = {c["cid"]: c["envelope_sha256"] for c in board["contexts"]}
    fam = {c["cid"]: c["family_id"] for c in board["contexts"]}
    values, x0ps, incomplete, bad = {}, {}, {}, []
    ids = [p.stem for p in sorted((bdir / "scores").glob("*.jsonl"))]  # ladder.py's order
    for eid in ids:
        recs = [json.loads(x) for x in (bdir / "scores" / f"{eid}.jsonl").read_text().splitlines() if x.strip()]
        cids = [r["cid"] for r in recs]
        if len(set(cids)) != len(cids) or any(
                r["cid"] not in env or r.get("board", board["board"]) != board["board"]
                or ("envelope_sha256" in r and r["envelope_sha256"] != env[r["cid"]]) for r in recs):
            bad.append(eid)
            continue
        got = {r["cid"]: float(r["utility"]) for r in recs}
        if set(got) != set(env):
            incomplete[eid] = len(got)
            continue
        values[eid] = got
        if eid == X0B:
            x0ps = {r["cid"]: [float(x) for x in (r.get("per_seed") or [])] for r in recs}
    return {"board": board, "env": env, "fam": fam, "values": values, "x0_per_seed": x0ps, "incomplete": incomplete,
            "bad": bad, "ids": ids, "files_ok": files_ok, "n_files": len(have)}


# ----------------------------------------------------------------------------------------------------------------------
# gates (PREREG §4): fills checks / gate; returns the harness context, or None when a precondition gate fails
# ----------------------------------------------------------------------------------------------------------------------
def collect(a, np, H, M, checks: dict, gate: dict):
    P = Path(a.pulled)
    lk = _json(a.lock)
    lock_sha = _sha(a.lock)
    design = _json(a.design)
    bm = [k for k in (lk.get("inputs_sha256") or {}) if Path(k).name == BOARD_MANIFEST_NAME]
    gate["lock"] = (lk.get("schema") == "jope.h418.lock/v1" and lk.get("entrant") == ENTRANT
                    and lk.get("seeds") == list(SEEDS) and lk.get("design") == DESIGN_NAME
                    and lk.get("design_sha256") == _sha(a.design) and len(bm) == 1
                    and lk["inputs_sha256"][bm[0]] == _sha(a.board_manifest))
    gate["design"] = design.get("schema") == "jope.h416.design/v1" and design.get("design") == DESIGN_NAME
    checks["lock_sha256"] = lock_sha

    # (1) board
    bd = load_board(Path(a.board), Path(a.board_manifest))
    env, fam, field = bd["env"], bd["fam"], bd["values"]
    lad = _json(Path(a.board) / "ladder.json")
    _, _, _, el0 = T16.elo_boot(np, H, field, fam, ANCHOR, LADDER_SEED, N_BOOT) if ANCHOR in field else (0, 0, 0, {})
    le = (lad.get("elo") or {}).get("elo") or {}
    dl = max((max(abs(el0[e]["elo"] - v["elo"]), abs(el0[e]["ci95"][0] - v["ci95"][0]),
                  abs(el0[e]["ci95"][1] - v["ci95"][1])) for e, v in le.items() if e in el0), default=math.inf)
    h416 = [e for e in bd["ids"] if e.startswith("h416.")]
    checks["board"] = {"files": bd["n_files"], "files_match_manifest": bd["files_ok"], "complete_entrants": len(field),
                       "incomplete": bd["incomplete"], "bad": bd["bad"], "h416_files": h416,
                       "ladder_reproduced_max_abs": float(dl)}
    gate["board_files"] = bd["files_ok"]
    b0 = bd["board"]
    gate["board_id"] = (b0.get("board") == "dev91-audit-v1" and b0.get("n_contexts") == N_CONTEXTS == len(env)
                        and b0.get("n_families") == N_FAMILIES == len(set(fam.values()))
                        and isinstance(b0.get("board_sha256"), str) and len(b0["board_sha256"]) == 64)
    gate["board_ladder"] = dl <= TOL_LADDER and set(le) == set(field)
    gate["board_entrants"] = (len(field) == N_STORED and not bd["incomplete"] and not bd["bad"] and not h416
                              and all(e in field for e in (X0B, XSTACK, ANCHOR)))

    # (2) contexts, (3) configs
    sel = _json(P / "cids91.json")
    cids = sorted(sel.get("cids") or [])
    cfg = {s: _json(P / t / f"config_{t}.json") for s, t in TAGS.items()}
    acfg = {n: _json(P / n / "config.json") for n in ARMS}
    gate["contexts"] = (len(cids) == N_CONTEXTS and cids == sorted(env) and len(set(fam.values())) == N_FAMILIES
                        and all((sel.get("family_of") or {}).get(c) == fam[c] for c in cids)
                        and all(sorted(x.get("cids") or []) == cids for x in list(cfg.values()) + list(acfg.values()))
                        and sel.get("lock_sha256") == lock_sha)
    gate["configs"] = (all(cfg[s].get("seed") == s and cfg[s].get("tag") == TAGS[s] and cfg[s].get("mode") == "confirm"
                           and cfg[s].get("design") == design and cfg[s].get("lock_sha256") == lock_sha
                           and cfg[s].get("entrant") == ENTRANT for s in SEEDS)
                       and all(acfg[n].get("arm") == n and acfg[n].get("seeds") == list(SEEDS)
                               and acfg[n].get("stage") == T16.STAGE_GP[n] and acfg[n].get("lock_sha256") == lock_sha
                               for n in ARMS))
    x0ps = bd["x0_per_seed"]
    gate["x0_board"] = bool(cids) and all(len(x0ps.get(c, [])) == 2 and abs(sum(x0ps[c]) / 2 - field[X0B][c]) <= TOL_MEAN
                                          for c in cids) if X0B in field else False
    if not all(gate.values()):
        return None
    x0 = {s: {c: x0ps[c][i] for c in cids} for i, s in enumerate(SEEDS)}

    # records
    recs16 = {s: {c: _result(P / TAGS[s] / "stages" / f"H416_{TAGS[s]}" / f"{c}__confirm.jsonl") for c in cids}
              for s in SEEDS}
    recs15 = {s: {c: _result(P / TAGS[s] / "stages" / f"H415_{recs16[s][c].get('recomputed_as') or TAGS[s]}"
                             / f"{c}__pool.jsonl") for c in cids} for s in SEEDS}
    gs = {s: {c: _out(_result(P / "GS" / "stages" / STAGE_GS / f"{c}__{s}.jsonl"), KEY) for c in cids} for s in SEEDS}
    arm = {n: {s: {c: _out(_result(P / n / "stages" / T16.STAGE_GP[n] / f"{c}__{s}.jsonl"), KEY) for c in cids}
               for s in SEEDS} for n in ARMS}
    bad, no_h = [], {s: [] for s in SEEDS}
    for s in SEEDS:
        for c, r in recs16[s].items():
            if r.get("cid") != c or r.get("tag") != TAGS[s] or r.get("mode") != "confirm":
                bad.append(f"{s}:{c[:12]}:identity")
            elif r.get("error"):
                if r["error"] in OK_ERRORS:
                    no_h[s].append(c)
                else:
                    bad.append(f"{s}:{c[:12]}:error")
            else:
                o = (r.get("outputs") or {}).get(SXB_KEY) or {}
                if o.get("invalid") or (o.get("raw") or {}).get("status") != "completed" or not _finite(o.get("value")):
                    bad.append(f"{s}:{c[:12]}:sxb")
    checks["records"] = {"units": 2 * len(cids), "no_usable_H": {str(s): len(no_h[s]) for s in SEEDS}, "bad": bad,
                         "recomputed_as": sum(1 for s in SEEDS for r in recs16[s].values() if r.get("recomputed_as"))}
    gate["records"] = not bad

    # (4) replays + the amendment-1 rule
    try:
        conf = T16.load_conflicts(a.conflicts, cids)
    except SystemExit as e:
        raise GateFail(str(e))
    exempt, rep = {}, {}
    for s in SEEDS:
        def dv(o, ref):
            return abs(float(o["value"]) - float(ref))
        gso = {c: _out(recs15[s][c], "GS.stock") for c in cids}
        ok = {c: dv(gso[c], gs[s][c]["value"]) <= TOL_REPLAY and dv(gso[c], x0[s][c]) <= TOL_REPLAY
              and _members(gso[c]) == _members(gs[s][c]) for c in cids}
        fail = sorted(c for c in cids if not ok[c])
        exempt[s] = sorted(c for c in fail if conf[c] > 0)
        rep[f"GS.{s}"] = {"max_abs_delta_vs_stored": max(dv(gso[c], gs[s][c]["value"]) for c in cids),
                          "max_abs_delta_vs_board": max(dv(gso[c], x0[s][c]) for c in cids),
                          "members_same": sum(1 for c in cids if _members(gso[c]) == _members(gs[s][c])),
                          "failed": [c[:12] for c in fail], "a1_exempt": [c[:12] for c in exempt[s]]}
        gate[f"gs_replay_{s}"] = fail == exempt[s]
        for n in ARMS:
            o = {c: _out(recs15[s][c], f"{n}.stock") for c in cids}
            d = max(dv(o[c], arm[n][s][c]["value"]) for c in cids)
            ms = sum(1 for c in cids if _members(o[c]) == _members(arm[n][s][c]))
            rep[f"{n}.{s}"] = {"max_abs_delta": d, "members_same": ms}
            gate[f"{n.lower()}_replay_{s}"] = d <= TOL_REPLAY and ms == len(cids)
    checks["replay"] = rep
    checks["memo_conflict_contexts"] = [c[:12] for c in cids if conf[c] > 0]

    # (5) AUDIT envelope
    ne = sum(1 for s in SEEDS for c in cids if recs16[s][c].get("audit_board_sha256") == env[c])
    checks["audit_envelope_matches_board"] = ne
    gate["audit_envelope"] = ne == 2 * len(cids)

    # (6) re-derivations from the npz columns (BCAR.mix only; the other designs are not computed)
    der = {}
    for s in SEEDS:
        rd = P / TAGS[s]
        ok16 = {c: r for c, r in recs16[s].items() if not r.get("error")}
        d_sxb = d_k = 0.0
        for c, r in ok16.items():
            try:
                z = T16.load_npz(np, r, rd)
            except SystemExit as e:
                raise GateFail(str(e))
            pred, _ = M.design_pred(DESIGN_NAME, z["Q"], z["y"], c, z["bag"], z["refit"], z["refs"], z["ym"], z["na"])
            d_sxb = max(d_sxb, abs(M.utility(pred, z["ya"], z["ba"])[0] - float(r["outputs"][SXB_KEY]["value"])))
            kb, kf = r.get("kcar100_bag_u"), r.get("kfb_u")
            d_k = max(d_k, abs(float(kb) - float(kf)) if _finite(kb) and _finite(kf) else math.inf)
        try:
            d_ref = T16.ref_check(np, M, ok16, recs15[s], rd)
        except SystemExit as e:
            raise GateFail(str(e))
        der[str(s)] = {"sxb_offline_max_abs_delta": float(d_sxb), "kcar100_bag_vs_kfb_max_abs_delta": float(d_k),
                       "ref_pred_max_abs_delta": float(d_ref), "units": len(ok16)}
        gate[f"sxb_offline_{s}"] = d_sxb <= TOL_DERIV
        gate[f"kcar100_bag_is_kfb_{s}"] = d_k <= TOL_DERIV
        gate[f"ref_preds_{s}"] = d_ref <= TOL_DERIV
    checks["derivations"] = der

    # run facts (counts only)
    runs = {}
    for d in list(TAGS.values()) + list(ARMS):
        sts = [_json(p) for p in sorted((P / d).glob("status_*.json"))]
        runs[d] = {"status_files": len(sts), "wall_s": sum(float(x.get("wall_s", 0)) for x in sts),
                   "physical_by_kind": dict(sum((collections.Counter(x.get("physical_by_kind") or {}) for x in sts),
                                                collections.Counter()))}
    checks["runs"] = runs
    return {"cids": cids, "env": env, "fam": fam, "field": field, "x0": x0, "recs16": recs16, "exempt": exempt,
            "no_h": no_h, "board": bd["board"], "lad": le, "lock": lk, "lock_sha": lock_sha}


# ----------------------------------------------------------------------------------------------------------------------
# readout (PREREG §5), after every gate passed
# ----------------------------------------------------------------------------------------------------------------------
def _elo_rank(np, H, field: dict, fam: dict, who: str) -> dict:
    arms, point, boot, elo = T16.elo_boot(np, H, _ladder_order(field), fam, ANCHOR, LADDER_SEED, N_BOOT)
    order = sorted(elo, key=lambda e: -elo[e]["elo"])
    return {"arms": arms, "point": point, "boot": boot, "elo": elo, "order": order, "rank": order.index(who) + 1,
            "n": len(order)}


def readout(np, H, k: dict, u: dict, sxb: dict, own: dict) -> dict:
    cids, fam, field = k["cids"], k["fam"], k["field"]
    new = _ladder_order({**field, ENTRANT: u})
    r = _elo_rank(np, H, new, fam, ENTRANT)
    elo, order = r["elo"], r["order"]
    ep = H.elo_panel(new, fam, ANCHOR, seed=LADDER_SEED, n_boot=N_BOOT)  # what tools/ladder.py writes
    d_ep = max(abs(ep["elo"][e]["elo"] - elo[e]["elo"]) for e in elo)
    top = next(e for e in order if e != ENTRANT)
    focus = [X0B, XSTACK, top]
    out = {"board_elo": elo, "rank": r["rank"], "n_entrants": r["n"], "order": order, "top_other": top,
           "family_mean_u": {e: H.family_mean(new[e], fam) for e in (ENTRANT, *focus)},
           "gap_elo": {f"vs_{e}": elo[ENTRANT]["elo"] - elo[e]["elo"] for e in focus},
           "gap_elo_vs_stored_ladder": {f"vs_{e}": elo[ENTRANT]["elo"] - k["lad"][e]["elo"] for e in focus},
           "delta_elo": {e: T16.delta_elo(np, r["arms"], r["point"], r["boot"], ENTRANT, e) for e in focus},
           "pairs": {e: T16.pair_stats(np, H, u, field[e], fam) for e in focus},
           "stored_entrants_elo_shift_max_abs": max(abs(elo[e]["elo"] - k["lad"][e]["elo"]) for e in field),
           "elo_panel_vs_boot_max_abs": float(d_ep)}
    desc = {}
    for s in SEEDS:  # one seed's SXB values against the stored board (descriptive)
        rs = _elo_rank(np, H, {**field, ENTRANT: sxb[s]}, fam, ENTRANT)
        desc[f"seed_{s}_only"] = {"elo": rs["elo"][ENTRANT], "rank": rs["rank"], "n": rs["n"],
                                  "gap_vs_" + XSTACK: rs["elo"][ENTRANT]["elo"] - rs["elo"][XSTACK]["elo"]}
    keep = [c for c in cids if c[:8] not in EXPOSED]
    fk = {c: fam[c] for c in keep}
    rs = _elo_rank(np, H, {e: {c: v[c] for c in keep} for e, v in new.items()}, fk, ENTRANT)
    desc["without_exposed_contexts"] = {"contexts": len(keep), "excluded": len(cids) - len(keep),
                                        "elo": rs["elo"][ENTRANT], "rank": rs["rank"], "n": rs["n"],
                                        "gap_vs_" + XSTACK: rs["elo"][ENTRANT]["elo"] - rs["elo"][XSTACK]["elo"]}
    if own:  # amendment-1 sensitivity: the deploy's own SXB values on the exempt (context, seed) units
        u2 = {c: sum(own.get((c, s), sxb[s][c]) for s in SEEDS) / len(SEEDS) for c in cids}
        rs = _elo_rank(np, H, {**field, ENTRANT: u2}, fam, ENTRANT)
        desc["a1_deploy_values"] = {"units": len(own), "elo": rs["elo"][ENTRANT], "rank": rs["rank"], "n": rs["n"]}
    out["descriptive"] = desc
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for k in ("pulled", "board", "board-manifest", "lock", "design", "conflicts", "out", "rows"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args(argv)
    out, rows = Path(a.out), Path(a.rows)
    if out.exists() or rows.exists():
        raise SystemExit(f"{out} or {rows} exists (create-once)")
    np, H = _stats()
    import meta_h416 as M

    checks, gate = {}, {}
    try:
        k = collect(a, np, H, M, checks, gate)
    except (GateFail, KeyError, TypeError, ValueError, IndexError, AttributeError) as e:
        gate["harness_inputs"] = False
        if isinstance(e, GateFail):
            checks["harness_inputs_error"] = str(e)
        else:  # a malformed harness input: the exception type and line only (its message could hold a value)
            tb = traceback.extract_tb(e.__traceback__)[-1]
            checks["harness_inputs_error"] = f"{type(e).__name__} at {Path(tb.filename).name}:{tb.lineno}"
        k = None
    base = {"schema": "jope.h418.dev91-analysis/v1", "wave": WAVE, "entrant": ENTRANT, "board": "dev91-audit-v1",
            "seeds": list(SEEDS), "tolerances": {"replay": TOL_REPLAY, "derivation": TOL_DERIV,
                                                  "ladder": TOL_LADDER}, "gate": gate, "checks": checks}
    failed = sorted(g for g, v in gate.items() if not v)
    if k is None or failed:
        base.update({"status": "INCOMPLETE", "gate_failed": failed})
        with open(out, "x") as f:
            f.write(json.dumps(base, indent=1, sort_keys=True, default=_jd) + "\n")
        print(f"INCOMPLETE: harness gate failed {failed}; no entrant value was read")
        print(f"checks: {json.dumps(checks, sort_keys=True, default=_jd)}")
        return 3

    cids, x0, recs16 = k["cids"], k["x0"], k["recs16"]
    sxb, why, own = {s: {} for s in SEEDS}, {s: collections.Counter() for s in SEEDS}, {}
    for s in SEEDS:
        ex = set(k["exempt"][s])
        for c in cids:
            r = recs16[s][c]
            if c in ex:
                sxb[s][c], w = x0[s][c], "a1_exempt_x0b"
                if not r.get("error"):
                    own[(c, s)] = float(r["outputs"][SXB_KEY]["value"])
            elif r.get("error"):
                sxb[s][c], w = x0[s][c], "no_usable_H_x0b"
            else:
                sxb[s][c], w = float(r["outputs"][SXB_KEY]["value"]), "sxb"
            why[s][w] += 1
    u = {c: sum(sxb[s][c] for s in SEEDS) / len(SEEDS) for c in cids}
    lk = k["lock"]
    src = {"wave": WAVE, "entrant_wave": "H416", "design": DESIGN_NAME, "design_commit": DESIGN_COMMIT,
           "design_sha256": lk["design_sha256"], "lock_sha256": k["lock_sha"], "prereg_commit": lk["prereg_commit"],
           "seeds": list(SEEDS), "x0": f"{X0B} per_seed (H335 Stage V stored {KEY} output)",
           "sources": {"GS": "H0.1-A: the stored H335 Stage V V_AGG run (memo replay)",
                       "GE": "H381-E2-S3fz: new run (H416 arm, E416_AGG)",
                       "GL": "H415-L: new run (H416 arm, E416_AGG)"},
           "fallback": "X0b per seed on a no-usable-H unit or an amendment-1 exemption"}
    if not all(math.isfinite(v) for v in u.values()):
        raise SystemExit("a non-finite board value after the gates (harness bug): nothing written")
    b, env = k["board"], k["env"]
    text = "".join(json.dumps({"board": b["board"], "board_sha256": b["board_sha256"], "cid": c,
                               "envelope_sha256": env[c], "model_id": ENTRANT, "utility": u[c],
                               "per_seed": [sxb[s][c] for s in SEEDS], "status": "completed",
                               "harness_version": HARNESS, "source": src}, sort_keys=True) + "\n" for c in cids)
    ro = readout(np, H, k, u, sxb, own)  # everything is computed before either create-once file is opened
    base.update({"status": "SCORED", "n_contexts": len(cids), "n_families": len(set(k["fam"].values())),
                 "value_source": {str(s): dict(why[s]) for s in SEEDS},
                 "a1_exempt": {str(s): [c[:12] for c in k["exempt"][s]] for s in SEEDS},
                 "rows": {"path": str(rows), "sha256": hashlib.sha256(text.encode()).hexdigest(), "n": len(cids)},
                 "readout": ro})
    report = json.dumps(base, indent=1, sort_keys=True, default=_jd) + "\n"
    with open(rows, "x") as f:
        f.write(text)
    with open(out, "x") as f:
        f.write(report)
    if _sha(rows) != base["rows"]["sha256"]:
        raise SystemExit(f"{rows} bytes differ from the computed rows")
    e, top = ro["board_elo"][ENTRANT], ro["top_other"]
    print(f"SCORED {ENTRANT} on dev91-audit-v1: AUDIT-Elo {e['elo']:.1f} [{e['ci95'][0]:.1f}, {e['ci95'][1]:.1f}], "
          f"rank {ro['rank']} of {ro['n_entrants']}; family-mean u {ro['family_mean_u'][ENTRANT]:.4f}")
    for x in (XSTACK, X0B, top):
        d, p = ro["delta_elo"][x], ro["pairs"][x]
        print(f"vs {x:28s} Elo gap {ro['gap_elo']['vs_' + x]:+7.1f} (stored ladder "
              f"{ro['gap_elo_vs_stored_ladder']['vs_' + x]:+7.1f}); dElo CI [{d['ci95'][0]:+.1f}, {d['ci95'][1]:+.1f}]; "
              f"families W/T/L {'/'.join(map(str, p['wtl']))}; gain {p['gain']:+.4f} ({100 * p['pct_rmse']:+.1f}% RMSE); "
              f"clipped {p['clipped_mean']:+.4f} p {p['signflip_p']:.3g} LOFO {p['lofo_min']:+.4f}")
    print(f"value sources {json.dumps(base['value_source'])}; a1 exempt {json.dumps(base['a1_exempt'])}")
    print(f"descriptive {json.dumps(ro['descriptive'], default=_jd)}")
    print(f"rows {rows} sha256 {base['rows']['sha256'][:16]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
