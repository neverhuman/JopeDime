"""Import the 12 pre-declared H335 AUDIT entrants onto a reference board as stored scores (no refit, no re-run).

Per context the value is exactly what H335's stats used: GP entrants = mean over the search seeds of the unit's
AUDIT output value (h335_lab.values_by_output), the 4 LLM Rust regressors = their single frontier_audit_h335 record.
Board ids are prefixed "h335." (e.g. h335.H335_AGG). Writes BOARD_DIR/scores/h335.<entrant>.jsonl; refuses to
overwrite an existing file.

  dev91-audit-v1        python3 import_h335.py dev91 BOARD_DIR
                        (reads only ~/jope-data/h335-stagev/run/panel_stage_v.json + llm_audit.jsonl: scores, no rows)
  p201b-train-audit-v1  inside sandbox_h335.sh (H335_CODE frozen copy, PYTHONPATH=$SRC):
                        python3 import_h335.py train BOARD_DIR
"""
import hashlib
import json
import sys
from pathlib import Path

ENTRANT_KEYS = {"H335_AGG": "AGG.g50.CV3@12", "OLD_MATCHED": "OLD.g50.single", "OLD_MATCHED_CV3": "OLD.g50.CV3@12",
                "H317_BASE": "H317_BASE.single", "H257_REPLICA": "H257_REPLICA.CV1",
                "BANK_NN8_PICK": "BANK_NN8_PICK.single", "BANK_NN8_PORTFOLIO": "BANK_NN8_PORTFOLIO.CV5",
                "BANK_NN1": "BANK_NN1.single"}
LLM = ("Opus5p5_Extra", "Gemini4_argon_high", "GPT6p1_Sol_Max", "GPT6_Astra_Max")
SV = Path("/home/ubuntu/jope-data/h335-stagev/run")
H335 = Path("/home/ubuntu/jope-data/h335-cl")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def write(board_dir: Path, values: dict, status: dict, per_seed: dict, source: dict) -> None:
    board = json.loads((board_dir / "BOARD.json").read_text())
    env = {c["cid"]: c["envelope_sha256"] for c in board["contexts"]}
    (board_dir / "scores").mkdir(exist_ok=True)
    for name, vals in values.items():
        if set(vals) != set(env):
            raise SystemExit(f"{name}: {len(set(env) ^ set(vals))} contexts differ from BOARD.json")
        out = board_dir / "scores" / f"h335.{name}.jsonl"
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        with open(out, "x") as f:
            for cid in sorted(vals):
                rec = {"board": board["board"], "board_sha256": board["board_sha256"], "cid": cid,
                       "envelope_sha256": env[cid], "model_id": f"h335.{name}", "utility": float(vals[cid]),
                       "status": status.get(name, {}).get(cid, "completed"), "harness_version": "h335-import-v1",
                       "source": source}
                if name in per_seed:
                    rec["per_seed"] = per_seed[name][cid]
                f.write(json.dumps(rec, sort_keys=True) + "\n")
    print(json.dumps({"board": board["board"], "entrants": len(values), "contexts": len(env)}))


def dev91(board_dir: Path) -> None:
    panel = json.loads((SV / "panel_stage_v.json").read_text())
    llm = [json.loads(x) for x in (SV / "llm_audit.jsonl").read_text().splitlines() if x]
    values, per_seed, status = {}, {}, {}
    for name in ENTRANT_KEYS:
        per_seed[name] = {c: [float(v) for v in panel[c]["gp"][name]] for c in panel}
        if any(len(v) != 2 for v in per_seed[name].values()):
            raise SystemExit(f"{name}: a context lacks one of the 2 search seeds")
        values[name] = {c: sum(v) / len(v) for c, v in per_seed[name].items()}
    for e in LLM:
        recs = {r["cid"]: r for r in llm if r["entrant"] == e}
        values[e] = {c: float(recs[c]["utility"]) for c in panel}
        status[e] = {c: recs[c]["status"] for c in panel}
        if any(abs(values[e][c] - float(panel[c]["llm"][e])) > 0 for c in panel):
            raise SystemExit(f"{e}: llm_audit.jsonl and panel_stage_v.json disagree")
    write(board_dir, values, status, per_seed,
          {"wave": "H335", "phase": "V", "panel_stage_v_sha256": sha(SV / "panel_stage_v.json"),
           "llm_audit_sha256": sha(SV / "llm_audit.jsonl"), "metrics_v_sha256": sha(SV / "metrics_v.json"),
           "seeds": [335101, 335102], "close_commit": "8b7c3041079aab54240d7972431d8527d4b5c5f7"})


def train(board_dir: Path) -> None:
    import h335_core as K  # noqa: F401  (installs the grammar group before any H317 import)
    import h335_lab as LAB
    import common_h317 as C
    cids = K.panel_p201b(C.train_rows())
    recs = LAB.collect([H335 / "runs" / "T_xbabe0", H335 / "runs" / "T_xbabe2"], "T", cids, K.SEEDS_T)
    vals = LAB.values_by_output(recs)
    values = {n: {c: float(v) for c, v in vals[k].items()} for n, k in ENTRANT_KEYS.items()}
    fr = H335 / "frontier" / "T" / "frontier_results.jsonl"
    llm = [json.loads(x) for x in fr.read_text().splitlines() if x]
    status = {}
    for e in LLM:
        r = {x["cid"]: x for x in llm if x["entrant"] == e}
        values[e] = {c: float(r[c]["utility"]) for c in cids}
        status[e] = {c: r[c]["status"] for c in cids}
    write(board_dir, values, status, {},
          {"wave": "H335", "phase": "T", "runs": ["runs/T_xbabe0", "runs/T_xbabe2"],
           "frontier_results_sha256": sha(fr), "decision_T_sha256": sha(H335 / "decision_T.json"),
           "seeds": list(K.SEEDS_T), "close_commit": "8b7c3041079aab54240d7972431d8527d4b5c5f7"})


if __name__ == "__main__":
    {"dev91": dev91, "train": train}[sys.argv[1]](Path(sys.argv[2]))
