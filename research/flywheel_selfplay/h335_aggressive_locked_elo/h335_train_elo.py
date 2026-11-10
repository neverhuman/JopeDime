"""H335 descriptive TRAIN (P201b) AUDIT-Elo ladder over the 12 pre-declared entrants (PREREG §3.3 'reported, not
gating'). Same functions as the dev91 readout (h335_stats.elo_panel, anchor H257_REPLICA = 1500); each P201b family
holds one context, so families = contexts here. Run in sandbox_h335.sh from the frozen copy:
  H335_CODE=/home/ubuntu/h335-stage/code ./sandbox_h335.sh env H335_GROUP=BASE PYTHONPATH=$SRC python3 h335_train_elo.py
"""
import json
import sys
from pathlib import Path

import h335_core as K  # noqa: F401  (installs the grammar group before any H317 import)
import h335_lab as LAB
import h335_stats as H
import common_h317 as C

DATA = Path("/home/ubuntu/jope-data/h335-cl")
dec = json.loads((DATA / "decision_T.json").read_text())
gstar, stack = dec["decision"]["G_star_after_time_cap"], dec["decision"]["stack"]
keys = {"H335_AGG": f"AGG.g{gstar}.{stack}", "OLD_MATCHED": f"OLD.g{gstar}.single",
        "OLD_MATCHED_CV3": f"OLD.g{gstar}.{K.STACK}", "H317_BASE": "H317_BASE.single",
        "H257_REPLICA": "H257_REPLICA.CV1", "BANK_NN8_PICK": "BANK_NN8_PICK.single",
        "BANK_NN8_PORTFOLIO": "BANK_NN8_PORTFOLIO.CV5", "BANK_NN1": "BANK_NN1.single"}
cids = K.panel_p201b(C.train_rows())
recs = LAB.collect([DATA / "runs" / "T_xbabe0", DATA / "runs" / "T_xbabe2"], "T", cids, K.SEEDS_T)
vals = LAB.values_by_output(recs)
ent = {n: vals[k] for n, k in keys.items()}
llm = [json.loads(x) for x in (DATA / "frontier" / "T" / "frontier_results.jsonl").read_text().splitlines() if x]
for e in sorted({r["entrant"] for r in llm}):
    ent[e] = {r["cid"]: float(r["utility"]) for r in llm if r["entrant"] == e}
fam_of = {r["context_sha256"]: r["family_id"] for r in C.train_rows()}
elo = H.elo_panel(ent, fam_of, K.ELO_ANCHOR)
out = {"schema": "jope.h335.train-elo/v1", "panel": "P201b", "locked_output": keys["H335_AGG"],
       "elo_audit_train": elo, "family_means_train": {n: H.family_mean(v, fam_of) for n, v in ent.items()},
       "llm_status": {e: sorted({r["status"] for r in llm if r["entrant"] == e}) for e in sorted({r["entrant"] for r in llm})}}
(DATA / "checks" / "train_elo.json").write_text(json.dumps(out, indent=1, sort_keys=True))
for n, v in sorted(elo["elo"].items(), key=lambda kv: -kv[1]["elo"]):
    print(f"{n:22s} {v['elo']:8.1f}  [{v['ci95'][0]:7.1f}, {v['ci95'][1]:7.1f}]  fam_mean={out['family_means_train'][n]:.4f}")
sys.exit(0)
