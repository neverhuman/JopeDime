"""H303 preflight (outer_train only): run_h303.member_picks / ens_picks at H302's constants (TAU 0.10, k 6, a 2) must
reproduce H302's own picks_bag and ENS_VOTE2 on each regression fold lock's 64 outer_train check contexts."""

import json
import sys

import numpy as np

import run_h303 as r

r.setup()
G = r.G
veto, core, bagged, ens, P8, rows = (G[x] for x in ("veto", "core", "bagged", "ens", "P8", "train_rows"))
SC = r.SCRATCH.parent / "h302_regress" / "scratch"
out = {}
for k in range(4):
    lock = json.loads((SC / f"fold_{k}" / f"fold_{k}_fit_lock.json").read_text())
    tr = sorted(c["context_sha256"] for c in rows if G["fam_to_fold"][c["family_id"]] != k)[:64]
    X = np.concatenate([P8[c]["X"] for c in tr])
    for rs in G["a302"].ROOT_TRAINING_SEEDS:
        outs = r.score_members(SC / f"fold_{k}", lock["members_by_root_seed"][str(rs)], X, len(tr), ens)
        ml = [r.member_picks(o, 0.10, 6, veto, core, bagged) for o in outs]
        ok_m = all(m == list(map(int, o["picks_bag"])) for m, o in zip(ml, outs))
        agg = ens.aggregate_ensemble_picks(outs, len(tr), q_fixed=6, tau=0.10)
        ok_e = r.ens_picks(ml, 2, ens) == list(map(int, agg["ENS_VOTE2"]))
        out[f"{k}:{rs}"] = dict(
            member_picks_match=ok_m, ens_vote2_match=ok_e, changed=int(sum(j != 0 for j in agg["ENS_VOTE2"]))
        )
print(json.dumps(out))
sys.exit(0 if all(v["member_picks_match"] and v["ens_vote2_match"] for v in out.values()) else 1)
