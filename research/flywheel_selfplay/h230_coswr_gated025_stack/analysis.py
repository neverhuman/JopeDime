"""H230 frozen H224/H217 estimators and gate, with preregistered H230 identities."""

import hashlib
import subprocess
import types

SOURCE = "e3c978d5d36f50a190b82543e592834b549e0844"
PATH = "research/flywheel_selfplay/h224_learned_switch_gate/analysis.py"
SHA256 = "9aecc0fc3a03bf63d5a1e37f6b98098d9787dd7e7e4e77f8499a8d549ee5f434"
raw = subprocess.check_output(
    ["git", "-C", "/home/ubuntu/JopeDime", "show", SOURCE + ":" + PATH]
)
if hashlib.sha256(raw).hexdigest() != SHA256:
    raise RuntimeError("frozen statistics mismatch")
parent = types.ModuleType("h230_frozen_statistics")
exec(compile(raw, PATH, "exec"), parent.__dict__)
SEEDS = (230101, 230102, 230103)
FOLD_SEEDS = dict(zip(SEEDS, (230141, 230142, 230143)))
ARMS = ("R882", "COSWR", "G025", "STACK")
BOOT_SEED = 230150
MARGIN = 0.002
EPS = 1e-12
parent.SEEDS, parent.FOLD_SEEDS = SEEDS, FOLD_SEEDS
parent.LCB_MARGIN = MARGIN
cell = parent.cell
fit_plan = parent.fit_plan
inner_model_for = parent.inner_model_for
seed_folds = parent.seed_folds
fold_manifest = parent.fold_manifest
partition_jaccard = parent.partition_jaccard
blend = parent.blend
zgap = parent.zgap
choose_tau = parent.choose_tau
gated = parent.gated
stream_seed = parent.stream_seed
shuffle_within_family = parent.shuffle_within_family


def contrast(panel, base, families, seed=BOOT_SEED):
    return parent.contrast(panel, base, families, seed=seed)


def elo_vs_reference(panel, base, families, seed=BOOT_SEED + 1):
    return parent.elo_vs_reference(panel, base, families, seed=seed)


def plateau(panel, base, families, seed=BOOT_SEED + 2):
    result = parent.plateau(panel, base, families, seed=seed)
    result["win"] = result["gain"] > MARGIN + EPS and result["ci95"][0] > EPS
    return result


def verdict(primary, secondary):
    passed = (
        primary["median_gain"] > MARGIN + EPS
        and primary["ci95"][0] > EPS
        and primary["harm_fraction"] <= 0.1
        and sum(x > EPS for x in primary["per_seed_gain"]) >= 2
    )
    return dict(
        verdict="ADVANCE" if passed and secondary["win"] else "NO_ADVANCE",
        primary_pass=passed,
        co_primary_plateau_pass=secondary["win"],
    )
