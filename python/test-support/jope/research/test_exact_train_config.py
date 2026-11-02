from pathlib import Path

import jope_spec_policy

from test_exact_common import atomic_json, prime_source_root, sha256_file

ROOT = Path("/home/ubuntu/jope-flywheel/test_exact_gen0")
CORPUS_DB = ROOT / "corpus" / "strict-test-exact.sqlite"
CATALOG_POINTER = ROOT / "catalog" / "current.json"
WARM = Path(
    "/home/ubuntu/jope-flywheel/training/round1/checkpoints/"
    "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592.model.pt"
)
WARM_SHA = "fbcd5459f2c18f6cd5e4c23eef76a508a91d653a82f27c6ea41bcee8cafa2592"
SCQ = Path("/home/ubuntu/smart_cluster/target/release/scq")
STRIDE_FLOATS = 1536 + 512
TRAINER_POLICY = jope_spec_policy.policy("trainer")


def render_resource_block() -> str:
    """Bind this renderer to the central trainer resource policy."""

    return jope_spec_policy.render_resource_block("trainer")


def render_placement_block() -> str:
    """Bind this renderer to the central trainer capability probes."""

    return jope_spec_policy.render_placement_block("trainer")

# ---------------------------------------------------------------------------
# EQUAL PER-DATASET REPRESENTATION
#
# DATASET_ROW_CAP (128) was an upper BOUND, not a quota. A dataset with 2,612
# strict observations contributed 128 rows and one with 6 contributed 6, so the
# deepest datasets outweighed the shallowest ~21x in every epoch — and the deep
# datasets are exactly the ones the model can memorise. Bounds do not equalise;
# quotas do.
#
# EQUAL_QUOTA is a QUOTA: every dataset contributes EXACTLY this many rows to
# every epoch (all of them, if it owns fewer). Because a static corpus file
# cannot express "a different 32 each epoch", the corpus holds a per-dataset
# POOL and `train()` takes the equal-quota draw from that pool once per epoch
# with a per-epoch seed. Over 24 epochs a deep dataset still exposes its whole
# pool; no single epoch sees more of it than of any other dataset.
#
# N=32 chosen from the measured per-dataset observation distribution over the
# 5,531 datasets that own strict observations (regression train median 806, max
# 2,265; binary train median 220, max 2,612):
#   - Measured on the live corpus after the encodability filter, 88.8-90.7% of
#     datasets clear 32 in every one of the four task/split cells (binary train
#     1239/1376, regression train 3284/3695, binary val 127/143, regression val
#     381/420). The next tier up (64) drops that to 80-85%.
#   - The regression sub-32 tail is the transient first-touch tier (datasets
#     sitting at exactly 25 or 31 observations because they have been evaluated
#     once); they cross 32 on their next wave, so 32 does not pin the quota to
#     an artefact the way 24 would.
#   - 32 divides both per-task batch quotas exactly (round(256*0.75)=192 -> 6
#     datasets/batch, 256-192=64 -> 2 datasets/batch), so every batch carries a
#     constant number of equally-sized reward-pair groups.
EQUAL_QUOTA = 32
VALIDATION_QUOTA = 32
# Rows FETCHED per dataset before the encodability filter. ~17% of observations
# resolve to no grammar engine and are dropped, so the quota must be taken from
# post-filter survivors or it is not a quota at all. Measured: a 128-wide
# candidate window already puts every dataset that CAN clear 32 encodable rows
# over the line (a 64-wide window is enough; wider windows add nothing), so this
# keeps the fetch and resolve cost identical to the old DATASET_ROW_CAP=128.
CANDIDATE_ROW_CAP = 128
# recent_plus_replay only: slots reserved for the newest observations before the
# rest of the pool is filled by an even spread over the dataset's history.
RECENCY_SLOTS = CANDIDATE_ROW_CAP // 2
# Sub-quota datasets: repeat their rows to reach EQUAL_QUOTA. OFF by default.
# There is no per-row weight channel that reaches the reward heads (see
# `equal_quota_draw`), so replication is the only mechanism that would equalise
# their total contribution — but it equalises it with DUPLICATES, which teaches
# the imitation head to memorise the very datasets that have the least evidence,
# and contributes nothing to the reward heads (a duplicated pair has a zero
# score gap, and self_play_bt_loss weights pairs by |gap|).
QUOTA_TAIL_REPLICATE = False
# Bumped whenever the sampler configuration changes: the snapshot directory is
# keyed on it, otherwise a rebuild at an unchanged watermark silently returns the
# corpus built by the PREVIOUS sampler.
SAMPLER_TAG = f"eq{EQUAL_QUOTA}v{VALIDATION_QUOTA}c{CANDIDATE_ROW_CAP}p2"

# Recipe E is intentionally additive. Recipes A-D keep their existing batching
# and objectives byte-for-byte. Fractions are enforced over every five
# 256-row batches because 15% and 10% are not both integral in one such batch.
RECIPE_E_BATCH_MIX = {
    "uniform_broad_replay": 0.50,
    "exact_current_parent_pairs": 0.25,
    "completed_descendant_trajectories": 0.15,
    "validity_instability_hard_negatives": 0.10,
}
RECIPE_E_LOSS_WEIGHTS = {
    "sequence_imitation": 0.20,
    "immediate_pairwise_listwise_reward": 0.25,
    "descendant_value": 0.30,
    "directional_edit_operator_policy": 0.15,
    "validity_stability_calibration": 0.10,
}
# The (task, metric) pairs the trainer accepts into the frozen corpus. This is
# the SINGLE authority: the freeze CTE derives its WHERE clause from it, the
# packed-corpus task guard derives its superset check from it, and the CLI
# derives --task choices from it. Before this existed the CTE, the guard, and
# the CLI each hard-coded regression/binary separately, so every multiclass and
# timeseries observation was silently excluded from every checkpoint -- waves
# evaluated them, the corpus stored them, and the model never saw one.
ENABLED_TASK_METRICS = {
    "regression": "r2",
    "binary": "auc",
    "multiclass": "macro_f1",
    "timeseries": "negative_mase",
}
# Tasks that must ALWAYS be present in a packed corpus. The optional tasks join
# as their observations arrive; their absence is a data condition, not an error.
MANDATORY_TASKS = ("regression", "binary")
# Optional tasks join training through JopeOptionalTaskReadiness.v2. Their
# task-specific dataset floors are deliberately not interchangeable.
#
# Evaluation is deliberately NOT gated on it: waves keep running the task, so
# the evidence that clears the floor can actually accumulate. Gating wave
# generation instead would be a deadlock dressed as a safeguard.
#
# The number guards the shared trunk. Task share tracks dataset share (rows are
# capped per dataset by CANDIDATE_ROW_CAP, not per task), so admitting a task
# with 29 datasets does not give it 0.1% of the gradient harmlessly -- it gives
# those 29 datasets unlimited repetition across epochs, which is how a trunk
# memorises a handful of datasets and calls it a task.
OPTIONAL_TASK_DATASET_FLOORS = {"multiclass": 500, "timeseries": 42}
OPTIONAL_TASK_READINESS = ROOT / "ops" / "optional-task-readiness.v2.json"

