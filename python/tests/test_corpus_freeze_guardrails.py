"""The two `ops/generalization-plan.md` guardrails, tested off the freeze path.

Both live in `scripts/test_exact_train.py` and both are only exercised by a real
freeze, which reads the whole 21 GB observations table and takes ~20 minutes.
That is far too expensive to be the only proof they work, and it is also the
wrong proof: a guardrail has to be shown FIRING, and a live freeze that fires
one of them quarantines a multi-GB snapshot to do it.

The plan asked for exactly two things (ops/generalization-plan.md:93-96):

    "alarm when the training corpus dataset count *falls* between freezes, or
     when median rows/dataset diverges by more than 2x between train and
     validation"
    "assert in the corpus builder that train and validation per-dataset medians
     are within 2x of each other, and fail the freeze if not"

Note the deliberate asymmetry, which these tests pin: the dataset-count check is
an ALARM (freeze, then shout) and the median check is an ASSERT (refuse the
freeze). Evidence only accumulates, so a fallen dataset count is always a
sampler or catalog defect — but refusing to publish on it would leave the
trainer pinned to the OLD corpus, which is the starvation the guardrail exists
to detect. A diverged median, by contrast, makes the run itself uninterpretable:
the historical failure was train median 6 against validation median 485, an 80x
inversion that manufactured ~80% of an apparent overfit gap and was read as a
model problem for a day.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))

from test_exact_train import (  # noqa: E402
    SPLIT_MEDIAN_RATIO_MAX,
    append_freeze_history,
    freeze_history_rows,
    publish_freeze_guard,
    quota_distribution,
    split_median_divergence,
)

# The live corpus at watermark 1,805,437: 2,857 datasets over four splits, every
# one of them drawing exactly its 32-row quota per epoch.
LIVE_EPOCH_DRAW = {
    "binary/train": {"datasets": 499, "median": 32, "rows": 15_968},
    "binary/validation": {"datasets": 43, "median": 32, "rows": 1_376},
    "regression/train": {"datasets": 2_072, "median": 32.0, "rows": 66_304},
    "regression/validation": {"datasets": 243, "median": 32, "rows": 7_776},
}


def record(watermark: int, datasets: dict[str, int], **extra) -> dict:
    return {
        "schema": "JopeCorpusFreezeHistory.v1",
        "watermark": watermark,
        "corpus_mode": "full_consolidation",
        "sampler": "eq32v32c128",
        "datasets_total": sum(datasets.values()),
        "per_split_datasets": datasets,
        **extra,
    }


LIVE_SPLITS = {
    "binary/train": 499,
    "binary/validation": 43,
    "regression/train": 2_072,
    "regression/validation": 243,
}


# ---------------------------------------------------------------------------
# Guardrail 1: the corpus dataset count must never fall between freezes
# ---------------------------------------------------------------------------


def test_the_first_freeze_has_nothing_to_compare_against(tmp_path):
    guard = publish_freeze_guard(record(1_805_437, LIVE_SPLITS), root=tmp_path)
    assert guard["regressed"] is False
    assert guard["severity"] == "ok"
    assert guard["previous_best_datasets"] == 0
    assert guard["threshold"] == "no prior freeze recorded"


def test_a_wider_freeze_is_healthy_and_moves_the_high_water_mark(tmp_path):
    """The unlock this workstream exists to deliver: 2,857 datasets -> 18,567."""
    append_freeze_history(record(1_805_437, LIVE_SPLITS), root=tmp_path)
    wider = {
        "binary/train": 2_964,
        "binary/validation": 323,
        "regression/train": 13_740,
        "regression/validation": 1_540,
    }
    guard = publish_freeze_guard(record(4_690_000, wider), root=tmp_path)
    assert guard["regressed"] is False
    assert guard["datasets_total"] == 18_567
    assert guard["previous_best_datasets"] == 2_857
    assert guard["severity"] == "ok"


def test_a_narrower_freeze_is_a_critical_alarm(tmp_path):
    """A sampler that silently drops datasets. Evidence only accumulates, so
    this can never be a property of the data."""
    append_freeze_history(record(1_805_437, LIVE_SPLITS), root=tmp_path)
    narrower = dict(LIVE_SPLITS, **{"regression/train": 400})
    guard = publish_freeze_guard(record(4_690_000, narrower), root=tmp_path)
    assert guard["regressed"] is True
    assert guard["severity"] == "critical"
    assert guard["previous_best_datasets"] == 2_857
    assert guard["previous_best_watermark"] == 1_805_437
    assert "1185 datasets at watermark 4690000" in guard["observed"]
    assert ">=2857 datasets" in guard["threshold"]


def test_the_high_water_mark_is_a_maximum_not_the_previous_row(tmp_path):
    """One bad freeze must not lower the bar for the next one."""
    append_freeze_history(record(1, {"regression/train": 2_857}), root=tmp_path)
    append_freeze_history(record(2, {"regression/train": 400}), root=tmp_path)
    guard = publish_freeze_guard(record(3, {"regression/train": 1_000}), root=tmp_path)
    assert guard["previous_best_datasets"] == 2_857
    assert guard["regressed"] is True


def test_the_guard_document_is_published_where_the_alarm_scan_reads_it(tmp_path):
    publish_freeze_guard(record(4_690_000, LIVE_SPLITS), root=tmp_path)
    published = json.loads(
        (tmp_path / "ops" / "corpus-freeze-guard.v1.json").read_text()
    )
    assert published["schema"] == "JopeCorpusFreezeGuard.v1"
    assert published["alarm"] == "corpus_dataset_count_fell"


def test_a_corrupt_history_line_never_blocks_a_freeze(tmp_path):
    """The ledger is append-only from more than one process. A torn line must
    cost the comparison, not the freeze."""
    path = tmp_path / "training" / "corpus-freeze-history.v1.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"watermark": 1, "datasets_total": 2857}\n{"watermark": 2, "dat\n')
    assert len(freeze_history_rows(tmp_path)) == 1
    guard = publish_freeze_guard(record(3, LIVE_SPLITS), root=tmp_path)
    assert guard["previous_best_datasets"] == 2_857


# ---------------------------------------------------------------------------
# Guardrail 2: train and validation per-dataset medians within 2x
# ---------------------------------------------------------------------------


def test_the_live_corpus_passes_with_ratio_one():
    scopes = split_median_divergence(LIVE_EPOCH_DRAW)
    assert set(scopes) == {"binary", "regression"}
    for task, values in scopes.items():
        assert values["ratio"] == 1.0, task


def test_the_historical_defect_is_caught():
    """train median 6 vs validation median 485 — the 80x inversion that
    manufactured most of an apparent overfit gap."""
    scopes = split_median_divergence(
        {
            "regression/train": {"datasets": 2_072, "median": 6},
            "regression/validation": {"datasets": 243, "median": 485},
            "binary/train": {"datasets": 499, "median": 32},
            "binary/validation": {"datasets": 43, "median": 32},
        }
    )
    assert scopes["regression"]["ratio"] > SPLIT_MEDIAN_RATIO_MAX
    assert round(scopes["regression"]["ratio"], 2) == 80.83
    # Per task, so a healthy task is not condemned by a sick one.
    assert scopes["binary"]["ratio"] == 1.0


def test_the_ratio_is_direction_free():
    """A train split that is deeper than validation is just as wrong as the
    reverse, and the assert must not care which way round it is."""
    high = split_median_divergence(
        {
            "regression/train": {"datasets": 10, "median": 32},
            "regression/validation": {"datasets": 10, "median": 4},
        }
    )
    low = split_median_divergence(
        {
            "regression/train": {"datasets": 10, "median": 4},
            "regression/validation": {"datasets": 10, "median": 32},
        }
    )
    assert high["regression"]["ratio"] == low["regression"]["ratio"] == 8.0


def test_exactly_two_x_is_allowed_and_just_over_is_not():
    at = split_median_divergence(
        {
            "binary/train": {"datasets": 1, "median": 16},
            "binary/validation": {"datasets": 1, "median": 32},
        }
    )
    over = split_median_divergence(
        {
            "binary/train": {"datasets": 1, "median": 15},
            "binary/validation": {"datasets": 1, "median": 32},
        }
    )
    assert at["binary"]["ratio"] == SPLIT_MEDIAN_RATIO_MAX
    assert not at["binary"]["ratio"] > SPLIT_MEDIAN_RATIO_MAX
    assert over["binary"]["ratio"] > SPLIT_MEDIAN_RATIO_MAX


def test_a_split_with_no_datasets_is_not_asserted_on():
    """`build_corpus` already refuses a freeze whose validation split is empty,
    so an absent side here must report `None` rather than divide by zero."""
    scopes = split_median_divergence(
        {
            "regression/train": {"datasets": 2_072, "median": 32},
            "regression/validation": {"datasets": 0, "median": 0},
        }
    )
    assert scopes["regression"]["ratio"] is None
    assert scopes["regression"]["validation"] is None


def test_pool_medians_would_have_failed_a_healthy_corpus():
    """Why the assert is on the EPOCH-DRAW medians and not the pool medians.

    The pool is 128-wide for train and 32-wide for validation BY DESIGN
    (test_exact_train.py:70, :63), so the live, healthy corpus has a pool ratio
    of 3.09x. Asserting there would quarantine every good freeze and say
    nothing about the failure the guardrail exists to catch.
    """
    pool = {
        "regression/train": {"datasets": 2_072, "median": 99.0},
        "regression/validation": {"datasets": 243, "median": 32},
    }
    assert split_median_divergence(pool)["regression"]["ratio"] > SPLIT_MEDIAN_RATIO_MAX
    assert split_median_divergence(LIVE_EPOCH_DRAW)["regression"]["ratio"] == 1.0


# ---------------------------------------------------------------------------
# The distribution the guardrails read
# ---------------------------------------------------------------------------


def test_quota_distribution_matches_the_live_manifest():
    """Same numbers as the frozen corpus manifest's binary/train pool."""
    counts = [94] * 100 + [104] * 300 + [117] * 99
    stats = quota_distribution(counts, 32)
    assert stats["datasets"] == 499
    assert stats["min"] == 94
    assert stats["median"] == 104
    assert stats["max"] == 117
    assert stats["below_quota_datasets"] == 0
    assert stats["above_quota_datasets"] == 499


def test_an_empty_split_reports_zero_rather_than_raising():
    assert quota_distribution([], 32) == {"datasets": 0, "rows": 0}
