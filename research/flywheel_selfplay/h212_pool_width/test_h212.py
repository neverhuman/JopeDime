"""Synthetic, outcome-free tests for H212's preregistered protocol."""

import numpy as np
import pytest

import analysis
import runner


REAL_CID = "002f7ead64684eb07b68e9a3fdd4c1e667224edc3af728a3d213050693a18031"
SELECTED_PATH = runner.INPUT / "selected.json"
RAW_TABLE_PATH = runner.INPUT / "tables" / f"{REAL_CID}.jtt1"
CACHED_GZ_PATH = runner.CACHE / "jtt1_gz" / f"{REAL_CID}.jtt1.gz"


def test_registered_seed_streams_are_stable_and_distinct():
    cid = "a" * 64
    values = [analysis.stream_seed(seed, cid) for seed in analysis.SEEDS]
    assert values == [analysis.stream_seed(seed, cid) for seed in analysis.SEEDS]
    assert len(set(values)) == 3
    with pytest.raises(ValueError):
        analysis.stream_seed(7, cid)


def test_family_folds_are_complete_disjoint_and_balanced():
    ids = [f"fam-{i:03d}" for i in range(205)]
    folds = analysis.family_folds(ids)
    assert [len(fold) for fold in folds] == [41] * 5
    assert len(set().union(*(set(x) for x in folds))) == 205
    assert all(not (set(folds[i]) & set(folds[j])) for i in range(5) for j in range(i))


def test_native_first_argmin_preserves_first_tie_and_gate_failure_cost():
    utility, index = analysis.first_argmin_utility(
        [0.2, 0.1, 0.1], [0.5, 0.8, 0.9], [True, False, True]
    )
    assert (utility, index) == (-2.0, 1)


def test_family_bootstrap_reports_strict_ties_as_not_wins():
    gain = np.zeros(205)
    gain[0] = 1.0
    gain[1] = -1.0
    ix = np.tile(np.arange(205), (8, 1))
    stats = analysis.family_gain_stats(gain, ix)
    assert stats["win_tie_loss"] == [1, 203, 1]
    assert stats["harm_fraction"] == 1 / 205
    assert stats["ci95"] == [0.0, 0.0]
    assert stats["p_one_sided_median_le_zero"] == 1.0


def test_holm_adjustment_is_monotone_and_bounded():
    adjusted = analysis.holm({"N32": 0.01, "N64": 0.04})
    assert adjusted == {"N32": 0.02, "N64": 0.04}
    with pytest.raises(ValueError):
        analysis.holm({"bad": 1.1})


def test_supplemental_pooling_takes_median_over_seed_level_family_means():
    gains = np.stack([np.full(205, x) for x in (0.1, 0.4, 0.8)])
    ix = np.tile(np.arange(205), (4, 1))
    stats = analysis.pooled_seed_gain_stats(gains, ix)
    assert stats["seed_family_mean_gain"] == [0.1, 0.4, 0.8]
    assert stats["median_over_three_seed_family_means"] == 0.4
    np.testing.assert_allclose(stats["ci95"], [0.4, 0.4], rtol=0, atol=1e-14)
    assert stats["p_one_sided_median_le_zero"] == 0.2


def test_primary_requires_all_seed_ci_harm_and_elo_gates():
    passing = {s: {"ci95": [0.01, 0.1], "harm_fraction": 0.2} for s in analysis.SEEDS}
    elo = {s: 1500.01 for s in analysis.SEEDS}
    assert analysis.decide(passing, elo)["verdict"] == "ADVANCE"
    passing[analysis.SEEDS[-1]] = {"ci95": [0.0, 0.1], "harm_fraction": 0.0}
    assert analysis.decide(passing, elo)["verdict"] == "NO_ADVANCE"
    assert analysis.decide({analysis.SEEDS[0]: passing[analysis.SEEDS[0]]}, elo)["verdict"] == "INCOMPLETE"


def test_width_outputs_are_validated_independently():
    arms = {
        width: [{"program_sha256": f"{width * 100 + i:064x}"} for i in range(width)]
        for width in analysis.WIDTHS
    }
    assert [len(analysis.validate_width_proposals(arms[w], w)) for w in analysis.WIDTHS] == [16, 32, 64]
    assert arms[16] != arms[64][:16]
    with pytest.raises(ValueError, match="exactly 16"):
        analysis.validate_width_proposals(arms[16][:15], 16)
    arms[64][-1] = arms[64][0]
    with pytest.raises(ValueError, match="unique"):
        analysis.validate_width_proposals(arms[64], 64)


def test_end_to_end_smoke_receipt_requires_all_widths_and_outputs():
    receipt = {
        "status": "PASS",
        "widths": list(analysis.WIDTHS),
        "rust_evaluation_completed": True,
        "analysis_row_written": True,
        "pool_record_count": len(analysis.WIDTHS),
    }
    runner.validate_smoke_receipt(receipt)
    with pytest.raises(RuntimeError, match="all registered widths"):
        runner.validate_smoke_receipt({**receipt, "widths": [16]})
    with pytest.raises(RuntimeError, match="Rust evaluation"):
        runner.validate_smoke_receipt({**receipt, "rust_evaluation_completed": False})


def test_dynamic_prediction_uses_explicit_context_identity():
    result = runner.predict_dynamic(
        None, None, REAL_CID, {"lm48_z": np.zeros((1, 48))}, None, None, [], None
    )
    assert result.size == 0


@pytest.mark.skipif(
    not all(path.is_file() for path in (SELECTED_PATH, RAW_TABLE_PATH, CACHED_GZ_PATH)),
    reason="approved H50/H58 TRAIN files are host-local",
)
def test_real_h58_train_file_matches_selected_raw_and_cv_envelope():
    import gzip
    import json

    selected = json.loads(SELECTED_PATH.read_text())
    record = next(row for row in selected["train"] if row["context_sha256"] == REAL_CID)
    raw = RAW_TABLE_PATH.read_bytes()
    compressed = CACHED_GZ_PATH.read_bytes()
    zipped_manifest = json.loads((runner.CACHE / "jtt1_gz_manifest.json").read_text())
    assert runner.sha(CACHED_GZ_PATH) == zipped_manifest[REAL_CID]
    envelope = runner.load_source(
        runner.H58_ENVELOPE_PATH,
        "h212_test_h58_envelope_positive",
        runner.H58_ENVELOPE_SHA,
    )
    runner.validate_train_cv_envelope(
        raw,
        gzip.decompress(compressed),
        record["train_only_jtt1_sha256"],
        envelope.training_cv_envelope,
    )


@pytest.mark.skipif(
    not all(path.is_file() for path in (SELECTED_PATH, RAW_TABLE_PATH, CACHED_GZ_PATH)),
    reason="approved H50/H58 TRAIN files are host-local",
)
def test_real_h58_train_envelope_rejects_flipped_byte():
    import gzip
    import json

    selected = json.loads(SELECTED_PATH.read_text())
    record = next(row for row in selected["train"] if row["context_sha256"] == REAL_CID)
    raw = RAW_TABLE_PATH.read_bytes()
    cached_envelope = gzip.decompress(CACHED_GZ_PATH.read_bytes())
    envelope = runner.load_source(
        runner.H58_ENVELOPE_PATH,
        "h212_test_h58_envelope_negative",
        runner.H58_ENVELOPE_SHA,
    )
    flipped = cached_envelope[:-1] + bytes([cached_envelope[-1] ^ 1])
    with pytest.raises(ValueError, match="CV envelope"):
        runner.validate_train_cv_envelope(
            raw, flipped, record["train_only_jtt1_sha256"], envelope.training_cv_envelope
        )
