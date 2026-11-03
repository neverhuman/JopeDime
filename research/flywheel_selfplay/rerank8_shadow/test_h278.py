import copy
import numpy as np
from h278_features import build_topk
from features import build_topk_feature_matrix_for_context, compute_sem24_numpy
from h278_metrics import advisory_gate, compare


def test_H276_feature_contract_bit_exact_to_existing_packet_builder():
    rng = np.random.default_rng(278)
    for n in [8, 16, 1152]:
        prog = rng.normal(size=(n, 412)).astype("f4")
        cost = rng.normal(size=n).astype("f4")
        prior = rng.normal(size=n).astype("f4")
        sem = compute_sem24_numpy(prog)
        proj = rng.normal(size=(n, 16)).astype("f4")
        lm = rng.normal(size=48).astype("f4")
        dope = rng.normal(size=629).astype("f4")
        order, old = build_topk_feature_matrix_for_context(
            cost, prior, prog, sem, proj, lm, dope
        )
        new_order, new = build_topk(cost, prior, prog[order], sem, proj, lm, dope, 8)
        assert np.array_equal(order, new_order) and np.array_equal(old, new)


def test_invalid_count_uses_gate_not_utility_value():
    r = compare([-2.0, 0.0], [-2.0, 1.0], ["a", "b"], [True, True], [True, True])
    assert r["shadow_invalid_picks"] == 0 and r["family_median_delta"] > 0.0
    r = compare([-2.0, 0.0], [-2.0, 1.0], ["a", "b"], [True, True], [False, True])
    assert not r["strict_positive_no_excess_invalid"]


def test_harms_and_ties_never_become_positive():
    r = compare([0.0, 0.0], [0.0, -0.02], ["a", "b"], [True] * 2, [True] * 2)
    assert r["harmed_families"] == 1 and r["harmed_families_gt_001"] == 1
    assert not r["strict_positive_no_excess_invalid"]


def row(n, gain=0.01):
    return {
        "round": n,
        "status": "SCORED",
        "completed_round": True,
        "seeds": {
            str(s): {
                "family_median_delta": gain,
                "shadow_invalid_picks": 0,
                "incumbent_invalid_picks": 0,
            }
            for s in [278101, 278102, 278103]
        },
    }


def test_gate_is_advisory_and_requires_P2_and_three_consecutive_rounds():
    rows = [row(n) for n in [1, 2, 3]]
    assert not advisory_gate(rows, "PENDING")["numerical_prerequisites_met"]
    g = advisory_gate(rows, "ADOPT_SHADOW_PASS")
    assert g["numerical_prerequisites_met"] and not g["adoption_authorized"]
    assert not advisory_gate([row(1), row(3), row(4)], "ADOPT_SHADOW_PASS")[
        "numerical_prerequisites_met"
    ]
    assert not advisory_gate([row(1), row(2), row(3, 0.0)], "ADOPT_SHADOW_PASS")[
        "numerical_prerequisites_met"
    ]
    bad = copy.deepcopy(rows)
    bad[-1]["seeds"]["278103"]["shadow_invalid_picks"] = 1
    assert not advisory_gate(bad, "ADOPT_SHADOW_PASS")["numerical_prerequisites_met"]


def test_H278_masks_all_nonbaseline_slots_before_margin():
    from h278_metrics import frozen_picks

    sc = np.array([[0.0, 4.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    pr = np.zeros_like(sc)
    pr[0, 1] = 0.10
    assert frozen_picks(sc, pr, 0.0).tolist() == [2]
    pr[0, 2] = 0.11
    assert frozen_picks(sc, pr, 0.0).tolist() == [0]
    pr[:] = 1.0
    assert frozen_picks(sc, pr, 0.0).tolist() == [0]
    assert frozen_picks(sc, np.zeros_like(pr), float("inf")).tolist() == [0]


def test_exported_features_stay_authoritative_with_one_ULP_context_norm_difference():
    from h278_shadow import validate_packet_alignment

    a = np.ones((91, 8, 568), np.float32)
    b = a.copy()
    b[0, :, 567] = np.nextafter(np.float32(1.0), np.float32(2.0))
    before = b.copy()
    r = validate_packet_alignment(b, a)
    assert r["exported_features_used_unmodified"] and np.array_equal(b, before)
    assert r["local_context_norm_max_float32_ULP_difference"] == 1
    b[0, 1, 40] = 2.0
    import pytest

    with pytest.raises(AssertionError):
        validate_packet_alignment(b, a)
