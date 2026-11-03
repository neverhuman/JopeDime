import copy
from pool_evidence import UniquePoolEvidence, pool_sha256

A = "a" * 64
B = "b" * 64
C = "c" * 64
D = "d" * 64
PIDS = [[f"{i:064x}" for i in range(8)], [f"{i + 8:064x}" for i in range(8)]]


def fingerprint(pids=None, anchor=A, inc=A):
    return pool_sha256(
        ["cid1", "cid2"], pids or PIDS, anchor, inc, "fixed-evaluator", D
    )


def row(n, key=A, gain=0.1, invalid=0):
    return {
        "round": n,
        "pool_sha256": key,
        "completed_round": True,
        "scoring_status": "SCORED",
        "record_mode": "LIVE_SHADOW",
        "panels": {
            "P2_GPU_EXACT_R3175_TOP8": {
                "seeds": {
                    str(s): {
                        "family_median_delta": gain,
                        "shadow_invalid_picks": invalid,
                        "incumbent_invalid_picks": 0,
                    }
                    for s in [278101, 278102, 278103]
                }
            }
        },
    }


def test_pool_identity_is_semantic_and_order_invariant():
    assert fingerprint() == fingerprint([list(reversed(x)) for x in PIDS])
    assert fingerprint() != fingerprint(anchor=B)
    assert fingerprint() != fingerprint(inc=B)
    changed = copy.deepcopy(PIDS)
    changed[0][0] = "f" * 64
    assert fingerprint() != fingerprint(changed)
    # Round numbers, timestamps, file paths, checkpoint serialization and reranker changes are not inputs.
    assert len(fingerprint()) == 64


def test_identical_rounds_only_first_counts():
    ledger = UniquePoolEvidence()
    out = [ledger.annotate(row(n)) for n in range(1, 6)]
    assert [r["counts_toward_gate"] for r in out] == [True, False, False, False, False]
    assert all(r["status"] == "REPEAT_NO_NEW_EVIDENCE" for r in out[1:])
    assert (
        ledger.gate()["unique_counted_pools"] == 1
        and not ledger.gate()["numerical_prerequisites_met"]
    )


def test_every_previously_counted_pool_is_remembered():
    ledger = UniquePoolEvidence()
    rows = [ledger.annotate(row(n, key)) for n, key in enumerate([A, B, A, C], 1)]
    assert [r["counts_toward_gate"] for r in rows] == [True, True, False, True]
    assert ledger.gate()["consecutive_positive_counted_rounds"] == 3
    assert (
        ledger.gate()["numerical_prerequisites_met"]
        and not ledger.gate()["adoption_authorized"]
    )


def test_negative_or_invalid_new_pool_resets_but_repeats_do_not_gain_credit():
    ledger = UniquePoolEvidence()
    ledger.annotate(row(1, A))
    ledger.annotate(row(2, B, gain=0.0))
    assert ledger.streak == 0
    r = ledger.annotate(row(3, B))
    assert not r["counts_toward_gate"] and ledger.streak == 0
    ledger.annotate(row(4, C, invalid=1))
    assert ledger.streak == 0


def test_historical_revisions_have_no_live_gate_credit():
    ledger = UniquePoolEvidence()
    out = [ledger.annotate(row(n), historical=True) for n in range(1, 25)]
    assert not any(r["counts_toward_gate"] for r in out)
    assert ledger.gate()["unique_counted_pools"] == 0
    assert ledger.annotate(row(25))["counts_toward_gate"]


def test_restart_replays_committed_records_not_just_state_file():
    ledger = UniquePoolEvidence()
    rows = [ledger.annotate(row(1, A)), ledger.annotate(row(2, A))]
    # Simulates process interruption between atomic round publication and state publication.
    recovered = UniquePoolEvidence.replay(rows)
    assert not recovered.annotate(row(3, A))["counts_toward_gate"]
    assert recovered.gate()["unique_counted_pools"] == 1


def test_unknown_round_gap_resets_positive_streak():
    ledger = UniquePoolEvidence()
    ledger.annotate(row(1, A))
    r = ledger.annotate(row(3, B))
    assert r["gap_reset_positive_streak"] and ledger.streak == 1
