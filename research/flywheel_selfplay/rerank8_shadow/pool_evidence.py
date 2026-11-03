"""Unique scored-pool evidence ledger; timestamps and serialization are not novelty."""

import hashlib
import copy
import json


def pool_sha256(
    contexts,
    candidate_pids,
    anchor_parameters_sha256,
    incumbent_parameters_sha256,
    evaluation_identity,
    dope_sha256,
):
    for value in [anchor_parameters_sha256, incumbent_parameters_sha256, dope_sha256]:
        assert len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    assert len(contexts) == len(candidate_pids) and len(set(contexts)) == len(contexts)
    rows = []
    for cid, pids in zip(contexts, candidate_pids):
        assert len(pids) == 8 and len(set(pids)) == 8
        rows.append({"cid": cid, "programs": sorted(str(p) for p in pids)})
    payload = {
        "schema": "jope.semantic-scored-pool/v1",
        "contexts": sorted(rows, key=lambda x: x["cid"]),
        "pool_anchor_parameter_sha256": anchor_parameters_sha256,
        "live_incumbent_parameter_sha256": incumbent_parameters_sha256,
        "evaluation_identity": evaluation_identity,
        "DOPE_sha256": dope_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class UniquePoolEvidence:
    def __init__(self, state=None):
        state = state or {}
        self.seen = set(state.get("counted_pool_sha256s", []))
        self.counted = state.get("counted_rounds", []).copy()
        self.streak = state.get("consecutive_positive_counted_rounds", 0)
        self.last = state.get("last_observed_round")

    def annotate(self, row, historical=False):
        assert row["completed_round"] and row["scoring_status"] == "SCORED"
        key = row["pool_sha256"]
        assert len(key) == 64
        if historical:
            row.update(
                counts_toward_gate=False,
                status="REPEAT_NO_NEW_EVIDENCE",
                gate_reason="RETROSPECTIVE_CONTRACT_REVISION_NO_LIVE_GATE_CREDIT",
            )
            self.last = row["round"]
            return row
        if self.last is not None and row["round"] != self.last + 1:
            self.streak = 0
            row["gap_reset_positive_streak"] = True
        self.last = row["round"]
        if key in self.seen:
            row.update(
                counts_toward_gate=False,
                status="REPEAT_NO_NEW_EVIDENCE",
                gate_reason="POOL_HASH_ALREADY_COUNTED",
            )
            return row
        self.seen.add(key)
        self.counted.append({"round": row["round"], "pool_sha256": key})
        results = row["panels"]["P2_GPU_EXACT_R3175_TOP8"]["seeds"]
        positive = set(results) == {"278101", "278102", "278103"} and all(
            s["family_median_delta"] > 0
            and s["shadow_invalid_picks"] <= s["incumbent_invalid_picks"]
            for s in results.values()
        )
        self.streak = self.streak + 1 if positive else 0
        row.update(
            counts_toward_gate=True,
            status="SCORED",
            gate_reason="FIRST_ELIGIBLE_LIVE_ROUND_FOR_POOL_HASH",
            strict_positive_no_excess_invalid=positive,
        )
        return row

    @classmethod
    def replay(cls, records):
        """Recover novelty from immutable committed round records, even if state write was interrupted."""
        ledger = cls()
        for original in sorted(records, key=lambda r: r["round"]):
            r = copy.deepcopy(original)
            ledger.annotate(r, historical=r.get("record_mode") != "LIVE_SHADOW")
            if "counts_toward_gate" in original:
                assert r["counts_toward_gate"] == original["counts_toward_gate"]
                assert r["status"] == original["status"]
        return ledger

    def state(self):
        return {
            "schema": "jope.unique-pool-evidence/v1",
            "counted_pool_sha256s": sorted(self.seen),
            "counted_rounds": self.counted,
            "consecutive_positive_counted_rounds": self.streak,
            "last_observed_round": self.last,
        }

    def gate(self):
        return {
            "schema": "jope.shadow-gate/unique-pool-v1",
            "H278_P2_ADOPT_SHADOW_PASS": True,
            "P2_qualification_source": "H281 GPU-exact r3175 replay and frozen H278 P2 rule",
            "unique_counted_pools": len(self.seen),
            "counted_rounds": self.counted,
            "consecutive_positive_counted_rounds": self.streak,
            "consecutive_strict_success_rounds": self.streak,
            "numerical_prerequisites_met": self.streak >= 3,
            "H280": "PENDING_QUALIFICATION",
            "H282": "FRESH_POOL_EVIDENCE_EXPECTED",
            "adoption_authorized": False,
            "promotion_wiring": False,
            "decision_owner": "watcher",
            "policy": "Only unique scored-pool identities count; repeats are ignored, fresh negative evidence resets streak",
        }
