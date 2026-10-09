"""Measured paired teacher proposals; integrity replay is never another sample."""

from dataclasses import asdict, dataclass
import math
import random
from statistics import fmean, median, stdev
import struct

from .arena.arms import Program
from .arena.game_spec import digest, require_sha


@dataclass(frozen=True)
class TeacherConfig:
    role_map_sha256: str
    evaluator_sha256: str
    canonicalizer_sha256: str
    numeric_margin: float = 1e-9
    minimum_refit_rows: int = 64
    bootstrap: int = 2000
    seed: int = 7
    protocol: str = "jope.flywheel-s1-inner-rmse/v1"

    def __post_init__(self):
        for name in ("role_map_sha256", "evaluator_sha256", "canonicalizer_sha256"):
            require_sha(getattr(self, name), name)
        if (
            not math.isfinite(self.numeric_margin)
            or self.numeric_margin < 0
            or type(self.minimum_refit_rows) is not int
            or self.minimum_refit_rows < 4
            or self.bootstrap != 2000
            or self.seed != 7
            or self.protocol != "jope.flywheel-s1-inner-rmse/v1"
        ):
            raise ValueError("invalid frozen teacher margin protocol")


def train_role(cid, config, role_check):
    if role_check(cid) != {"role": "train", "role_map_sha256": config.role_map_sha256}:
        raise ValueError("teacher proposals require fresh train-role authority")


def measurement(row, config, verify):
    if (
        row.get("role") != "train"
        or row.get("source") != "measured"
        or row.get("outcome_class") != "qualified_complete"
        or any(k in row for k in ("fitness", "predicted_q", "q_hat"))
    ):
        raise ValueError("teacher comparison requires qualified measured train rows")
    for name in (
        "role_map_sha256",
        "evaluator_sha256",
        "canonicalizer_sha256",
        "protocol",
    ):
        if row.get(name) != getattr(config, name):
            raise ValueError("teacher comparison protocol mismatch")
    for name in ("hwm_key_sha256", "receipt_sha256"):
        require_sha(row.get(name), name)
    program = Program.from_dict(row["program"])
    q, opt, grid = row.get("q"), row.get("q_opt"), row.get("q_grid")
    if (
        not isinstance(q, (float, int))
        or not math.isfinite(q)
        or type(grid) is not int
        or grid != round(q * 1e6)
        or opt != max(-math.log(1000), min(math.log(1000), q))
    ):
        raise ValueError("teacher measured raw/clipped/grid quality mismatch")
    proof = verify(row)
    for name in (
        "context_id",
        "lineage_id",
        "hwm_key_sha256",
        "receipt_sha256",
        "q",
        "q_opt",
        "q_grid",
    ):
        if proof.get(name) != row[name]:
            raise ValueError("teacher receipt identity/value mismatch")
    if (
        proof.get("program_sha256") != program.program_sha256
        or proof.get("closure_verified") is not True
        or proof.get("parity_verified") is not True
        or proof.get("qualified") is not True
        or proof.get("parity_tolerance") != 1e-9
    ):
        raise ValueError("teacher native/Rust closure required")
    folds = proof["folds"]
    if len(folds) != 2 or [f["fold_id"] for f in folds] != [0, 1]:
        raise ValueError("two original paired folds required")
    ids = []
    for fold in folds:
        if (
            not fold["row_ids"]
            or len(set(fold["row_ids"])) != len(fold["row_ids"])
            or not len(fold["row_ids"])
            == len(fold["targets"])
            == len(fold["predictions"])
            or not all(
                math.isfinite(v) for v in (*fold["targets"], *fold["predictions"])
            )
            or not math.isfinite(fold["rmse"])
            or fold["rmse"] < 0
        ):
            raise ValueError("complete finite saved out-of-fold predictions required")
        ids.extend(fold["row_ids"])
    if (
        len(set(ids)) != len(ids)
        or not math.isfinite(proof["epsilon"])
        or proof["epsilon"] <= 0
    ):
        raise ValueError("disjoint original folds and pinned positive epsilon required")
    return proof


def paired_folds(incumbent, challenger):
    if incumbent["epsilon"] != challenger["epsilon"]:
        raise ValueError("paired comparison epsilon differs")
    for old, new in zip(incumbent["folds"], challenger["folds"]):
        if old["row_ids"] != new["row_ids"] or bits(old["targets"]) != bits(
            new["targets"]
        ):
            raise ValueError("paired original row membership/target bytes differ")
    eps = incumbent["epsilon"]
    return tuple(
        math.log(max(old["rmse"], eps) / max(new["rmse"], eps))
        for old, new in zip(incumbent["folds"], challenger["folds"])
    )


def bits(values):
    return b"".join(struct.pack(">d", float(x)) for x in values)


def paired_row_se(incumbent, challenger, config):
    """Resample original OOF rows within each fold, pairing both predictions."""
    paired_folds(incumbent, challenger)
    rng = random.Random(config.seed)
    samples = []
    eps = incumbent["epsilon"]
    for _ in range(config.bootstrap):
        ds = []
        for old, new in zip(incumbent["folds"], challenger["folds"]):
            rows = [rng.randrange(len(old["row_ids"])) for _ in old["row_ids"]]
            old_rmse = math.sqrt(
                fmean((old["targets"][i] - old["predictions"][i]) ** 2 for i in rows)
            )
            new_rmse = math.sqrt(
                fmean((new["targets"][i] - new["predictions"][i]) ** 2 for i in rows)
            )
            ds.append(math.log(max(old_rmse, eps) / max(new_rmse, eps)))
        samples.append(fmean(ds))
    return stdev(samples)


def fresh_bitwise(row, measured, config, verify_fresh):
    fresh_row = verify_fresh(row)
    if fresh_row.get("verification_kind") != "fresh-native-rust-recompile":
        raise ValueError("fresh native execution and Rust recompile required")
    fresh = measurement(fresh_row["measurement"], config, lambda r: fresh_row["proof"])
    if fresh["receipt_sha256"] == measured["receipt_sha256"] or any(
        fresh[k] != measured[k]
        for k in (
            "context_id",
            "lineage_id",
            "hwm_key_sha256",
            "program_sha256",
            "q",
            "q_opt",
            "q_grid",
            "epsilon",
        )
    ):
        raise ValueError(
            "fresh replay must be distinct evidence for the identical measurement"
        )
    for old, new in zip(measured["folds"], fresh["folds"]):
        if (
            old["row_ids"] != new["row_ids"]
            or bits(old["targets"]) != bits(new["targets"])
            or bits(old["predictions"]) != bits(new["predictions"])
            or bits([old["rmse"]]) != bits([new["rmse"]])
        ):
            raise ValueError(
                "fresh native/Rust re-verification is not bitwise identical"
            )
    return fresh["receipt_sha256"]


def dry_run(pairs, config, *, role_check, verify_receipt, verify_fresh, verify_refits):
    """Pure reporter. Verifiers wrap pinned artifacts; no teacher writes exist."""
    proposals = []
    seen = set()
    for pair in pairs:
        cid = pair["context_id"]
        train_role(cid, config, role_check)
        if cid in seen:
            raise ValueError("one registered teacher comparison per context")
        seen.add(cid)
        rows = [pair["incumbent"], pair["challenger"]]
        if any(row.get("context_id") != cid for row in rows):
            raise ValueError("teacher context identity mismatch")
        old, new = [measurement(r, config, verify_receipt) for r in rows]
        if (
            old["hwm_key_sha256"] != new["hwm_key_sha256"]
            or old["lineage_id"] != new["lineage_id"]
        ):
            raise ValueError(
                "same exact comparison identity and train lineage required"
            )
        differences = paired_folds(old, new)
        se_row = paired_row_se(old, new, config)
        se_pair = max(se_row, abs(differences[0] - differences[1]) / 2)
        margin = max(config.numeric_margin, 0.01, 1.96 * se_pair)
        passed = fmean(differences) > margin
        fresh_receipts = []
        refits = None
        scope = "promoted_within_protocol_only"
        if passed:
            train_role(cid, config, role_check)
            fresh_receipts = [
                fresh_bitwise(r, proof, config, verify_fresh)
                for r, proof in zip(rows, (old, new))
            ]
            n = sum(len(f["row_ids"]) for f in old["folds"])
            if n >= config.minimum_refit_rows:
                refits = verify_refits(pair)
                registration = pair.get("refit_registration")
                if (
                    not isinstance(registration, dict)
                    or registration.get("scope") != "train-fitting-only"
                    or registration.get("context_id") != cid
                    or registration.get("hwm_key_sha256") != old["hwm_key_sha256"]
                    or registration.get("program_sha256s")
                    != [old["program_sha256"], new["program_sha256"]]
                    or len(set(registration.get("split_ids", ()))) != 5
                    or len(refits) != 5
                ):
                    raise ValueError("five preregistered train-fitting refits required")
                require_sha(pair.get("refit_registration_sha256"), "refit registration")
                if digest(registration) != pair["refit_registration_sha256"]:
                    raise ValueError("refit registration digest mismatch")
                if [r.get("split_id") for r in refits] != registration["split_ids"]:
                    raise ValueError("refit inventory differs from registration")
                for r in refits:
                    require_sha(r.get("receipt_sha256"), "qualified refit receipt")
                    if (
                        r.get("registration_sha256")
                        != pair["refit_registration_sha256"]
                        or r.get("closure_verified") is not True
                        or r.get("parity_verified") is not True
                        or r.get("parity_tolerance") != 1e-9
                        or r.get("scope") != "train-fitting-only"
                        or r.get("context_id") != cid
                        or r.get("program_sha256s") != registration["program_sha256s"]
                        or len(r.get("fold_d", ())) != 2
                        or not all(math.isfinite(x) for x in r["fold_d"])
                    ):
                        raise ValueError("qualified paired refit closure required")
                passed = median(fmean(r["fold_d"]) for r in refits) > 0
                scope = "robust_cross_split" if passed else "refit_rejected"
        train_role(cid, config, role_check)
        proposals.append(
            {
                "context_id": cid,
                "lineage_id": old["lineage_id"],
                "old_program_sha256": old["program_sha256"],
                "new_program_sha256": new["program_sha256"],
                "hwm_key_sha256": old["hwm_key_sha256"],
                "measured_receipt_sha256s": [
                    old["receipt_sha256"],
                    new["receipt_sha256"],
                ],
                "fold_d": differences,
                "paired_d": fmean(differences),
                "row_bootstrap_se": se_row,
                "se_pair": se_pair,
                "margin": margin,
                "would_replace_shadow_teacher": passed,
                "qualification_scope": scope if passed else "no_replace",
                "fresh_integrity_receipt_sha256s": fresh_receipts,
                "refits": refits,
                "statistical_observations": 2,
                "comparisons": 1,
            }
        )
    manifest = {
        "schema": "jope.rlfw-teacher-dry-run/v1",
        "config": asdict(config),
        "proposals": proposals,
        "writes": 0,
        "benchmark_valid": False,
        "scope": "train-only research proposals; owner adoption decision still required",
    }
    return {"manifest": manifest, "sha256": digest(manifest)}
