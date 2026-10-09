"""Conservative measured-set and raw-iid targets for train-role contexts only.

The tilt guarantee applies only to the measured set. Shared-network distillation
is approximate; clipping changes the operator. Executed matched games remain
the acceptance test. No inferred head value is a reward.
"""

from dataclasses import asdict, dataclass
import math

from .arena.arms import Program
from .arena.game_spec import digest, require_sha


@dataclass(frozen=True)
class TargetConfig:
    anchor_sha256: str
    trie_sha256: str
    tokenizer_sha256: str
    normalizer_sha256: str
    role_map_sha256: str
    evaluator_sha256: str
    canonicalizer_sha256: str
    protocol: str = "jope.flywheel-s1-inner-rmse/v1"
    temperature: float = 0.2
    alpha: float = 1.0
    advantage_clip: float = 5.0
    form: str = "set"

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name.endswith("sha256"):
                require_sha(value, name)
        if (
            self.protocol != "jope.flywheel-s1-inner-rmse/v1"
            or self.form not in ("set", "raw_iid")
            or not math.isfinite(self.temperature)
            or self.temperature <= 0
            or not math.isfinite(self.alpha)
            or self.alpha < 0
            or not math.isfinite(self.advantage_clip)
            or self.advantage_clip <= 0
        ):
            raise ValueError("invalid measured-tilt configuration")


def tilt_weights(q, q_opt, anchor_logprob, config):
    """The baseline is the unweighted mean of clipped measured quality."""
    if not q or not len(q) == len(q_opt) == len(anchor_logprob):
        raise ValueError("aligned nonempty measured values required")
    if not all(math.isfinite(x) for x in (*q, *q_opt, *anchor_logprob)):
        raise ValueError("finite measured values and anchor likelihood required")
    baseline = math.fsum(q_opt) / len(q_opt)
    # Raw anchor draws already contribute one power of the sampling density.
    # Generalized targets therefore use alpha - 1 on each iid draw.
    exponent = config.alpha if config.form == "set" else config.alpha - 1.0
    logits = [
        exponent * lp
        + max(
            -config.advantage_clip,
            min(config.advantage_clip, (value - baseline) / config.temperature),
        )
        for value, lp in zip(q, anchor_logprob)
    ]
    peak = max(logits)
    weights = [math.exp(x - peak) for x in logits]
    total = math.fsum(weights)
    return tuple(x / total for x in weights), baseline


def prefix_targets(programs, weights):
    """Store prefix reach mass and conditional next-token mass, including EOS."""
    masses = {}
    for program, weight in zip(programs, weights):
        tokens = program.tokens
        for position, token in enumerate(tokens[1:], 1):
            children = masses.setdefault(tokens[:position], {})
            children[token] = children.get(token, 0.0) + weight
    return tuple(
        {
            "prefix": prefix,
            "reach_mass": math.fsum(children.values()),
            "children": tuple(
                (token, mass / math.fsum(children.values()))
                for token, mass in sorted(children.items())
            ),
        }
        for prefix, children in sorted(masses.items())
    )


def check_train(context_id, config, role_check):
    authority = role_check(context_id)
    if authority != {"role": "train", "role_map_sha256": config.role_map_sha256}:
        raise ValueError("fresh train-role authority required; dev/sealed data refused")


def build_targets(rows, config, *, trie, role_check, verify_receipt, anchor_logprob):
    """Verification callbacks must use the pinned native/Rust artifact closure.

    Unknown rows can yield zero support, never substitute a score. Raw-iid groups
    must be complete and unselected; a partial group contributes no actor target.
    The caller supplies PR1 anchor likelihoods and the exact registered trie.
    """
    if trie.identity != config.trie_sha256:
        raise ValueError("anchor trie support identity mismatch")
    grouped = {}
    for original in rows:
        row = dict(original)
        cid = row["context_id"]
        # Do this before eligibility filtering, including unqualified dev rows.
        check_train(cid, config, role_check)
        if row.get("role") != "train":
            raise ValueError("development or sealed row in training input")
        if any(k in row for k in ("predicted_q", "q_hat", "fitness")):
            raise ValueError("only measured q may construct a tilt")
        grouped.setdefault(cid, []).append(row)
    contexts = []
    for cid, members in sorted(grouped.items()):
        valid = []
        keys = set()
        for row in members:
            if row.get("outcome_class") != "qualified_complete":
                if any(row.get(k) is not None for k in ("q", "q_opt", "q_grid")):
                    raise ValueError(
                        "unqualified outcomes cannot carry numeric rewards"
                    )
                continue
            if row.get("source") != "measured":
                raise ValueError("measured provenance required")
            for name in (
                "evaluator_sha256",
                "canonicalizer_sha256",
                "protocol",
                "role_map_sha256",
            ):
                if row.get(name) != getattr(config, name):
                    raise ValueError("target comparison identity mismatch: " + name)
            require_sha(row.get("hwm_key_sha256"), "comparison identity")
            require_sha(row.get("receipt_sha256"), "qualified receipt")
            q, opt = row.get("q"), row.get("q_opt")
            if (
                type(row.get("q_grid")) is not int
                or not isinstance(q, (float, int))
                or not isinstance(opt, (float, int))
                or not math.isfinite(q)
                or not math.isfinite(opt)
                or round(q * 1_000_000) != row["q_grid"]
                or opt != max(-math.log(1000), min(math.log(1000), q))
            ):
                raise ValueError("exact measured raw/clipped/grid quality required")
            proof = verify_receipt(row)
            expected = {
                k: row[k]
                for k in (
                    "context_id",
                    "lineage_id",
                    "hwm_key_sha256",
                    "receipt_sha256",
                    "q",
                    "q_opt",
                    "q_grid",
                )
            }
            p = Program.from_dict(row["program"])
            expected["program_sha256"] = p.program_sha256
            if proof != {
                **expected,
                "qualified": True,
                "closure_verified": True,
                "parity_verified": True,
                "parity_tolerance": 1e-9,
            }:
                raise ValueError("qualified native/Rust receipt closure required")
            if p.tokens not in trie.programs:
                continue  # Zero actor support for a measured program outside the anchor trie.
            catalog = trie.programs[p.tokens]
            if (
                not isinstance(catalog, dict)
                or catalog.get("dsl") != p.dsl
                or catalog.get("program_sha256") != p.program_sha256
                or ("tokens" in catalog and tuple(catalog["tokens"]) != p.tokens)
            ):
                raise ValueError("measured program does not match canonical trie leaf")
            keys.add((row["hwm_key_sha256"], row["lineage_id"]))
            valid.append((row, p))
        if len(keys) > 1:
            raise ValueError(
                "same-context targets require one exact comparison identity"
            )
        if config.form == "set":
            distinct = {}
            for row, p in valid:
                old = distinct.get(p.program_sha256)
                if old and (old[0]["q_grid"], old[0]["q"], old[0]["q_opt"]) != (
                    row["q_grid"],
                    row["q"],
                    row["q_opt"],
                ):
                    raise ValueError("conflicting measured duplicate")
                distinct.setdefault(p.program_sha256, (row, p))
            valid = [distinct[k] for k in sorted(distinct)]
        elif valid:
            groups = {
                (r.get("iid_group_id"), r.get("iid_group_size")) for r, p in valid
            }
            if len(groups) != 1:
                raise ValueError(
                    "raw-iid targets require one complete registered group"
                )
            group, size = next(iter(groups))
            if (
                not group
                or type(size) is not int
                or size != len(members)
                or any(
                    r.get("sampling_checkpoint_sha256") != config.anchor_sha256
                    or r.get("selection_rule") != "iid-trie-sample"
                    for r, p in valid
                )
            ):
                raise ValueError("raw-iid provenance must be unfiltered anchor draws")
            if len(valid) != size:
                valid = []
            elif sorted(r.get("draw_index") for r, p in valid) != list(range(size)):
                raise ValueError("raw-iid draw inventory is incomplete or duplicated")
            else:
                valid.sort(key=lambda item: item[0]["draw_index"])
        eligible = (
            len({p.program_sha256 for r, p in valid}) >= 2
            and len({r["q_grid"] for r, p in valid}) >= 2
        )
        item = {
            "context_id": cid,
            "role": "train",
            "eligible": eligible,
            "lineage_id": next(iter(keys))[1] if keys else members[0]["lineage_id"],
            "programs": (),
            "prefixes": (),
            "certified_anchor_mass": 0.0,
            "baseline_q_opt": None,
        }
        if eligible:
            logprobs = [anchor_logprob(cid, p.tokens) for r, p in valid]
            if any(not math.isfinite(lp) or lp > 0 for lp in logprobs):
                raise ValueError(
                    "finite summed EOS-inclusive anchor likelihood required"
                )
            weights, baseline = tilt_weights(
                [r["q"] for r, p in valid],
                [r["q_opt"] for r, p in valid],
                logprobs,
                config,
            )
            unique_mass = {p.program_sha256: lp for (r, p), lp in zip(valid, logprobs)}
            mass = math.fsum(math.exp(lp) for lp in unique_mass.values())
            if mass > 1 + 1e-9:
                raise ValueError(
                    "certified support mass exceeds the anchor probability"
                )
            item.update(
                programs=tuple(
                    {
                        "program": p.to_dict(),
                        "weight": w,
                        "q": r["q"],
                        "q_opt": r["q_opt"],
                        "q_grid": r["q_grid"],
                        "receipt_sha256": r["receipt_sha256"],
                        "hwm_key_sha256": r["hwm_key_sha256"],
                        "anchor_logprob": lp,
                    }
                    for (r, p), w, lp in zip(valid, weights, logprobs)
                ),
                prefixes=prefix_targets([p for r, p in valid], weights),
                certified_anchor_mass=mass,
                baseline_q_opt=baseline,
            )
        check_train(cid, config, role_check)
        contexts.append(item)
    manifest = {
        "schema": "jope.rlfw-exit-targets/v1",
        "config": asdict(config),
        "contexts": contexts,
        "benchmark_valid": False,
        "scope": "train-only partial measured support; matched games required",
    }
    return {"target_sha256": digest(manifest), "manifest": manifest}
