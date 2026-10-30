from __future__ import annotations

import random
import re
from collections.abc import Iterable
from dataclasses import dataclass

KNOWN_VALID_FALLBACK = "(pipe X (robust_scale) (ridge))"

TRANSFORMS = {
    "standardize": {},
    "robust_scale": {},
    "rank": {},
    "blom": {},
    "quantile_bins": {"bins": [4, 8, 16, 32]},
    "bin_onehot": {"bins": [4, 8, 12, 16]},
    "hinge": {"knots": [2, 4, 6, 8]},
    "winsor_clip": {"q": [0.01, 0.02, 0.05, 0.1]},
    "cheb": {"degree": [2, 3, 4, 6]},
    "poly_interact": {},
    "rff_cos": {"n": [8, 16, 32, 64], "gamma": [0.5, 1.0, 2.0], "seed": [2, 7, 11, 23]},
    "rff_sin": {"n": [8, 16, 32, 64], "gamma": [0.5, 1.0, 2.0], "seed": [3, 5, 13, 29]},
    "rand_proj": {"k": [8, 16, 32, 64], "seed": [7, 17, 31]},
    "oblique_hinge": {"n": [8, 16, 32], "seed": [7, 19]},
    "row_stats": {},
    "pair_prod": {"topk": [4, 8, 12]},
    "target_bin_encode": {"bins": [4, 8, 16], "prior": [1.0, 5.0, 10.0]},
    "sigmoid": {},
}

HEADS = {
    "ridge": {},
    "elasticnet": {"alpha": [0.0005, 0.001, 0.01], "l1_ratio": [0.2, 0.5, 0.8]},
    "histgbm": {"depth": [2, 3, 4], "lr": [0.05, 0.1, 0.15], "rounds": [80, 120, 180]},
    "histgbm2": {
        "leaves": [15, 31, 63],
        "lr": [0.03, 0.05, 0.08],
        "rounds": [300, 500],
        "bags": [3, 5],
    },
    "gpr": {"kernel": ["rbf", "rbf+linear", "rbf*periodic"], "m": [300, 400], "steps": [30, 40]},
    "ksmooth": {"k": [16, 32, 64], "gamma": [0.5, 1.0, 2.0]},
}


@dataclass(frozen=True)
class DecodeChoice:
    transforms: tuple[str, ...] = ()
    head: str = "ridge"


def sample_genome(rng: random.Random | None = None, *, kind: str | None = None) -> str:
    rng = rng or random.Random()
    if kind not in (None, "pipe"):
        raise ValueError("Jope V1 active grammar is pipeline-only")
    return sample_pipeline(rng)


def sample_pipeline(rng: random.Random | None = None, max_transforms: int = 6) -> str:
    rng = rng or random.Random()
    n = rng.randint(0, max(0, max_transforms))
    ops = rng.sample(list(TRANSFORMS), k=min(n, len(TRANSFORMS)))
    stages = [_format_stage(op, TRANSFORMS[op], rng, allow_append=True) for op in ops]
    head = rng.choice(list(HEADS))
    stages.append(_format_stage(head, HEADS[head], rng, allow_append=False))
    return "(pipe X " + " ".join(stages) + ")"


def decode_choice(choice: DecodeChoice, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    stages = [_format_stage(op, TRANSFORMS.get(op, {}), rng, allow_append=True) for op in choice.transforms]
    stages.append(_format_stage(choice.head, HEADS.get(choice.head, {}), rng, allow_append=False))
    return "(pipe X " + " ".join(stages) + ")"


def corrupt_genome(genome: str, rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    edits = [
        lambda g: g.replace("(pipe X", "(pipe Z", 1),
        lambda g: g.replace("(ridge", "(not_a_head", 1),
        lambda g: g[:-1] if g.endswith(")") else g + ")",
        lambda g: '(sym affine "col(0)")',
        _move_head_to_front,
    ]
    return rng.choice(edits)(genome)


def iter_sampled(n: int, seed: int = 0, invalid: int = 0) -> Iterable[tuple[str, bool]]:
    rng = random.Random(seed)
    for _ in range(n):
        yield sample_genome(rng), True
    for _ in range(invalid):
        yield corrupt_genome(sample_genome(rng), rng), False


def estimate_complexity(genome: str) -> int:
    if genome.startswith("(sym "):
        return max(1, len(re.findall(r"[A-Za-z_][A-Za-z0-9_]*|[-+*/^]", genome)))
    return genome.count("(") - 1


def _format_stage(
    op: str,
    params: dict[str, list[object]],
    rng: random.Random,
    *,
    allow_append: bool,
) -> str:
    parts = [op]
    for key, values in params.items():
        value = rng.choice(values)
        if isinstance(value, str):
            if key == "kernel" and value == "rbf":
                continue
            value_s = value if re.match(r"^[A-Za-z0-9_+*.-]+$", value) else f'"{value}"'
        elif isinstance(value, bool):
            value_s = "true" if value else "false"
        else:
            value_s = str(value)
        parts.append(f"{key}={value_s}")
    if allow_append and rng.random() < 0.25:
        parts.append("append=true")
    return "(" + " ".join(parts) + ")"


def _move_head_to_front(genome: str) -> str:
    match = re.search(r" \((ridge|elasticnet|histgbm|histgbm2|gpr|ksmooth)[^)]*\)", genome)
    if not match or not genome.startswith("(pipe X"):
        return genome + " trailing"
    head = match.group(0)
    without = genome[: match.start()] + genome[match.end() :]
    return without.replace("(pipe X", "(pipe X" + head, 1)
