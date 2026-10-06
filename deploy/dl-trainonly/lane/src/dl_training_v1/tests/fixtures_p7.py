"""Shared deterministic synthetic fixtures for every P7 test (PLAN §5 P7 'synthetic fixtures'; §6.1 units and groups,
§6.2 partitions, §6.3 labels). FIXTURE_VERSION 'jope.dl-p7-fixture/v1'.

Everything is a pure function of the constants below. The only randomness is h(tag), the first 8 bytes of
sha256('jope.dl-p7-fixture/v1:' + tag), and u(tag) = h(tag)/2**64. No global, numpy or torch RNG is touched, nothing
is read or written on disk, and no set or hash-seed order is ever iterated, so every call returns fresh objects with
identical content in every process. dltrain modules are imported lazily inside the functions that need them, so a
test that needs only the corpus never imports the model or the trainer.

Layout: 13 families fam-00..fam-12 (fam-00..07 train_fit, fam-08 train_nonpool, fam-09/10 dev, fam-11 heldout,
fam-12 sealed_excluded); train families' fold-3 contexts are the split diagnostic ('diag'). Each context holds
12 + ctx_idx % 9 toy `pipe` programs (three shared seeds first) with fake two-fold inner-CV metrics and outcomes,
and seven special train_fit contexts exercise the §6.3 label edge cases (see SPECIAL_NAMES).
"""
from __future__ import annotations

import copy
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np

FIXTURE_VERSION = "jope.dl-p7-fixture/v1"

VOCAB = (("<PAD>", "<BOS>", "<EOS>", "(", ")", "pipe", "X", "std", "log1p", "winsor_clip", "pca", "ridge", "lasso",
          "alpha=", ".") + tuple(str(d) for d in range(10)) + tuple(f"<UNUSED_{i}>" for i in range(7)))
V = len(VOCAB)
TOKEN_ID = {tok: i for i, tok in enumerate(VOCAB)}
PAD, BOS, EOS = TOKEN_ID["<PAD>"], TOKEN_ID["<BOS>"], TOKEN_ID["<EOS>"]
LPAREN, RPAREN, PIPE, X_TOK = TOKEN_ID["("], TOKEN_ID[")"], TOKEN_ID["pipe"], TOKEN_ID["X"]
ALPHA, DOT = TOKEN_ID["alpha="], TOKEN_ID["."]
OPS = ("std", "log1p", "winsor_clip", "pca")
HEADS = ("ridge", "lasso")
OP_IDS = frozenset(TOKEN_ID[o] for o in OPS)
HEAD_IDS = frozenset(TOKEN_ID[h] for h in HEADS)
DIGIT_IDS = frozenset(TOKEN_ID[str(d)] for d in range(10))
MAX_STAGES = 3
MAX_SEQ = 24
ENC_DIM = 8
Y_TRAIN_ROWS = 24

N_DATASETS = (1, 2, 3, 1, 2, 3, 1, 2, 1, 2, 1, 2, 1)
FAMILY_IDS = tuple(f"fam-{f:02d}" for f in range(len(N_DATASETS)))
ASSIGNMENT = {**{f"fam-{f:02d}": "train_fit" for f in range(8)}, "fam-08": "train_nonpool", "fam-09": "dev",
              "fam-10": "dev", "fam-11": "heldout", "fam-12": "sealed_excluded"}
TRAIN_ASSIGNMENTS = ("train_fit", "train_nonpool")

SEED_DSLS = ("(pipe X (std) (ridge alpha=1.00))", "(pipe X (log1p) (lasso alpha=0.10))",
             "(pipe X (pca) (std) (ridge alpha=0.50))")
SEED_SPECS = ((("std",), "ridge", "100"), (("log1p",), "lasso", "010"), (("pca", "std"), "ridge", "050"))
SPECIAL_NAMES = ("degenerate", "few_skeletons", "no_detail", "exact_fit", "nonfinite", "ties", "detailless")
SPECIAL_MIN_PROGRAMS = 14

COMPLETED = "completed"
FAILED = "failed:compile_error"
TIMEOUT = "timeout@90000ms"
ABANDONED = "abandoned:runtime_error"
MISSING = "missing"
POISON_FAILED = "failed:poison"
CONTEXT_SCHEMA = "fixture/v1"
TIE_A = 0.75
TIE_DELTA = 2e-7

RUN_SEED = hashlib.sha256(b"jope.dl-p7-fixture/v1 run_seed").digest()


def h(tag: str) -> int:
    """Fixture randomness: the first 8 bytes of sha256('jope.dl-p7-fixture/v1:' + tag), big-endian."""
    return int.from_bytes(hashlib.sha256((FIXTURE_VERSION + ":" + tag).encode()).digest()[:8], "big")


def u(tag: str) -> float:
    return h(tag) / 2**64


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------------------------- vocabulary and grammar

def vocab() -> tuple[str, ...]:
    return VOCAB


def _allowed(state: str, k: int) -> frozenset[int]:
    if state == "slot":                      # '(' at the stage/head slot with k completed stages
        return (OP_IDS if k < MAX_STAGES else frozenset()) | (HEAD_IDS if k >= 1 else frozenset())
    return {"bos": frozenset({LPAREN}), "open": frozenset({PIPE}), "pipe": frozenset({X_TOK}),
            "x": frozenset({LPAREN}), "op": frozenset({RPAREN}), "stage_close": frozenset({LPAREN}),
            "head": frozenset({ALPHA}), "alpha": DIGIT_IDS, "d1": frozenset({DOT}), "dot": DIGIT_IDS,
            "d2": DIGIT_IDS, "d3": frozenset({RPAREN}), "head_close": frozenset({RPAREN}),
            "pipe_close": frozenset({EOS}), "eos": frozenset()}[state]


def _advance(state: str, k: int, tok: int) -> tuple[str, int]:
    if state == "slot":
        return ("op", k) if tok in OP_IDS else ("head", k)
    if state == "op":
        return "stage_close", k + 1
    nxt = {"bos": "open", "open": "pipe", "pipe": "x", "x": "slot", "stage_close": "slot", "head": "alpha",
           "alpha": "d1", "d1": "dot", "dot": "d2", "d2": "d3", "d3": "head_close", "head_close": "pipe_close",
           "pipe_close": "eos"}
    return nxt[state], k


def legal_mask(tokens: Sequence[int]) -> np.ndarray:
    """Grammar mask (len-1, V): row i is the set of tokens allowed after tokens[:i+1]. ValueError unless tokens is
    one complete program BOS ... EOS of the toy grammar."""
    toks = [int(t) for t in tokens]
    if len(toks) < 2 or toks[0] != BOS:
        raise ValueError("a program starts with <BOS> and has at least two tokens")
    mask = np.zeros((len(toks) - 1, V), dtype=np.bool_)
    state, k = "bos", 0
    for i in range(len(toks) - 1):
        allowed = _allowed(state, k)
        mask[i, sorted(allowed)] = True
        if toks[i + 1] not in allowed:
            raise ValueError(f"token {toks[i + 1]} is illegal at position {i + 1}")
        state, k = _advance(state, k, toks[i + 1])
    if state != "eos":
        raise ValueError("the program does not end with <EOS>")
    return mask


def program(stages: Sequence[str], head: str, digits: str | int) -> dict:
    """{dsl, tokens, program_sha256, skeleton_sha256} of '(pipe X (op)... (head alpha=d.dd))'; the skeleton is the dsl
    with the alpha literal replaced by '#'."""
    stages = tuple(stages)
    if not 1 <= len(stages) <= MAX_STAGES or any(op not in OPS for op in stages) or head not in HEADS:
        raise ValueError(f"bad program spec {stages!r} {head!r}")
    if isinstance(digits, int):
        digits = f"{digits:03d}"
    if len(digits) != 3 or any(c not in "0123456789" for c in digits):
        raise ValueError(f"digits must be three decimal digits, got {digits!r}")
    stage_text = "".join(f"({op}) " for op in stages)
    dsl = f"(pipe X {stage_text}({head} alpha={digits[0]}.{digits[1:]}))"
    skeleton = f"(pipe X {stage_text}({head} alpha=#))"
    tokens = ([BOS, LPAREN, PIPE, X_TOK] + [t for op in stages for t in (LPAREN, TOKEN_ID[op], RPAREN)]
              + [LPAREN, TOKEN_ID[head], ALPHA, TOKEN_ID[digits[0]], DOT, TOKEN_ID[digits[1]], TOKEN_ID[digits[2]],
                 RPAREN, RPAREN, EOS])
    return {"dsl": dsl, "tokens": tokens, "program_sha256": _sha(dsl), "skeleton_sha256": _sha(skeleton)}


# ------------------------------------------------------------------------------------------------ units and labels

def assignment() -> dict[str, str]:
    return dict(ASSIGNMENT)


def _n_contexts(f: int, j: int) -> int:
    return 1 + ((3 * f + 2 * j) % 5)


def _n_prog(ctx_idx: int) -> int:
    return 12 + ctx_idx % 9


def _split_role(assigned: str, fold: int) -> str:
    if assigned in TRAIN_ASSIGNMENTS:
        return "diag" if fold == 3 else "train_fit"
    return assigned                                   # 'dev', 'heldout' or 'sealed_excluded'


def _base_contexts() -> list[dict]:
    rows = []
    for f, fam in enumerate(FAMILY_IDS):
        for j in range(N_DATASETS[f]):
            ds = _sha(f"fixture-dataset:{fam}:{j}")
            for fold in range(_n_contexts(f, j)):
                rows.append({"context_sha256": _sha(f"fixture-context:{ds}:{fold}"), "dataset_hash": ds,
                             "family_id": fam, "split_role": _split_role(ASSIGNMENT[fam], fold), "fold": fold,
                             "frozen_fold0": fold == 0})
    rows.sort(key=lambda r: r["context_sha256"])
    return rows


def _special_indices(base: Sequence[Mapping]) -> dict[str, int]:
    """name -> ctx_idx: the first train_fit contexts by ctx_idx with >= 14 programs, in SPECIAL_NAMES order."""
    picked = [i for i, row in enumerate(base)
              if row["split_role"] == "train_fit" and _n_prog(i) >= SPECIAL_MIN_PROGRAMS][:len(SPECIAL_NAMES)]
    if len(picked) != len(SPECIAL_NAMES):
        raise RuntimeError("fixture has too few train_fit contexts for the specials")
    return dict(zip(SPECIAL_NAMES, picked))


def _y_train(ctx_idx: int, special: str | None) -> list[float]:
    if special == "degenerate":
        return [2.5] * Y_TRAIN_ROWS
    return [((ctx_idx % 7) - 3) + (0.5 + 0.25 * (ctx_idx % 5)) * math.sin(0.37 * i + 0.11 * ctx_idx)
            for i in range(Y_TRAIN_ROWS)]


def contexts() -> list[dict]:
    """ContextRow dicts sorted by context_sha256 (ctx_idx = position)."""
    base = _base_contexts()
    special_of = {i: name for name, i in _special_indices(base).items()}
    for i, row in enumerate(base):
        row["y_train"] = _y_train(i, special_of.get(i))
    return base


def special_contexts() -> dict[str, str]:
    base = _base_contexts()
    return {name: base[i]["context_sha256"] for name, i in _special_indices(base).items()}


def _outcome(j: int, n_prog: int) -> str:
    if j % 10 == 7:
        return FAILED
    if j % 10 == 8:
        return TIMEOUT
    if j == 9 and n_prog >= 14:
        return ABANDONED
    if j == 10 and n_prog >= 16:
        return MISSING
    return COMPLETED


def _generated_program(prefix: str, used: set[str], *, seed_skeletons: bool = False,
                       avoid_skeleton: str | None = None) -> dict:
    """A program from tags '<prefix>:k|op<i>|head|d', re-drawn with prefix + ':r' until its sha is new in the context
    (and its skeleton differs from avoid_skeleton). seed_skeletons draws only the three seed skeletons."""
    while True:
        if seed_skeletons:
            stages, head, _ = SEED_SPECS[h(f"{prefix}:sk") % len(SEED_SPECS)]
        else:
            stages = tuple(OPS[h(f"{prefix}:op{i}") % len(OPS)] for i in range(1 + h(f"{prefix}:k") % MAX_STAGES))
            head = HEADS[h(f"{prefix}:head") % len(HEADS)]
        prog = program(stages, head, h(f"{prefix}:d") % 1000)
        if prog["program_sha256"] not in used and prog["skeleton_sha256"] != avoid_skeleton:
            return prog
        prefix += ":r"


def _set_completed_metrics(row: dict, a: float, base: Sequence[float]) -> None:
    rmse = [base[0] * math.exp(-a), base[1] * math.exp(-a + 0.02)]
    ratios = [rmse[0] / base[0], rmse[1] / base[1]]
    logs = [math.log(ratios[0]), math.log(ratios[1])]
    row["rmse_j"] = rmse
    row["cv_mean_log_ratio"] = (logs[0] + logs[1]) / 2
    row["cv_mean_ratio"] = (ratios[0] + ratios[1]) / 2
    row["cv_log_ratio_sd"] = abs(logs[0] - logs[1]) / 2     # population sd of two values


def _record(ctx: Mapping, prog: Mapping, outcome: str, j: int, base: Sequence[float] | None, shared: bool) -> dict:
    return {"context_sha256": ctx["context_sha256"], "context_schema": CONTEXT_SCHEMA,
            "dataset_hash": ctx["dataset_hash"], "family_id": ctx["family_id"], "split_role": ctx["split_role"],
            "program_sha256": prog["program_sha256"], "skeleton_sha256": prog["skeleton_sha256"],
            "tokens": list(prog["tokens"]), "length": len(prog["tokens"]), "outcome": outcome, "rmse_j": None,
            "baseline_rmse_j": None if base is None else list(base), "cv_mean_log_ratio": None,
            "cv_log_ratio_sd": None, "cv_mean_ratio": None, "wall_us": None,
            "backend": "remote" if j % 6 == 5 else "local", "shared_seed": shared}


def _context_records(ctx_idx: int, ctx: Mapping, special: str | None) -> list[dict]:
    from dltrain import labels
    sha = ctx["context_sha256"]
    n_prog = _n_prog(ctx_idx)
    s_c = float(labels.robust_scale(ctx["y_train"]))
    base = [s_c * 1.0, s_c * 1.2]
    used: set[str] = set()
    skeleton_of: dict[int, str] = {}
    rows = []
    for j in range(n_prog):
        outcome = _outcome(j, n_prog)
        tag = f"{sha}:{j}"
        if j < len(SEED_SPECS):
            prog = program(*SEED_SPECS[j])
        else:
            prog = _generated_program(tag, used, seed_skeletons=special == "few_skeletons" and outcome == COMPLETED,
                                      avoid_skeleton=skeleton_of.get(3) if special == "ties" and j == 4 else None)
        used.add(prog["program_sha256"])
        skeleton_of[j] = prog["skeleton_sha256"]
        unlabelled = outcome in (ABANDONED, MISSING)
        row = _record(ctx, prog, outcome, j, None if unlabelled else base, j < len(SEED_SPECS))
        if outcome == COMPLETED:
            if special == "degenerate":
                row["rmse_j"] = [0.0, 0.0]
            else:
                if special == "ties":
                    a = TIE_A if j == 3 else TIE_A + TIE_DELTA if j == 4 else 0.7 - 1.5 * u(f"{tag}:a")
                else:
                    a = 0.8 - 1.6 * u(f"{tag}:a")
                _set_completed_metrics(row, a, base)
            row["wall_us"] = int(1e5 + 4e7 * u(f"{tag}:w"))
        elif outcome == FAILED:
            row["wall_us"] = int(5e4 + 1e6 * u(f"{tag}:w"))
        elif outcome == TIMEOUT:
            row["wall_us"] = 90_000_000
        rows.append(row)
    if special == "no_detail":
        for row in rows:
            row["rmse_j"] = None
            row["baseline_rmse_j"] = None
    elif special == "exact_fit":
        rows[3].update(rmse_j=[0.0, 0.0], cv_mean_ratio=0.0, cv_mean_log_ratio=None, cv_log_ratio_sd=None)
    elif special == "nonfinite":
        rows[3].update(rmse_j=[math.nan, base[1]], cv_mean_ratio=None, cv_mean_log_ratio=None, cv_log_ratio_sd=None)
        rows[4].update(rmse_j=[math.inf, base[1]], cv_mean_ratio=None, cv_mean_log_ratio=None, cv_log_ratio_sd=None)
    elif special == "detailless":
        eps = float(labels.epsilon(ctx["y_train"], s_c))
        cap = float(labels.q_cap(base, eps))
        rows[3].update(rmse_j=None, cv_mean_log_ratio=-(cap + 1.0), cv_mean_ratio=math.exp(-(cap + 1.0)),
                       cv_log_ratio_sd=None)
        rows[4].update(rmse_j=None, cv_mean_log_ratio=-math.inf, cv_mean_ratio=0.0, cv_log_ratio_sd=None)
    return rows


def records() -> list[dict]:
    """TrainRecord dicts for every context, context by context in ctx_idx order, programs in j order."""
    ctxs = contexts()
    special_of = {i: name for name, i in _special_indices(ctxs).items()}
    out: list[dict] = []
    for i, ctx in enumerate(ctxs):
        out.extend(_context_records(i, ctx, special_of.get(i)))
    return out


def encodings() -> dict[str, np.ndarray]:
    """float32 [ENC_DIM] per context: z[k] = 2·u('enc:<ctx>:<k>') - 1."""
    return {row["context_sha256"]: np.array([2.0 * u(f"enc:{row['context_sha256']}:{k}") - 1.0 for k in range(ENC_DIM)],
                                            dtype=np.float32)
            for row in _base_contexts()}


# --------------------------------------------------------------------------------------- corpus, configs and data

def corpus():
    from dltrain import sampler
    return sampler.Corpus.build(contexts(), records(), encodings(), assignment())


def sampler_config(**kw: Any):
    from dltrain import sampler
    return sampler.SamplerConfig(**{"b_ctx": 4, "group_size": 8, **kw})


def model_config(**kw: Any):
    from dltrain import model
    return model.PolicyConfig(**{"vocab_size": V, "ctx_dim": ENC_DIM, "d_model": 32, "n_layers": 2, "n_heads": 2,
                                 "d_ff": 64, "max_seq": MAX_SEQ, "dropout": 0.1, "context_free_head": False, **kw})


def train_config(**kw: Any):
    from dltrain import train
    return train.TrainConfig(**{"lr": 3e-3, "lr_min": 3e-4, "warmup": 10, "betas": (0.9, 0.999), "eps": 1e-8,
                                "weight_decay": 0.01, "clip_norm": 1.0, "commit_every_updates": 20,
                                "commit_every_seconds": None, "eval_every_epochs": None, "autocast_dtype": None,
                                "num_threads": 4, "ctx_drop_prob": 0.25, "ctx_noise_std": 0.05,
                                "live_interval_s": 3.0, "receipt_interval_s": 2.0, **kw})


def plan_kwargs(**kw: Any) -> dict:
    """Keyword arguments for train.make_run_plan; a keyword replaces the default of the same name."""
    out = {"run_id": "p7-fixture", "run_seed": RUN_SEED, "arm": "fixture", "objective": "successor-v1",
           "max_updates": 80, "deadline_utc_us": None, "storage_cap_bytes": 1 << 30, "refs": None, **kw}
    for key, make in (("model", model_config), ("sampler", sampler_config), ("train", train_config)):
        if key not in out:
            out[key] = make()
    return out


def train_data(**kw: Any):
    """TrainData(corpus(), fit(corpus, sampler_config(**kw)), sampler_config(**kw), legal_mask, V, ENC_DIM)."""
    from dltrain import sampler
    cfg = sampler_config(**kw)
    corp = corpus()
    return sampler.TrainData(corpus=corp, fit=sampler.fit(corp, cfg), config=cfg, legal_mask_fn=legal_mask,
                             vocab_size=V, ctx_dim=ENC_DIM)


# ------------------------------------------------------------------------------------------------------- poison

def poison(contexts: Sequence[Mapping], records: Sequence[Mapping],
           encodings: Mapping[str, Any]) -> tuple[list[dict], list[dict], dict[str, np.ndarray]]:
    """Perturb every unit whose split_role is not 'train_fit' (dev, diag, heldout, sealed_excluded): y_train·3 + 7,
    every rmse_j·0.37, outcomes rotated (completed <-> failed), two extra completed records with new programs, and
    negated encodings. train_fit units are untouched; the inputs are never mutated (PLAN §5 P7 test 7)."""
    ctxs = [copy.deepcopy(dict(c)) for c in contexts]
    targets = {c["context_sha256"] for c in ctxs if c["split_role"] != "train_fit"}
    for c in ctxs:
        if c["context_sha256"] in targets:
            c["y_train"] = [3.0 * y + 7.0 for y in c["y_train"]]
    recs: list[dict] = []
    used: dict[str, set[str]] = {}
    base_of: dict[str, list[float]] = {}
    for r in records:
        r = copy.deepcopy(dict(r))
        sha = r["context_sha256"]
        used.setdefault(sha, set()).add(r["program_sha256"])
        if sha in targets:
            if r.get("baseline_rmse_j") is not None:
                base_of.setdefault(sha, list(r["baseline_rmse_j"]))
            if r.get("rmse_j") is not None:
                r["rmse_j"] = [None if x is None else x * 0.37 for x in r["rmse_j"]]
            if r["outcome"] == COMPLETED:
                r["outcome"] = POISON_FAILED
            elif r["outcome"].startswith("failed:"):
                r["outcome"] = COMPLETED
        recs.append(r)
    for c in ctxs:
        sha = c["context_sha256"]
        if sha not in targets:
            continue
        base = base_of.get(sha)
        for i in range(2):
            tag = f"poison:{sha}:{i}"
            prog = _generated_program(tag, used.setdefault(sha, set()))
            used[sha].add(prog["program_sha256"])
            row = _record(c, prog, COMPLETED, 0, base, False)
            if base is not None and min(base) > 0:
                _set_completed_metrics(row, 0.8 - 1.6 * u(f"{tag}:a"), base)
            row["wall_us"] = int(1e5 + 4e7 * u(f"{tag}:w"))
            recs.append(row)
    enc = {k: np.array(-np.asarray(v, dtype=np.float32) if k in targets else v, dtype=np.float32, copy=True)
           for k, v in encodings.items()}
    return ctxs, recs, enc
