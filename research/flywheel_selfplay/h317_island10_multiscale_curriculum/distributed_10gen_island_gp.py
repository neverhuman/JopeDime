"""H317 GP track: cross-family bank, context codes, Gen-0 seeding (H0.4 Seed A-D) and 10-generation evolution
(H0.5 Evo A panmictic vs Evo B 4-island MAP-Elites + epsilon-lexicase), plus the H257 replica loop.

Everything here selects programs from SEARCH results only and runs inside `selection_scope` (AUDIT requests raise).

Bank invariant (stats red team, hard): for a target context c every bank candidate excludes all champion / winner
records whose family_id == family(c) or whose dataset_id == dataset(c) (also drops c's own champion, which was found on
all rows including AUDIT). `assert_bank_disjoint` runs at every seeding entry point and in every evolution generation;
`BANK_COUNTERS["same_family_bank_hits"]` must stay 0.

Ported logic (copied/adapted, cited): `propose_flywheel_candidates` mix (champion retrieval, EditZero kinds, synthesis)
from JopeDimeMicro rust_evaluator.py; H257 `propose_gen0_from_11view_batch` roles (view-0 anchor + landmark affinity,
UCB diversity explorers) and `screen_child_proposals_with_causal_trajectory` pick rule (k-2 exploit + 2 UCB) from
origin/main research/flywheel_selfplay/h257_phase2_subsample_gp_curriculum/batched_subsample_gp_evolution.py.
"""

from __future__ import annotations

import json
import math
import random
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import common_h317 as C  # noqa: E402

import numpy as np  # noqa: E402

import gp_grammar_h317 as G  # noqa: E402
import screener_h317 as S  # noqa: E402
from gp_engine_h317 import ContextEnvs, EvalEngine, SearchResult, selection_scope  # noqa: E402

EVO_GENS = 10
PROPOSALS_PER_GEN = 48
LIVE_PER_GEN = 6
MIGRATION_GENS = (3, 6, 9)
LINE_SEARCH_GENS = (8, 9, 10)
AUDIT_GENS = (0, 3, 6, 10)
SHORTLIST_BANK_NN = 96
SHORTLIST_GRAMMAR_PER_ISLAND = 16
LEXICASE_CASES = ("lr0", "lr1", "tail_lr", "complexity")


class BankLeakError(AssertionError):
    pass


BANK_COUNTERS = {"same_family_bank_hits": 0, "excluded_sha_collisions_filtered": 0}
_BANK_LOCK = threading.Lock()


# ----------------------------------------------------------------------------------------------------------------------
# Bank
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class BankEntry:
    dsl: str
    sha: str
    island: int
    cids: Tuple[str, ...]
    families: Tuple[str, ...]
    dataset_ids: Tuple[str, ...]
    kinds: Tuple[str, ...]


def build_bank(rows: Sequence[dict]) -> Tuple[List[BankEntry], dict]:
    """TRAIN champions (champions.json, dl_split DL-Train) + TRAIN winners (winners.json joined to TRAIN rows)."""
    fam_of = {r["context_sha256"]: r["family_id"] for r in rows}
    ds_of = {r["context_sha256"]: r["dataset_id"] for r in rows}
    recs: Dict[str, dict] = {}
    info = {
        "champions_in": 0,
        "winners_in": 0,
        "winners_non_train_dropped": 0,
        "noncanonical_dropped": 0,
    }

    def add(dsl: str, cid: str, kind: str):
        if cid not in fam_of:
            info["winners_non_train_dropped"] += kind == "winner"
            return
        try:
            if G.canonical(dsl) != dsl:
                info["noncanonical_dropped"] += 1
                return
        except G.GrammarError:
            info["noncanonical_dropped"] += 1
            return
        sha = G.sha256_text(dsl)
        r = recs.setdefault(sha, {"dsl": dsl, "cids": [], "kinds": []})
        r["cids"].append(cid)
        r["kinds"].append(kind)

    for r in json.loads((C.RES01 / "champions.json").read_text()):
        if r.get("dl_split") == "DL-Train":
            info["champions_in"] += 1
            add(r["dsl"], r["context_sha256"], "champion")
    for r in json.loads((C.RES01 / "winners.json").read_text())[
        "all_90_qualified_novel_winner_records"
    ]:
        info["winners_in"] += 1
        add(r["dsl"], r["context_sha256"], "winner")
    bank = []
    for sha in sorted(recs):
        r = recs[sha]
        cids = tuple(r["cids"])
        bank.append(
            BankEntry(
                r["dsl"],
                sha,
                G.island_of(r["dsl"]),
                cids,
                tuple(fam_of[c] for c in cids),
                tuple(ds_of[c] for c in cids),
                tuple(r["kinds"]),
            )
        )
    info["entries"] = len(bank)
    return bank, info


class TargetBank:
    """Eligible bank for one target context (family- and dataset-disjoint)."""

    def __init__(
        self,
        bank: Sequence[BankEntry],
        cid: str,
        fam_of: Dict[str, str],
        ds_of: Dict[str, str],
    ):
        self.cid = cid
        fam, ds = fam_of[cid], ds_of[cid]
        same_fam = [b for b in bank if fam in b.families]
        same_ds_other_fam = [
            b for b in bank if ds in b.dataset_ids and fam not in b.families
        ]
        excl = {b.sha for b in same_fam} | {b.sha for b in same_ds_other_fam}
        self.eligible = [b for b in bank if b.sha not in excl]
        self.excluded_shas: Set[str] = excl
        self.excluded_cids: Set[str] = {c for c, f in fam_of.items() if f == fam} | {
            c for c, d in ds_of.items() if d == ds
        }
        self.n_excluded_same_family = len(same_fam)
        self.n_excluded_same_dataset_other_family = len(same_ds_other_fam)
        self.by_sha = {b.sha: b for b in self.eligible}

    def exclusion_record(self) -> dict:
        return {
            "n_excluded": len(self.excluded_shas),
            "same_family": self.n_excluded_same_family,
            "same_dataset_other_family": self.n_excluded_same_dataset_other_family,
            "eligible": len(self.eligible),
        }

    def filter(self, cands: Iterable[G.Candidate]) -> List[G.Candidate]:
        out = []
        for c in cands:
            if c.sha in self.excluded_shas:
                with _BANK_LOCK:
                    BANK_COUNTERS["excluded_sha_collisions_filtered"] += 1
                continue
            out.append(c)
        return out


def _lineage(c: G.Candidate) -> Tuple[str, ...]:
    for k, v in c.meta:
        if k == "bank_cids":
            return tuple(x for x in v.split("|") if x)
    return ()


def assert_bank_disjoint(
    cid: str, candidates: Iterable[G.Candidate], tb: TargetBank
) -> None:
    """Hard invariant: no candidate is (or descends from) a same-family / same-dataset bank record of `cid`."""
    if tb.cid != cid:
        raise BankLeakError(f"TargetBank for {tb.cid} used for {cid}")
    for c in candidates:
        if c.sha in tb.excluded_shas or any(x in tb.excluded_cids for x in _lineage(c)):
            with _BANK_LOCK:
                BANK_COUNTERS["same_family_bank_hits"] += 1
            raise BankLeakError(
                f"{cid}: candidate {c.sha[:12]} ({c.operator}) touches an excluded bank record"
            )


def bank_candidate(b: BankEntry, operator: str = "bank_nn") -> G.Candidate:
    return G.Candidate(
        b.dsl, b.sha, operator, (b.sha,), b.island, (("bank_cids", "|".join(b.cids)),)
    )


def merge_lineage(*cands: G.Candidate) -> str:
    seen: List[str] = []
    for c in cands:
        for x in _lineage(c):
            if x not in seen:
                seen.append(x)
    return "|".join(seen[:64])


# ----------------------------------------------------------------------------------------------------------------------
# Context codes (bank nearest neighbours, BANK_NN1)
# ----------------------------------------------------------------------------------------------------------------------
LM12 = slice(
    320 + 629, 320 + 629 + 12
)  # lm48_z[:12] inside pack_view_feats [tokens 320 | dope629 | lm48_z 48]
DOPE = slice(320, 320 + 629)


class ContextCoder:
    """codes(cid, role) -> {"lm12": [12], "dope": [D]}; role "target" must use SEARCH rows only."""

    name = "abstract"

    def target(self, cid: str) -> dict:
        raise NotImplementedError

    def bank(self, cid: str) -> dict:
        raise NotImplementedError


class NeuralContextCoder(ContextCoder):
    """Target: view 0 of the SEARCH-envelope views (screener encoding). Bank contexts: exact float32 view 0 of the
    neural track's full-table encodings, from the merged file $H317_DATA/neural/views_c32_train1219.npz when present
    (else the per-context $H317_DATA/views/<cid>.npz). Bank contexts are never the context being audited; a target's
    own family (hence its own full-table encoding) is excluded by TargetBank before any distance is computed."""

    name = "dope_view0"

    def __init__(
        self,
        search_views: Callable[[str], np.ndarray],
        bank_view_dir: Path = C.DATA_ROOT / "views",
        merged_path: Optional[Path] = C.DATA_ROOT
        / "neural"
        / "views_c32_train1219.npz",
    ):
        self.search_views, self.bank_view_dir = search_views, Path(bank_view_dir)
        self.merged_path = Path(merged_path) if merged_path else None
        self._merged: Optional[Dict[str, np.ndarray]] = None
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()

    def _merged_view0(self) -> Optional[Dict[str, np.ndarray]]:
        if (
            self._merged is None
            and self.merged_path is not None
            and self.merged_path.exists()
        ):
            import importlib

            V = importlib.import_module("crop_shuffle_synthetic_views")
            m = V.load_merged_views(self.merged_path)
            self._merged = {
                str(c): np.asarray(v, np.float64)
                for c, v in zip(m["cids"], m["view0_f32"])
            }
        return self._merged

    def target(self, cid):
        v = np.asarray(self.search_views(cid), np.float64)[0]
        return {"lm12": v[LM12], "dope": v[DOPE]}

    def bank(self, cid):
        with self._lock:
            hit = self._cache.get(cid)
        if hit is None:
            import importlib

            with self._lock:
                merged = self._merged_view0()
            if merged is not None:
                v = merged[cid]
            else:
                V = importlib.import_module("crop_shuffle_synthetic_views")
                v = np.asarray(
                    V.context_view_feats(
                        cid, "full", views=[0], view_dir=self.bank_view_dir
                    )[0],
                    np.float64,
                )
            hit = {"lm12": v[LM12], "dope": v[DOPE]}
            with self._lock:
                self._cache[cid] = hit
        return hit


def numpy_landmarks(x: np.ndarray, y: np.ndarray, folds: np.ndarray) -> np.ndarray:
    """12 cheap meta-features (tests / timing fallback only). Inputs must already be SEARCH rows for a target."""
    x = np.nan_to_num(np.asarray(x, np.float64))
    y = np.asarray(y, np.float64)
    n, p = x.shape
    xs = (x - x.mean(0)) / np.maximum(x.std(0), 1e-9)
    ys = (y - y.mean()) / max(y.std(), 1e-9)
    corr = np.abs(xs.T @ ys) / n
    sub = xs[:, : min(p, 40)]
    with np.errstate(invalid="ignore", divide="ignore"):
        cc = (
            np.nan_to_num(np.abs(np.corrcoef(sub.T)))
            if sub.shape[1] > 1
            else np.zeros((1, 1))
        )
    off = cc[~np.eye(len(cc), dtype=bool)] if len(cc) > 1 else np.zeros(1)

    def r2(feats):
        a, b = feats[folds == 0], feats[folds == 1]
        ya, yb = ys[folds == 0], ys[folds == 1]
        out = []
        for tr, ytr, te, yte in ((a, ya, b, yb), (b, yb, a, ya)):
            A = np.c_[tr, np.ones(len(tr))]
            w = np.linalg.solve(A.T @ A + 1.0 * np.eye(A.shape[1]), A.T @ ytr)
            pr = np.c_[te, np.ones(len(te))] @ w
            out.append(1.0 - np.mean((pr - yte) ** 2) / max(np.var(yte), 1e-9))
        return float(np.mean(out))

    lin = r2(xs[:, :60])
    quad = r2(np.c_[xs[:, :30], xs[:, :30] ** 2])
    rk = r2(((np.argsort(np.argsort(x[:, :60], 0), 0) / max(n - 1, 1)) - 0.5) * 3.4)
    skew = float(np.mean(ys**3))
    kurt = float(np.mean(ys**4))
    return np.nan_to_num(
        np.array(
            [
                math.log(n),
                math.log(p),
                skew,
                math.log(max(kurt, 1e-3)),
                len(np.unique(y)) / n,
                float(corr.mean()),
                float(corr.max()),
                lin,
                quad - lin,
                rk - lin,
                float(off.mean()),
                float(np.mean([len(np.unique(x[:, j])) <= 2 for j in range(p)])),
            ]
        )
    )


class NumpyContextCoder(ContextCoder):
    """Fallback for tests and timing runs (no neural encodings yet)."""

    name = "numpy_landmarks12"

    def __init__(self):
        self._cache: Dict[Tuple[str, str], dict] = {}
        self._lock = threading.Lock()

    def _code(self, cid: str, role: str) -> dict:
        with self._lock:
            hit = self._cache.get((cid, role))
        if hit is None:
            env = C.read_jtt1(C.table_path(cid))
            if role == "target":
                env = C.search_and_audit_envelopes(cid, env)[0]
            v = numpy_landmarks(env.mat[:, :-1], env.y, env.folds)
            hit = {"lm12": v, "dope": v}
            with self._lock:
                self._cache[(cid, role)] = hit
        return hit

    def target(self, cid):
        return self._code(cid, "target")

    def bank(self, cid):
        return self._code(cid, "bank")


def _zdist(t: np.ndarray, M: np.ndarray) -> np.ndarray:
    mu, sd = M.mean(0), np.maximum(M.std(0), 1e-9)
    return np.sqrt(np.sum(((M - mu) / sd - (t - mu) / sd) ** 2, axis=1))


def bank_nn(
    tb: TargetBank, coder: ContextCoder, k: int, key: str = "lm12"
) -> List[Tuple[float, BankEntry]]:
    """Eligible entries by distance of their nearest eligible source context (z-scored over eligible contexts only)."""
    ctx_ids = sorted(
        {c for b in tb.eligible for c in b.cids if c not in tb.excluded_cids}
    )
    M = np.stack([coder.bank(c)[key] for c in ctx_ids])
    d = dict(zip(ctx_ids, _zdist(np.asarray(coder.target(tb.cid)[key], np.float64), M)))
    scored = []
    for b in tb.eligible:
        ds = [d[c] for c in b.cids if c in d]
        if ds:
            scored.append((float(min(ds)), b))
    scored.sort(key=lambda t: (t[0], t[1].sha))
    return scored[:k]


def bank_champions_by_dope(
    tb: TargetBank, coder: ContextCoder, k: int, operator: str
) -> List[G.Candidate]:
    """Champions of the k nearest eligible contexts by Euclidean distance of z-scored DOPE view-0 encodings (the
    screener's context encoding; target = SEARCH rows only). Distinct DSLs, nearest first."""
    champs = [b for b in tb.eligible if "champion" in b.kinds]
    owner: Dict[str, BankEntry] = {}
    for b in champs:
        for c, kd in zip(b.cids, b.kinds):
            if kd == "champion" and c not in tb.excluded_cids:
                owner.setdefault(c, b)
    ctx_ids = sorted(owner)
    if not ctx_ids:
        return []
    M = np.stack([coder.bank(c)["dope"] for c in ctx_ids])
    dist = _zdist(np.asarray(coder.target(tb.cid)["dope"], np.float64), M)
    out: List[G.Candidate] = []
    for j in np.argsort(dist, kind="stable"):
        b = owner[ctx_ids[int(j)]]
        if all(c.sha != b.sha for c in out):
            out.append(bank_candidate(b, operator))
        if len(out) >= k:
            break
    assert_bank_disjoint(tb.cid, out, tb)
    return out


def bank_nn1(tb: TargetBank, coder: ContextCoder) -> Optional[G.Candidate]:
    """BANK_NN1: champion of the nearest eligible context (1 eval, no search)."""
    out = bank_champions_by_dope(tb, coder, 1, "bank_nn1")
    return out[0] if out else None


# ----------------------------------------------------------------------------------------------------------------------
# Per-(context, search seed) GP context
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class GPContext:
    ctx: ContextEnvs
    engine: EvalEngine
    screener: S.Screener
    tb: TargetBank
    coder: ContextCoder
    seed: int
    corrector: S.ResidualCorrector = field(default_factory=S.ResidualCorrector)

    @property
    def cid(self) -> str:
        return self.ctx.cid

    def rng(self, *parts) -> random.Random:
        return random.Random(C.stream_seed("H317-gp", self.seed, self.cid, *parts))

    def evaluate(self, cands: Sequence[G.Candidate]) -> List[SearchResult]:
        assert_bank_disjoint(self.cid, cands, self.tb)
        return self.engine.search(self.ctx, cands)


def rank_key(r: SearchResult):
    return (0 if r.gate else 1, r.fitness, r.sha)


def best_by_search(results: Sequence[SearchResult]) -> Optional[SearchResult]:
    done = [r for r in results if r.status == "completed"]
    return min(done, key=rank_key) if done else None


def unique_by_sha(cands: Iterable[G.Candidate]) -> List[G.Candidate]:
    seen, out = set(), []
    for c in cands:
        if c.sha not in seen:
            seen.add(c.sha)
            out.append(c)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Shortlist: bank NN + grammar proposals (M ~ 160)
# ----------------------------------------------------------------------------------------------------------------------
def grammar_shortlist(
    g: GPContext,
    parents: Sequence[G.Candidate],
    per_island: int = SHORTLIST_GRAMMAR_PER_ISLAND,
    tag: str = "gen0",
) -> List[G.Candidate]:
    rng = g.rng("grammar", tag)
    w = g.ctx.width
    out: List[G.Candidate] = []
    seen = {c.sha for c in parents}
    pg = [p.genome() for p in parents]
    for isl in range(4):
        made, tries = 0, 0
        while made < per_island and tries < per_island * 25:
            tries += 1
            u = rng.random()
            par: List[G.Candidate] = []
            if isl == 3 and u < 0.5 and len(pg) >= 2:
                i, j = rng.sample(range(len(pg)), 2)
                gen, op, par = (
                    G.op_portfolio([pg[i], pg[j]]),
                    "nn_portfolio",
                    [parents[i], parents[j]],
                )
            elif u < 0.5 and pg:
                i = rng.randrange(len(pg))
                gen, op, par = (
                    G.head_swap(pg[i], isl, rng),
                    "nn_head_swap",
                    [parents[i]],
                )
            elif u < 0.7 and pg:
                i = rng.randrange(len(pg))
                base = (
                    G.head_swap(pg[i], isl, rng) if parents[i].island != isl else pg[i]
                )
                gen = rng.choice(
                    [
                        G.op_parameter,
                        lambda x, r: G.op_insert_stage(x, r),
                        G.op_delete_stage,
                    ]
                )(base, rng)
                op, par = "nn_mutate", [parents[i]]
            else:
                gen, op = G.op_synth(isl, rng), "grammar_synth"
            cand = G.safe_candidate(
                gen,
                f"{tag}_{op}_i{isl}",
                [p.sha for p in par],
                w,
                bank_cids=merge_lineage(*par),
            )
            if cand is None or cand.sha in seen or cand.island != isl:
                continue
            seen.add(cand.sha)
            out.append(cand)
            made += 1
    return g.tb.filter(out)


def build_shortlist(
    g: GPContext, n_bank: int = SHORTLIST_BANK_NN, with_grammar: bool = True
) -> List[G.Candidate]:
    nn = [bank_candidate(b) for _, b in bank_nn(g.tb, g.coder, n_bank)]
    nn = [c for c in nn if G.width_ok(c.dsl, g.ctx.width)]
    out = list(nn)
    if with_grammar:
        out += grammar_shortlist(g, nn[:12])
    out = unique_by_sha(out)
    assert_bank_disjoint(g.cid, out, g.tb)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# H0.4 Gen-0 seeding schemes
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class SeedRun:
    scheme: str
    order: List[G.Candidate]
    results: List[SearchResult]
    n_reveals: int = 0
    info: dict = field(default_factory=dict)

    @property
    def evals_used(self) -> int:
        return len({r.sha for r in self.results})

    def budget_matched(self, k: int = 8) -> Optional[SearchResult]:
        return best_by_search(self.results[:k])

    def best(self) -> Optional[SearchResult]:
        return best_by_search(self.results)

    def search_metrics(self) -> dict:
        passing = [r for r in self.results if r.gate]
        b, bm = self.best(), self.budget_matched()
        return {
            "evals_used": self.evals_used,
            "n_reveals": self.n_reveals,
            "best_search_utility": b.utility if b else None,
            "best_sha": b.sha if b else None,
            "bm_best_search_utility": bm.utility if bm else None,
            "bm_best_sha": bm.sha if bm else None,
            "mean_passing_search_utility": float(np.mean([r.utility for r in passing]))
            if passing
            else None,
            "n_passing": len(passing),
            "n_failed": sum(r.status != "completed" for r in self.results),
            "n_families_passing": len({G.head_family(r.dsl) for r in passing}),
            "n_islands_passing": len({G.island_of(r.dsl) for r in passing}),
            "order_shas": [c.sha for c in self.order],
        }


def _pick(
    order_scores: np.ndarray, chosen: List[int], allowed: Optional[Sequence[int]] = None
) -> Optional[int]:
    idx = np.argsort(-order_scores, kind="stable")
    allow = set(allowed) if allowed is not None else None
    for j in idx:
        j = int(j)
        if j not in chosen and (allow is None or j in allow):
            return j
    return None


def _min_dist(D: np.ndarray, j: int, chosen: Sequence[int]) -> float:
    if not chosen:
        return 0.0
    return float(np.min(np.linalg.norm(D[chosen] - D[j][None, :], axis=1)))


def seed_A(g: GPContext, n: int = 8, n_nn: int = 24) -> SeedRun:
    """H257 baseline: bank-only shortlist of landmark NNs, canonical view-0 score combined with landmark affinity
    (total = z(view0) - 0.35 z(lm distance)), slot 0 = argmax, then UCB structural-diversity explorers."""
    with selection_scope("H0.4.seedA"):
        nn = [(d, bank_candidate(b)) for d, b in bank_nn(g.tb, g.coder, n_nn)]
        nn = [(d, c) for d, c in nn if G.width_ok(c.dsl, g.ctx.width)]
        cands = [c for _, c in nn]
        assert_bank_disjoint(g.cid, cands, g.tb)
        s0 = S.zscore_rows(g.screener.score(g.cid, [0], [c.dsl for c in cands]))[0]
        dist = np.array([d for d, _ in nn])
        total = s0 - 0.35 * (dist - dist.mean()) / max(dist.std(), 1e-9)
        D = np.stack([G.ext_descriptor(c.dsl) for c in cands])
        chosen: List[int] = []
        j = _pick(total, chosen)
        if j is not None:
            chosen.append(j)
        while len(chosen) < min(n, len(cands)):
            acq = np.array(
                [
                    total[k] + 0.30 * _min_dist(D, k, chosen)
                    if k not in chosen
                    else -1e18
                    for k in range(len(cands))
                ]
            )
            chosen.append(int(np.argmax(acq)))
        order = [cands[k] for k in chosen]
        res = g.evaluate(order)
    return SeedRun("A", order, res, info={"shortlist": len(cands)})


def _regime_scores(Sm: np.ndarray, lo: int, hi: int) -> np.ndarray:
    blk = Sm[lo:hi]
    return blk.mean(0) - 0.35 * blk.std(0)


def seed_B(g: GPContext, per_regime: int = 4) -> SeedRun:
    """16 bank seeds, 4 per view-regime sub-batch (0-7, 8-15, 16-23, 24-31); regime score = mean - 0.35 sd."""
    with selection_scope("H0.4.seedB"):
        cands = build_shortlist(g, with_grammar=False)
        Sm = S.screen_matrix(g.screener, g.cid, [c.dsl for c in cands])
        per: List[List[int]] = []
        chosen: List[int] = []
        for lo, hi in S.REGIMES:
            rs = _regime_scores(Sm, lo, hi)
            mine = []
            for _ in range(per_regime):
                j = _pick(rs, chosen)
                if j is None:
                    break
                chosen.append(j)
                mine.append(j)
            per.append(mine)
        order_idx = [
            per[r][k] for k in range(per_regime) for r in range(4) if k < len(per[r])
        ]  # round-robin
        order = [cands[k] for k in order_idx]
        res = g.evaluate(order)
    return SeedRun("B", order, res, info={"shortlist": len(cands)})


def _island_roles(
    Sm: np.ndarray,
    cons: np.ndarray,
    cands: Sequence[G.Candidate],
    D: np.ndarray,
    chosen: List[int],
    per_island: int,
) -> List[List[int]]:
    """Plan §3.3 island roles over the [32, M] matrix (pool roles replaced by island consensus: no pool DSLs)."""
    sd_all = Sm.std(0)
    by_isl = {i: [k for k, c in enumerate(cands) if c.island == i] for i in range(4)}
    m = lambda vs: Sm[list(vs)].mean(0)  # noqa: E731
    roles = {
        0: [Sm[0], m(range(19, 22)), m(range(0, 8)) - 0.35 * Sm[1:8].std(0), cons],
        1: [m(range(1, 4)), Sm[17], None, cons],  # None = UCB explorer
        2: [m(range(8, 16)), m((16, 18, 23)), cons, cons - 1.0 * sd_all],
        3: [m(range(24, 32)), "hybrid", "hybrid", "explore"],
    }
    out: List[List[int]] = []
    for isl in range(4):
        allowed = by_isl[isl]
        mine: List[int] = []
        rl = list(roles[isl])
        while len(rl) < per_island:
            rl.append(cons)
        for role in rl[:per_island]:
            if isinstance(role, np.ndarray):
                j = _pick(role, chosen, allowed)
            elif role == "hybrid":
                hyb = [k for k in allowed if G.head_family(cands[k].dsl) == "portfolio"]
                j = _pick(cons, chosen, hyb or allowed)
            else:  # UCB / max-distance explorer
                acq = np.full(len(cands), -1e18)
                for k in allowed:
                    if k not in chosen:
                        acq[k] = (
                            cons[k] + 0.5 * sd_all[k] if role is None else 0.0
                        ) + 0.3 * _min_dist(D, k, chosen)
                j = int(np.argmax(acq)) if acq.max() > -1e17 else None
            if j is None:  # island exhausted: fill with the overall consensus best
                j = _pick(cons, chosen)
            if j is None:
                break
            chosen.append(j)
            mine.append(j)
        out.append(mine)
    return out


def _round_robin(per: List[List[int]]) -> List[int]:
    k_max = max((len(p) for p in per), default=0)
    return [p[k] for k in range(k_max) for p in per if k < len(p)]


def seed_C(
    g: GPContext, per_island: int = 4, shortlist: Optional[List[G.Candidate]] = None
) -> SeedRun:
    """16 seeds across 4 islands from one [32, M~160] scoring of bank NN + grammar proposals."""
    with selection_scope("H0.4.seedC"):
        cands = shortlist if shortlist is not None else build_shortlist(g)
        assert_bank_disjoint(g.cid, cands, g.tb)
        Sm = S.screen_matrix(g.screener, g.cid, [c.dsl for c in cands])
        cons = S.consensus_weights(Sm.shape[0]) @ Sm
        D = np.stack([G.ext_descriptor(c.dsl) for c in cands])
        per = _island_roles(Sm, cons, cands, D, [], per_island)
        order = [cands[k] for k in _round_robin(per)]
        res = g.evaluate(order)
    return SeedRun(
        "C",
        order,
        res,
        info={
            "shortlist": len(cands),
            "shortlist_islands": [sum(c.island == i for c in cands) for i in range(4)],
        },
    )


def seed_D(
    g: GPContext, per_island: int = 5, shortlist: Optional[List[G.Candidate]] = None
) -> SeedRun:
    """Active 2+2: reveal top-2 by consensus, correct, reveal 2 by corrected UCB, correct; then 20 seeds (5/island)."""
    with selection_scope("H0.4.seedD"):
        cands = shortlist if shortlist is not None else build_shortlist(g)
        assert_bank_disjoint(g.cid, cands, g.tb)
        dsls = [c.dsl for c in cands]
        Sm = S.screen_matrix(g.screener, g.cid, dsls)
        cons = S.consensus_weights(Sm.shape[0]) @ Sm
        revealed: List[int] = []
        rev_res: List[SearchResult] = []
        for _ in range(2):
            j = _pick(cons, revealed)
            if j is not None:
                revealed.append(j)
        rev_res += g.evaluate([cands[k] for k in revealed])
        S_corr, pred, _ = S.corrected_matrix(
            g.screener, g.cid, dsls, [(r.dsl, r.utility) for r in rev_res], g.corrector
        )
        ucb = pred + 0.5 * S_corr.std(0)
        r2 = []
        for _ in range(2):
            j = _pick(ucb, revealed + r2)
            if j is not None:
                r2.append(j)
        rev_res += g.evaluate([cands[k] for k in r2])
        revealed += r2
        S_corr, pred, info = S.corrected_matrix(
            g.screener, g.cid, dsls, [(r.dsl, r.utility) for r in rev_res], g.corrector
        )
        D = np.stack([G.ext_descriptor(d) for d in dsls])
        per = _island_roles(S_corr, pred, cands, D, list(revealed), per_island - 1)
        # 5th seed per island: the island's best corrected prediction not yet chosen
        taken = list(revealed) + [k for p in per for k in p]
        for isl in range(4):
            j = _pick(pred, taken, [k for k, c in enumerate(cands) if c.island == isl])
            if j is None:
                j = _pick(pred, taken)
            if j is not None:
                per[isl].append(j)
                taken.append(j)
        seeds = [cands[k] for k in _round_robin(per)]
        res = g.evaluate(seeds)
        order = [cands[k] for k in revealed] + seeds
    return SeedRun(
        "D",
        order,
        rev_res + res,
        n_reveals=len(revealed),
        info={
            "shortlist": len(cands),
            "corrector": {k: float(v) for k, v in info.items()},
        },
    )


def registry_candidates(width: int) -> List[G.Candidate]:
    out = []
    for name in G.REGISTRY_E_ORDER:
        gen = G.cap_gpr(G.parse(G.REGISTRY_PRESETS[name]), width)
        out.append(G.Candidate.from_genome(gen, f"registry:{name}"))
    return out


def seed_E(g: GPContext, seed_a: SeedRun) -> SeedRun:
    """8 worker-registry presets + Seed A's 8 bank seeds in a fixed interleaved order (preset, bank, preset, ...).
    Presets are screener-ordered only if feat412 covers every preset token; otherwise the declared fixed order."""
    with selection_scope("H0.4.seedE"):
        presets = registry_candidates(g.ctx.width)
        covered = [not G.feat412_blind(c.dsl) for c in presets]
        mode = "fixed"
        if all(covered):
            sc = S.zscore_rows(g.screener.score(g.cid, [0], [c.dsl for c in presets]))[
                0
            ]
            presets = [presets[int(j)] for j in np.argsort(-sc, kind="stable")]
            mode = "screener_view0"
        bank = list(seed_a.order[:8])
        order: List[G.Candidate] = []
        for k in range(max(len(presets), len(bank))):
            if k < len(presets):
                order.append(presets[k])
            if k < len(bank):
                order.append(bank[k])
        order = unique_by_sha(order)
        res = g.evaluate(order)
    return SeedRun(
        "E",
        order,
        res,
        info={
            "preset_order": mode,
            "featurize_covered": int(sum(covered)),
            "presets": [c.operator for c in presets],
        },
    )


SCHEMES = ("A", "B", "C", "D", "E")


def run_h04(g: GPContext) -> Dict[str, SeedRun]:
    with selection_scope("H0.4.shortlist"):
        sl = build_shortlist(g)
    a = seed_A(g)
    return {
        "A": a,
        "B": seed_B(g),
        "C": seed_C(g, shortlist=sl),
        "D": seed_D(g, shortlist=sl),
        "E": seed_E(g, a),
    }


def run_scheme(g: GPContext, scheme: str) -> SeedRun:
    """One locked seeding scheme (Stage V); identical picks to the same scheme inside run_h04."""
    if scheme == "A":
        return seed_A(g)
    if scheme == "B":
        return seed_B(g)
    if scheme == "E":
        return seed_E(g, seed_A(g))
    with selection_scope("shortlist"):
        sl = build_shortlist(g)
    return seed_C(g, shortlist=sl) if scheme == "C" else seed_D(g, shortlist=sl)


# ----------------------------------------------------------------------------------------------------------------------
# CV5 PORTFOLIO_NNLS, BAG_FINAL, SEL_MEANLR, NOISE_PROBE, BANK_NN8 references
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class OneProgram:
    """A single deployable program built after a search (CV5 / BAG_FINAL): SEARCH-evaluated once, AUDIT once."""

    name: str
    cand: Optional[G.Candidate]
    result: Optional[SearchResult]
    fallback: bool
    info: dict = field(default_factory=dict)

    @property
    def live_evals(self) -> int:
        return 0 if self.result is None else 1


def portfolio_nnls(
    g: GPContext,
    results: Sequence[SearchResult],
    cand_of: Dict[str, G.Candidate],
    name: str = "CV5",
) -> OneProgram:
    """Best-by-SEARCH gate-passing program per head class (tree / linear / kernel, + best non-portfolio meta pipe)
    -> `(portfolio (member A) (member B) ... (blend folds=5 guard=clip kind=nnls repeats=auto))`, one SEARCH eval.
    Fallback (single best tree) when < 2 classes pass, the portfolio does not parse, or its SEARCH eval fails."""
    with selection_scope(f"{name}.build"):
        best: Dict[int, SearchResult] = {}
        for r in sorted([r for r in results if r.gate and r.ok], key=rank_key):
            isl = G.island_of(r.dsl)
            if isinstance(G.parse(r.dsl), G.Portfolio):
                continue
            best.setdefault(isl, r)
        members = [best[i] for i in sorted(best)][:4]
        if len(members) < 2:
            return OneProgram(
                name,
                None,
                None,
                True,
                {"reason": "fewer_than_2_classes", "classes": sorted(best)},
            )
        gen = G.op_portfolio([G.parse(m.dsl) for m in members])
        parents = [cand_of[m.sha] for m in members if m.sha in cand_of]
        cand = G.safe_candidate(
            gen,
            f"{name}_portfolio_nnls",
            [m.sha for m in members],
            g.ctx.width,
            cost_exempt=True,
            bank_cids=merge_lineage(*parents),
        )
        if cand is None:
            return OneProgram(name, None, None, True, {"reason": "parse_error"})
        if cand.sha in g.tb.excluded_shas:
            return OneProgram(name, None, None, True, {"reason": "excluded_sha"})
        res = g.evaluate([cand])[0]
        info = {
            "classes": [G.island_of(m.dsl) for m in members],
            "member_shas": [m.sha for m in members],
        }
        if res.status != "completed":
            return OneProgram(
                name, cand, res, True, {**info, "reason": f"search_failed:{res.error}"}
            )
        return OneProgram(name, cand, res, False, info)


def bag_final_program(
    g: GPContext, chosen: Optional[SearchResult], cand_of: Dict[str, G.Candidate]
) -> OneProgram:
    """BAG_FINAL secondary (report-only): bagged histgbm2 twin of a chosen tree-head program."""
    with selection_scope("BAG_FINAL.build"):
        if chosen is None or G.island_of(chosen.dsl) != 0:
            return OneProgram(
                "BAG_FINAL", None, None, True, {"reason": "not_a_tree_head"}
            )
        gen = G.bag_final(G.parse(chosen.dsl))
        par = [cand_of[chosen.sha]] if chosen.sha in cand_of else []
        cand = G.safe_candidate(
            gen,
            "bag_final",
            [chosen.sha],
            g.ctx.width,
            cost_exempt=True,
            bank_cids=merge_lineage(*par),
        )
        if cand is None or cand.sha in g.tb.excluded_shas:
            return OneProgram("BAG_FINAL", None, None, True, {"reason": "invalid"})
        res = g.evaluate([cand])[0]
        return OneProgram(
            "BAG_FINAL", cand, res, res.status != "completed", {"from": chosen.sha}
        )


def sel_meanlr(results: Sequence[SearchResult]) -> Optional[SearchResult]:
    """SEL_MEANLR secondary: finalist by lowest SEARCH mean log-ratio (gate-passing) instead of fitness."""
    ps = [r for r in results if r.gate and r.ok]
    return min(ps, key=lambda r: (r.mean_lr, r.sha)) if ps else None


def noise_probe(
    g: GPContext,
    seeds: Sequence[G.Candidate],
    base: Sequence[SearchResult],
    splits: Sequence[str] = ("noise1", "noise2"),
) -> dict:
    """Re-evaluate Gen-0 seeds under extra y-stratified random 2-fold splits of the SEARCH rows (diagnostic).

    Decomposes mean_lr[c, s] = offset_s + candidate effect + residual: reports the SD of the shared per-split offsets
    and the mean candidate-specific SD across splits after removing the offset."""
    with selection_scope("NOISE_PROBE"):
        rows = {"search": {r.sha: r for r in base}}
        for sp in splits:
            rows[sp] = {r.sha: r for r in g.engine.search_alt(g.ctx, sp, list(seeds))}
    names = ["search"] + list(splits)
    shas = [
        c.sha for c in seeds if all(rows[n][c.sha].status == "completed" for n in names)
    ]
    if len(shas) < 2:
        return {"n_candidates": len(shas)}
    M = np.array([[rows[n][s].mean_lr for n in names] for s in shas])  # [C, S]
    U = np.array([[rows[n][s].utility for n in names] for s in shas])
    off = M.mean(0)
    resid = M - off[None, :]
    return {
        "n_candidates": len(shas),
        "splits": names,
        "shared_offset_sd": float(np.std(off, ddof=1)),
        "candidate_specific_sd": float(np.mean(np.std(resid, axis=1, ddof=1))),
        "raw_candidate_sd": float(np.mean(np.std(M, axis=1, ddof=1))),
        "utility_raw_candidate_sd": float(np.mean(np.std(U, axis=1, ddof=1))),
        "rank_agreement_spearman": float(
            np.mean([_spearman(M[:, 0], M[:, j]) for j in range(1, M.shape[1])])
        ),
        "n_extra_evals": len(seeds) * len(splits),
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float("nan")


def bank_nn8_pick(g: GPContext, k: int = 8) -> SeedRun:
    """BANK_NN8_PICK: the 8 nearest cross-family bank champions (DOPE view-0), all evaluated, best-by-SEARCH."""
    with selection_scope("BANK_NN8_PICK"):
        order = [
            c
            for c in bank_champions_by_dope(g.tb, g.coder, k * 2, "bank_nn8")
            if G.width_ok(c.dsl, g.ctx.width)
        ][:k]
        res = g.evaluate(order)
    return SeedRun("NN8", order, res)


# ----------------------------------------------------------------------------------------------------------------------
# Evolution
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class EvoRun:
    arm: str
    results: List[SearchResult]
    gen_of: Dict[str, int]
    cand_of: Dict[str, G.Candidate]
    live_per_gen: Dict[int, int]
    screened_per_gen: Dict[int, int]
    line_search_live: Dict[int, int] = field(default_factory=dict)
    migration_live: Dict[int, int] = field(default_factory=dict)
    island_hist: Dict[int, List[int]] = field(default_factory=dict)
    archive_cells: Dict[int, int] = field(default_factory=dict)

    def best_after(self, gen: int) -> Optional[SearchResult]:
        return best_by_search([r for r in self.results if self.gen_of[r.sha] <= gen])

    @property
    def live_evals(self) -> int:
        return sum(self.live_per_gen.values())

    def summary(self) -> dict:
        curve = {}
        for gen in sorted(set(self.gen_of.values())):
            b = self.best_after(gen)
            curve[gen] = None if b is None else b.utility
        return {
            "arm": self.arm,
            "live_per_gen": self.live_per_gen,
            "screened_per_gen": self.screened_per_gen,
            "line_search_live": self.line_search_live,
            "migration_live": self.migration_live,
            "best_search_curve": curve,
            "island_hist": self.island_hist,
            "archive_cells": self.archive_cells,
            "n_results": len(self.results),
            "n_failed": sum(r.status != "completed" for r in self.results),
            "result_shas": [r.sha for r in self.results],
        }


class _Pop:
    def __init__(
        self, seeds: Sequence[G.Candidate], seed_results: Sequence[SearchResult]
    ):
        self.results: List[SearchResult] = []
        self.gen_of: Dict[str, int] = {}
        self.cand_of: Dict[str, G.Candidate] = {}
        for c, r in zip(seeds, seed_results):
            if r.sha not in self.gen_of:
                self.results.append(r)
                self.gen_of[r.sha] = 0
                self.cand_of[r.sha] = c

    def add(self, cands, results, gen):
        for c, r in zip(cands, results):
            if r.sha in self.gen_of:
                raise AssertionError("re-evaluation inside an arm")
            self.results.append(r)
            self.gen_of[r.sha] = gen
            self.cand_of[r.sha] = c

    def ranked(self) -> List[SearchResult]:
        return sorted(self.results, key=rank_key)


def _cand_with_lineage(
    gen, op, parents: Sequence[G.Candidate], width: int
) -> Optional[G.Candidate]:
    return G.safe_candidate(
        gen, op, [p.sha for p in parents], width, bank_cids=merge_lineage(*parents)
    )


def _fill_unique(
    g: GPContext,
    pool: List[G.Candidate],
    need: int,
    seen: Set[str],
    island: Optional[int],
    rng: random.Random,
    tag: str,
) -> List[G.Candidate]:
    out = []
    tries = 0
    while len(out) < need and tries < need * 50:
        tries += 1
        isl = island if island is not None else rng.randrange(4)
        c = G.safe_candidate(G.op_synth(isl, rng), f"{tag}_synth", [], g.ctx.width)
        if c is None or c.sha in seen or c.sha in g.tb.excluded_shas:
            continue
        seen.add(c.sha)
        out.append(c)
    return out


def propose_panmictic(
    g: GPContext,
    pop: _Pop,
    n: int,
    rng: random.Random,
    gen: int,
    eligible_nn: List[G.Candidate],
) -> List[G.Candidate]:
    """H257 / rust_evaluator propose_flywheel_candidates mix on the top-4 fertile parents."""
    seen = set(pop.gen_of)
    fertile = [pop.cand_of[r.sha] for r in pop.ranked()[:4]]
    out: List[G.Candidate] = []

    def add(c: Optional[G.Candidate]):
        if c is not None and c.sha not in seen and c.sha not in g.tb.excluded_shas:
            seen.add(c.sha)
            out.append(c)

    for c in (
        eligible_nn
    ):  # A. champion retrieval (up to min(4, n // 3) unseen NN bank records)
        if len([x for x in out if x.operator == "champion_retrieval"]) >= min(
            4, n // 3
        ):
            break
        if c.sha not in seen:
            add(
                G.Candidate(
                    c.dsl, c.sha, "champion_retrieval", c.parents, c.island, c.meta
                )
            )
    fg = [f.genome() for f in fertile]
    step = 0
    while (
        len(out) < n and step < n * 6 and fg
    ):  # B. EditZero kinds cycling over parents
        i = step % len(fg)
        par, pgen = fertile[i], fg[i]
        kind = step % 5
        if kind == 0:
            child, op = G.op_parameter(pgen, rng), "edit_parameter"
        elif kind == 1:
            child, op = G.op_replace_head(pgen, rng), "edit_replace_head"
        elif kind == 2:
            child, op = G.op_insert_stage(pgen, rng), "edit_insert_stage"
        elif kind == 3:
            child, op = G.op_delete_stage(pgen, rng), "edit_delete_stage"
        else:
            j = (i + 1) % len(fg)
            child, op = G.op_crossover(pgen, fg[j], rng), "edit_crossover"
            add(
                _cand_with_lineage(
                    child, f"g{gen}_{op}", [par, fertile[j]], g.ctx.width
                )
            )
            step += 1
            continue
        add(_cand_with_lineage(child, f"g{gen}_{op}", [par], g.ctx.width))
        step += 1
    out += _fill_unique(
        g, out, n - len(out), seen, None, rng, f"g{gen}"
    )  # C. synthesis fill
    return out[:n]


def _screen(
    g: GPContext, props: Sequence[G.Candidate], pop: _Pop
) -> Tuple[np.ndarray, np.ndarray]:
    revealed = [(r.dsl, r.utility) for r in pop.results]
    S_corr, pred, _ = S.corrected_matrix(
        g.screener, g.cid, [c.dsl for c in props], revealed, g.corrector
    )
    return pred, S_corr


def _zs(v: np.ndarray) -> np.ndarray:
    return (v - v.mean()) / max(v.std(), 1e-9)


def pick_exploit_ucb(
    props: Sequence[G.Candidate], pred: np.ndarray, pop: _Pop, k: int
) -> List[int]:
    """H257 pick: (k-2) exploit by z(pred) + 0.25 cos-to-best - 0.08 complexity/12, then 2 UCB by min distance."""
    if len(props) <= k:
        return list(range(len(props)))
    D = np.stack([G.ext_descriptor(c.dsl) for c in props])
    best = pop.ranked()[0]
    db = G.ext_descriptor(best.dsl)
    cos = (D @ db) / np.maximum(np.linalg.norm(D, axis=1) * np.linalg.norm(db), 1e-9)
    comp = np.array([G.complexity(c.dsl) / 12.0 for c in props])
    score = _zs(pred) + 0.25 * cos - 0.08 * comp
    chosen = [int(j) for j in np.argsort(-score, kind="stable")[: max(1, k - 2)]]
    ref = np.stack([G.ext_descriptor(r.dsl) for r in pop.results])
    while len(chosen) < k:
        R = np.concatenate([ref, D[chosen]], axis=0)
        ucb = np.array(
            [
                score[j] + 0.35 * float(np.min(np.linalg.norm(R - D[j][None], axis=1)))
                if j not in chosen
                else -1e18
                for j in range(len(props))
            ]
        )
        chosen.append(int(np.argmax(ucb)))
    return chosen


def evo_panmictic(
    g: GPContext,
    seed_cands: Sequence[G.Candidate],
    seed_results: Sequence[SearchResult],
    gens: int = EVO_GENS,
    n_prop: int = PROPOSALS_PER_GEN,
    n_live: int = LIVE_PER_GEN,
    arm: str = "evoA",
) -> EvoRun:
    with selection_scope(f"H0.5.{arm}"):
        pop = _Pop(seed_cands, seed_results)
        nn = [
            bank_candidate(b, "champion_retrieval")
            for _, b in bank_nn(g.tb, g.coder, 96)
        ]
        nn = [c for c in nn if G.width_ok(c.dsl, g.ctx.width)]
        live, screened = {}, {}
        for gen in range(1, gens + 1):
            rng = g.rng(arm, "gen", gen)
            props = propose_panmictic(g, pop, n_prop, rng, gen, nn)
            assert_bank_disjoint(g.cid, props, g.tb)
            pred, _ = _screen(g, props, pop)
            pick = [props[j] for j in pick_exploit_ucb(props, pred, pop, n_live)]
            res = g.evaluate(pick)
            pop.add(pick, res, gen)
            live[gen], screened[gen] = len(pick), len(props)
        return EvoRun(
            arm,
            pop.results,
            pop.gen_of,
            pop.cand_of,
            live,
            screened,
            island_hist={
                gen: [
                    sum(
                        1
                        for r in pop.results
                        if pop.gen_of[r.sha] == gen and G.island_of(r.dsl) == i
                    )
                    for i in range(4)
                ]
                for gen in range(gens + 1)
            },
        )


# -- Evo B: islands, MAP-Elites archive, epsilon-lexicase, migration, line search -------------------------------------
_NONLIN = {
    "cheb",
    "hinge",
    "poly_interact",
    "pair_prod",
    "rff_sin",
    "rff_cos",
    "oblique_hinge",
    "bin_onehot",
}


def elite_cell(dsl: str) -> Tuple[int, int]:
    c = G.complexity(dsl)
    cb = 0 if c <= 3 else 1 if c <= 5 else 2 if c <= 7 else 3
    gen = G.parse(dsl)
    pipes = gen.members if isinstance(gen, G.Portfolio) else (gen,)
    nl = int(any(s.op in _NONLIN for p in pipes for s in p.transforms))
    return cb, nl


class MapElites:
    def __init__(self):
        self.cells: Dict[int, Dict[Tuple[int, int], SearchResult]] = {
            i: {} for i in range(4)
        }

    def add(self, r: SearchResult) -> None:
        if r.status != "completed":
            return
        isl = G.island_of(r.dsl)
        cell = elite_cell(r.dsl)
        cur = self.cells[isl].get(cell)
        if cur is None or rank_key(r) < rank_key(cur):
            self.cells[isl][cell] = r

    def elites(self, isl: int) -> List[SearchResult]:
        return sorted(self.cells[isl].values(), key=rank_key)

    def champion(self, isl: int) -> Optional[SearchResult]:
        e = [r for r in self.elites(isl) if r.gate]
        return e[0] if e else None


def case_values(r: SearchResult) -> np.ndarray:
    if r.status != "completed":
        return np.full(len(LEXICASE_CASES), np.inf)
    v = np.array([r.lr0, r.lr1, r.tail_lr, float(r.complexity)], dtype=np.float64)
    return np.where(np.isfinite(v), v, np.inf)


def epsilon_lexicase(pool: Sequence[SearchResult], rng: random.Random) -> SearchResult:
    """Automatic epsilon-lexicase (La Cava et al. 2016): epsilon = MAD per case over the pool; lower is better."""
    if not pool:
        raise ValueError("empty pool")
    V = np.stack([case_values(r) for r in pool])
    eps = []
    for c in range(V.shape[1]):
        col = V[:, c][np.isfinite(V[:, c])]
        eps.append(float(np.median(np.abs(col - np.median(col)))) if col.size else 0.0)
    alive = list(range(len(pool)))
    cases = list(range(V.shape[1]))
    rng.shuffle(cases)
    for c in cases:
        best = min(V[i, c] for i in alive)
        if not math.isfinite(best):
            continue
        alive = [i for i in alive if V[i, c] <= best + eps[c]]
        if len(alive) == 1:
            break
    return pool[alive[rng.randrange(len(alive))]]


def migration_splices(arch: MapElites, g: GPContext, gen: int) -> List[G.Candidate]:
    """Best of islands 0-2 spliced into island 3 with the explicit portfolio combinator (all three + pairs)."""
    bests = [arch.champion(i) for i in range(3)]
    bests = [b for b in bests if b is not None]
    out = []
    combos = (
        [bests] + [[a, b] for k, a in enumerate(bests) for b in bests[k + 1 :]]
        if len(bests) >= 2
        else []
    )
    for combo in combos:
        gen_ = G.op_portfolio([G.parse(b.dsl) for b in combo])
        c = G.safe_candidate(
            gen_,
            f"g{gen}_migration_splice",
            [b.sha for b in combo],
            g.ctx.width,
            cost_exempt=True,
        )
        if c is not None:
            out.append(c)
    return out


def propose_island(
    g: GPContext,
    isl: int,
    arch: MapElites,
    pop: _Pop,
    n: int,
    rng: random.Random,
    gen: int,
    seen: Set[str],
    nn_by_isl: Dict[int, List[G.Candidate]],
    ring_in: Optional[SearchResult],
) -> Tuple[List[G.Candidate], List[G.Candidate]]:
    """12 proposals for island `isl`; returns (proposals, line_search_subset)."""
    out: List[G.Candidate] = []
    ls: List[G.Candidate] = []

    def add(c: Optional[G.Candidate], into=None) -> bool:
        if c is None or c.sha in seen or c.sha in g.tb.excluded_shas:
            return False
        seen.add(c.sha)
        out.append(c)
        if into is not None:
            into.append(c)
        return True

    if gen in LINE_SEARCH_GENS:
        champ = arch.champion(isl)
        if champ is not None:
            nb = G.line_search_neighbors(G.parse(champ.dsl))
            rng.shuffle(nb)
            for gen_ in nb:
                if len(ls) >= n // 2:
                    break
                add(
                    _cand_with_lineage(
                        gen_,
                        f"g{gen}_line_search",
                        [pop.cand_of[champ.sha]],
                        g.ctx.width,
                    ),
                    ls,
                )
    if gen in MIGRATION_GENS and ring_in is not None and isl != 3:
        mig = G.head_swap(G.parse(ring_in.dsl), isl, rng)
        add(
            _cand_with_lineage(
                mig, f"g{gen}_ring_migrant", [pop.cand_of[ring_in.sha]], g.ctx.width
            )
        )
    pool = arch.elites(isl)
    if len(pool) < 2:
        pool = [
            r
            for r in pop.results
            if G.island_of(r.dsl) == isl and r.status == "completed"
        ] or pool
    tries = 0
    while len(out) < n and tries < n * 12:
        tries += 1
        if pool:
            p1 = epsilon_lexicase(pool, rng)
            c1 = pop.cand_of[p1.sha]
            g1 = G.parse(p1.dsl)
            u = rng.random()
            if u < 0.30:
                child, op, par = G.op_parameter(g1, rng), "edit_parameter", [c1]
            elif u < 0.45:
                child, op, par = (
                    G.op_replace_head(g1, rng, isl),
                    "edit_replace_head",
                    [c1],
                )
            elif u < 0.65:
                child, op, par = G.op_insert_stage(g1, rng), "edit_insert_stage", [c1]
            elif u < 0.75:
                child, op, par = G.op_delete_stage(g1, rng), "edit_delete_stage", [c1]
            elif u < 0.92 and len(pool) >= 2:
                p2 = epsilon_lexicase(pool, rng)
                child, op, par = (
                    G.op_crossover(g1, G.parse(p2.dsl), rng),
                    "edit_crossover",
                    [c1, pop.cand_of[p2.sha]],
                )
            else:
                child, op, par = G.op_synth(isl, rng), "grammar_synth", []
        else:
            nn = nn_by_isl.get(isl) or []
            if nn and rng.random() < 0.5:
                b = rng.choice(nn)
                child, op, par = G.op_parameter(G.parse(b.dsl), rng), "nn_mutate", [b]
            else:
                child, op, par = G.op_synth(isl, rng), "grammar_synth", []
        c = _cand_with_lineage(child, f"g{gen}_{op}_i{isl}", par, g.ctx.width)
        if c is not None and isl != 3 and c.island != isl:
            continue  # keep island children inside their estimator family (island 3 accepts any hybrid)
        add(c)
    out += _fill_unique(g, out, n - len(out), seen, isl, rng, f"g{gen}_i{isl}")
    return out[:n], ls


def evo_islands(
    g: GPContext,
    seed_cands: Sequence[G.Candidate],
    seed_results: Sequence[SearchResult],
    gens: int = EVO_GENS,
    n_prop: int = PROPOSALS_PER_GEN,
    n_live: int = LIVE_PER_GEN,
    arm: str = "evoB",
) -> EvoRun:
    per_island_prop = n_prop // 4
    with selection_scope(f"H0.5.{arm}"):
        pop = _Pop(seed_cands, seed_results)
        arch = MapElites()
        for r in pop.results:
            arch.add(r)
        nn_all = [bank_candidate(b, "nn") for _, b in bank_nn(g.tb, g.coder, 96)]
        nn_by_isl = {
            i: [c for c in nn_all if c.island == i and G.width_ok(c.dsl, g.ctx.width)]
            for i in range(4)
        }
        live, screened, ls_live, mig_live, cells = {}, {}, {}, {}, {}
        for gen in range(1, gens + 1):
            rng = g.rng(arm, "gen", gen)
            seen = set(pop.gen_of)
            props: List[G.Candidate] = []
            ls_set: Set[str] = set()
            for isl in range(4):
                ring_src = (
                    arch.champion((isl - 1) % 4) if gen in MIGRATION_GENS else None
                )
                p, ls = propose_island(
                    g,
                    isl,
                    arch,
                    pop,
                    per_island_prop,
                    rng,
                    gen,
                    seen,
                    nn_by_isl,
                    ring_src,
                )
                if isl == 3 and gen in MIGRATION_GENS:
                    spl = [
                        c
                        for c in migration_splices(arch, g, gen)
                        if c.sha not in seen and c.sha not in g.tb.excluded_shas
                    ]
                    for c in spl:
                        seen.add(c.sha)
                    p = (spl + p)[:per_island_prop]
                props += p
                ls_set |= {c.sha for c in ls}
            if len(props) != n_prop:
                props += _fill_unique(
                    g, props, n_prop - len(props), seen, None, rng, f"g{gen}_fill"
                )
            assert len(props) == n_prop, (len(props), n_prop)
            assert_bank_disjoint(g.cid, props, g.tb)
            pred, _ = _screen(g, props, pop)
            z = _zs(pred)
            isl_of = [c.island for c in props]
            chosen: List[int] = []
            for isl in range(4):
                idx = [j for j in range(len(props)) if isl_of[j] == isl]
                if isl == 3 and gen in MIGRATION_GENS:
                    spl = [j for j in idx if "migration_splice" in props[j].operator]
                    idx = spl or idx
                if idx and len(chosen) < n_live:
                    chosen.append(max(idx, key=lambda j: (z[j], -j)))
            for j in np.argsort(-z, kind="stable"):
                if len(chosen) >= n_live:
                    break
                j = int(j)
                if j not in chosen and sum(isl_of[k] == isl_of[j] for k in chosen) < 2:
                    chosen.append(j)
            for j in np.argsort(
                -z, kind="stable"
            ):  # cannot happen with 4 islands x 12, kept for exactness
                if len(chosen) >= n_live:
                    break
                if int(j) not in chosen:
                    chosen.append(int(j))
            pick = [props[j] for j in chosen]
            res = g.evaluate(pick)
            pop.add(pick, res, gen)
            for r in res:
                arch.add(r)
            live[gen], screened[gen] = len(pick), len(props)
            ls_live[gen] = sum(c.sha in ls_set for c in pick)
            mig_live[gen] = sum(
                "migration" in c.operator or "ring_migrant" in c.operator for c in pick
            )
            cells[gen] = sum(len(arch.cells[i]) for i in range(4))
        return EvoRun(
            arm,
            pop.results,
            pop.gen_of,
            pop.cand_of,
            live,
            screened,
            ls_live,
            mig_live,
            island_hist={
                gen: [
                    sum(
                        1
                        for r in pop.results
                        if pop.gen_of[r.sha] == gen and G.island_of(r.dsl) == i
                    )
                    for i in range(4)
                ]
                for gen in range(gens + 1)
            },
            archive_cells=cells,
        )


ARMS = {"evoA": evo_panmictic, "evoB": evo_islands}


def h257_replica_evo(g: GPContext, seed_cands, seed_results) -> EvoRun:
    """H257 `run_3gen_live_gp_evolution_for_context`: 3 generations, 20 proposals, 5 children per generation."""
    return evo_panmictic(
        g, seed_cands, seed_results, gens=3, n_prop=20, n_live=5, arm="h257"
    )


def screen_topk_pick(
    g: GPContext, k: int, shortlist: Optional[List[G.Candidate]] = None
) -> SeedRun:
    """SCREEN_TOPK_PICK: one screening of bank NN + grammar proposals, live-evaluate the top-K by consensus."""
    with selection_scope("SCREEN_TOPK_PICK"):
        cands = shortlist if shortlist is not None else build_shortlist(g)
        if len(cands) < k:  # extend the grammar part until K candidates exist
            extra = grammar_shortlist(
                g,
                cands[:12],
                per_island=int(math.ceil((k - len(cands)) / 4)) + 4,
                tag="topk",
            )
            cands = unique_by_sha(cands + extra)
        assert_bank_disjoint(g.cid, cands, g.tb)
        Sm = S.screen_matrix(g.screener, g.cid, [c.dsl for c in cands])
        cons = S.consensus_weights(Sm.shape[0]) @ Sm
        order = [cands[int(j)] for j in np.argsort(-cons, kind="stable")[:k]]
        res = g.evaluate(order)
    return SeedRun("TOPK", order, res, info={"k": k, "shortlist": len(cands)})
