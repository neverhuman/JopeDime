"""Train-fold-fitted features: G3 (op-pair synergy), skeleton encoding, G7 (retrieval transfer).

EVERY statistic in this module is estimated from training-fold rows ONLY and applied to all rows.
That is the whole point of the fit/transform split -- these are the features that would leak if
fitted on the full table, and they are also the ones the plan expects to pay the most.

Two guards are built in rather than left to the caller:
  * `fit()` takes an explicit boolean train mask and never looks at rows outside it.
  * `transform()` cannot see labels at all; it only reads the fitted tables.

G7's self-hit hazard is made auditable rather than silently accepted. On newmath, 8 anchor genomes
appear on all 2,090 datasets, so a genome retrieving its own past results on a neighbouring dataset
is the common case, not the exception. `g7_self_hit_count` exposes exactly how often that fired so
the ablation can distinguish transfer from memorization.
"""

from __future__ import annotations

import itertools

import numpy as np

from .parse import ParsedGenome, dominant_family

# A pair must be seen this many times in the training folds before its lift is estimated at all.
MIN_PAIR_SUPPORT = 40
MIN_SKELETON_SUPPORT = 5
SYNERGY_THRESHOLD = 0.03
DEFAULT_K = 32

# Empirical-Bayes shrinkage of pair lifts toward 0 (the additive null).
#
# MEASURED: split-half reliability of the per-pair lift is only r = +0.50 with 68% sign agreement,
# and it does NOT improve with support (r = 0.50 / 0.48 / 0.60 across the 80-160 / 160-400 / 400+
# support bands). So the instability is genuine heterogeneity across datasets, not small samples --
# a pair's non-additivity really does depend on which dataset it lands on. By contrast op_mean
# (r = 0.82) and head_mean (r = 0.95) are highly stable.
#
# Shrinking by count/(count + PRIOR) keeps the aggregate signal, which is real, while damping the
# per-pair estimates that flip sign. Individual g3_pair_* columns should be treated as weak and
# left for the sweep's sentinel floor to accept or reject on evidence.
PAIR_SHRINK_PRIOR = 200.0

G3_NAMES = [
    "g3_pair_synergy_sum",
    "g3_pair_synergy_max",
    "g3_pair_antagonism_min",
    "g3_n_synergistic_pairs",
    "g3_n_antagonistic_pairs",
    "g3_adjacent_synergy_sum",
    "g3_pair_coverage",
    "g3_n_pairs",
    "g3_has_discretize_then_linear",
    "g3_has_randproj_then_tree",
    "g3_skeleton_mean",
    "g3_skeleton_count",
    "g3_skeleton_rarity",
    "g3_head_mean",
    "g3_op_mean_sum",
]

DISCRETIZERS = {"bin_onehot", "quantile_bins", "target_bin_encode"}
LINEAR_FAMILY = "linear"
TREE_FAMILY = "tree"
PROJECTION_OPS = {"rand_proj", "svd_project", "proj", "nystrom", "count_sketch", "tensor_sketch"}


def _pairs(pg: ParsedGenome) -> list[tuple[str, str]]:
    return list(itertools.combinations(sorted(set(pg.ops())), 2))


class SynergyFeatures:
    """G3: how well do this genome's op pairs, skeleton, and head do on TRAINING datasets.

    'Synergy' is the pair's observed mean percentile minus what an additive model of the two ops'
    individual means would predict. Measured examples: elasticnet+step +0.097, histgbm+rand_proj
    -0.078 -- non-additivity of that size is larger than the entire gap between prior art and the
    null, which is why it gets its own feature family.
    """

    def __init__(self) -> None:
        self.op_mean: dict[str, float] = {}
        self.pair_lift: dict[tuple[str, str], float] = {}
        self.skeleton: dict[str, tuple[float, int]] = {}
        self.head_mean: dict[str, float] = {}
        self.grand_mean = 0.5
        self.n_train = 0

    def fit(self, parses: list[ParsedGenome], y: np.ndarray, train: np.ndarray) -> SynergyFeatures:
        idx = np.flatnonzero(train)
        self.n_train = len(idx)
        self.grand_mean = float(y[idx].mean()) if len(idx) else 0.5

        op_acc: dict[str, list[float]] = {}
        pair_acc: dict[tuple[str, str], list[float]] = {}
        skel_acc: dict[str, list[float]] = {}
        head_acc: dict[str, list[float]] = {}
        for i in idx:
            pg = parses[i]
            if not pg.ok:
                continue
            yi = float(y[i])
            for op in set(pg.ops()):
                op_acc.setdefault(op, []).append(yi)
            for p in _pairs(pg):
                pair_acc.setdefault(p, []).append(yi)
            skel_acc.setdefault(pg.skeleton, []).append(yi)
            if pg.head:
                head_acc.setdefault(pg.head.op, []).append(yi)

        self.op_mean = {o: float(np.mean(v)) for o, v in op_acc.items() if len(v) >= MIN_PAIR_SUPPORT}
        self.head_mean = {h: float(np.mean(v)) for h, v in head_acc.items() if len(v) >= MIN_PAIR_SUPPORT}
        self.skeleton = {
            s: (float(np.mean(v)), len(v)) for s, v in skel_acc.items() if len(v) >= MIN_SKELETON_SUPPORT
        }
        # lift = observed - (mean_a + mean_b - grand_mean), the additive expectation, then shrunk
        # toward 0 by support. See PAIR_SHRINK_PRIOR for why shrinkage is not optional here.
        self.pair_lift = {}
        for p, v in pair_acc.items():
            if len(v) < MIN_PAIR_SUPPORT or p[0] not in self.op_mean or p[1] not in self.op_mean:
                continue
            expected = self.op_mean[p[0]] + self.op_mean[p[1]] - self.grand_mean
            raw = float(np.mean(v)) - expected
            self.pair_lift[p] = raw * (len(v) / (len(v) + PAIR_SHRINK_PRIOR))
        return self

    def feature_names(self) -> list[str]:
        return list(G3_NAMES)

    def extract(self, pg: ParsedGenome) -> dict[str, float]:
        out = {n: 0.0 for n in G3_NAMES}
        if not pg.ok:
            out["g3_skeleton_rarity"] = float(np.log(max(self.n_train, 1)))
            return out

        pairs = _pairs(pg)
        lifts = [self.pair_lift[p] for p in pairs if p in self.pair_lift]
        out["g3_n_pairs"] = float(len(pairs))
        out["g3_pair_coverage"] = float(len(lifts) / len(pairs)) if pairs else 0.0
        if lifts:
            out["g3_pair_synergy_sum"] = float(sum(lifts))
            out["g3_pair_synergy_max"] = float(max(lifts))
            out["g3_pair_antagonism_min"] = float(min(lifts))
            out["g3_n_synergistic_pairs"] = float(sum(1 for x in lifts if x > SYNERGY_THRESHOLD))
            out["g3_n_antagonistic_pairs"] = float(sum(1 for x in lifts if x < -SYNERGY_THRESHOLD))

        ops = pg.ops()
        adj = 0.0
        for a, b in zip(ops, ops[1:]):
            adj += self.pair_lift.get(tuple(sorted((a, b))), 0.0)
        out["g3_adjacent_synergy_sum"] = float(adj)

        head = pg.head
        head_fams = set(head.families) if head else set()
        out["g3_has_discretize_then_linear"] = float(
            bool(DISCRETIZERS & set(ops)) and LINEAR_FAMILY in head_fams
        )
        # the measured -0.078 anti-motif: a random projection feeding a tree head
        out["g3_has_randproj_then_tree"] = float(bool(PROJECTION_OPS & set(ops)) and TREE_FAMILY in head_fams)

        mean_count = self.skeleton.get(pg.skeleton)
        if mean_count:
            out["g3_skeleton_mean"], cnt = mean_count[0], mean_count[1]
            out["g3_skeleton_count"] = float(cnt)
            out["g3_skeleton_rarity"] = float(np.log(max(self.n_train, 1) / cnt))
        else:
            out["g3_skeleton_mean"] = self.grand_mean
            out["g3_skeleton_rarity"] = float(np.log(max(self.n_train, 1)))

        out["g3_head_mean"] = self.head_mean.get(head.op, self.grand_mean) if head else self.grand_mean
        out["g3_op_mean_sum"] = float(
            sum(self.op_mean.get(o, self.grand_mean) - self.grand_mean for o in set(ops))
        )
        return out


class RetrievalFeatures:
    """G7: how did genomes LIKE this one score on the nearest TRAINING datasets?

    This is the only mechanism in the build that converts a dataset-constant embedding into a
    genome-varying signal, which is exactly what a within-dataset percentile target requires. It is
    also where the 4,168-d sketch's measured retrieval edge (2.1x over hyperion on op-skeleton
    retrieval) is supposed to cash in -- the ablation decides whether it actually does.
    """

    def __init__(self, space: str, k: int = DEFAULT_K) -> None:
        self.space = space
        self.k = k
        self.names = [
            f"g7_{space}_{s}"
            for s in (
                "skeleton_mean",
                "skeleton_max",
                "skeleton_count",
                "head_mean",
                "domfam_mean",
                "ngram_weighted",
                "sim_mean",
                "sim_decay",
                "nbr_best_mean",
                "nbr_best_std",
                "self_hit_count",
                "coverage",
            )
        ]
        self._train_ds: list[str] = []
        self._emb: np.ndarray | None = None
        self._by_ds_skel: dict[str, dict[str, float]] = {}
        self._by_ds_head: dict[str, dict[str, float]] = {}
        self._by_ds_fam: dict[str, dict[str, float]] = {}
        self._ds_best: dict[str, float] = {}
        self._genome_ds: dict[str, set[str]] = {}

    def feature_names(self) -> list[str]:
        return list(self.names)

    def fit(
        self,
        dataset: np.ndarray,
        parses: list[ParsedGenome],
        y: np.ndarray,
        train: np.ndarray,
        emb_by_dataset: dict[str, np.ndarray],
    ) -> RetrievalFeatures:
        idx = np.flatnonzero(train)
        skel: dict[str, dict[str, list[float]]] = {}
        head: dict[str, dict[str, list[float]]] = {}
        fam: dict[str, dict[str, list[float]]] = {}
        best: dict[str, list[float]] = {}
        for i in idx:
            pg, d, yi = parses[i], str(dataset[i]), float(y[i])
            best.setdefault(d, []).append(yi)
            if not pg.ok:
                continue
            skel.setdefault(d, {}).setdefault(pg.skeleton, []).append(yi)
            if pg.head:
                head.setdefault(d, {}).setdefault(pg.head.op, []).append(yi)
            fam.setdefault(d, {}).setdefault(dominant_family(pg), []).append(yi)
            self._genome_ds.setdefault(pg.text, set()).add(d)

        self._by_ds_skel = {d: {k: float(np.mean(v)) for k, v in m.items()} for d, m in skel.items()}
        self._by_ds_head = {d: {k: float(np.mean(v)) for k, v in m.items()} for d, m in head.items()}
        self._by_ds_fam = {d: {k: float(np.mean(v)) for k, v in m.items()} for d, m in fam.items()}
        self._ds_best = {d: float(np.max(v)) for d, v in best.items()}

        # The index holds TRAIN-FOLD datasets only. A test dataset must never be its own neighbour.
        self._train_ds = sorted(self._by_ds_skel)
        mat = np.stack([emb_by_dataset[d] for d in self._train_ds]).astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        self._emb = mat / np.maximum(norms, 1e-8)
        return self

    def _neighbours(self, vec: np.ndarray, exclude: str | None):
        v = vec.astype(np.float32)
        v = v / max(float(np.linalg.norm(v)), 1e-8)
        sims = self._emb @ v
        if exclude is not None:
            try:
                sims[self._train_ds.index(exclude)] = -np.inf
            except ValueError:
                pass
        top = np.argpartition(-sims, min(self.k, len(sims) - 1))[: self.k]
        top = top[np.argsort(-sims[top])]
        return [(self._train_ds[j], float(sims[j])) for j in top if np.isfinite(sims[j])]

    def extract(self, pg: ParsedGenome, dataset: str, vec: np.ndarray) -> dict[str, float]:
        out = {n: 0.0 for n in self.names}
        if self._emb is None or not pg.ok:
            return out
        nbrs = self._neighbours(vec, exclude=dataset)
        if not nbrs:
            return out

        p = self.space
        sims = np.array([s for _, s in nbrs], dtype=np.float64)
        w = np.maximum(sims, 0.0)
        wsum = float(w.sum()) or 1.0

        skel_vals, head_vals, fam_vals, best_vals = [], [], [], []
        skel_w, hits = [], 0
        fam_key = dominant_family(pg)
        head_op = pg.head.op if pg.head else ""
        for (d, s), wi in zip(nbrs, w):
            best_vals.append(self._ds_best.get(d, 0.5))
            sv = self._by_ds_skel.get(d, {}).get(pg.skeleton)
            if sv is not None:
                skel_vals.append(sv)
                skel_w.append(wi)
            hv = self._by_ds_head.get(d, {}).get(head_op)
            if hv is not None:
                head_vals.append(hv * wi)
            fv = self._by_ds_fam.get(d, {}).get(fam_key)
            if fv is not None:
                fam_vals.append(fv * wi)
            if d in self._genome_ds.get(pg.text, ()):
                hits += 1

        out[f"g7_{p}_skeleton_count"] = float(len(skel_vals))
        out[f"g7_{p}_coverage"] = float(len(skel_vals) / len(nbrs))
        if skel_vals:
            out[f"g7_{p}_skeleton_mean"] = float(np.average(skel_vals, weights=skel_w or None))
            out[f"g7_{p}_skeleton_max"] = float(np.max(skel_vals))
        out[f"g7_{p}_head_mean"] = float(sum(head_vals) / wsum) if head_vals else 0.0
        out[f"g7_{p}_domfam_mean"] = float(sum(fam_vals) / wsum) if fam_vals else 0.0
        out[f"g7_{p}_ngram_weighted"] = out[f"g7_{p}_skeleton_mean"] * out[f"g7_{p}_coverage"]
        out[f"g7_{p}_sim_mean"] = float(sims.mean())
        out[f"g7_{p}_sim_decay"] = float(sims[0] - sims[-1]) if len(sims) > 1 else 0.0
        out[f"g7_{p}_nbr_best_mean"] = float(np.mean(best_vals))
        out[f"g7_{p}_nbr_best_std"] = float(np.std(best_vals))
        out[f"g7_{p}_self_hit_count"] = float(hits)
        return out
