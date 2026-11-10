#!/usr/bin/env python3
"""H419 column injection and real-table tasks (addendum 1).

inject_columns   H317's view regime 3 ("appended synthetic columns") on the GPU, per task and per column one of:
                 0 Gaussian / uniform distractor            4 monotone warp: sign(x) ln(1+|x|) / normal scores of ranks
                 1 Student-t3 / Cauchy distractor           5 quantile step: 1[x > q25] / 1[x > q75]
                 2 noisy collinear duplicate                6 random Fourier: sin / cos(w z + b)
                   a z_a + (1-a) z_b + 0.15 e
                 3 bilinear z_a z_b / quadratic z_a^2 - 1
                 (z = standardised source column). The target never depends on an injected column, so distractors are
                 pure noise and the others are redundant re-expressions of real signal.
sample_synth     a synthetic-prior batch where, with probability p, 20-60% of the feature slots are injected columns.
RealBank         tasks cut from FIT-ICL tables (icl_bank.py fit): a row crop, a column crop in random order, injected
                 columns filling the remaining feature slots, and one of three targets: the table's own target (50%),
                 another real column (25%), or a synthetic teacher function of the real features (25%). Real targets
                 get 1.5-3% Huberised noise in 30% of tasks (H317 regime 4).
"""
import math

import numpy as np
import torch

N_KINDS = 7


def inject_columns(prior, x, k):
    """x [B,N,F0] (NaN = missing) -> [B,N,k] injected columns."""
    B, N, F0 = x.shape
    z = prior.std(torch.nan_to_num(x))
    ia = prior.randint(0, F0, (B, 1, k)).expand(-1, N, -1)
    ib = prior.randint(0, F0, (B, 1, k)).expand(-1, N, -1)
    za, zb = z.gather(2, ia), z.gather(2, ib)
    e, u = prior.randn(B, N, k), prior.rand(B, N, k)
    kind = prior.randint(0, N_KINDS, (B, 1, k))
    alt = prior.rand(B, 1, k) < 0.5
    a = prior.rand(B, 1, k)
    t3 = e / torch.sqrt((prior.randn(B, N, k, 3) ** 2).mean(-1).clamp(min=1e-3))
    cauchy = torch.tan(math.pi * (u.clamp(1e-4, 1 - 1e-4) - 0.5))
    rank = (torch.argsort(torch.argsort(za, 1), 1).float() + 0.5) / N
    scores = math.sqrt(2.0) * torch.erfinv((2 * rank - 1).clamp(-0.999999, 0.999999))
    q = torch.quantile(za, torch.tensor([0.25, 0.75], device=x.device), dim=1, keepdim=True)
    w = prior.randn(B, 1, k) * prior.logu(0.5, 3.0, B, 1, k)
    ph = prior.rand(B, 1, k) * 2 * math.pi
    cols = (torch.where(alt, e, (u * 2 - 1) * 1.732),
            torch.where(alt, t3, cauchy),
            a * za + (1 - a) * zb + 0.15 * e,
            torch.where(alt, za * zb, za * za - 1),
            torch.where(alt, torch.sign(za) * torch.log1p(za.abs()), scores),
            torch.where(alt, (za > q[0]).float(), (za > q[1]).float()),
            torch.where(alt, torch.sin(w * za + ph), torch.cos(w * za + ph)))
    out = cols[0]
    for i in range(1, N_KINDS):
        out = torch.where(kind == i, cols[i], out)
    return out


def _shuffle_columns(prior, x):
    B, N, F = x.shape
    perm = torch.argsort(prior.rand(B, F), 1)
    return x.gather(2, perm.unsqueeze(1).expand(-1, N, -1))


def sample_synth(prior, rng, B, N, Fn, p=0.5):
    """Synthetic-prior batch; with probability p a share of the Fn feature slots holds injected columns."""
    k = 0
    if Fn >= 4 and rng.random() < p:
        k = min(Fn - 2, max(1, int(round(rng.uniform(0.2, 0.6) * Fn))))
    x, y, kinds = prior.sample(B, N, Fn - k)
    if k:
        x = _shuffle_columns(prior, torch.cat([x, inject_columns(prior, x, k)], -1))
    return x, y, kinds, k


class RealBank:
    def __init__(self, path, device, prior):
        z = np.load(path, allow_pickle=False)
        if set(z["cohort"].tolist()) != {"FIT-ICL"}:
            raise SystemExit("the training bank holds a table outside FIT-ICL")
        self.prior, self.dev = prior, torch.device(device)
        data, shape, off = z["data"], z["shape"], z["offset"]
        self.tabs = [torch.from_numpy(data[o:o + r * c].reshape(r, c).copy()).to(self.dev)
                     for (r, c), o in zip(shape.tolist(), off.tolist())]
        self.family = z["family"].tolist()
        self.n_tables = len(self.tabs)
        self.manifest_sha256 = str(z["manifest_sha256"])

    def eligible(self, N):
        return [i for i, t in enumerate(self.tabs) if t.shape[0] >= N]

    def _teacher(self, xs):
        """Synthetic target on real features xs [1,N,k] (standardised, missing = 0)."""
        p = self.prior
        u = float(p.rand(1))
        if u < 0.4:
            f = p.gp(xs)
        elif u < 0.7:
            f = p.tree(xs)
        else:
            f = p.std(p.gp(xs)) * float(p.rand(1)) + p.std(p.tree(xs))
        f = p.std(f)
        first = p.rand(1, 1) < 0.5
        return torch.where(first, p.warp_y(p.noise(f, xs)), p.noise(p.std(p.warp_y(f)), xs))

    def sample(self, rng, B, N, Fn):
        """Returns x [B,N,Fn], y [B,N], modes [B] (0 own target, 1 other column, 2 teacher), mean injected share."""
        p = self.prior
        ok = self.eligible(N)
        if not ok:
            return None
        xs, ys, modes, inj = [], [], [], 0
        for _ in range(B):
            t = self.tabs[rng.choice(ok)]
            r, c = t.shape
            u = rng.random()
            mode = 0 if u < 0.5 else 1 if u < 0.75 else 2
            tcol = c - 1 if mode == 0 else rng.randrange(c - 1) if mode == 1 else -1
            feats = [j for j in range(c) if j != tcol]
            if mode == 2:
                feats = feats[:-1] if rng.random() < 0.5 else feats      # the real target may serve as a feature
            k_real = rng.randint(min(2, len(feats)), min(len(feats), Fn))
            pick = torch.tensor(rng.sample(feats, k_real), device=self.dev)
            rows = torch.randperm(r, device=self.dev, generator=p.g)
            if mode < 2:
                rows = rows[torch.isfinite(t[rows, tcol])]
                if rows.numel() < N:
                    mode, tcol = 2, -1
                    rows = torch.randperm(r, device=self.dev, generator=p.g)
            rows = rows[:N]
            x = t[rows][:, pick].unsqueeze(0)                         # [1,N,k_real]
            if mode < 2:
                y = t[rows, tcol].unsqueeze(0)
                if float(torch.nan_to_num(y).std()) < 1e-12:
                    mode = 2
                elif rng.random() < 0.3:
                    e = p.randn(1, N) / torch.sqrt((p.randn(1, N, 3) ** 2).mean(-1).clamp(min=1e-3)) / 1.732
                    y = y + y.std() * rng.uniform(0.015, 0.03) * e.clamp(-1.345, 1.345)
            if mode == 2:
                y = self._teacher(p.std(torch.nan_to_num(x)))
                if not bool(torch.isfinite(y).all()) or float(y.std()) < 1e-12:
                    y = torch.nan_to_num(x[..., 0]) + 0.1 * p.randn(1, N)
            k_inj = Fn - k_real
            if k_inj:
                x = torch.cat([x, inject_columns(p, x, k_inj)], -1)
                inj += k_inj / Fn
            xs.append(_shuffle_columns(p, x)); ys.append(y.float()); modes.append(mode)
        return torch.cat(xs, 0), torch.cat(ys, 0), modes, inj / B


class RealVal:
    """FIT-SEL SEARCH-row monitor (icl_bank.py val): query = fold 0 of deploy_h415.kfold_labels, context = folds 1-4."""

    def __init__(self, path, device):
        z = np.load(path, allow_pickle=False)
        if set(z["cohort"].tolist()) != {"FIT-SEL"}:
            raise SystemExit("the monitor bank holds a context outside FIT-SEL")
        data, shape, off, lab = z["data"], z["shape"], z["offset"], z["lab"]
        self.items, lo = [], 0
        for (r, c), o in zip(shape.tolist(), off.tolist()):
            m = torch.from_numpy(data[o:o + r * c].reshape(r, c).copy())
            fold = torch.from_numpy(lab[lo:lo + r].astype(np.int64))
            lo += r
            order = torch.cat([torch.nonzero(fold != 0).squeeze(1), torch.nonzero(fold == 0).squeeze(1)])
            self.items.append((m[order].to(device), int((fold != 0).sum())))
        self.family = z["family"].tolist()
