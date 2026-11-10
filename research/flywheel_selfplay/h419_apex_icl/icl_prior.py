#!/usr/bin/env python3
"""H419 synthetic task prior, generated on the training device. No file is read: every task is drawn from a seeded
torch.Generator. A batch holds B regression tasks with one shape (N rows, F features).

Function families (PROBS):
  scm  - random MLP structural causal graphs, per-unit mixed activations, node noise; features and target are random
         graph nodes (so features may be causes, effects or unrelated);
  gp   - Gaussian-process draws by random Fourier features (RBF and Matern-like spectra, ARD, sparse relevance,
         sums and products, optional linear and periodic parts);
  tree - random oblivious trees and forests with data-dependent thresholds;
  mix  - a GP draw combined with a forest draw (sum, gate or product).
Version 2 (addendum 6) adds three simple families, because real tables are much easier than the first four:
  lin    - a linear function of the features (dense with decaying coefficients, or sparse);
  add    - an additive model: a sum of random univariate functions (linear, saturating, hinge, V, quadratic, slow
           sine, step, log, sigmoid), optionally blended toward linear;
  addint - an additive model plus 1-5 pairwise products of its univariate terms.
Then: target warps, noise (30% of tasks noise-free; homo- or heteroscedastic; normal, Student-t or Laplace), feature
quantisation, monotone feature warps, missing cells, and a random affine map of every column and of the target.
"""
import math

import torch

KINDS = ("scm", "gp", "tree", "mix", "lin", "add", "addint")
PROBS = (0.35, 0.25, 0.20, 0.20)                       # version 1: the first four families (addenda 1-5)
PROBS_V2 = (0.20, 0.15, 0.15, 0.10, 0.10, 0.20, 0.10)  # version 2 (addendum 6): 40% simple families
PROBS_SIMPLE = (0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.3)     # the simple families alone (validation block)


def draw_features(rng):
    """Feature count: 50% 3-12, 30% 13-32, 20% 33-100."""
    u = rng.random()
    if u < 0.5:
        return rng.randint(3, 12)
    if u < 0.8:
        return rng.randint(13, 32)
    return rng.randint(33, 100)


class Prior:
    def __init__(self, device, seed):
        self.dev = torch.device(device)
        self.g = torch.Generator(device=self.dev)
        self.g.manual_seed(seed)

    # ---- primitives ----
    def randn(self, *s):
        return torch.randn(*s, generator=self.g, device=self.dev)

    def rand(self, *s):
        return torch.rand(*s, generator=self.g, device=self.dev)

    def randint(self, lo, hi, s):
        return torch.randint(lo, hi, s, generator=self.g, device=self.dev)

    def logu(self, lo, hi, *s):
        return torch.exp(self.rand(*s) * (math.log(hi) - math.log(lo)) + math.log(lo))

    def pick(self, lo, hi):
        return int(self.randint(lo, hi + 1, (1,)).item())

    @staticmethod
    def std(v, dim=1):
        return (v - v.mean(dim, keepdim=True)) / (v.std(dim, keepdim=True) + 1e-6)

    # ---- inputs ----
    def base_x(self, B, N, K):
        x = self.randn(B, N, K)
        kind = self.randint(0, 4, (B, 1, 1))
        uni = (self.rand(B, N, K) * 2 - 1) * 1.732
        cent = self.randn(B, 4, K) * 1.5
        asg = self.randint(0, 4, (B, N))
        mixt = cent.gather(1, asg.unsqueeze(-1).expand(-1, -1, K)) + 0.5 * x
        heavy = x * torch.exp(0.8 * self.randn(B, N, 1).abs())
        x = torch.where(kind == 0, x, torch.where(kind == 1, uni, torch.where(kind == 2, mixt, heavy)))
        alpha = self.rand(B, 1, 1) * (self.rand(B, 1, 1) < 0.5)
        A = self.randn(B, K, K) / math.sqrt(K)
        return (1 - alpha) * x + alpha * (x @ A)

    def relevance(self, B, Fn, kmax):
        """[B,F] 0/1 mask with 1..min(F,kmax) relevant features per task, biased to few."""
        kmax = min(Fn, kmax)
        k = 1 + torch.floor(self.rand(B, 1) ** 2 * kmax).clamp(max=kmax - 1)
        rank = torch.argsort(torch.argsort(self.rand(B, Fn), 1), 1)
        return (rank < k).float()

    # ---- function families ----
    def mixed_act(self, z):
        B, N, H = z.shape
        ku = self.randint(0, 8, (B, 1, H))
        kd = self.randint(0, 8, (B, 1, 1)).expand(-1, -1, H)
        k = torch.where(self.rand(B, 1, 1) < 0.5, kd, ku).expand(-1, N, -1)
        outs = torch.stack([torch.tanh(z), torch.relu(z), torch.sin(z), z, z.abs(), torch.sigmoid(3 * z),
                            (z > 0).to(z.dtype), z * z.abs().clamp(max=4)], 0)
        return outs.gather(0, k.unsqueeze(0)).squeeze(0)

    def scm(self, B, N, Fn):
        L = self.pick(2, 5)
        H = max((16, 32, 64)[self.pick(0, 2)], math.ceil(2 * (Fn + 1) / (L + 1)))
        h = self.std(self.base_x(B, N, H))
        nodes = [h]
        keep = self.rand(B, 1, 1) * 0.85 + 0.15
        wscale = self.logu(0.5, 3.0, B, 1, 1)
        noise = self.logu(1e-3, 0.3, B, 1, 1) * (self.rand(B, 1, 1) < 0.8)
        for _ in range(L):
            W = self.randn(B, H, H) * (self.rand(B, H, H) < keep) / torch.sqrt(keep * H) * wscale
            h = self.mixed_act(h @ W + self.randn(B, 1, H))
            h = self.std(h + noise * self.randn(B, N, H))
            nodes.append(h)
        allN = torch.cat(nodes, -1)
        idx = torch.argsort(self.rand(B, allN.shape[-1]), 1)[:, :Fn + 1]
        sel = allN.gather(2, idx.unsqueeze(1).expand(-1, N, -1))
        return sel[..., :Fn], sel[..., Fn]

    def _rff(self, xs, M=256):
        B, N, Fn = xs.shape
        rel = self.relevance(B, Fn, 10)
        ell = self.logu(0.3, 6.0, B, 1, Fn)
        nu = self.randint(0, 4, (B, 1, 1))                       # 0 rbf, 1..3 Matern-like with 1/3/5 degrees
        dof = (2 * nu - 1).clamp(min=1).float()
        chi = (self.randn(B, M, 5) ** 2 * (torch.arange(5, device=self.dev) < dof)).sum(-1, keepdim=True) / dof
        tail = torch.where(nu == 0, torch.ones_like(chi), chi.clamp(min=1e-3).rsqrt())
        w = self.randn(B, M, Fn) / ell * rel.unsqueeze(1) * tail
        f = torch.cos(xs @ w.transpose(1, 2) + self.rand(B, 1, M) * 2 * math.pi) @ (self.randn(B, M, 1) / math.sqrt(M))
        return f.squeeze(-1), rel

    def gp(self, xs):
        B, N, Fn = xs.shape
        f1, rel = self._rff(xs)
        f2, _ = self._rff(xs)
        mode = self.rand(B, 1)
        f = torch.where(mode < 0.5, f1, torch.where(mode < 0.75, f1 + f2, self.std(f1) * self.std(f2)))
        f = self.std(f)
        lin = ((xs * rel.unsqueeze(1)) @ self.randn(B, Fn, 1)).squeeze(-1)
        f = f + self.std(lin) * self.rand(B, 1) * 2 * (self.rand(B, 1) < 0.3)
        j = self.randint(0, Fn, (B, 1, 1)).expand(-1, N, -1)
        per = torch.cos(2 * math.pi * xs.gather(2, j).squeeze(-1) / self.logu(0.3, 3.0, B, 1) + self.rand(B, 1) * 6.283)
        return f + per * self.rand(B, 1) * 2 * (self.rand(B, 1) < 0.15)

    def tree(self, xs):
        B, N, Fn = xs.shape
        T = int(round(2 ** (self.rand(1).item() * 6)))
        D = self.pick(1, 6)
        kmax = min(Fn, 12)
        k = 1 + torch.floor(self.rand(B, 1) ** 2 * kmax).clamp(max=kmax - 1)
        rel_idx = torch.argsort(self.rand(B, Fn), 1)[:, :kmax]
        feat = rel_idx.gather(1, torch.floor(self.rand(B, T * D) * k).long())
        xv = xs.gather(2, feat.unsqueeze(1).expand(-1, N, -1))
        thr = xv.gather(1, self.randint(0, N, (B, 1, T * D)))
        bits = (xv > thr).view(B, N, T, D).long()
        leaf = (bits * (2 ** torch.arange(D, device=self.dev))).sum(-1)
        decay = torch.exp(-torch.arange(T, device=self.dev).float() * self.logu(1e-3, 0.3, B, 1)).unsqueeze(-1)
        vals = self.randn(B, T, 2 ** D) * decay
        return vals.gather(2, leaf.transpose(1, 2)).sum(1)

    # ---- simple families (version 2) ----
    def _coef(self, B, Fn, kmax):
        """[B,F] coefficients: half of the tasks dense, half sparse (relevance); magnitudes decay over a random order."""
        dense = self.rand(B, 1) < 0.5
        rel = torch.where(dense, torch.ones(B, Fn, device=self.dev), self.relevance(B, Fn, kmax))
        order = torch.argsort(torch.argsort(self.rand(B, Fn), 1), 1).float()
        return self.randn(B, Fn) * rel * torch.exp(-self.logu(0.01, 1.0, B, 1) * order)

    def lin(self, xs):
        return (xs @ self._coef(xs.shape[0], xs.shape[2], 12).unsqueeze(-1)).squeeze(-1)

    def _uni(self, xs):
        """[B,N,F] -> one random univariate function of each feature, standardised over rows."""
        B, N, Fn = xs.shape
        k = self.randint(0, 9, (B, 1, Fn))
        t = self.randn(B, 1, Fn) * 0.8
        s = self.logu(0.5, 3.0, B, 1, Fn)
        z = xs - t
        outs = [xs, torch.tanh(s * z), torch.relu(z), z.abs(), z * z, torch.sin(s * xs + 3 * t), (z > 0).to(xs.dtype),
                torch.sign(xs) * torch.log1p(xs.abs()), torch.sigmoid(3 * s * z)]
        g = outs[0]
        for i in range(1, len(outs)):
            g = torch.where(k == i, outs[i], g)
        return self.std(g)

    def add(self, xs, inter=False):
        B, N, Fn = xs.shape
        g = self._uni(xs)
        share = self.rand(B, 1, 1) * (self.rand(B, 1, 1) < 0.5)   # half of the tasks are blended toward linear
        g = (1 - share) * g + share * xs
        f = (g @ self._coef(B, Fn, 20).unsqueeze(-1)).squeeze(-1)
        if inter:
            P = self.pick(1, 5)
            ja = self.randint(0, Fn, (B, 1, P)).expand(-1, N, -1)
            jb = self.randint(0, Fn, (B, 1, P)).expand(-1, N, -1)
            prod = self.std(g.gather(2, ja) * g.gather(2, jb))
            f = self.std(f) + (prod @ self.randn(B, P, 1)).squeeze(-1) * self.logu(0.05, 0.7, B, 1) / math.sqrt(P)
        return f

    # ---- post-processing ----
    def warp_y(self, f):
        B = f.shape[0]
        k = self.randint(0, 20, (B, 1))                          # 0..6 warp, 7..19 none (35% warped)
        s = self.logu(0.3, 1.5, B, 1)
        p = self.logu(0.5, 3.0, B, 1)
        lev = self.randint(3, 51, (B, 1)).float()
        rank = torch.argsort(torch.argsort(f, 1), 1).float() / f.shape[1]
        outs = [torch.exp(s * f), torch.sign(f) * f.abs() ** p, torch.sinh(s * f.clamp(-6, 6)), torch.tanh(s * f),
                torch.round(torch.sigmoid(f) * lev) / lev, rank, -torch.exp(-s * f)]
        out = f
        for i, o in enumerate(outs):
            out = torch.where(k == i, o, out)
        return out

    def noise(self, f, xs):
        B, N = f.shape
        sd = self.logu(0.01, 1.0, B, 1) * (self.rand(B, 1) >= 0.3)
        het = self.std((xs @ self.randn(B, xs.shape[-1], 1)).squeeze(-1))
        sd = sd * torch.where(self.rand(B, 1) < 0.2, torch.exp(0.7 * het.clamp(-3, 3)), torch.ones_like(het))
        e = self.randn(B, N)
        k = self.rand(B, 1)
        t3 = e / torch.sqrt((self.randn(B, N, 3) ** 2).mean(-1).clamp(min=1e-3)) / 1.732
        lap = torch.sign(e) * -torch.log(self.rand(B, N).clamp(min=1e-9)) / 1.414
        e = torch.where(k < 0.8, e, torch.where(k < 0.9, t3, lap))
        return f + sd * e

    def warp_x(self, x):
        B, N, Fn = x.shape
        xs = self.std(x)
        r = self.rand(B, 1, Fn)
        qon = (self.rand(B, 1, 1) < 0.4) & (r < 0.3)
        lev = self.randint(2, 21, (B, 1, Fn)).float()
        x = torch.where(qon, torch.floor(torch.sigmoid(1.5 * xs) * lev).clamp(max=lev - 1), x)
        x = torch.where((r >= 0.3) & (r < 0.4), torch.exp(xs.clamp(-6, 6)), x)
        x = torch.where((r >= 0.4) & (r < 0.5), xs ** 3, x)
        scale = self.logu(1e-2, 1e2, B, 1, Fn)
        x = x * scale + self.randn(B, 1, Fn) * scale * 3
        mon = (self.rand(B, 1, 1) < 0.1) & (self.rand(B, 1, Fn) < 0.3)
        miss = mon & (self.rand(B, N, Fn) < self.rand(B, 1, Fn) * 0.3)
        return torch.where(miss, torch.nan, x)

    # ---- one batch ----
    def sample(self, B, N, Fn, kinds=None, probs=PROBS):
        """Returns x [B,N,F] (NaN = missing), y [B,N], kind [B] (index into KINDS). probs: family probabilities
        (PROBS = version 1; with it this function draws exactly what it drew before version 2 existed)."""
        if kinds is None:
            kinds = torch.multinomial(torch.tensor(probs, device=self.dev), B, replacement=True, generator=self.g)
        xs_all, f_all = [], []
        for ki, name in enumerate(KINDS):
            b = int((kinds == ki).sum().item())
            if b == 0:
                continue
            if name == "scm":
                x, f = self.scm(b, N, Fn)
            else:
                x = self.base_x(b, N, Fn)
                xs = self.std(x)
                if name == "gp":
                    f = self.gp(xs)
                elif name == "tree":
                    f = self.tree(xs)
                elif name == "lin":
                    f = self.lin(xs)
                elif name in ("add", "addint"):
                    f = self.add(xs, inter=name == "addint")
                else:
                    f1, f2 = self.std(self.gp(xs)), self.std(self.tree(xs))
                    a = self.rand(b, 1)
                    j = self.randint(0, Fn, (b, 1, 1)).expand(-1, N, -1)
                    gate = xs.gather(2, j).squeeze(-1) > self.randn(b, 1) * 0.5
                    m = self.rand(b, 1)
                    f = torch.where(m < 0.5, a * f1 + (1 - a) * f2, torch.where(m < 0.8, torch.where(gate, f1, f2), f1 * f2))
            xs_all.append(x); f_all.append(f)
        order = torch.argsort(torch.argsort(kinds, stable=True), stable=True)
        x = torch.cat(xs_all, 0)[order]
        f = self.std(torch.cat(f_all, 0)[order])
        xs = self.std(x)
        first = self.rand(B, 1) < 0.5                            # noise before or after the target warp
        y = torch.where(first, self.warp_y(self.noise(f, xs)), self.noise(self.std(self.warp_y(f)), xs))
        bad = ~torch.isfinite(y).all(1, keepdim=True) | (torch.nan_to_num(y).std(1, keepdim=True) < 1e-6)
        y = torch.where(bad, xs[..., 0] + 0.1 * self.randn(B, N), y)
        y = self.std(y) * self.logu(1e-2, 1e3, B, 1) + self.randn(B, 1) * self.logu(1e-2, 1e3, B, 1)
        x = self.warp_x(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
        return x, y, kinds
