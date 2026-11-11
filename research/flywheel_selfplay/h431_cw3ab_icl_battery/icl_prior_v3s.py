#!/usr/bin/env python3
"""H431 (CW-3 Part B): H419 synthetic prior extended with five tabular-structure families (--prior v3s).

Preserves 100% byte-for-byte identical outputs and RNG state transitions to h419/r9/code/icl_prior.py whenever
`probs` is PROBS (v1, length 4) or PROBS_V2 (v2, length 7) or PROBS_SIMPLE.

New structure families in KINDS_V3S (CW3_CURRICULUM.md section 3 Part B):
  key     - 8 to N/4 discrete groups (uniform or Zipf sizes), group effect random (50%) or smooth in group index (50%),
            integer key id relabelled at random in 50% of tasks, 0..3 group-level covariates, 30% zero within-group term;
  keylin  - y = a_g + b_g * x_j per group g;
  cat     - 2..min(12, F) unordered factor columns with 2..20 levels and random integer codes, additive effects plus
            0..3 pair interactions and optional additive numeric part;
  arith   - y = h(x_a op x_b) with op in (-, *, /, log-ratio) plus weak additive remainder (<= 20% variance);
  mult    - y = exp(sum of factor and numeric effects).
"""
from __future__ import annotations

import math

import torch

KINDS = ("scm", "gp", "tree", "mix", "lin", "add", "addint")
KINDS_V3S = KINDS + ("key", "keylin", "cat", "arith", "mult")
PROBS = (0.35, 0.25, 0.20, 0.20)
PROBS_V2 = (0.20, 0.15, 0.15, 0.10, 0.10, 0.20, 0.10)
PROBS_SIMPLE = (0.0, 0.0, 0.0, 0.0, 0.3, 0.4, 0.3)
NEW_SHARES_DEFAULT = (0.07, 0.03, 0.06, 0.06, 0.03)   # key, keylin, cat, arith, mult (sum = 0.25)
PROBS_V3S = tuple(0.75 * p for p in PROBS_V2) + NEW_SHARES_DEFAULT


def probs_v3s_from_shares(shares: tuple[float, float, float, float, float]) -> tuple[float, ...]:
    s = sum(shares)
    normed = tuple(0.25 * (v / s) for v in shares) if s > 0 else NEW_SHARES_DEFAULT
    return tuple(0.75 * p for p in PROBS_V2) + normed


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

    # ---- primitives (identical to r9) ----
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

    # ---- inputs (identical to r9) ----
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
        kmax = min(Fn, kmax)
        k = 1 + torch.floor(self.rand(B, 1) ** 2 * kmax).clamp(max=max(0, kmax - 1))
        rank = torch.argsort(torch.argsort(self.rand(B, Fn), 1), 1)
        return (rank < k).float()

    # ---- v1 & v2 function families (identical to r9) ----
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
        nu = self.randint(0, 4, (B, 1, 1))
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

    def _coef(self, B, Fn, kmax):
        dense = self.rand(B, 1) < 0.5
        rel = torch.where(dense, torch.ones(B, Fn, device=self.dev), self.relevance(B, Fn, kmax))
        order = torch.argsort(torch.argsort(self.rand(B, Fn), 1), 1).float()
        return self.randn(B, Fn) * rel * torch.exp(-self.logu(0.01, 1.0, B, 1) * order)

    def lin(self, xs):
        return (xs @ self._coef(xs.shape[0], xs.shape[2], 12).unsqueeze(-1)).squeeze(-1)

    def _uni(self, xs):
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
        share = self.rand(B, 1, 1) * (self.rand(B, 1, 1) < 0.5)
        g = (1 - share) * g + share * xs
        f = (g @ self._coef(B, Fn, 20).unsqueeze(-1)).squeeze(-1)
        if inter:
            P = self.pick(1, 5)
            ja = self.randint(0, Fn, (B, 1, P)).expand(-1, N, -1)
            jb = self.randint(0, Fn, (B, 1, P)).expand(-1, N, -1)
            prod = self.std(g.gather(2, ja) * g.gather(2, jb))
            f = self.std(f) + (prod @ self.randn(B, P, 1)).squeeze(-1) * self.logu(0.05, 0.7, B, 1) / math.sqrt(P)
        return f

    # ---- v3s structure families (CW-3 Part B) ----
    def _draw_groups(self, B: int, N: int, max_g: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw group assignments [B, N] in 0..G_b-1 (uniform 50% or Zipf 50%) and relabelled integer codes [B, N]."""
        gmax = max(8, min(max_g, N // 4))
        G = int(round(math.exp(self.rand(1).item() * (math.log(gmax) - math.log(8.0)) + math.log(8.0))))
        zipf = self.rand(1).item() < 0.5
        if zipf:
            s = 0.5 + self.rand(1).item() * 0.9
            w = (torch.arange(1, G + 1, device=self.dev, dtype=torch.float32) ** (-s)).expand(B, G)
        else:
            w = torch.ones((B, G), device=self.dev, dtype=torch.float32)
        g_id = torch.multinomial(w, N, replacement=True, generator=self.g)  # [B, N]
        # Relabel group ids at random in half of tasks
        perm = torch.argsort(self.rand(B, G), dim=1)
        relabel = self.rand(B, 1) < 0.5
        codes = torch.where(relabel, perm.gather(1, g_id), g_id).float()
        return g_id, codes

    def key_task(self, B: int, N: int, Fn: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.base_x(B, N, Fn)
        xs = self.std(x)
        g_id, codes = self._draw_groups(B, N, max_g=max(16, N // 4))
        G = int(g_id.max().item()) + 1
        # Half random group effect, half smooth in group index
        eff_rand = self.randn(B, G)
        grid = torch.linspace(-2.5, 2.5, G, device=self.dev).unsqueeze(0).expand(B, G)
        eff_smooth = torch.sin(self.logu(0.6, 2.5, B, 1) * grid + self.rand(B, 1) * 6.283) + 0.4 * grid
        smooth_mask = self.rand(B, 1) < 0.5
        eff = torch.where(smooth_mask, eff_smooth, eff_rand)
        f_grp = eff.gather(1, g_id)
        # Put key code in column 0, plus 0..3 group-level covariates, then permute columns
        disc = torch.zeros((B, Fn), dtype=torch.bool, device=self.dev)
        x[:, :, 0] = codes
        disc[:, 0] = True
        n_cov = min(Fn - 1, self.pick(0, 3))
        for c in range(1, 1 + n_cov):
            cov_table = self.randn(B, G)
            x[:, :, c] = cov_table.gather(1, g_id)
            disc[:, c] = True
        # Within-group term: none in 30% of tasks
        has_within = (self.rand(B, 1) >= 0.30).float()
        within = self.std(self.lin(xs)) * self.logu(0.05, 0.35, B, 1) * has_within
        f = self.std(f_grp) + within
        perm = torch.argsort(self.rand(B, Fn), dim=1)
        x = x.gather(2, perm.unsqueeze(1).expand(-1, N, -1))
        disc = disc.gather(1, perm)
        return x, f, disc

    def keylin_task(self, B: int, N: int, Fn: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.base_x(B, N, Fn)
        xs = self.std(x)
        g_id, codes = self._draw_groups(B, N, max_g=max(12, N // 8))
        G = int(g_id.max().item()) + 1
        a_g = self.randn(B, G)
        b_g = self.randn(B, G)
        x[:, :, 0] = codes
        j_s = 1 if Fn > 1 else 0
        slope_x = xs[:, :, j_s]
        f = a_g.gather(1, g_id) + b_g.gather(1, g_id) * slope_x
        disc = torch.zeros((B, Fn), dtype=torch.bool, device=self.dev)
        disc[:, 0] = True
        perm = torch.argsort(self.rand(B, Fn), dim=1)
        x = x.gather(2, perm.unsqueeze(1).expand(-1, N, -1))
        disc = disc.gather(1, perm)
        return x, f, disc

    def cat_task(self, B: int, N: int, Fn: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.base_x(B, N, Fn)
        xs = self.std(x)
        k_cat = min(Fn, self.pick(2, 12))
        f = torch.zeros((B, N), device=self.dev)
        disc = torch.zeros((B, Fn), dtype=torch.bool, device=self.dev)
        cat_ids = []
        cat_levs = []
        for j in range(k_cat):
            lev = self.pick(2, 20)
            gid = self.randint(0, lev, (B, N))
            perm_c = torch.argsort(self.rand(B, lev), dim=1)
            codes = perm_c.gather(1, gid).float() * self.randint(1, 5, (B, 1)).float()
            x[:, :, j] = codes
            disc[:, j] = True
            eff = self.randn(B, lev)
            f = f + eff.gather(1, gid)
            cat_ids.append(gid)
            cat_levs.append(lev)
        n_int = self.pick(0, min(3, k_cat // 2))
        for m in range(n_int):
            ia, ib = 2 * m % k_cat, (2 * m + 1) % k_cat
            la, lb = cat_levs[ia], cat_levs[ib]
            pair_id = cat_ids[ia] * lb + cat_ids[ib]
            pair_eff = self.randn(B, la * lb) * 0.6
            f = f + pair_eff.gather(1, pair_id)
        if k_cat < Fn:
            has_num = (self.rand(B, 1) < 0.5).float()
            f = self.std(f) + has_num * 0.4 * self.std(self.add(xs[:, :, k_cat:], inter=False))
        perm = torch.argsort(self.rand(B, Fn), dim=1)
        x = x.gather(2, perm.unsqueeze(1).expand(-1, N, -1))
        disc = disc.gather(1, perm)
        return x, f, disc

    def arith_task(self, B: int, N: int, Fn: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.base_x(B, N, Fn)
        xs = self.std(x)
        ja = 0
        jb = 1 if Fn > 1 else 0
        op = self.randint(0, 4, (B, 1))   # 0 sub, 1 mul, 2 div, 3 log-ratio
        sc = self.logu(0.5, 10.0, B, 1)
        sh = self.randn(B, 1) * sc * 3.0
        xa_sub = xs[:, :, ja] * sc + sh
        xb_sub = xs[:, :, jb] * sc + sh
        pos_a = torch.exp(0.6 * xs[:, :, ja].clamp(-3.0, 3.0)) * sc
        pos_b = torch.exp(0.6 * xs[:, :, jb].clamp(-3.0, 3.0)) * self.logu(0.5, 5.0, B, 1)
        xa = torch.where(op >= 2, pos_a, torch.where(op == 0, xa_sub, xs[:, :, ja] * sc + sh * 0.2))
        xb = torch.where(op >= 2, pos_b, torch.where(op == 0, xb_sub, xs[:, :, jb] * sc + sh * 0.2))
        x[:, :, ja] = xa
        x[:, :, jb] = xb
        d = torch.where(
            op == 0, xa - xb,
            torch.where(op == 1, xa * xb, torch.where(op == 2, xa / xb.clamp_min(1e-4), torch.log(xa.clamp_min(1e-4) / xb.clamp_min(1e-4))))
        )
        dz = self.std(d)
        hk = self.randint(0, 4, (B, 1))
        h = torch.where(hk == 0, dz, torch.where(hk == 1, torch.tanh(1.5 * dz), torch.where(hk == 2, dz * dz - 1.0, torch.sign(dz) * torch.log1p(dz.abs()))))
        rem_scale = torch.sqrt(self.rand(B, 1) * 0.20)   # at most 20% variance remainder
        f = self.std(h) + rem_scale * self.std(self.lin(xs))
        disc = torch.zeros((B, Fn), dtype=torch.bool, device=self.dev)
        perm = torch.argsort(self.rand(B, Fn), dim=1)
        x = x.gather(2, perm.unsqueeze(1).expand(-1, N, -1))
        return x, f, disc

    def mult_task(self, B: int, N: int, Fn: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_cat, f_cat, disc = self.cat_task(B, N, Fn)
        xs = self.std(x_cat)
        log_f = 0.7 * self.std(f_cat) + 0.5 * self.std(self.lin(xs))
        f = torch.exp(0.85 * self.std(log_f).clamp(-3.5, 3.5))
        return x_cat, f, disc

    # ---- post-processing ----
    def warp_y(self, f):
        B = f.shape[0]
        k = self.randint(0, 20, (B, 1))
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

    def warp_x(self, x, protect_discrete: torch.Tensor | None = None):
        B, N, Fn = x.shape
        xs = self.std(x)
        r = self.rand(B, 1, Fn)
        qon = (self.rand(B, 1, 1) < 0.4) & (r < 0.3)
        lev = self.randint(2, 21, (B, 1, Fn)).float()
        if protect_discrete is not None:
            prot = protect_discrete.unsqueeze(1)   # [B, 1, Fn]
            qon = qon & (~prot)
            exp_mask = (r >= 0.3) & (r < 0.4) & (~prot)
            cub_mask = (r >= 0.4) & (r < 0.5) & (~prot)
        else:
            exp_mask = (r >= 0.3) & (r < 0.4)
            cub_mask = (r >= 0.4) & (r < 0.5)
        x = torch.where(qon, torch.floor(torch.sigmoid(1.5 * xs) * lev).clamp(max=lev - 1), x)
        x = torch.where(exp_mask, torch.exp(xs.clamp(-6, 6)), x)
        x = torch.where(cub_mask, xs ** 3, x)
        scale = self.logu(1e-2, 1e2, B, 1, Fn)
        x = x * scale + self.randn(B, 1, Fn) * scale * 3
        mon = (self.rand(B, 1, 1) < 0.1) & (self.rand(B, 1, Fn) < 0.3)
        miss = mon & (self.rand(B, N, Fn) < self.rand(B, 1, Fn) * 0.3)
        return torch.where(miss, torch.nan, x)

    # ---- one batch ----
    def sample(self, B, N, Fn, kinds=None, probs=PROBS):
        is_v3s = (probs is not None and len(probs) == len(KINDS_V3S)) or (
            kinds is not None and int(kinds.max().item()) >= len(KINDS)
        )
        if not is_v3s:
            # Exact r9 execution path (zero extra ops or RNG draws)
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
            first = self.rand(B, 1) < 0.5
            y = torch.where(first, self.warp_y(self.noise(f, xs)), self.noise(self.std(self.warp_y(f)), xs))
            bad = ~torch.isfinite(y).all(1, keepdim=True) | (torch.nan_to_num(y).std(1, keepdim=True) < 1e-6)
            y = torch.where(bad, xs[..., 0] + 0.1 * self.randn(B, N), y)
            y = self.std(y) * self.logu(1e-2, 1e3, B, 1) + self.randn(B, 1) * self.logu(1e-2, 1e3, B, 1)
            x = self.warp_x(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0))
            return x, y, kinds

        # v3s path: includes the 5 new structure families and protects discrete key/factor columns from coarse binning
        if kinds is None:
            kinds = torch.multinomial(torch.tensor(probs, device=self.dev), B, replacement=True, generator=self.g)
        xs_all, f_all, disc_all = [], [], []
        for ki, name in enumerate(KINDS_V3S):
            b = int((kinds == ki).sum().item())
            if b == 0:
                continue
            disc = torch.zeros((b, Fn), dtype=torch.bool, device=self.dev)
            if name == "scm":
                x, f = self.scm(b, N, Fn)
            elif name == "key":
                x, f, disc = self.key_task(b, N, Fn)
            elif name == "keylin":
                x, f, disc = self.keylin_task(b, N, Fn)
            elif name == "cat":
                x, f, disc = self.cat_task(b, N, Fn)
            elif name == "arith":
                x, f, disc = self.arith_task(b, N, Fn)
            elif name == "mult":
                x, f, disc = self.mult_task(b, N, Fn)
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
            xs_all.append(x); f_all.append(f); disc_all.append(disc)
        order = torch.argsort(torch.argsort(kinds, stable=True), stable=True)
        x = torch.cat(xs_all, 0)[order]
        f = self.std(torch.cat(f_all, 0)[order])
        disc = torch.cat(disc_all, 0)[order]
        xs = self.std(x)
        first = self.rand(B, 1) < 0.5
        # Forarith, protect raw pair relationship from nonlinear feature warp on columns 0..Fn-1
        is_arith = (kinds == KINDS_V3S.index("arith")).unsqueeze(-1)
        disc = disc | is_arith
        y = torch.where(first, self.warp_y(self.noise(f, xs)), self.noise(self.std(self.warp_y(f)), xs))
        bad = ~torch.isfinite(y).all(1, keepdim=True) | (torch.nan_to_num(y).std(1, keepdim=True) < 1e-6)
        y = torch.where(bad, xs[..., 0] + 0.1 * self.randn(B, N), y)
        y = self.std(y) * self.logu(1e-2, 1e3, B, 1) + self.randn(B, 1) * self.logu(1e-2, 1e3, B, 1)
        x = self.warp_x(torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), protect_discrete=disc)
        return x, y, kinds
