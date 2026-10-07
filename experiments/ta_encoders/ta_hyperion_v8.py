"""TA_Hyperion_v8: target-aware in-context head on top of frozen Hyperion v7 row embeddings.

Changes vs Hyperion v7 (which pools X-only row embeddings into [mean,std,p10,p90]):
  1. Row tokens = LayerNorm(v7 row embedding 512) -> 256, plus a LABEL CHANNEL: support rows add Emb([y, 1]); query rows add a
     learned [MASK] token (their y is hidden).
  2. 6-layer pre-norm transformer over the context's rows; keys are support rows only, so every row (support or query) reads
     the support set's (x, y) pairs (TabPFN-style in-context learning), and query targets can never leak into the context.
  3. Context embedding (fixed width 1024) = projection of
       a) label-conditioned attention pooling: 8 learned seeds attend over the transformer's support-row states (which mix y),
       b) explicit X->y summaries on the projected row embeddings z (256): per-dimension corr(z_d, y) over support rows,
          per-dimension cov(z_d, residual) of a closed-form in-context ridge fit y ~ z, the ridge fit's LOO R^2, log n.
  4. Row-level embeddings: the transformer's per-row states (256) are returned for every row.
Objective (self-supervised, no GP labels): masked-y regression of held-out query rows inside each context's outer-train set
+ 0.1 * InfoNCE between context embeddings computed from two disjoint support subsets of the same context.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def ridge_summ(z, y, msk, lam=1.0):
    """z [B,N,d], y [B,N], msk [B,N] support mask -> corr [B,d], cov(z, resid) [B,d], loo_r2 [B]. Support rows only."""
    m = msk.float(); n = m.sum(1).clamp(min=2)
    zm = (z * m[..., None]).sum(1) / n[:, None]; ym = (y * m).sum(1) / n
    zc = (z - zm[:, None]) * m[..., None]; yc = (y - ym[:, None]) * m
    zs = torch.sqrt((zc ** 2).sum(1) / n[:, None] + 1e-6); ys = torch.sqrt((yc ** 2).sum(1) / n + 1e-6)
    corr = (zc * yc[..., None]).sum(1) / n[:, None] / (zs * ys[:, None])
    zn = zc / zs[:, None]
    # dual ridge (N x N) on support rows: K = Zn Zn^T / d, alpha = (K + lam I)^-1 y ; LOO residual = alpha / diag((K+lam I)^-1)
    d = z.shape[-1]
    K = zn @ zn.transpose(1, 2) / d
    eye = torch.eye(K.shape[1], device=z.device, dtype=K.dtype)
    A = K + lam * eye + (1 - m)[:, :, None] * eye * 1e3          # padded / query rows decoupled
    Ainv = torch.linalg.inv(A.float()).to(K.dtype)
    alpha = (Ainv @ yc[..., None]).squeeze(-1)
    loo = alpha / torch.diagonal(Ainv, dim1=1, dim2=2).clamp(min=1e-6)
    loo_r2 = 1 - ((loo * m) ** 2).sum(1) / ((yc ** 2).sum(1) + 1e-6)
    fit = yc - lam * alpha
    res = (yc - fit) * m
    cov_res = (zn * res[..., None]).sum(1) / n[:, None]
    return corr, cov_res, loo_r2.clamp(-1, 1)


class TAHyperionV8(nn.Module):
    def __init__(self, d_in=512, d=256, layers=6, heads=8, seeds=8, out_dim=1024):
        super().__init__()
        self.ln_in = nn.LayerNorm(d_in); self.inp = nn.Linear(d_in, d)
        self.yemb = nn.Sequential(nn.Linear(2, d), nn.GELU(), nn.Linear(d, d)); self.mask_tok = nn.Parameter(torch.zeros(d))
        lay = nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(lay, layers, enable_nested_tensor=False)
        self.ln_out = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.seeds = nn.Parameter(torch.randn(seeds, d) * 0.02); self.pma = nn.MultiheadAttention(d, heads, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(seeds * d + 2 * d + 2, out_dim), nn.GELU(), nn.Linear(out_dim, out_dim))
        self.cproj = nn.Sequential(nn.Linear(out_dim, 256), nn.GELU(), nn.Linear(256, 128))
        self.out_dim = out_dim

    def _rowtok(self, R):
        return self.inp(self.ln_in(R))

    def forward(self, R, y, sup, valid):
        """R [B,N,512], y [B,N] (standardised with support stats), sup/valid [B,N] bool. Returns yhat [B,N], ctx [B,out], rows [B,N,d]."""
        z = self._rowtok(R)
        ych = self.yemb(torch.stack([y * sup, sup.float()], -1))
        tok = z + torch.where(sup[..., None], ych, self.mask_tok.expand_as(z))
        keypad = ~(sup & valid)
        h = self.ln_out(self.enc(tok, src_key_padding_mask=keypad))
        yhat = self.head(h).squeeze(-1)
        q = self.seeds[None].expand(R.shape[0], -1, -1)
        pooled, _ = self.pma(q, h, h, key_padding_mask=keypad)
        with torch.autocast("cuda", enabled=False):
            corr, covr, loo = ridge_summ(z.float(), y.float(), sup & valid)
        n = (sup & valid).float().sum(1)
        feats = torch.cat([pooled.flatten(1), corr.to(pooled.dtype), covr.to(pooled.dtype), loo[:, None].to(pooled.dtype), torch.log1p(n)[:, None].to(pooled.dtype)], 1)
        ctx = self.proj(feats)
        return yhat, ctx, h


# ------------------------------------------------------------------------------------------------ v8b variants
class TAHyperionV8Raw(TAHyperionV8):
    """v8b-raw: same head, but each row token also gets the row's raw values (the same stored, train-only, importance-ordered
    first 64 features the v7 cache saw), standardised with SUPPORT-row stats, plus a feature-presence mask."""

    def __init__(self, raw_dim=64, **kw):
        super().__init__(**kw)
        d = self.inp.out_features; self.raw_dim = raw_dim
        self.rawp = nn.Sequential(nn.Linear(2 * raw_dim, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, R, y, sup, valid, Xr, fvalid):
        with torch.autocast("cuda", enabled=False):
            m = (sup & valid).float()[..., None]; n = m.sum(1).clamp(min=2)
            mu = (Xr * m).sum(1) / n; sd = torch.sqrt((((Xr - mu[:, None]) * m) ** 2).sum(1) / n + 1e-6)
            xs = ((Xr - mu[:, None]) / sd[:, None]).clamp(-6, 6) * fvalid[:, None].float() * valid[..., None].float()
            raw = torch.cat([xs, fvalid[:, None].float().expand_as(xs)], -1)
        self._raw_tok = self.rawp(raw)
        return super().forward(R, y, sup, valid)

    def _rowtok(self, R):
        return self.inp(self.ln_in(R)) + self._raw_tok


def trunk_embed(trunk, x, m, p_id, p_val, k_train):
    """Hyperion v7 embed_only with the first (layers - k_train) blocks frozen (no_grad) and the last k_train blocks + the
    embedding head trainable. Same math as hyperion_reference.MatrixGhostUnifiedStandalone._rowcol/embed_only."""
    L = len(trunk.blocks)

    def ln(layer, t):
        return layer(t.float()).to(t.dtype)
    with torch.no_grad():
        cell = trunk.val(torch.stack([x, m], dim=-1))
        s = trunk.scale_emb(p_id).to(cell.dtype)
        cell = cell + s[:, None, None, :]
        row_h = trunk.row_pool(cell, m, 2) + s[:, None, :]
        col_h = trunk.col_pool(cell, m, 1) + s[:, None, :]
        r_pad, c_pad = m.sum(2) == 0, m.sum(1) == 0
        for blk in trunk.blocks[: L - k_train]:
            r, c = row_h, col_h
            r_att, _ = blk['r2c'](r, c, c, key_padding_mask=c_pad, need_weights=False)
            c_att, _ = blk['c2r'](c, r, r, key_padding_mask=r_pad, need_weights=False)
            row_h = ln(blk['lnr'], r + r_att + blk['ffr'](r)); col_h = ln(blk['lnc'], c + c_att + blk['ffc'](c))
            row_h = row_h.masked_fill(r_pad[..., None], 0); col_h = col_h.masked_fill(c_pad[..., None], 0)
    def tail(row_h, col_h):
        for blk in trunk.blocks[L - k_train:]:
            r, c = row_h, col_h
            r_att, _ = blk['r2c'](r, c, c, key_padding_mask=c_pad, need_weights=False)
            c_att, _ = blk['c2r'](c, r, r, key_padding_mask=r_pad, need_weights=False)
            row_h = ln(blk['lnr'], r + r_att + blk['ffr'](r)); col_h = ln(blk['lnc'], c + c_att + blk['ffc'](c))
            row_h = row_h.masked_fill(r_pad[..., None], 0); col_h = col_h.masked_fill(c_pad[..., None], 0)

        def gstats(t, mask1d):
            t = t.float(); mask = mask1d.unsqueeze(-1).float(); cnt = mask.sum(1).clamp(min=1.0)
            mean = (t * mask).sum(1) / cnt; mx = t.masked_fill(mask < 0.5, -1e4).max(1).values
            diff = (t - mean.unsqueeze(1)) * mask; std = torch.sqrt((diff * diff).sum(1) / cnt + 1e-6)
            empty = mask.sum(1) == 0
            mean, std, mx = [u.masked_fill(empty, 0.0) for u in (mean, std, mx)]
            return torch.cat([mean, mx, std], dim=1)
        g = torch.cat([gstats(row_h, m.sum(2) > 0), gstats(col_h, m.sum(1) > 0), (p_val.float().view(-1, 1) / float(trunk.max_patch))], 1)
        with torch.autocast("cuda", enabled=False):
            return trunk.emb_head(g.float(), p_val)
    if torch.is_grad_enabled():   # activation checkpointing: keep only the trainable-region inputs, recompute in backward
        from torch.utils.checkpoint import checkpoint
        return checkpoint(tail, row_h, col_h, use_reentrant=False)
    return tail(row_h, col_h)


class TAHyperionV8FT(TAHyperionV8):
    """v8b-ft: the pinned Hyperion v7 trunk (init = pinned weights) runs inside the model; its last k_train blocks and the
    hard-gated embedding head are fine-tuned (lower LR), everything else frozen. Row embeddings follow the v7 recipe used for
    the cache (1 sweep, all six window sizes, per-column q01/q99 quantisation over the context's sampled rows, cell mean,
    per-context cell min-max, row mean). No raw values reach the head."""

    def __init__(self, trunk, cfg, k_train=2, chunk=384, **kw):
        super().__init__(**kw)
        self.trunk, self.cfg, self.k_train, self.chunk = trunk, cfg, k_train, chunk
        for p in self.trunk.parameters(): p.requires_grad_(False)
        L = len(self.trunk.blocks)
        self.trained_blocks = [f"blocks.{i}" for i in range(L - k_train, L)] + ["emb_head"]
        for i in range(L - k_train, L):
            for p in self.trunk.blocks[i].parameters(): p.requires_grad_(True)
        for p in self.trunk.emb_head.parameters(): p.requires_grad_(True)

    def trunk_params(self):
        return [p for p in self.trunk.parameters() if p.requires_grad]

    def head_params(self):
        tid = {id(p) for p in self.trunk.parameters()}
        return [p for p in self.parameters() if id(p) not in tid]

    def rows_embed(self, X, valid, fvalid):
        """X [B,N,P] raw stored values (rows = the context's sampled outer-train rows) -> R [B,N,512]."""
        cfg = self.cfg; ws_list, P, D = tuple(cfg["ws_list"]), int(cfg["max_patch"]), int(cfg["emb_dim"])
        pad, qs, mp = int(cfg["pad_u8"]), float(cfg["quant_scale"]), int(cfg["min_patch"])
        B, N, _ = X.shape; dev = X.device
        U, shapes = [], []
        with torch.no_grad():
            for b in range(B):
                n, p = int(valid[b].sum()), int(fvalid[b].sum())
                x = X[b, :n, :p].float()
                lo = torch.quantile(x, cfg["q_low"] / 100, dim=0); hi = torch.quantile(x, cfg["q_high"] / 100, dim=0)
                span = torch.where(hi - lo < 1e-9, torch.ones_like(hi), hi - lo)
                U.append(((x - lo) / span * 255).clamp(0, 255).floor()); shapes.append((n, p))
        sums = [torch.zeros(n, p, D, device=dev) for n, p in shapes]
        for ws in ws_list:
            tiles, meta = [], []
            for b, (u, (n, p)) in enumerate(zip(U, shapes)):
                xp = torch.nn.functional.pad(u, (0, (-p) % ws, 0, (-n) % ws), value=pad)
                nr, nc = xp.shape[0] // ws, xp.shape[1] // ws
                tiles.append(xp.reshape(nr, ws, nc, ws).permute(0, 2, 1, 3).reshape(nr * nc, ws, ws)); meta.append((b, nr, nc))
            T = torch.cat(tiles)
            canv = torch.full((len(T), P, P), float(pad), device=dev); canv[:, :ws, :ws] = T
            mask = torch.zeros(len(T), P, P, device=dev); mask[:, :ws, :ws] = 1
            xt = canv.div(255).mul(2).sub(1).mul(qs)
            pv = torch.full((len(T),), ws, device=dev, dtype=torch.long)
            Z = torch.cat([trunk_embed(self.trunk, xt[s:s + self.chunk], mask[s:s + self.chunk], pv[s:s + self.chunk] - mp, pv[s:s + self.chunk], self.k_train)
                           for s in range(0, len(T), self.chunk)])
            o = 0
            for b, nr, nc in meta:
                n, p = shapes[b]; z = Z[o:o + nr * nc].view(nr, nc, D); o += nr * nc
                sums[b] = sums[b] + z.repeat_interleave(ws, 0).repeat_interleave(ws, 1)[:n, :p]
        R = torch.zeros(B, N, D, device=dev)
        for b, (n, p) in enumerate(shapes):
            cells = sums[b] / len(ws_list)
            vmin = cells.amin((0, 1)); vmax = cells.amax((0, 1)); sp = vmax - vmin
            cells = torch.where(sp[None, None] < 1e-9, torch.zeros_like(cells), (cells - vmin) / torch.where(sp < 1e-9, torch.ones_like(sp), sp))
            R[b, :n] = cells.mean(1)
        return R

    def forward(self, X, y, sup, valid, fvalid, R=None):
        if R is None:
            R = self.rows_embed(X, valid, fvalid)
        return super().forward(R, y, sup, valid)
