"""DL_wave0 architectures. All PURE end-to-end: inputs are only the clean encoder blocks + LV4 stats (block L).
No GBM prediction, anchor or distillation term exists in any of these modules (wave-1 adds a matched anchored twin explicitly).
Every model takes (xL [B,dL], mL [B,dL] missing mask, xe: list of [B,d_b] encoder blocks, me [B,nb] block-missing flags)
and returns dict(fit_mu, fit_ls, gap_mu, gap_ls, b10) — heteroscedastic heads for fitness/gap, logit for beat10."""
import math, torch, torch.nn as nn, torch.nn.functional as F
class Heads(nn.Module):
    def __init__(s, d):
        super().__init__(); s.n = nn.LayerNorm(d); s.o = nn.Linear(d, 5); nn.init.zeros_(s.o.weight); nn.init.zeros_(s.o.bias)   # start at the train mean, unit variance
    def forward(s, h):
        o = s.o(s.n(h)); return {"fit_mu": o[:, 0], "fit_ls": o[:, 1].clamp(-6, 4), "gap_mu": o[:, 2], "gap_ls": o[:, 3].clamp(-6, 4), "b10": o[:, 4]}
def block_drop(me, p, training):
    """Randomly mark whole encoder blocks missing during training (never all of them). Returns updated missing flags."""
    if not training or p <= 0: return me
    d = (torch.rand_like(me) < p).float(); d[(d + me).clamp(max=1).sum(1) >= me.shape[1]] = 0; return (me + d).clamp(max=1)
class BlockProj(nn.Module):
    """Per-block projection d_b -> k with a learned 'missing' vector used when the block is absent."""
    def __init__(s, dims, k, p=0.1):
        super().__init__(); s.p = nn.ModuleList([nn.Sequential(nn.Dropout(p), nn.Linear(d, k), nn.GELU(), nn.Linear(k, k)) for d in dims])
        s.miss = nn.Parameter(torch.zeros(len(dims), k))
    def forward(s, xe, me):
        if not s.p: return me.new_zeros(me.shape[0], 0, s.miss.shape[1])   # feature-subset arms with no encoder block
        out = []
        for b, (x, f) in enumerate(zip(xe, me.unbind(1))):
            h = s.p[b](x); f = f[:, None]; out.append(h * (1 - f) + s.miss[b] * f)
        return torch.stack(out, 1)   # [B, nb, k]
class FeatTok(nn.Module):
    """FT-Transformer style numeric tokenizer with periodic (PLE-lite) features for the LV4 block."""
    def __init__(s, dL, d, nfreq=8):
        super().__init__(); s.freq = nn.Parameter(torch.randn(dL, nfreq) * 0.5); s.lin = nn.Parameter(torch.randn(dL, 2 * nfreq + 1, d) * (1 / math.sqrt(2 * nfreq + 1)))
        s.bias = nn.Parameter(torch.zeros(dL, d)); s.miss = nn.Parameter(torch.zeros(dL, d))
    def forward(s, x, m):
        z = 2 * math.pi * x[..., None] * s.freq; f = torch.cat([x[..., None], torch.sin(z), torch.cos(z)], -1)   # [B,dL,2nf+1]
        t = torch.einsum("bjf,jfd->bjd", f, s.lin) + s.bias; m = m[..., None]; return t * (1 - m) + s.miss * m
class BlockFormer(nn.Module):
    """W0-A: block-token transformer. Tokens = [CLS] + one token per LV4 column (FeatTok) + 2 tokens per encoder block."""
    def __init__(s, dL, dims, d=128, L=4, heads=8, drop=0.1, bdrop=0.2, ntok=2):
        super().__init__(); s.ft = FeatTok(dL, d); s.bp = BlockProj(dims, d * ntok, 0.1); s.ntok = ntok; s.bdrop = bdrop
        s.cls = nn.Parameter(torch.zeros(1, 1, d)); s.btype = nn.Parameter(torch.zeros(1, len(dims) * ntok, d)); nn.init.normal_(s.btype, std=0.02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, drop, activation="gelu", batch_first=True, norm_first=True), L, enable_nested_tensor=False)
        s.h = Heads(d)
    def forward(s, xL, mL, xe, me):
        me = block_drop(me, s.bdrop, s.training); B = xL.shape[0]
        e = s.bp(xe, me).reshape(B, -1, s.cls.shape[-1]) + s.btype
        t = torch.cat([s.cls.expand(B, -1, -1), s.ft(xL, mL), e], 1); return s.h(s.enc(t)[:, 0])
class ResMLP(nn.Module):
    """W0-B: residual MLP on the concatenated clean features (encoder blocks projected per block, LV4 + missing mask raw)."""
    def __init__(s, dL, dims, k=64, w=512, depth=4, drop=0.2, bdrop=0.2):
        super().__init__(); s.bp = BlockProj(dims, k, 0.1); s.bdrop = bdrop
        s.inp = nn.Sequential(nn.Linear(2 * dL + len(dims) * (k + 1), w), nn.GELU())
        s.blocks = nn.ModuleList([nn.Sequential(nn.LayerNorm(w), nn.Linear(w, 2 * w), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * w, w)) for _ in range(depth)])
        for b in s.blocks: nn.init.zeros_(b[-1].weight); nn.init.zeros_(b[-1].bias)   # residual branches start as identity
        s.h = Heads(w)
    def forward(s, xL, mL, xe, me):
        me = block_drop(me, s.bdrop, s.training); e = s.bp(xe, me).flatten(1)
        h = s.inp(torch.cat([xL * (1 - mL), mL, e, me], 1))
        for b in s.blocks: h = h + b(h)
        return s.h(h)
class GatedSet(nn.Module):
    """W0-C: gated set / Perceiver over blocks. Each block (LV4 incl.) -> embedding; a per-row sigmoid gate weights blocks;
    2 latent queries cross-attend over the gated block set, then 2 self-attn layers over [latents + blocks]. Gates are exported."""
    def __init__(s, dL, dims, d=192, nlat=4, heads=8, drop=0.1, bdrop=0.2):
        super().__init__(); s.bp = BlockProj(dims, d, 0.1); s.bdrop = bdrop
        s.lp = nn.Sequential(nn.Linear(2 * dL, 2 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(2 * d, d))
        s.bid = nn.Parameter(torch.randn(1, len(dims) + 1, d) * 0.02); s.gate = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
        s.lat = nn.Parameter(torch.randn(1, nlat, d) * 0.02); s.xa = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True); s.nx = nn.LayerNorm(d)
        s.sa = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 2 * d, drop, activation="gelu", batch_first=True, norm_first=True), 2, enable_nested_tensor=False)
        s.h = Heads(d); s.last_gate = None
    def forward(s, xL, mL, xe, me):
        me = block_drop(me, s.bdrop, s.training); B = xL.shape[0]
        toks = torch.cat([s.lp(torch.cat([xL * (1 - mL), mL], 1))[:, None], s.bp(xe, me)], 1) + s.bid      # [B, nb+1, d]
        g = torch.sigmoid(s.gate(toks)).squeeze(-1); mfull = torch.cat([me.new_zeros(me.shape[0], 1), me], 1)   # L token always present (also with 0 encoder blocks)
        g = g * (1 - mfull) + 1e-4; s.last_gate = g.detach()
        kv = s.nx(toks * g[..., None]); lat, _ = s.xa(s.lat.expand(B, -1, -1), kv, kv, key_padding_mask=mfull.bool())
        h = s.sa(torch.cat([lat, toks * g[..., None]], 1)); return s.h(h[:, :lat.shape[1]].mean(1))
ARCH = {"blockformer": BlockFormer, "resmlp": ResMLP, "gatedset": GatedSet}
