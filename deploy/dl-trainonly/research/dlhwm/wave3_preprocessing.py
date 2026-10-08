"""Inner-fit transformations with complete inference state; GPU cache, mmap raw blocks."""
import numpy as np
import torch
from dlhwm.train import rankgauss_fit, rankgauss_apply


PCA_PROJECTION = {'rows': 32, 'allow_tf32': False}


def project_pca(z, center, components):
    """Use identical IEEE-float32 GEMM shapes for corpus and feature-only inference.

    TF32 and shape-dependent GEMM selection can change bf16 model predictions on
    reload. Padding also keeps singleton/tail requests on the same numerical path.
    This changes numerical execution only; fitting and selected columns stay inner-fit.
    """
    size = PCA_PROJECTION['rows']
    result = z.new_empty((len(z), components.shape[1]))
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        for offset in range(0, len(z), size):
            part = z[offset:offset + size]
            count = len(part)
            padded = z.new_zeros((size, z.shape[1]))
            padded[:count] = part
            result[offset:offset + count] = ((padded - center) @ components)[:count]
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    return result


def prepare(raw_l, raw_e, blocks, columns, overlap, fit, spec, device, seed, state=None, fit_targets=None):
    """Reload needs raw features and overlap flags only, never labels or a fitting split."""
    fitting = state is None
    if fitting:
        state = {'schema': 'jope.wave3-preprocessor/v2', 'blocks': blocks, 'columns': columns,
                 'encoder': {}, 'fit_rows': np.asarray(fit), 'tabdpt_policy': 'overlap_missing'}
        if spec['pca']:
            state['pca_projection'] = dict(PCA_PROJECTION)
    else:
        if state['blocks'] != blocks or state['columns'] != columns:
            raise ValueError('inference feature order mismatch')
        if spec['pca'] and state.get('pca_projection') != PCA_PROJECTION:
            raise ValueError('incompatible PCA numerical policy; use matching checkpoint source')
    ll = np.array(raw_l, dtype=np.float64, copy=True)
    tc = [i for i, c in enumerate(columns) if c.startswith('Z:tabdpt13') or c == 'ZL:2']
    ll[np.ix_(np.asarray(overlap, bool), tc)] = np.nan
    if fitting:
        state['QL'] = rankgauss_fit(ll[fit])
    xl = torch.as_tensor(rankgauss_apply(ll, state['QL']), device=device)
    ml = torch.as_tensor((~np.isfinite(ll)).astype(np.float32), device=device)
    xe, misses = [], []
    for bi, block in enumerate(blocks):
        # Copy a single raw block to GPU, retaining no eager CPU corpus copies.
        x = torch.tensor(np.asarray(raw_e[block]), dtype=torch.float32, device=device)
        if block == 'tabdpt13':
            x[torch.as_tensor(overlap, dtype=torch.bool, device=device)] = float('nan')
        miss = ~torch.isfinite(x).all(1)
        if fitting:
            med = torch.empty(x.shape[1], device=device); scale = torch.empty_like(med)
            rows = torch.as_tensor(fit, device=device)
            # Keep each quantile input below torch's 2**24-element limit.
            chunk = max(1, min(1024, (2**24 - 1) // len(fit)))
            for offset in range(0, x.shape[1], chunk):
                c = x[rows, offset:offset + chunk]
                q = torch.nanquantile(c, torch.tensor([.25, .5, .75], device=device), dim=0)
                mean = torch.nanmean(c, 0); sd = torch.sqrt(torch.nanmean((c - mean).square(), 0))
                iqr = (q[2] - q[0]) / 1.349
                med[offset:offset + chunk] = torch.nan_to_num(q[1], nan=0.)
                scale[offset:offset + chunk] = torch.where(iqr > 1e-9, iqr, torch.where(sd > 1e-9, sd, torch.ones_like(sd)))
            state['encoder'][block] = {'median': med.cpu().numpy(), 'scale': scale.cpu().numpy()}
        es = state['encoder'][block]
        z = torch.nan_to_num(((x - torch.as_tensor(es['median'], device=device)) / torch.as_tensor(es['scale'], device=device)).clamp(-8, 8), nan=0.)
        del x
        if spec['pca']:
            if fitting:
                rows = torch.as_tensor(fit, device=device); rows = rows[~miss[rows]]
                nk = min(spec['pca'], len(rows) - 1, z.shape[1])
                if nk < 1:
                    raise ValueError('insufficient present inner-fit rows for PCA: ' + block)
                center = z[rows].mean(0)
                devices = [torch.cuda.current_device()] if device == 'cuda' else []
                with torch.random.fork_rng(devices=devices):
                    torch.manual_seed(seed + 1009 * (bi + 1))
                    _, _, v = torch.pca_lowrank(z[rows] - center, q=nk, center=False, niter=4)
                sign = torch.sign(v.gather(0, v.abs().argmax(0, keepdim=True))).squeeze(0)
                v = v * sign
                ps = project_pca(z[rows], center, v).std(0).clamp(min=1e-6)
                es.update(center=center.cpu().numpy(), components=v.cpu().numpy(), pc_scale=ps.cpu().numpy())
            z = project_pca(z, torch.as_tensor(es['center'], device=device),
                            torch.as_tensor(es['components'], device=device)) / torch.as_tensor(es['pc_scale'], device=device)
            z = z.clamp(-8, 8)
        z[miss] = 0
        xe.append(z); misses.append(miss.float())
    me = torch.stack(misses, 1) if misses else xl.new_zeros((len(xl), 0))
    if spec['topk']:
        c = torch.cat([xl] + xe, 1)
        cm = torch.cat([ml] + [mi[:, None].expand(-1, z.shape[1]) for z, mi in zip(xe, misses)], 1)
        names = list(columns) + [f'{b}:pc{i}' for b, z in zip(blocks, xe) for i in range(z.shape[1])]
        if fitting:
            if fit_targets is None:
                raise ValueError('top-K fitting needs inner-fit labels')
            import lightgbm as lgb
            cx = c[torch.as_tensor(fit, device=device)].cpu().numpy(); importance = np.zeros(cx.shape[1])
            for head in ('fitness', 'beat10'):
                yy = fit_targets[head]; valid = np.isfinite(yy)
                g = lgb.LGBMRegressor(n_estimators=300, learning_rate=.05, num_leaves=15, min_child_samples=20,
                                     colsample_bytree=.5, subsample=.8, subsample_freq=1, n_jobs=2, random_state=seed, verbose=-1)
                g.fit(cx[valid], yy[valid])
                imp = np.abs(g.booster_.predict(cx[valid], pred_contrib=True)[:, :-1]).mean(0)
                importance += imp / max(imp.sum(), 1e-12)
            state['selected'] = np.argsort(-importance, kind='stable')[:spec['topk']]
            state['selected_names'] = [names[i] for i in state['selected']]
        xl, ml = c[:, state['selected']], cm[:, state['selected']]
        xe, me = [], xl.new_zeros((len(xl), 0))
    return (xl, ml, xe, me), state
