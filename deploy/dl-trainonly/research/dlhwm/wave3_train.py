#!/usr/bin/env python3
"""Wave3: immutable arm spec, exhaustive batches, grouped validation and inference checkpoints."""
import argparse
import contextlib
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dlhwm.common import PFX, assert_clean, log, put_create_only, sha256b
from dlhwm.models import ARCH
from dlhwm.train import group_folds
from dlhwm.wave3_objectives import objective, metrics, lineage_weights, validate_spec
from dlhwm.wave3_preprocessing import prepare


def clean_json(obj):
    if isinstance(obj, dict): return {k: clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)): return [clean_json(v) for v in obj]
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (float, np.floating)): return float(obj) if np.isfinite(obj) else None
    return obj


def atomic_json(path, obj):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(clean_json(obj), indent=1, allow_nan=False) + '\n')
    os.replace(tmp, path)


def atomic_npy(path, arr):
    path = Path(path); tmp = path.with_name(path.name + '.tmp')
    with tmp.open('wb') as f: np.save(f, arr)
    os.replace(tmp, path)


def row_hash(rows):
    return sha256b(np.sort(np.asarray(rows, dtype='<i8')).tobytes())


def diagnostic_panel(rows, groups, limit=2048, seed=0):
    """Label-independent lineage round-robin; cover every group before adding views."""
    rng = np.random.default_rng(104729 + seed)
    ug = rng.permutation(np.unique(groups[rows]))
    buckets = [list(rng.permutation(rows[groups[rows] == g])) for g in ug]
    panel = []
    while len(panel) < min(limit, len(rows)):
        for bucket in buckets:
            if bucket: panel.append(bucket.pop())
            if len(panel) >= min(limit, len(rows)): break
    return np.sort(panel)


def exposure(rows, expected):
    rows, expected = np.asarray(rows), np.asarray(expected)
    unique, counts = np.unique(rows, return_counts=True)
    good = np.array_equal(unique, np.sort(expected)) and np.all(counts == 1)
    if not good: raise RuntimeError('epoch omitted or repeated inner-fit rows')
    return {'rows': len(rows), 'unique_rows': len(unique), 'count_min': int(counts.min()),
            'count_max': int(counts.max()), 'row_sha256': row_hash(rows), 'omitted': 0, 'duplicated': 0}


def load_data(data):
    meta = json.loads((data / 'meta.json').read_text())
    blocks = [b for b in meta['blocks'] if b != 'L']; assert_clean(['L'] + blocks)
    l = np.load(data / 'blocks_L.npy', mmap_mode='r')
    e = {b: np.load(data / f'blocks_{b}.npy', mmap_mode='r') for b in blocks}
    targets = {h: np.asarray(v, dtype=np.float64) for h, v in dict(np.load(data / 'targets.npz')).items()}
    fo, gid = np.load(data / 'folds.npy'), np.load(data / 'gid.npy')
    n = meta['n']
    if l.shape[0] != n or len(gid) != n or fo.shape != (3, n) or len(set(meta['view_id'])) != n:
        raise RuntimeError('row identity/shape mismatch')
    for h in ('fitness', 'gap', 'beat10'):
        if targets[h].shape != (n,) or not np.isfinite(targets[h]).all():
            raise RuntimeError('this wave3 bundle requires finite targets: ' + h)
    if set(np.unique(targets['beat10'])) - {0, 1}: raise RuntimeError('nonbinary beat10')
    return meta, blocks, l, e, targets, fo, gid


def infer_checkpoint(ck, l, e, overlap, device='cpu', native_units=False):
    """Public feature-only checkpoint inference. No targets, group IDs or fitting split."""
    model = ARCH[ck['arch']](ck['dL'], ck['dims'], **ck['kw']).to(device)
    model.load_state_dict(ck['state_dict']); model.eval()
    tensors, _ = prepare(l, e, ck['blocks'], ck['L_cols'], overlap, None, ck['configuration'],
                         device, 0, state=ck['preprocessor'])
    xl, ml, xe, me = tensors
    with torch.no_grad(), torch.autocast(device, dtype=torch.bfloat16, enabled=device == 'cuda'):
        out = model(xl, ml, xe, me)
    p = np.column_stack([out[k].float().cpu().numpy() for k in ('fit_mu', 'fit_ls', 'gap_mu', 'gap_ls', 'b10')])
    p = mask_outputs(p, ck['configuration'])
    if native_units:
        for head, col in [('fitness', 0), ('gap', 2)]:
            center, scale = ck['target_norm'][head]
            p[:, col] = p[:, col] * scale + center
            p[:, col + 1] = p[:, col + 1] + 2 * math.log(scale)
    return p


def mask_outputs(p, spec):
    p = p.copy()
    if not spec['uncertainty_trained']: p[:, [1, 3]] = np.nan
    if 'gap' not in spec['active_heads']: p[:, [2, 3]] = np.nan
    if 'beat10' not in spec['active_heads']: p[:, 4] = np.nan
    return p


def main():
    ap = argparse.ArgumentParser()
    for name in ('data', 'out', 'run-id', 'arch', 'spec', 'git-sha'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--repeats', default='0,1,2')
    ap.add_argument('--folds', default='0,1,2,3,4'); ap.add_argument('--s3', action='store_true')
    ap.add_argument('--qualify-epochs', type=int, default=0)
    a = ap.parse_args(); arm = json.loads(Path(a.spec).read_text()); spec = arm['configuration']; validate_spec(spec)
    if a.arch != spec['arch'] or len(a.git_sha) != 40 or any(c not in '0123456789abcdef' for c in a.git_sha):
        raise RuntimeError('source/config identity mismatch')
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(int(os.environ.get('DLHWM_TORCH_THREADS', '2')))
    torch.backends.cuda.matmul.allow_tf32 = True
    data, out = Path(a.data), Path(a.out) / a.run_id
    out.mkdir(parents=True, exist_ok=False)
    lock = (out / 'run.lock').open('x'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    meta, blocks, l, e, targets, fo, gid = load_data(data); n = meta['n']
    config = dict(spec['cfg']); epochs = a.qualify_epochs or config['epochs']
    summary = {'run_id': a.run_id, 'arch': a.arch, 'wave': 3, 'arm': arm['id'], 'seed': a.seed,
               'pure': True, 'gbm_anchor': False, 'gbm_input': False, 'distill': False, 'optimistic': False,
               'git_sha': a.git_sha, 'host': socket.gethostname(), 'blocks': ['L'] + blocks,
               'cfg': dict(config, kw=spec['kw']), 'configuration': spec, 'active_heads': spec['active_heads'],
               'uncertainty_trained': spec['uncertainty_trained'], 'pca': spec['pca'], 'topk': spec['topk'],
               'data_manifest_sha256': sha256b((data / 'MANIFEST.json').read_bytes()),
               'arm_spec_sha256': sha256b(Path(a.spec).read_bytes()), 'rows': n, 'lineages': len(np.unique(gid)),
               'qualification_only': bool(a.qualify_epochs), 'folds': {}, 's3': [], 'state': 'running'}
    atomic_json(out / 'summary.json', summary)
    batches = (out / 'batch_metrics.jsonl').open('x', buffering=1)
    ledger = (out / 'exposure.jsonl').open('x', buffering=1)
    for r in map(int, a.repeats.split(',')):
        oof, raw = np.full((n, 5), np.nan), np.full((n, 2), np.nan)
        for k in map(int, a.folds.split(',')):
            started = time.time(); te = np.where(fo[r] == k)[0]; outer = np.where(fo[r] != k)[0]
            inner = group_folds(gid[outer], 10, 777 + 31 * r + k)
            fit, va = outer[inner != 0], outer[inner == 0]
            if any(set(gid[x]) & set(gid[y]) for x, y in ((fit, va), (fit, te), (va, te))):
                raise RuntimeError('lineage leakage')
            if not np.array_equal(np.sort(np.concatenate([fit, va, te])), np.arange(n)):
                raise RuntimeError('fold roles omit rows')
            seed = a.seed * 100 + r * 10 + k
            features, state = prepare(l, e, blocks, meta['L_cols'], meta['tabdpt_overlap'], fit, spec, dev, seed,
                                      fit_targets={h: targets[h][fit] for h in ('fitness', 'beat10')})
            xl, ml, xe, me = features
            norm = {}
            yy = {}
            for h in ('fitness', 'gap'):
                center, scale = np.median(targets[h][fit]), np.std(targets[h][fit])
                if not np.isfinite(scale) or scale <= 1e-12: raise RuntimeError('degenerate target scale')
                norm[h] = [float(center), float(scale)]; yy[h] = ((targets[h] - center) / scale).astype(np.float32)
            yy['beat10'] = targets['beat10'].astype(np.float32)
            gy = {h: torch.as_tensor(v, device=dev) for h, v in yy.items()}
            gw = None
            if spec['sample_weight'] == 'lineage':
                gw = {}
                for h in gy:
                    full = np.zeros(n, np.float32)
                    full[fit] = lineage_weights(gid[fit], np.isfinite(yy[h][fit]))
                    gw[h] = torch.as_tensor(full, device=dev)
            torch.manual_seed(seed); rng = np.random.RandomState(seed)
            model = ARCH[a.arch](xl.shape[1], [z.shape[1] for z in xe], **spec['kw']).to(dev)
            ema = torch.optim.swa_utils.AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(spec['ema']))
            opt = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'], betas=(.9, .98))
            total_steps = config['epochs'] * math.ceil(len(fit) / config['bs']); warm = max(1, int(.05 * total_steps))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * .5 * (1 + math.cos(math.pi * min(1, s / total_steps))))
            panel = diagnostic_panel(fit, gid, seed=seed)
            def forward(m, idx):
                ii = torch.as_tensor(idx, device=dev)
                with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == 'cuda'):
                    return m(xl[ii], ml[ii], [z[ii] for z in xe], me[ii])
            @torch.no_grad()
            def predict(m, idx):
                m.eval(); values = []
                # Same batch partition for pre/post serialization comparisons.
                for off in range(0, len(idx), 256):
                    o = forward(m, idx[off:off + 256])
                    for head, keys in [('fitness', ['fit_mu']), ('gap', ['gap_mu']), ('beat10', ['b10'])]:
                        if head in spec['active_heads'] and any(not torch.isfinite(o[h]).all() for h in keys):
                            raise RuntimeError('nonfinite active prediction')
                    values.append(np.column_stack([o[h].float().cpu().numpy() for h in ('fit_mu', 'fit_ls', 'gap_mu', 'gap_ls', 'b10')]))
                return mask_outputs(np.concatenate(values), spec)
            @torch.no_grad()
            def evaluate(m, idx, epoch):
                m.eval(); pred = predict(m, idx)
                met = metrics(pred, {h: yy[h][idx] for h in yy}, gid[idx], spec)
                # Full panel objective aggregates fixed target support, independent rank RNG.
                eval_rng = torch.Generator(device=dev); eval_rng.manual_seed(1000003 + seed)
                sums = dict.fromkeys(('fitness', 'gap', 'beat10', 'rank'), 0.)
                supports = dict.fromkeys(('fitness', 'gap', 'beat10'), 0)
                weighted_sums = dict.fromkeys(('fitness', 'gap', 'beat10'), 0.) if gw is not None and np.isin(idx, fit).all() else None
                for off in range(0, len(idx), 256):
                    ids = idx[off:off + 256]; ii = torch.as_tensor(ids, device=dev)
                    o = forward(m, ids)
                    _, parts, sup = objective(o, {h: gy[h][ii] for h in gy}, spec, generator=eval_rng)
                    if weighted_sums is not None:
                        _, weighted_parts, _ = objective(o, {h: gy[h][ii] for h in gy}, spec,
                                                         weights={h: gw[h][ii] for h in gw}, generator=eval_rng)
                        for h in weighted_sums: weighted_sums[h] += float(weighted_parts[h]) * sup[h]
                    for h in supports: sums[h] += float(parts[h]) * sup[h]; supports[h] += sup[h]
                    sums['rank'] += float(parts['rank']) * len(ids)
                terms = {h: sums[h] / max(1, supports[h]) for h in supports}
                terms['rank'] = sums['rank'] / len(idx)
                met['objective'] = sum(spec['head_weights'][h] * terms[h] for h in terms)
                if not np.isfinite(met['objective']): raise RuntimeError('nonfinite evaluation objective')
                met['objective_parts'] = terms
                if weighted_sums is not None:
                    weighted_parts = {h: weighted_sums[h] / max(1, supports[h]) for h in weighted_sums}
                    met['fixed_lineage_weighted_objective'] = sum(spec['head_weights'][h] * weighted_parts[h] for h in weighted_parts)
                    met['fixed_lineage_weighted_parts'] = weighted_parts
                return met, pred
            hist, best_state, best_ep, best_crit, bad, step = [], None, -1, -float('inf'), 0, 0
            for ep in range(epochs):
                model.train(); perm = rng.permutation(fit); seen = []; sums = dict.fromkeys(('fitness', 'gap', 'beat10', 'rank'), 0.)
                counts = dict.fromkeys(('fitness', 'gap', 'beat10'), 0)
                before = model.h.o.weight.detach().clone()
                for off in range(0, len(perm), config['bs']):
                    ids = perm[off:off + config['bs']]; ii = torch.as_tensor(ids, device=dev)
                    o = forward(model, ids)
                    loss, parts, sup = objective(o, {h: gy[h][ii] for h in gy}, spec, weights=None if gw is None else {h: gw[h][ii] for h in gw})
                    if not torch.isfinite(loss): raise RuntimeError('nonfinite training objective')
                    opt.zero_grad(set_to_none=True); loss.backward()
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    opt.step(); sched.step(); ema.update_parameters(model); step += 1; seen.extend(ids.tolist())
                    for h in counts: sums[h] += float(parts[h].detach()) * sup[h]; counts[h] += sup[h]
                    sums['rank'] += float(parts['rank'].detach()) * len(ids)
                    batches.write(json.dumps({'r': r, 'k': k, 'ep': ep, 'step': step, 'rows': len(ids),
                        'loss': float(loss.detach()), 'parts': {h: float(v.detach()) for h, v in parts.items()}, 'supports': sup,
                        'lr': sched.get_last_lr()[0], 'gn': float(gn), 'cumulative_rows': ep * len(fit) + len(seen), 'at': time.time()}, allow_nan=False) + '\n')
                proof = exposure(seen, fit)
                changed = not torch.equal(before, model.h.o.weight)
                if ep == 0 and not changed: raise RuntimeError('optimizer did not change parameters')
                proof.update(r=r, k=k, ep=ep, steps=step, changed_parameters=changed,
                             valid_supports=counts, fit_row_sha256=row_hash(fit), va_row_sha256=row_hash(va), te_row_sha256=row_hash(te))
                ledger.write(json.dumps(proof) + '\n')
                vm, _ = evaluate(ema.module, va, ep)
                tm, _ = evaluate(ema.module, panel, ep) if ep % 5 == 0 else (None, None)
                if spec['checkpoint'] == 'lineage_macro_fitness_mse': crit = -vm['fitness_macro_mse']
                else: crit = vm['fitness_r2'] + (vm['beat10_auc'] or 0.)
                improved = np.isfinite(crit) and crit > best_crit + spec['min_delta']
                if improved:
                    best_crit, best_ep, bad = crit, ep, 0
                    best_state = {h: p.detach().cpu().clone() for h, p in ema.module.state_dict().items()}
                else: bad += 1
                optimizer_terms = {h: sums[h] / max(1, counts[h]) for h in counts}
                optimizer_terms['rank'] = sums['rank'] / len(fit)
                optimizer_loss = sum(spec['head_weights'][h] * optimizer_terms[h] for h in optimizer_terms)
                hist.append({'ep': ep, 'train_loss': optimizer_loss, 'val_loss': vm['objective'],
                    'train_eval': tm, 'val_eval': vm, 'val_r2': vm['fitness_r2'], 'val_auc': vm['beat10_auc'],
                    'train_eval_fitness_mse': None if tm is None else tm['fitness_mse'], 'val_eval_fitness_mse': vm['fitness_mse'],
                    'val_macro_fitness_mse': vm['fitness_macro_mse'], 'checkpoint': spec['checkpoint'],
                    'best_ep': best_ep, 'best_val_crit': best_crit, 'checkpoint_improved': bool(improved),
                    'optimizer_parts': optimizer_terms, 'optimizer_supports': counts, 'lr': sched.get_last_lr()[0],
                    'gn': float(gn), 'exposure': proof, 'step': step, 'at': time.time()})
                atomic_json(out / f'hist_r{r}f{k}.json', hist)
                if ep % 10 == 0: log('wave3', arm['id'], f'r{r}f{k} ep{ep} steps{step} rows{len(seen)} trainEMA={None if tm is None else tm["fitness_mse"]} valEMA={vm["fitness_mse"]:.5f} macro={vm["fitness_macro_mse"]:.5f}')
                if bad >= config['patience']: break
            if best_state is None: raise RuntimeError('no finite checkpoint selected')
            ema.module.load_state_dict(best_state)
            tm, _ = evaluate(ema.module, fit, best_ep); vm, _ = evaluate(ema.module, va, best_ep)
            p = predict(ema.module, te); testm = metrics(p, {h: yy[h][te] for h in yy}, gid[te], spec)
            oof[te] = p; raw[te, 0] = p[:, 0] * norm['fitness'][1] + norm['fitness'][0]
            if 'gap' in spec['active_heads']: raw[te, 1] = p[:, 2] * norm['gap'][1] + norm['gap'][0]
            ck = {'schema': 'jope.wave3-checkpoint/v1', 'state_dict': best_state, 'arch': a.arch, 'kw': spec['kw'],
                  'dL': xl.shape[1], 'dims': [z.shape[1] for z in xe], 'blocks': blocks, 'L_cols': meta['L_cols'],
                  'preprocessor': state, 'target_norm': norm, 'configuration': spec, 'fold': [r, k],
                  'run_id': a.run_id, 'git_sha': a.git_sha, 'bundle_sha256': summary['data_manifest_sha256']}
            buf = io.BytesIO(); torch.save(ck, buf); body = buf.getvalue()
            loaded = torch.load(io.BytesIO(body), map_location='cpu', weights_only=False)
            checkrows = te[:32]
            pr = infer_checkpoint(loaded, l[checkrows], {b: e[b][checkrows] for b in blocks},
                                  np.asarray(meta['tabdpt_overlap'])[checkrows], dev)
            expected = predict(ema.module, checkrows); finite = np.isfinite(expected)
            error = float(np.abs(pr[finite] - expected[finite]).max())
            if not np.array_equal(np.isfinite(pr), finite) or not np.allclose(pr[finite], expected[finite], atol=2e-5, rtol=2e-4):
                raise RuntimeError('feature-only checkpoint roundtrip mismatch: ' + str(error))
            rec = {'r': r, 'k': k, 'n_tr': len(fit), 'n_va': len(va), 'n_te': len(te), 'best_ep': best_ep,
                   'epochs_run': len(hist), 'best_val_crit': best_crit, 'checkpoint': spec['checkpoint'],
                   'train_eval_best': tm, 'val_eval_best': vm, 'test_r2': testm['fitness_r2'], 'test_auc': testm['beat10_auc'],
                   'secs': time.time() - started, 'target_norm': norm, 'ckpt_sha256': sha256b(body),
                   'preprocessor_roundtrip': {'rows': len(checkrows), 'max_abs_error': error, 'passed': True},
                   'diagnostic_panel': {'rows': len(panel), 'lineages': len(np.unique(gid[panel])), 'row_sha256': row_hash(panel)},
                   'exposure': {'epochs': len(hist), 'rows_per_epoch': len(fit), 'all_rows_once_per_epoch': True,
                                'total_rows': len(fit) * len(hist), 'fit_sha256': row_hash(fit), 'va_sha256': row_hash(va), 'te_sha256': row_hash(te)},
                   'topk_features': state.get('selected_names'), 'active_heads': spec['active_heads']}
            path = out / f'model_r{r}f{k}.pt'
            with path.open('xb') as f: f.write(body)
            if a.s3:
                base = f'{PFX}models/hwm-dlwave/{a.run_id}/r{r}f{k}/'
                rec['s3_ckpt'] = put_create_only(base + 'model.pt', body)
                summary['s3'].append(rec['s3_ckpt'])
                summary['s3'].append(put_create_only(base + 'fold_manifest.json', json.dumps(clean_json(rec), sort_keys=True, allow_nan=False).encode()))
            summary['folds'][f'r{r}f{k}'] = rec
            atomic_npy(out / f'oof_r{r}.npy', oof); atomic_npy(out / f'oofraw_r{r}.npy', raw)
            atomic_json(out / 'summary.json', summary)
            log('DONE wave3', arm['id'], f'r{r}f{k} best_ep{best_ep} R2{testm["fitness_r2"]} AUC{testm["beat10_auc"]} {rec["secs"]:.0f}s')
            del features, xl, ml, xe, me, model, ema, opt, sched, gy, gw, ck, loaded, best_state
            if dev == 'cuda': torch.cuda.empty_cache()
        # Fitness/gap standardization varies by fold: report native OOF scores.
        native = oof.copy(); native[:, 0] = raw[:, 0]; native[:, 2] = raw[:, 1]
        mm = metrics(native, targets, gid, spec)
        summary[f'oof_r{r}'] = {'fit_r2': mm['fitness_r2'], 'gap_r2': mm['gap_r2'], 'b10_auc': mm['beat10_auc']}
        atomic_json(out / 'summary.json', summary)
    summary['state'] = 'all_done'; summary['completed_at'] = time.time()
    if a.s3:
        summary['s3'].append(put_create_only(f'{PFX}models/hwm-dlwave/{a.run_id}/run_manifest.json', json.dumps(clean_json(summary), sort_keys=True, allow_nan=False).encode()))
    atomic_json(out / 'summary.json', summary); log('ALL DONE wave3', a.run_id)


if __name__ == '__main__': main()
