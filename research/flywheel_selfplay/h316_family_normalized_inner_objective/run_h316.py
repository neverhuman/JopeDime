"""H316: paired TRAIN-only BASE regression, then frozen-member NORM objective.

PREREG.md defines the in-sample inner-TRAIN normalization interpretation.
The imported H303/H302/H308 sources are byte pinned; no dev91 scoring.
"""
from __future__ import annotations
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CUSTODY = Path(os.environ.get('H316_CUSTODY', str(HERE.parent)))
FS = Path(os.environ.get('H316_FS', str(CUSTODY / 'vendor')))
DATA = Path(os.environ.get('H316_DATA', '/home/ubuntu/jope-data/h316-solhwv'))
PARTS = ('P0', 'P1', 'P2')
SEEDS = (30001, 30002, 30003)
R = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def durable(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.new')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def outer_fold(part, family, a302):
    if part == 'P0':
        return a302.outer_fold_of_family(family)
    salt = {'P1': b'H308-outer-v1\x00', 'P2': b'H308-outer-v2\x00'}[part]
    return hashlib.sha256(salt + family.encode()).digest()[-1] % 4


def normalized_fold(picks, cids, packets, families):
    """One fold's inner-TRAIN gains, scales and q10/q90 clipping only."""
    values = {}
    for c in cids:
        values.setdefault(families[c], []).append(float(packets[c]['utility_k'][0]))
    scales = {}
    for family, u in values.items():
        u = np.asarray(u, dtype=np.float64)
        scales[family] = max(float(np.median(np.abs(u - np.median(u)))), 1e-3)
    gain = np.array([(float(packets[c]['utility_k'][int(j)]) - float(packets[c]['utility_k'][0])) /
                     scales[families[c]] for c, j in zip(cids, picks)], dtype=np.float64)
    lo, hi = np.quantile(gain, [0.1, 0.9])
    return np.clip(gain, lo, hi), dict(q10=float(lo), q90=float(hi), scales=scales)


def norm_objective(records):
    by_family = {}
    for family, gain in records:
        by_family.setdefault(family, []).append(float(gain))
    return float(np.median([np.mean(v) for v in by_family.values()]))


def choose(table, key):
    best = max(x[key] for x in table)
    tied = [x for x in table if x[key] >= best - 1e-4]
    return sorted(tied, key=lambda x: (x['inner_switches'], -x['k'], -x['a'], x['tau']))[0], len(tied)


def cached_members(*args):
    path = Path(args[0]) / 'members.json'
    if path.exists():
        return json.loads(path.read_text())
    members = R['fit_original'](*args)
    durable(path, members)
    return members


def prepare():
    receipt = json.loads((CUSTODY / 'prereg-published.json').read_text())
    assert receipt['published']
    assert sha(HERE / 'PREREG.md') == json.loads((HERE / 'manifest.json').read_text())['prereg_sha256']
    manifest = json.loads((HERE / 'manifest.json').read_text())
    for rel, digest in manifest['sources'].items():
        assert sha(FS / rel) == digest, rel
    sys.path.insert(0, str(FS / 'h303_ensvote2_fully_nested_hparams'))
    sys.path.insert(1, str(FS / 'h308_ensvote2_trainonly_repartition'))
    import run_h303 as r303
    import trainonly_loader as tl
    selected = json.loads(Path('/home/ubuntu/jope-data/h50/input/selected.json').read_text())
    r303.np = tl.TrainOnlyNumpy({r['context_sha256'] for r in selected['train']})
    r303.SCRATCH = DATA / 'setup'
    r303.setup()
    r303.np = np
    assert tl.COUNTERS['dlval_feature_rows_read'] == 0
    assert all(r.get('has_meta_test') is not True and r['task'] == 'regression' for r in r303.G['train_rows'])
    R.update(r303=r303, loader=tl, loader_counters=dict(tl.COUNTERS), fit_original=r303.fit_members)
    r303.fit_members = cached_members
    return r303


def set_part(r303, part):
    G = r303.G
    G['fam_to_fold'] = {f: outer_fold(part, f, G['a302']) for f in set(G['fam_by_cid'].values())}


def task(args):
    phase, part, k, seed = args
    r303 = R['r303']
    G = r303.G
    set_part(r303, part)
    tdir = DATA / part / f'fold_{k}' / f'root_{seed}'
    normlock = tdir / 'norm_fit_lock.json'
    if phase == 'norm' and normlock.exists():
        return {'task': args, 'existing_lock': sha(normlock)}
    full = G['P8']
    forbidden = {r['context_sha256'] for r in G['train_rows'] if G['fam_to_fold'][r['family_id']] == k}
    guarded = R['loader'].GuardedPackets(full, forbidden)
    G['P8'] = guarded
    start = time.time()
    try:
        if phase == 'base':
            r303.SCRATCH = DATA / part
            lock = r303.task((k, seed))
            durable(tdir / 'base_guard.json', dict(outer_test_packet_accesses_before_lock=0,
                    outer_test_packets_built_before_lock=len(forbidden), guard_denials=guarded.denied,
                    fit_lock_sha256=sha(tdir / 'fit_lock.json')))
            from archive_h316 import compress_inner
            compress_inner(tdir, durable)
            result = dict(chosen=lock['chosen'])
        else:
            assert json.loads((DATA / 'regression.json').read_text())['status'] == 'PASS'
            from archive_h316 import restore_inner, remove_restored_inner
            restore_inner(tdir)
            rows = [r for r in G['train_rows'] if G['fam_to_fold'][r['family_id']] != k]
            inner = {r['family_id']: r303.inner_fold(k, r['family_id']) for r in rows}
            keys = [(t, kk, a) for t in r303.TAU_GRID for kk in r303.K_GRID for a in r303.A_GRID]
            records = {key: [] for key in keys}
            switches = {key: 0 for key in keys}
            normlog = {}
            for tau in r303.TAU_GRID:
                for i in range(4):
                    train = [r for r in rows if inner[r['family_id']] != i]
                    cids = [r['context_sha256'] for r in train]
                    idir = tdir / f'inner_tau{tau}_fold{i}'
                    assert (idir / 'members.json').exists(), 'BASE members must exist; no hidden refit'
                    members = json.loads((idir / 'members.json').read_text())
                    outs = r303.score_members(idir, members, np.concatenate([guarded[c]['X'] for c in cids]), len(cids), G['ens'])
                    for kk in r303.K_GRID:
                        ml = [r303.member_picks(o, tau, kk, G['veto'], G['core'], G['bagged']) for o in outs]
                        for a in r303.A_GRID:
                            key = (tau, kk, a)
                            picks = r303.ens_picks(ml, a, G['ens'])
                            gain, log = normalized_fold(picks, cids, guarded, G['fam_by_cid'])
                            records[key].extend((G['fam_by_cid'][c], v) for c, v in zip(cids, gain))
                            switches[key] += sum(j != 0 for j in picks)
                            normlog[f'{tau}|{kk}|{a}|{i}'] = log
            table = [dict(tau=t, k=kk, a=a, inner_obj=norm_objective(records[(t, kk, a)]),
                          inner_switches=switches[(t, kk, a)]) for t, kk, a in keys]
            chosen, n_tied = choose(table, 'inner_obj')
            base = json.loads((tdir / 'fit_lock.json').read_text())
            refitdir = tdir / ('refit' if chosen['tau'] == base['chosen']['tau'] else f"norm_refit_tau{chosen['tau']}")
            members = base['members'] if refitdir.name == 'refit' else cached_members(refitdir, k, seed, rows, guarded, G['fam_by_cid'], chosen['tau'])
            lock = dict(schema='jope.h316.fit-lock/v1', partition=part, outer_fold=k, root_seed=seed,
                        chosen=chosen, n_tied=n_tied, inner_table=table, normalization=normlog,
                        members=members, refit_dir=refitdir.name,
                        base_fit_lock_sha256=sha(tdir / 'fit_lock.json'),
                        outer_test_packet_accesses_before_lock=0, guard_denials=guarded.denied,
                        outer_test_packets_built_before_lock=len(forbidden), elapsed_sec=time.time()-start)
            assert guarded.denied == 0
            durable(normlock, lock)
            remove_restored_inner(tdir)
            result = dict(chosen=chosen, n_tied=n_tied)
        assert guarded.denied == 0
    finally:
        G['P8'] = full
    out = dict(task=args, seconds=round(time.time()-start, 2), **result)
    print(json.dumps(out), flush=True)
    return out


def score(part, arm):
    r303 = R['r303']; G = r303.G
    set_part(r303, part)
    cids = [r['context_sha256'] for r in G['train_rows']]
    picks = np.full((3, len(cids)), -1, dtype=np.int32)
    hashes = {}
    for k in range(4):
        test = [i for i,c in enumerate(cids) if G['fam_to_fold'][G['fam_by_cid'][c]] == k]
        for s, seed in enumerate(SEEDS):
            tdir = DATA / part / f'fold_{k}' / f'root_{seed}'
            lp = tdir / ('fit_lock.json' if arm == 'BASE' else 'norm_fit_lock.json')
            lock = json.loads(lp.read_text()) # lock verified BEFORE outer packet access
            hashes[f'{k}:{seed}'] = sha(lp)
            marker = tdir / (arm.lower() + '_score_once.json')
            assert not marker.exists(), 'outer fold already scored; replay stored picks instead'
            durable(marker, dict(lock_sha256=sha(lp), state='STARTED'))
            X = np.concatenate([G['P8'][cids[i]]['X'] for i in test])
            refit = 'refit' if arm == 'BASE' else lock['refit_dir']
            outs = r303.score_members(tdir/refit, lock['members'], X, len(test), G['ens'])
            ch = lock['chosen']
            ml = [r303.member_picks(o, ch['tau'], ch['k'], G['veto'], G['core'], G['bagged']) for o in outs]
            p = r303.ens_picks(ml, ch['a'], G['ens'])
            picks[s,test] = p
            durable(marker, dict(lock_sha256=sha(lp), state='COMPLETE', contexts=len(test),
                                 picks_sha256=hashlib.sha256(np.asarray(p,dtype=np.int32).tobytes()).hexdigest()))
    assert (picks >= 0).all()
    np.save(DATA / f'{arm}_{part}.npy', picks)
    durable(DATA/f'{arm}_{part}_locks.json', hashes)
    return picks


def main():
    import evaluate_h316 as ev
    assert os.uname().nodename.split('.')[0] == 'xbabe2' and os.getpriority(os.PRIO_PROCESS,0) == 19
    r303 = prepare()
    tasks = [(p,k,s) for p in PARTS for k in range(4) for s in SEEDS]
    with ProcessPoolExecutor(max_workers=6) as pool:
        list(pool.map(task, [('base',*a) for a in tasks]))
    base = {p: score(p,'BASE') for p in PARTS}
    ev.regress(r303,base,DATA)
    with ProcessPoolExecutor(max_workers=6) as pool:
        list(pool.map(task, [('norm',*a) for a in tasks]))
    norm = {p: score(p,'NORM') for p in PARTS}
    ev.evaluate(r303,base,norm,DATA)

if __name__ == '__main__':
    main()
