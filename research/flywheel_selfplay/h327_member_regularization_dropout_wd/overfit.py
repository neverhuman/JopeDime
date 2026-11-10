"""Post-fit TRAIN-only final-member resubstitution versus selected inner-OOF gap."""
import math
import numpy as np


def family_gain(cids, picks, packets, families):
    values = {}
    for c, pick in zip(cids, picks):
        values.setdefault(families[c], []).append(
            float(packets[c]['utility_k'][int(pick)]) - float(packets[c]['utility_k'][0]))
    return math.fsum(math.fsum(x) / len(x) for x in values.values()) / len(values)


def measure(directory, member, cids, packets, families, G):
    import lightgbm as lgb
    import hashlib
    rank_path = directory / member['ranker']['file']
    gate_path = directory / member['veto']['file']
    for p, record in [(rank_path, member['ranker']), (gate_path, member['veto'])]:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == record['sha256']
    X = np.asarray(np.concatenate([packets[c]['X'] for c in cids]), dtype=np.float32)
    ranker = lgb.Booster(model_file=str(rank_path))
    gate = lgb.Booster(model_file=str(gate_path))
    raw = ranker.predict(X, num_iteration=int(member['selected_n_estimators']), num_threads=1).reshape(len(cids), 8)
    inv = gate.predict(X, num_threads=1).reshape(len(cids), 8)
    masked = G['veto'].mask(raw, inv, tau=float(member['selected_tau']))
    # The receipt's inner gain uses raw_margin before strict admission.
    picks = np.array([G['core'].select_from_top4_scores(x, float(member['raw_margin'])) for x in masked], dtype=np.int32)
    train = family_gain(cids, picks, packets, families)
    inner = float(member['inner_fit_family_gain'])
    return dict(train_gain=train, inner_val_gain=inner, gap=train-inner,
                member_seed=member['seed'], member_index=member['member_index'],
                contexts=len(cids), families=len(set(families[c] for c in cids)),
                raw_margin=member['raw_margin'], tau=member['selected_tau'],
                selected_n=member['selected_n_estimators'], selected_lr=member['selected_lr'],
                ranker_sha256=member['ranker']['sha256'], veto_sha256=member['veto']['sha256'],
                picks_sha256=hashlib.sha256(picks.tobytes()).hexdigest()), picks


def summarize(rows):
    assert len(rows) == 216
    result = dict(status='PASS', definition='Final member raw-margin train resubstitution minus selected inner-OOF family-equal gain; no bag/ensemble gap, selected OOF optimistic.', arms={}, cells=[], members=rows)
    for arm in ['BASE', 'REG']:
        group = [r for r in rows if r['arm'] == arm]
        assert len(group) == 108
        result['arms'][arm] = {how+'_'+key: float(fn([r[key] for r in group]))
                              for how, fn in [('mean', np.mean), ('median', np.median)]
                              for key in ['train_gain', 'inner_val_gain', 'gap']}
        for part in ['P0', 'P1', 'P2']:
            for seed in [30001, 30002, 30003]:
                cell = [r for r in group if r['partition'] == part and r['root_seed'] == seed]
                assert len(cell) == 12
                result['cells'].append(dict(arm=arm, partition=part, root_seed=seed,
                    **{key: float(np.mean([r[key] for r in cell])) for key in ['train_gain', 'inner_val_gain', 'gap']}))
    pairs = {}
    for row in rows:
        key = (row['partition'], row['fold'], row['root_seed'], row['member_index'])
        pairs.setdefault(key, {})[row['arm']] = row['gap']
    diff = [x['REG']-x['BASE'] for x in pairs.values()]
    assert len(diff) == 108
    result['paired_gap_change'] = dict(mean=float(np.mean(diff)), median=float(np.median(diff)), members=108)
    return result
