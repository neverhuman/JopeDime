"""Once-only SOFTVOTE evaluation after all 36 independently audited score locks."""
from runtime import *
import math
import softvote


def score_member_outputs(G, directory, member, X, n, chosen, temperatures):
    with softvote.temperatures(temperatures) as trace:
        out = G['ens'].score_single_member_from_disk(directory, member, X, n, G['ens'].K)
        out['_strict_margin'] = math.inf if member['strict_margin'] == 'inf' else float(member['strict_margin'])
        picks = r303.member_picks(out, chosen['tau'], chosen['k'], G['veto'], G['core'], G['bagged'])
    return out, np.asarray(picks, dtype=np.int32), trace


def main():
    assert json.loads((D / 'calibration_done.json').read_text())['status'] == 'PASS'
    assert json.loads((P / 'source/lock_review.json').read_text())['status'] == 'PASS'
    G = initialize()
    treatment = softvote.install(G['bagged'])
    assert treatment == json.loads((P / 'source/vote_transform.json').read_text())
    rows = G['train_rows']
    cids = [r['context_sha256'] for r in rows]
    families = np.array([r['family_id'] for r in rows])
    allcids = set(cids)
    index = {c: i for i, c in enumerate(cids)}
    locks = {}
    for part in PARTS:
        for k in range(4):
            for seed in SEEDS:
                td = D / part / f'fold_{k}' / f'root_{seed}'
                lock = json.loads((td / 'score_lock_h337.json').read_text())
                assert lock['outer_test_rows_read_before_lock'] == lock['outer_test_packets_built_before_lock'] == lock['denied_reads'] == 0
                for name, digest in lock['source_hashes'].items():
                    assert sha(P / 'source' / name) == digest, name
                assert sha(td / 'calibration_deltas.npz') == lock['calibration_sha256']
                ref = D / 'references' / part / f'fold_{k}' / f'root_{seed}'
                assert sha(ref / 'fit_lock.json') == lock['base_fit_lock_sha256']
                for name, digest in lock['model_sha256'].items():
                    assert sha(ref / 'refit' / name) == digest
                locks[f'{part}/{k}/{seed}'] = dict(sha256=sha(td / 'score_lock_h337.json'), models=30)
    durable(P / 'source/lock_receipt.json', dict(status='PASS', count=36, models=1080, locks=locks,
                                               outer_rows_read_before_lock=0, model_fits=0))
    with np.load(D / 'references/out/panel.npz', allow_pickle=False) as z:
        assert z['cids'].tolist() == cids and z['families'].tolist() == families.tolist()
        utility, invalid, picks = z['u'].copy(), z['invalid'].copy(), z['picks'].copy()
        utility[:, :, 2] = np.nan
        picks[:, :, 1] = -1
    receipts = []
    for pi, part in enumerate(PARTS):
        for k in range(4):
            test = [r for r in rows if fold(part, r['family_id']) == k]
            tc = [r['context_sha256'] for r in test]
            ix = [index[c] for c in tc]
            for seed in SEEDS:
                td = D / part / f'fold_{k}' / f'root_{seed}'
                assert not (td / 'outer_once.json').exists()
                durable(td / 'outer_once.json', dict(state='STARTED', score_lock_sha256=locks[f'{part}/{k}/{seed}']['sha256']))
            scope = Scope(tc, allcids)
            sys.addaudithook(scope.hook)
            try:
                packet = packets(test, scope)
                X = np.asarray(np.concatenate([packet[c]['X'] for c in tc]), dtype=np.float32)
                for si, seed in enumerate(SEEDS):
                    td = D / part / f'fold_{k}' / f'root_{seed}'
                    lock = json.loads((td / 'score_lock_h337.json').read_text())
                    directory = D / 'references' / part / f'fold_{k}' / f'root_{seed}' / 'refit'
                    member_picks = []
                    raw_hashes = {}
                    for member in lock['members']:
                        mi = member['member_index']
                        out, mp, trace = score_member_outputs(G, directory, member, X, len(tc), lock['soft_chosen'], lock['temperatures'][mi])
                        raw_path = td / f'outer_raw_m{mi}.npz'
                        np.savez_compressed(raw_path, cids=np.array(tc), raw_scores=out['raw_scores'],
                                            p_inv=out['p_inv'], bag_scores=out['bag_scores'],
                                            member_picks=mp, soft_sums=np.asarray(trace, dtype=np.float64))
                        raw_hashes[raw_path.name] = sha(raw_path)
                        member_picks.append(mp.tolist())
                    result = np.asarray(r303.ens_picks(member_picks, lock['soft_chosen']['a'], G['ens']), dtype=np.int32)
                    cell = pi * 3 + si
                    for n, c in enumerate(tc):
                        j = ix[n]
                        assert utility[cell, j, 0] == packet[c]['utility_k'][0]
                        assert bool(invalid[cell, j, 0]) == (not bool(packet[c]['gate_k'][0]))
                        utility[cell, j, 2] = packet[c]['utility_k'][int(result[n])]
                        invalid[cell, j, 2] = not bool(packet[c]['gate_k'][int(result[n])])
                    picks[cell, ix, 1] = result
                    durable(td / 'outer_once.json', dict(state='COMPLETE', model_scores=1, contexts=len(tc),
                                                        score_lock_sha256=locks[f'{part}/{k}/{seed}']['sha256'],
                                                        softvote_pick_sha256=hashlib.sha256(result.tobytes()).hexdigest(),
                                                        raw_outputs=raw_hashes))
                receipts.append(dict(partition=part, fold=k, contexts=tc, ranges=scope.ranges,
                                     packet_hashes=scope.packet_hashes, bytes=scope.bytes))
            finally:
                scope.active = False
    assert np.isfinite(utility).all() and (picks >= 0).all()
    with np.load(D / 'references/regression_picks.npz', allow_pickle=False) as z:
        assert all(np.array_equal(picks[i * 3:i * 3 + 3, :, 0], z[part]) for i, part in enumerate(PARTS))
    (D / 'out').mkdir(exist_ok=False)
    np.savez_compressed(D / 'out/panel.npz', u=utility, invalid=invalid, picks=picks, cids=np.array(cids), families=families)
    durable(D / 'out/outer_read_receipts.json', receipts)
    durable(P / 'source/score_receipt.json', dict(status='PASS', outer_model_score_calls=36, score_locks=36,
                                                final_models_unchanged=1080, model_fits=0, dev91_rows_read=0,
                                                panel_sha256=sha(D / 'out/panel.npz'), r3175_train='NOT_MEASURED'))
    print('PASS: 36 once-only SOFTVOTE scores with all BASE models unchanged')


if __name__ == '__main__':
    main()
