"""Fail closed on incomplete cached inner policies; never fit, select, or score."""
from runtime import *
from selection import require_grid_cache, CacheCoverageError


def main():
    assert json.loads((P / 'source/regression_gate.json').read_text())['status'] == 'PASS'
    G = initialize()
    configs = [(t,k,a) for t in r303.TAU_GRID for k in r303.K_GRID for a in r303.A_GRID]
    cells = []
    for part in PARTS:
        for k in range(4):
            expected = [r['context_sha256'] for r in G['train_rows'] if fold(part,r['family_id']) != k]
            for seed in SEEDS:
                d = D / 'references' / part / f'fold_{k}' / f'root_{seed}'
                lock = json.loads((d / 'fit_lock.json').read_text())
                p = d / 'inner_selected_oof.npz'
                with np.load(p, allow_pickle=False) as z:
                    keys = sorted(z.files)
                    cids = z['cids'].tolist()
                    picks = z['picks']
                    shape = list(picks.shape)
                assert cids == expected
                assert picks.shape == (len(cids),)
                archive = json.loads((d / 'model_archive.json').read_text())
                inner_names = [n for n in archive['members'] if n.startswith('inner_tau')]
                metadata = [n for n in inner_names if not n.endswith('.txt')]
                assert len(inner_names) == 360 and not metadata
                try:
                    require_grid_cache(picks,cids,expected,configs,configs)
                except CacheCoverageError as error:
                    reason = str(error)
                else:
                    raise RuntimeError('Unexpected complete grid; do not label it incomplete')
                cells.append(dict(partition=part,fold=k,root_seed=seed,cache_keys=keys,cache_shape=shape,
                                  cache_sha256=sha(p),available_config=lock['chosen'],available_configs=1,
                                  required_configs=18,missing_configs=17,reason=reason,
                                  archived_inner_model_files=len(inner_names),inner_member_policy_metadata_files=metadata,
                                  source_model_archive=dict(bucket=archive['bucket'],key=archive['key'],version_id=archive['version_id'],sha256=archive['sha256']),
                                  allowed_contexts=len(expected),context_closure_exact=True))
    receipt = dict(wave_id='H326',status='INCOMPLETE_INPUT_COVERAGE',reason='MISSING_GRID_OOF_AND_INNER_MEMBER_POLICY_METADATA',
                   cells=cells,required_config_cells=648,available_config_cells=36,missing_config_cells=612,
                   member_refits=0,selection_calls=0,bsel_fit_locks=0,bsel_outer_score_calls=0,
                   no_partial_scoring=True,baseline_regression='PASS',
                   disposition='No unapproved inner refit, fabricated abstain count, or scientific NOT_ADVANCED verdict.',
                   finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    durable(P / 'source/cache_coverage.json',receipt)
    durable(D / 'cache_coverage.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='cells'}),flush=True)
    raise SystemExit(2)


if __name__ == '__main__':
    main()
