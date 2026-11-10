"""Temperatures from archived inner-heldout bag margins; never fit a model."""
from runtime import *
import gc
import io
import math
import tarfile
import time

import boto3
import lightgbm as lgb
from softvote import temperature_from_deltas, transform
import inspect


def extract_inner(cell, base_lock, index, s3):
    ref = D / 'references' / cell
    receipt = json.loads((ref / 'model_archive.json').read_text())
    assert sha(ref / 'model_archive.json') == index['receipt_sha256']
    for key in ['bucket', 'key', 'version_id', 'sha256', 'bytes']:
        assert receipt[key] == index[key]
    assert receipt['members']['fit_lock.json'] == sha(ref / 'fit_lock.json')
    for member in base_lock['members']:
        for model in [member['ranker'], member['veto'], *member['bags']]:
            assert receipt['members']['refit/' + model['file']] == model['sha256']
    reply = s3.get_object(Bucket=index['bucket'], Key=index['key'], VersionId=index['version_id'])
    assert reply['VersionId'] == index['version_id']
    raw = reply['Body'].read()
    reply['Body'].close()
    assert len(raw) == index['bytes'] and hashlib.sha256(raw).hexdigest() == index['sha256']
    tau = base_lock['chosen']['tau']
    wanted = {f'inner_tau{tau}_fold{i}/{bag["file"]}'
              for i in range(4) for member in base_lock['members'] for bag in member['bags']}
    assert len(wanted) == 96 and wanted <= receipt['members'].keys()
    work = D / 'inner_work'
    work.mkdir(exist_ok=False)
    original_identity = None
    with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as tf:
        assert sorted(tf.getnames()) == sorted(receipt['members'])
        for member in tf:
            assert member.isfile() and not member.issym()
            assert not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
            content = tf.extractfile(member).read()
            assert hashlib.sha256(content).hexdigest() == receipt['members'][member.name]
            if member.name == 'input_identity.json':
                original_identity = json.loads(content)
            if member.name in wanted:
                path = work / member.name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                path.chmod(0o444)
    assert len(list(work.glob('*/*.txt'))) == 96
    assert original_identity is not None
    return work, receipt, original_identity, dict(cell=cell, **{k: index[k] for k in ['bucket', 'key', 'version_id', 'sha256', 'bytes']},
                               exact_version_get_verified=True, all_members_verified=len(receipt['members']),
                               extracted_members={name: receipt['members'][name] for name in sorted(wanted)})


def main():
    assert json.loads((P / 'source/regression_gate.json').read_text())['status'] == 'PASS'
    assert json.loads((P / 'implementation-published.json').read_text())['published']
    G = initialize()
    _, treatment = transform(inspect.getsource(G['bagged'].vote_counts))
    durable(P / 'source/vote_transform.json', treatment)
    rows = G['train_rows']
    allcids = {r['context_sha256'] for r in rows}
    refhash = json.loads((D / 'references/regression_input_reads.json').read_text())['packet_hashes']
    indices = {x['cell']: x for x in json.loads((P / 'source/base_model_archive_index.json').read_text())['archives']}
    session = boto3.Session(profile_name='veox-jepson-prod')
    assert session.client('sts').get_caller_identity()['Account'] == '714044927654'
    s3 = session.client('s3', region_name='us-east-2')
    source_hashes = json.loads(Path(os.environ['H337_SOURCE_RECEIPT']).read_text())['source_sha256']
    results = []
    for part in PARTS:
        for k in range(4):
            train = [r for r in rows if fold(part, r['family_id']) != k]
            cids = [r['context_sha256'] for r in train]
            allowed = set(cids)
            inner = np.array([r303.inner_fold(k, r['family_id']) for r in train], dtype=np.int8)
            scope = Scope(allowed, allcids, fit=True)
            sys.addaudithook(scope.hook)
            packet = {}
            try:
                packet = packets(train, scope)
                assert scope.packet_hashes == {c: refhash[c] for c in cids}
                for seed in SEEDS:
                    started = time.time()
                    cell = f'{part}/fold_{k}/root_{seed}'
                    td = D / cell
                    td.mkdir(parents=True, exist_ok=False)
                    ref = D / 'references' / cell
                    base = json.loads((ref / 'fit_lock.json').read_text())
                    assert base['outer_test_rows_read_before_lock'] == 0
                    model_hashes = {}
                    for member in base['members']:
                        assert len(member['bags']) == 8
                        for model in [member['ranker'], member['veto'], *member['bags']]:
                            path = ref / 'refit' / model['file']
                            assert sha(path) == model['sha256']
                            model_hashes[model['file']] = model['sha256']
                    assert len(model_hashes) == 30
                    work, old_receipt, original_identity, recovery = extract_inner(cell, base, indices[cell], s3)
                    assert set(original_identity['allowed_contexts']) == allowed
                    assert original_identity['packet_hashes'] == scope.packet_hashes
                    delta = np.full((3, 8, len(train), 7), np.nan, dtype=np.float64)
                    scored_models = []
                    for hold in range(4):
                        ids = np.flatnonzero(inner == hold)
                        assert len(ids) > 0
                        X = np.asarray(np.concatenate([packet[cids[i]]['X'] for i in ids]), dtype=np.float32)
                        for member in base['members']:
                            mi = member['member_index']
                            for bag in member['bags']:
                                bi = bag['bag_index']
                                name = f'inner_tau{base["chosen"]["tau"]}_fold{hold}/{bag["file"]}'
                                path = work / name
                                assert sha(path) == old_receipt['members'][name]
                                model = lgb.Booster(model_file=str(path))
                                assert model.num_feature() == 568
                                prediction = np.asarray(model.predict(X, num_iteration=-1, num_threads=1), dtype=np.float64).reshape(len(ids), 8)
                                assert np.isfinite(prediction).all()
                                delta[mi, bi, ids] = prediction[:, 1:] - prediction[:, :1]
                                scored_models.append(dict(inner_fold=hold, member_index=mi, bag_index=bi,
                                                          path=name, sha256=old_receipt['members'][name],
                                                          native_tree_count=model.num_trees(), contexts=len(ids)))
                                del model
                    assert len(scored_models) == 96 and np.isfinite(delta).all()
                    temps = np.array([[temperature_from_deltas(delta[m, b]) for b in range(8)] for m in range(3)])
                    np.savez_compressed(td / 'calibration_deltas.npz', cids=np.array(cids), inner_folds=inner, deltas=delta, temperatures=temps)
                    durable(td / 'input_identity.json', dict(allowed_contexts=cids, forbidden_contexts=sorted(allcids - allowed),
                                                           packet_hashes=scope.packet_hashes, ranges=scope.ranges))
                    durable(td / 'inner_recovery.json', dict(**recovery, scored_models=scored_models))
                    assert scope.contexts == allowed and scope.denied == 0
                    lock = dict(schema='jope.h337.score-lock/v1', partition=part, outer_fold=k, root_seed=seed,
                                aggregator='sigmoid_margin_vote_sum', bag_count=8, temperatures=temps.tolist(),
                                calibration_population='selected-TAU inner-heldout, all j1..7 candidates, ddof0',
                                candidate_count_per_temperature=len(train) * 7, soft_chosen=base['chosen'],
                                members=base['members'], base_fit_lock_sha256=sha(ref / 'fit_lock.json'),
                                model_sha256=model_hashes, calibration_sha256=sha(td / 'calibration_deltas.npz'),
                                input_identity_sha256=sha(td / 'input_identity.json'), recovery_sha256=sha(td / 'inner_recovery.json'),
                                source_hashes=source_hashes, treatment_ast_sha256=treatment['treatment_ast_sha256'],
                                outer_test_rows_read_before_lock=0, outer_test_packets_built_before_lock=0,
                                denied_reads=0, model_fits=0,
                                finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
                    durable(td / 'score_lock_h337.json', lock)
                    for path in work.glob('*/*.txt'):
                        name = str(path.relative_to(work))
                        assert sha(path) == old_receipt['members'][name]
                        path.unlink()
                    for directory in work.iterdir():
                        directory.rmdir()
                    work.rmdir()
                    result = dict(cell=cell, lock_sha256=sha(td / 'score_lock_h337.json'), temperatures=24,
                                  inner_models=96, seconds=time.time() - started, extraction_reclaimed=True)
                    results.append(result)
                    print(json.dumps(result), flush=True)
                    del delta, temps
                    gc.collect()
            finally:
                scope.active = False
                del packet
                gc.collect()
    assert len(results) == 36
    durable(D / 'calibration_done.json', dict(status='PASS', cells=36, temperatures=864, inner_models=3456,
                                             model_fits=0, outer_rows_before_lock=0, results=results))
    print('PASS: 36 score locks, 864 temperatures, 3456 archived inner bag predictions; zero refits', flush=True)


if __name__ == '__main__':
    main()
