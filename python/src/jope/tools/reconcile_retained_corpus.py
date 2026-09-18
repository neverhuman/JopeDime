#!/usr/bin/env python3
"""Reconcile exact retained ledgers and old label identities after recovery packing."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from resume_expired_cycle import atomic, read, require, sha


def labels(corpus, keys):
    index = read(corpus / 'index.json')
    contexts = {i: (c['dataset'], c['cut']) for i, c in enumerate(index['contexts'])
                if (c['dataset'], c['cut']) in keys}
    result = {}
    with (corpus / 'labels.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['context'] in contexts:
                key = (*contexts[row['context']], hashlib.sha256(row['program'].encode()).hexdigest())
                result[key] = row['rmse_cut_normalized']
    return result


def reconcile(root, audit_path, clean, projected, recovered, contexts, output):
    audit = read(audit_path)
    require(len(audit['sources']) == 437, 'retained ledger inventory changed')
    provenance = read(recovered / 'provenance.json')
    source_map = {row['path']: row for row in provenance['sources']}
    checks = []
    for row in audit['sources']:
        actual = source_map.get(row['path'])
        require(actual is not None and actual['sha256'] == row['sha256']
                and actual['observations'] == row['observations'], 'retained ledger missing from packed provenance: ' + row['path'])
        require(sha(root / row['path']) == row['sha256'], 'retained ledger bytes changed')
        require(sha((root / row['path']).with_name('search.json')) == row['search_sha256'], 'retained search receipt changed')
        checks.append(dict(path=row['path'], sha256=row['sha256'], observations=row['observations']))
    keys = {(g['dataset'], g['cut']) for g in audit['group_summary']['group_differences'] if g['delta'] < 0}
    old = labels(clean, keys)
    projection = labels(projected, keys)
    missing = {k: v for k, v in old.items() if k not in projection}
    require(len(missing) == 65294, 'projected old-label deficit differs from audited 65,294')
    new = labels(recovered, keys)
    absent = [k for k in missing if k not in new]
    rescored = [dict(dataset=k[0], cut=k[1], program_sha256=k[2], old=missing[k], new=new[k])
                for k in missing if k in new and missing[k] != new[k]]
    index = read(recovered / 'index.json')
    cohort = read(contexts)
    eligible = {(c['dataset'], c['cut']): c for c in cohort['encodings'] if c['partition'] in ('meta_train', 'meta_validation')}
    for group in index['groups']:
        c = index['contexts'][group['context']]
        bound = eligible.get((c['dataset'], c['cut']))
        require(bound and all(c[k] == bound[k] for k in ('family', 'partition', 'prepared', 'prepared_sha256', 'sha256')),
                'withdrawn, out-of-cohort or sealed supervision entered recovery')
    result = dict(schema='parallel.retained_corpus_reconciliation.v1', timestamp=int(time.time()),
         passed=not absent, corpus_sha256=sha(recovered / 'index.json'),
         audit_sha256=sha(audit_path), contexts_sha256=sha(contexts),
         retained_ledgers_verified=len(checks), retained_observations=sum(x['observations'] for x in checks),
         older_labels_missing_from_projection=len(missing), recovered_old_labels=len(missing)-len(absent),
         missing=absent, retained_identity_with_different_score=rescored, sources=checks,
         excluded_supervision_verified=True, final_tests_opened=False)
    atomic(output, result)
    require(result['passed'], 'recovery corpus did not retain every audited old label')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','missing','retained_identity_with_different_score')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'audit', 'clean', 'projected', 'recovered', 'contexts', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    reconcile(args.root, args.audit, args.clean, args.projected, args.recovered, args.contexts, args.output)
