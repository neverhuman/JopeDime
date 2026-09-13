#!/usr/bin/env python3
"""Freeze fresh-group replay and lineage-dependent sampling from verified corpora."""
import argparse
import hashlib
import mmap
import json
import math
import teaching_manifest
from pathlib import Path

from promotion_gate import canonical, immutable
from resume_expired_cycle import atomic, read, require, sha


def groups(corpus):
    index = read(corpus/'index.json')
    if index.get('schema') == 'parallel.teaching_shards.v1':
        return sharded_groups(corpus)
    provenance = read(corpus/'provenance.json')
    require(sha(corpus/'index.json') == provenance['index_sha256'], 'corpus index hash changed')
    require(sha(corpus/'labels.jsonl') == provenance['labels_sha256'], 'original labels hash changed')
    require(index['record_bytes'] == 532, 'unknown packed record layout')
    require(sha(corpus/'records.bin') == index['records_sha256'], 'corpus record hash changed')
    result = {}
    # Discoveries come from original observations, never moving percentiles,
    # normalization, grammar IDs, runtime noise or a format-only rebuild.
    labels = {}
    with (corpus/'labels.jsonl').open() as stream:
        for line in stream:
            label = json.loads(line)
            record = label['record']
            require(record not in labels, 'duplicate label record')
            labels[record] = label
    with (corpus/'records.bin').open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as records:
        require(len(records) == index['records'] * index['record_bytes'], 'truncated records')
        for number, group in enumerate(index['groups']):
            require(group['partition'] in ('meta_train','meta_validation'), 'sealed group in teaching corpus')
            context = index['contexts'][group['context']]
            key = (context['prepared_sha256'], context['cut'], group['family'], group['partition'])
            require(key not in result and context['partition'] == group['partition']
                    and context['family'] == group['family'], 'duplicate or mismatched group identity')
            h = hashlib.sha256()
            programs = {}
            for record in group['records']:
                require(type(record) is int and 0 <= record < index['records'], 'invalid record offset')
                start = record*index['record_bytes']
                # Token IDs and numeric targets are semantic. Legal-set IDs and
                # context positions may change when unrelated groups are added.
                h.update(records[start:start+256])
                h.update(records[start+512:start+514])
                h.update(records[start+520:start+532])
                label = labels[record]
                require(label['context'] == group['context'], 'label context mismatch')
                program = label['program']
                require(program not in programs, 'duplicate program in teaching group')
                score = label['rmse_cut_normalized']
                require(type(score) in (int, float) and math.isfinite(score) and score >= 0,
                        'invalid original-precision label')
                programs[program] = dict(record=record, rmse=score)
            result[key] = dict(index=number, digest=h.hexdigest(), records=len(group['records']), programs=programs)
    return index, result


def sharded_groups(corpus):
    index,segments=teaching_manifest.load(teaching_manifest.workspace(corpus),corpus)
    result={}
    for segment,group in zip(segments,index['groups']):
        context=index['contexts'][group['context']]
        key=(context['prepared_sha256'],context['cut'],group['family'],group['partition'])
        require(key not in result,'duplicate teaching group identity')
        programs={};h=hashlib.sha256();count=0
        with (segment['directory']/'records.bin').open('rb') as binary, (segment['directory']/'labels.jsonl').open() as labels:
            for local,line in enumerate(labels):
                row=json.loads(line);record=binary.read(532)
                require(len(record)==532 and row['record']==local and row['context']==group['context'],
                        'sharded label coordinates changed')
                h.update(record[:256]);h.update(record[512:514]);h.update(record[520:532])
                score=row['rmse_cut_normalized'];program=row['program']
                require(program not in programs and type(score) in (int,float) and math.isfinite(score) and score>=0,
                        'invalid or duplicate original-precision label')
                programs[program]=dict(record=segment['offset']+local,rmse=score)
                count+=1
            require(count==segment['records'] and not binary.read(1),'sharded label count changed')
        result[key]=dict(index=segment['group'],digest=h.hexdigest(),records=count,programs=programs)
    return index,result


def teacher_beating_records(current_programs, previous_programs):
    """Programs that beat the previous group's best RMSE, or every program on a new group."""
    if not previous_programs:
        return sorted(p['record'] for p in current_programs.values())
    best = min(p['rmse'] for p in previous_programs.values())
    return sorted(p['record'] for p in current_programs.values() if p['rmse'] < best)


def freeze(corpus, baseline, checkpoint, replay, optimizer, corpora=None, block_identity='0',
           discoveries='any_new_or_improved_program'):
    require(discoveries in ('any_new_or_improved_program', 'teacher_improved'),
            'unknown discovery rule')
    (current, current_groups), (before, prior_groups) = corpora or (groups(corpus), groups(baseline))
    require(current['vocabulary_sha256'] == before['vocabulary_sha256']
            and current['context_index_sha256'] == before['context_index_sha256'], 'replay corpus cohort/vocabulary changed')
    state = read(checkpoint/'state.json')
    require(state['vocabulary_sha256'] == current['vocabulary_sha256'], 'checkpoint vocabulary changed')
    fresh_records = {}
    for key, row in current_groups.items():
        if key[3] != 'meta_train' or row['records'] < 8:
            continue
        previous = prior_groups.get(key, {}).get('programs', {})
        if discoveries == 'teacher_improved':
            added = teacher_beating_records(row['programs'], previous)
        else:
            added = [p['record'] for program, p in row['programs'].items()
                     if program not in previous or p['rmse'] < previous[program]['rmse']]
        if added:
            fresh_records[str(row['index'])] = sorted(added)
    fresh = sorted(map(int, fresh_records))
    lineage = dict(corpus_sha256=sha(corpus/'index.json'), parent_checkpoint_sha256=sha(checkpoint/'state.json'),
                   baseline_corpus_sha256=sha(baseline/'index.json'),
                   replay=replay, sampling_version='lineage_fresh_program_v2', block_identity=str(block_identity))
    if discoveries == 'teacher_improved':
        lineage['discoveries'] = discoveries
        lineage['sampling_version'] = 'lineage_teacher_improved_v1'
    seed = int.from_bytes(hashlib.sha256(canonical(lineage)).digest()[:8], 'little')
    blocked = []
    if optimizer == 'carry':
        for name,key in [('optimizer.ot','optimizer_sha256'), ('optimizer.json','optimizer_metadata_sha256')]:
            if not (checkpoint/name).is_file() or sha(checkpoint/name) != state.get(key):
                blocked.append('missing or incompatible optimizer state: '+name)
        if not state.get('ranking_backward') or not state.get('runtime_objective'):
            blocked.append('optimizer objective identity absent from parent checkpoint')
    return dict(schema='parallel.replay_policy.v2', **lineage,
        vocabulary_sha256=current['vocabulary_sha256'], fresh_records=fresh_records,
        sampling_seed=seed, fresh_groups=sorted(fresh), optimizer=optimizer,
        fresh_fraction=.5 if replay == 'fresh_balanced' and fresh else 0,
        fallback='no fresh training groups; broad replay' if replay == 'fresh_balanced' and not fresh else None,
        fresh_training_groups=len(fresh), historical_training_groups=sum(k[3] == 'meta_train' for k in current_groups),
        eligible=not blocked, blocked_reasons=blocked, final_tests_opened=False)


def materialize_view(corpus, policy, dest):
    """Reject retired group cloning; native replay preserves real family identity.

    Pass the frozen policy directly to replay_training via --replay-policy.
    Manufacturing corpus families cannot implement scientific sampling weights.
    """
    raise RuntimeError('legacy replay group cloning is retired; use native --replay-policy')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('corpus','baseline','checkpoint','output'):
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--replay', choices=['broad','fresh_balanced'], required=True)
    p.add_argument('--optimizer', choices=['reset','carry'], required=True)
    p.add_argument('--block-identity', required=True)
    p.add_argument('--discoveries', choices=['any_new_or_improved_program', 'teacher_improved'],
                   default='any_new_or_improved_program')
    args = p.parse_args()
    immutable(args.output, freeze(args.corpus, args.baseline, args.checkpoint, args.replay, args.optimizer,
                                 block_identity=args.block_identity, discoveries=args.discoveries))
