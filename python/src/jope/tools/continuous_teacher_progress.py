#!/usr/bin/env python3
"""Current collection gains from committed continuous-generation quality receipts."""
import argparse
import heapq
import os
import threading
import time
from collections import Counter
from pathlib import Path

from continuous_quality_report import search_row
from continuous_status import better, finite, summarize_quality, within
from promotion_gate import immutable
from resume_expired_cycle import atomic, read, require, sha


def summarize(reports, now):
    champions, events, sources = {}, [], []
    latest_source = None
    for generation, report, jobs in sorted(reports, key=lambda item: item[1]['timestamp']):
        if report['schema'] == 'parallel.continuous_search_quality.v1':
            require(len(report['rows']) == len(jobs['jobs']) == 1, 'per-search report cardinality')
            row, job = report['rows'][0], jobs['jobs'][0]
            require(row['output'] == job['output'] and row['rng_seed'] == job['rng_seed']
                    and row['partition'] in ('meta_train', 'meta_validation')
                    and row['units'] == 'original_target', 'per-search quality identity changed')
            require(all(row[k] is None or finite(row[k]) for k in
                ('best_final_gp', 'best_retained_teacher', 'best_initial_neural_seed')), 'invalid per-search metric')
        else:
            summarize_quality(report, jobs)
        sources.append(dict(generation=generation['number'], jobs_sha256=report['jobs_sha256']))
        latest_source = max(latest_source or 0, report['timestamp'])
        grouped = {}
        for row in report['rows']:
            # Legacy reports lack evaluator/cut bytes. Keep them separate, never
            # merge their scores with fully bound current evaluation identities.
            key = tuple(row.get(k) for k in ('dataset', 'family', 'partition', 'cut',
                'prepared_sha256', 'training_sha256', 'validation_sha256', 'evaluator_sha256', 'vocabulary_sha256'))
            grouped.setdefault(key, []).append(row)
        for key, rows in grouped.items():
            previous = champions.get(key)
            teachers = [r['best_retained_teacher'] for r in rows if finite(r['best_retained_teacher'])]
            if teachers:
                previous = min([previous, *teachers] if previous is not None else teachers)
            eligible = [r for r in rows if finite(r['best_final_gp'])]
            if not eligible:
                continue
            best = min(r['best_final_gp'] for r in eligible)
            if previous is not None and better(best, previous):
                winners = [r for r in eligible if not better(best, r['best_final_gp'])]
                origins = sorted({r.get('best_final_origin', 'unavailable') for r in winners})
                events.append(dict(generation=generation['number'], dataset=key[0], family=key[1],
                    partition=key[2], cut=key[3], old_rmse=previous, new_rmse=best,
                    relative_improvement=(previous - best) / previous,
                    origin=origins[0] if len(origins) == 1 else 'shared_or_multiple_origins',
                    arms=sorted(r['arm'] for r in winners), checkpoint=generation['checkpoint'],
                    timestamp=max(r.get('completed_at', report['timestamp']) for r in winners),
                    identity_status='verified' if all(v is not None for v in key) else 'legacy_partial',
                    search_receipts=[dict(output=r['output'], sha256=r['search_sha256']) for r in winners]))
            champions[key] = min(best, previous) if previous is not None else best
    last = max((e['timestamp'] for e in events), default=None)
    return dict(schema='parallel.continuous_teacher_progress.v1', timestamp=now,
        sources=sources, reported_generations=len({g['number'] for g,_,_ in reports}), source_timestamp=latest_source,
        source_age_seconds=None if latest_source is None else max(0, now-latest_source),
        new_champions=len(events), unique_improved_families=len({e['family'] for e in events}),
        unique_improved_cuts=len({(e['dataset'], e['cut']) for e in events}),
        cuts_with_teacher=len({key[:4] for key in champions}), evaluator_cut_identities=len(champions),
        new_bests=len(events), matched_cut_improvements=len({(e['dataset'], e['cut']) for e in events}),
        top_improvements=sorted(events,key=lambda e:-e['relative_improvement'])[:20],
        partitions=dict(Counter(e['partition'] for e in events)),
        champion_reuse=dict(supported=True,status='continuous generation receipts; retained teachers remain separate'),
        evaluation_count_scope='continuous-generation improvements only; no historical service totals',
        discovery_origins=dict(Counter(e['origin'] for e in events)), improvements=events,
        last_meaningful_improvement_at=last,
        seconds_since_meaningful_improvement=None if last is None else max(0, now-last),
        scope='Incremental continuous collection gains against retained/prior teachers; adaptive, not transfer qualification.',
        independent_evaluation=False, final_tests_opened=False)


def publish(root, manifest_path, now=None, status=None):
    now = time.time() if now is None else now
    manifest = read(manifest_path)
    state = read(within(root, manifest['output']) / 'state.json')
    require(state['manifest_sha256'] == sha(manifest_path), 'teacher manifest changed')
    reports, overdue = [], []
    status = {} if status is None else status
    status.update(phase='inventory',processed_searches=0,pending_searches=None,
                  inventory_at=now,oldest_pending_completed_at=None)
    pending=[]
    for generation in state['generations']:
        if generation['stage'] != 4:continue
        directory=within(root,generation['path'])
        jobs_sha=sha(directory/'jobs.json')
        for job in read(directory/'jobs.json')['jobs']:
            source=within(root,job['output'])/'search.json'
            if source.is_file() and not (directory/'search-reports'/(sha(source)+'-'+jobs_sha+'.json')).is_file():
                pending.append((job['output'],source.stat().st_mtime))
    outstanding=dict(pending)
    ages=[(timestamp,path) for path,timestamp in pending];heapq.heapify(ages)
    status.update(phase='reporting',pending_searches=len(outstanding),
                  oldest_pending_completed_at=min(outstanding.values(),default=None))
    first_seen = []
    for generation in state['generations']:
        directory = within(root, generation['path'])
        if generation['stage'] != 4:
            continue
        jobs_sha = sha(directory / 'jobs.json')
        for job in read(directory / 'jobs.json')['jobs']:
            source = within(root, job['output'])
            if not (source / 'search.json').is_file():
                continue
            search_sha, observation_sha = sha(source / 'search.json'), sha(source / 'observations.json')
            saved = directory / 'search-reports' / (search_sha + '-' + jobs_sha + '.json')
            if saved.is_file():
                row = read(saved)
                require(row['search_sha256'] == search_sha and row['observation_sha256'] == observation_sha,
                        'published teacher ledger changed')
            else:
                row = search_row(root, directory, job)
                immutable(saved, row)
                first_seen.append(max(0, now-row['completed_at']))
            report = dict(schema='parallel.continuous_search_quality.v1', timestamp=row['completed_at'],
                          jobs_sha256=jobs_sha, rows=[row])
            reports.append((generation, report, dict(jobs=[job])))
            outstanding.pop(job['output'],None)
            while ages and ages[0][1] not in outstanding:heapq.heappop(ages)
            status.update(processed_searches=len(reports),pending_searches=len(outstanding),
                          oldest_pending_completed_at=ages[0][0] if ages else None)
    value = summarize(reports, now)
    value.update(manifest_sha256=sha(manifest_path), source_stale=False, unreported_completed=overdue,
                 reported_searches=len(reports), reporter_heartbeat_at=now,
                 unprocessed_receipt_age_seconds=0,
                 oldest_new_receipt_lag_seconds=max(first_seen, default=0),
                 receipt_inventory_at=now, pending_inventory_count=len(outstanding),
                 unprocessed_age_scope='receipts found in the last completed inventory',
                 report_protocol='per_search_v2')
    atomic(root / 'runs/teacher-progress.json', value)
    status.update(phase='waiting',last_publication_at=time.time())
    return value


def heartbeat(root,status,stop):
    while not stop.is_set():
        now=time.time();row=dict(status)
        oldest=row.get('oldest_pending_completed_at')
        atomic(root/'runs/teacher-reporter.json',dict(row,schema='parallel.teacher_reporter.v1',
            timestamp=now,pid=os.getpid(),unprocessed_receipt_age_seconds=None if oldest is None else max(0,now-oldest)))
        stop.wait(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--interval', type=int, default=0)
    args = parser.parse_args()
    require(args.interval == 0 or args.interval >= 5, 'report interval must be at least five seconds')
    status={};stop=threading.Event()
    reporter=threading.Thread(target=heartbeat,args=(args.root.resolve(),status,stop),daemon=True)
    reporter.start()
    try:
        while True:
            publish(args.root.resolve(), args.manifest,status=status)
            if not args.interval:break
            time.sleep(args.interval)
    finally:
        stop.set();reporter.join()
