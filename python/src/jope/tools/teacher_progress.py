#!/usr/bin/env python3
"""Publish compact durable teacher progress; never read observation ledgers."""
import argparse
import hashlib
import json
import time
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), sort_keys=True).encode()).hexdigest()


def publish(root):
    path = root / 'runs/teacher-progress.json'
    old = read(path) if path.exists() else {}
    files = list((root / 'runs/champions').glob('*/current.json'))
    entries = [read(p) for p in files]
    identities, cuts, improved_cuts, improvements = set(), set(), set(), []
    for file, c in zip(files, entries):
        assert c['schema'] == 'parallel.cut_champion.v1'
        identity = c['identity']
        # Observation values were validated and fsynced by the Rust writer.
        for field in ['dataset', 'family', 'partition', 'cut', 'prepared_sha256',
                      'training_sha256', 'validation_sha256', 'evaluator_sha256', 'vocabulary_sha256']:
            assert c['observation'][field] == identity[field]
        identities.add(tuple(identity[k] for k in sorted(identity)))
        cuts.add(tuple(identity[k] for k in ['dataset', 'cut', 'prepared_sha256', 'training_sha256', 'validation_sha256']))
        previous = c.get('previous_rmse_original_units')
        if previous is None:
            continue
        assert c['rmse_original_units'] < previous
        improved_cuts.add(tuple(identity[k] for k in ['dataset', 'cut', 'prepared_sha256', 'training_sha256', 'validation_sha256']))
        output = root / c['source']
        progress_path = output.parent / 'search.json'
        if not progress_path.exists():
            progress_path = output.parent / 'progress.json'
        progress = read(progress_path) if progress_path.exists() else {}
        seed_path = output.parent.parent / 'seeds.json'
        checkpoint = None
        if seed_path.exists() and hashlib.sha256(seed_path.read_bytes()).hexdigest() == c['observation'].get('seeds_sha256'):
            checkpoint = read(seed_path).get('checkpoint_sha256')
        trajectory, event = [], c
        for _ in range(16):
            assert event['identity'] == identity
            trajectory.append({'timestamp': event['timestamp'], 'rmse': event['rmse_original_units'],
                               'evaluations': event['observation'].get('evaluation_index', -1) + 1,
                               'wall_seconds': event['observation'].get('completed_search_seconds'),
                               'source': event['source'], 'evaluation_count_scope': 'within_source_search_at_discovery'})
            previous_event = event.get('previous_event_sha256')
            if not previous_event:
                break
            event = read(file.parent / f'event-{previous_event}.json')
        trajectory.reverse()
        assert all(a['rmse'] > b['rmse'] for a,b in zip(trajectory, trajectory[1:]))
        improvements.append({k: identity[k] for k in ['dataset', 'family', 'cut']} | {
            'old_rmse': previous, 'new_rmse': c['rmse_original_units'],
            'relative_improvement': (previous-c['rmse_original_units'])/previous if previous > 0 else None,
            'evaluations': progress.get('completed_evaluations', progress.get('evaluations')),
            'wall_seconds': progress.get('wall_seconds'), 'source_checkpoint': checkpoint,
            'source_checkpoint_status': 'verified_seed_receipt' if checkpoint else 'unavailable_or_generic_seed',
            'identity_sha256': file.parent.name, 'evaluator_sha256': identity['evaluator_sha256'],
            'trajectory': trajectory,
            'evaluation_bias': 'adaptive_cut_heldout_teacher'})
    improvements.sort(key=lambda c: (-c['relative_improvement'], c['dataset'], c['cut']))
    new_bests = sum(c['improvement_number'] for c in entries)
    gp_path = root / 'runs/gp-service.json'
    gp = read(gp_path) if gp_path.exists() else {}
    history = old.get('history', [])
    row = {'timestamp': int(time.time()), 'best_count': len(cuts), 'new_bests': new_bests}
    if not history or (history[-1]['best_count'], history[-1]['new_bests']) != (len(cuts), new_bests):
        history.append(row)
    result = {'schema': 'parallel.teacher_progress.v1', 'timestamp': int(time.time()),
              'cuts_with_teacher': len(cuts), 'evaluator_cut_identities': len(identities),
              'new_bests': new_bests, 'matched_cut_improvements': len(improved_cuts),
              'total_evaluations': gp.get('completed_evaluations'),
              'running_evaluations': gp.get('running_evaluations'),
              'evaluation_count_scope': 'entire active GP service; not restricted to champion-bearing searches',
              'history': history[-1000:], 'top_improvements': improvements[:20],
              'champion_reuse': {'supported': True,
                  'status': 'durable incumbent store populated' if entries else 'awaiting completed-observation import',
                  'limitations': 'Best found under recorded budgets, not a global optimum. Both proposal arms share a frozen snapshot. Programs from older evaluator versions are always refitted; scores never cross cuts or evaluator identities.'},
              'qualification_only': True, 'evaluation_bias': 'adaptive_cut_heldout_teacher',
              'observation_ledgers_read': False, 'original_final_test_targets_opened': False}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(result, indent=2) + '\n')
    temporary.replace(path)
    return {'cuts_with_teacher': len(cuts), 'new_bests': new_bests, 'receipt': str(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--interval', type=int, default=0)
    args = ap.parse_args()
    assert args.interval == 0 or args.interval >= 5
    while True:
        print(json.dumps(publish(args.root)), flush=True)
        if not args.interval:
            return
        time.sleep(args.interval)


if __name__ == '__main__':
    main()
