#!/usr/bin/env python3
"""F11 read-out: the first monitored learning checkpoint after a hold release decides the rollback clause.

Reads the controller journal (learning/journal.jsonl) for the acknowledgement event named by the decision receipt and
the first diagnostic_accepted / diagnostic_rejected event after it, loads that monitor receipt (parallel.frozen_monitoring.v1)
and the guard state, and writes an immutable parallel.hold_release_readout.v1 under successor/decisions/ that says, per
F11: `continue` (accepted: the loop runs on under sampler A) or `rollback_required` (rejected: training must stop through
a revision with automatic_training false and re-parent to the last diagnostic_accepted checkpoint named here). Nothing here
changes any state; root implements the revision.
"""
import argparse
import json
from pathlib import Path

import promotion_gate
from resume_expired_cycle import read, require, sha

SCHEMA = 'parallel.hold_release_readout.v1'
DECISION_SCHEMA = 'parallel.hold_release_decision.v1'


def journal_events(path):
    with open(path, 'rb') as stream:
        for raw in stream:
            try:
                yield json.loads(raw)
            except ValueError:
                continue


def readout(root, run, decision_path):
    decision = read(decision_path)
    require(decision.get('schema') == DECISION_SCHEMA, 'decision receipt schema')
    nonce = 'hold-release-' + sha(decision_path)[:16]
    ack = None
    verdict = None
    last_accepted = None
    for event in journal_events(run / 'learning/journal.jsonl'):
        kind = event.get('event')
        if ack is None:
            if kind == 'diagnostic_accepted':
                last_accepted = dict(checkpoint_sha256=event.get('checkpoint_sha256'), receipt=event.get('receipt'), timestamp=event.get('timestamp'))
            if kind == 'diagnostic_hold_acknowledged' and (event.get('acknowledgement') or {}).get('nonce') == nonce:
                ack = event
            continue
        if kind in ('diagnostic_accepted', 'diagnostic_rejected'):
            verdict = event
            break
    require(ack is not None, 'no diagnostic_hold_acknowledged event with nonce %s' % nonce)
    if verdict is None:
        return dict(schema=SCHEMA, status='pending', decision=dict(path=str(decision_path), sha256=sha(decision_path)), acknowledgement=ack,
                    last_accepted_before_release=last_accepted, final_tests_opened=False)
    receipt_path = root / verdict['receipt'] if isinstance(verdict.get('receipt'), str) else None
    receipt = read(receipt_path) if receipt_path and receipt_path.is_file() else None
    state = read(run / 'learning/state.json')
    guard = (state or {}).get('training_diagnostic') or {}
    outcome = 'continue' if verdict['event'] == 'diagnostic_accepted' else 'rollback_required'
    return dict(schema=SCHEMA, status='decided', outcome=outcome, rule='F11: first monitored checkpoint after release accepted -> continue under sampler A; rejected -> stop via automatic_training:false and re-parent to the last diagnostic_accepted checkpoint',
                decision=dict(path=str(decision_path), sha256=sha(decision_path)), acknowledgement=dict(timestamp=ack.get('timestamp'), nonce=nonce),
                verdict=dict(event=verdict['event'], timestamp=verdict.get('timestamp'), checkpoint_sha256=verdict.get('checkpoint_sha256'), receipt=verdict.get('receipt'),
                             receipt_sha256=sha(receipt_path) if receipt_path and receipt_path.is_file() else None),
                monitor={k: (receipt or {}).get(k) for k in ('evaluated_step', 'sequence_nll', 'ranking_bt', 'ranking_listwise', 'runtime_huber')} if receipt else None,
                guard=dict(baseline_ranking_bt=guard.get('baseline_ranking_bt'), ranking_bt=guard.get('ranking_bt'), consecutive_rejections=guard.get('consecutive_rejections'), held=guard.get('held'), last_step=guard.get('last_step')),
                reparent_to=last_accepted if outcome == 'rollback_required' else None, final_tests_opened=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--run', default='runs/eight-a100-recovery-20260912')
    parser.add_argument('--decision', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='immutable receipt path (written only when decided)')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    result = readout(root, root / args.run, args.decision)
    if result['status'] == 'decided' and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = dict(result, receipt_sha256=promotion_gate.immutable(args.output, result))
    print(json.dumps(result, sort_keys=True))
    return result


if __name__ == '__main__':
    main()
