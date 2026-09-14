#!/usr/bin/env python3
"""Derive one population arm's replay policy from the base block policy (PREREGISTRATION `population` block).

Arms (pre-registered): 0 fresh_balanced/carry/s0 (incumbent), 1 fresh_balanced/carry/s1 (null pair -> eta), 2 fresh_balanced/reset/s0,
3 broad/carry/s0, 4 broad/reset/s0, 5 fresh_balanced/carry/s2, 6 fresh_balanced/carry/s3 (until the concentration sampler is
generalised), 7 fresh_balanced/reset/s1 (reserved for B1). Seed rule: s_k = u64_le(sha256(base_policy_sha256 ':' arm_index ':' k)[:8]).
Only `sampling_seed`, `optimizer` and `replay` may differ from the base policy; membership (fresh_groups, fresh_records, corpus and
vocabulary digests) is byte-identical, and the receipt proves it. The trainer reads `sampling_seed` (source-v3 training.rs:429-432),
so arms 0/1/5/6 need no trainer rebuild; `replay: broad` arms depend on Codex's sampler support and are emitted with a flag.

  population_arm_policy.py --base <policy.json> --arm K --output policy-arm-K.json [--receipt PATH]
"""
import argparse
import hashlib
import json
from pathlib import Path

import promotion_gate
from resume_expired_cycle import atomic, read, require, sha

SCHEMA = 'parallel.population_arm_policy.v1'
ARMS = {0: dict(replay='fresh_balanced', optimizer='carry', seed=0, role='incumbent'),
        1: dict(replay='fresh_balanced', optimizer='carry', seed=1, role='null pair (eta)'),
        2: dict(replay='fresh_balanced', optimizer='reset', seed=0, role='restart-from-base'),
        3: dict(replay='broad', optimizer='carry', seed=0, role='variant'),
        4: dict(replay='broad', optimizer='reset', seed=0, role='variant'),
        5: dict(replay='fresh_balanced', optimizer='carry', seed=2, role='second replicate'),
        6: dict(replay='fresh_balanced', optimizer='carry', seed=3, role='sampler B placeholder until the concentration sampler is generalised'),
        7: dict(replay='fresh_balanced', optimizer='reset', seed=1, role='reserved for B1 (after Tier-B F4)')}
CHANGEABLE = ('sampling_seed', 'optimizer', 'replay')
MEMBERSHIP = ('fresh_groups', 'fresh_records', 'corpus_sha256', 'vocabulary_sha256', 'baseline_corpus_sha256', 'parent_checkpoint_sha256')


def arm_seed(base_sha256, arm, k):
    return int.from_bytes(hashlib.sha256(('%s:%d:%d' % (base_sha256, arm, k)).encode()).digest()[:8], 'little')


def derive(base_path, arm):
    require(arm in ARMS, 'unknown arm %r' % arm)
    base = read(base_path)
    require(base.get('schema') == 'parallel.replay_policy.v2', 'base policy schema')
    require(all(k in base for k in MEMBERSHIP + CHANGEABLE), 'base policy lacks membership or sampling keys')
    base_sha = sha(base_path)
    spec = ARMS[arm]
    policy = dict(base)
    policy['sampling_seed'] = arm_seed(base_sha, arm, spec['seed'])
    policy['optimizer'] = spec['optimizer']
    policy['replay'] = spec['replay']
    unchanged = {k: v for k, v in base.items() if k not in CHANGEABLE}
    derived_rest = {k: v for k, v in policy.items() if k not in CHANGEABLE}
    require(json.dumps(unchanged, sort_keys=True) == json.dumps(derived_rest, sort_keys=True), 'membership changed')
    membership_sha = hashlib.sha256(json.dumps({k: base[k] for k in MEMBERSHIP}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    receipt = dict(schema=SCHEMA, arm=arm, role=spec['role'], base_policy=dict(path=str(base_path), sha256=base_sha), seed_index=spec['seed'], sampling_seed=policy['sampling_seed'],
                   seed_rule="u64_le(sha256(base_policy_sha256 ':' arm_index ':' k)[:8])", changed={k: dict(base=base[k], arm=policy[k]) for k in CHANGEABLE if base[k] != policy[k]},
                   membership_sha256=membership_sha, membership_identical=True, needs_sampler_support=spec['replay'] != base['replay'], final_tests_opened=False)
    return policy, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--arm', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args(argv)
    policy, receipt = derive(args.base, args.arm)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic(args.output, policy)
    receipt['policy'] = dict(path=str(args.output), sha256=sha(args.output))
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt['receipt_sha256'] = promotion_gate.immutable(args.receipt, receipt)
    print(json.dumps({k: receipt[k] for k in ('arm', 'role', 'sampling_seed', 'changed', 'membership_sha256', 'needs_sampler_support', 'policy')}, sort_keys=True))
    return receipt


if __name__ == '__main__':
    main()
