"""Versioned flywheel identities and measured, same-task comparison rules.

Pure functions only: this module does not execute programs or publish HWM updates.
It is an additional experimental protocol, not a rewrite of historical S1 q labels.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re

SCHEMA = 'jope.flywheel-comparison/v1'
HEX = re.compile(r'[0-9a-f]{64}\Z')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def sha256(value):
    if not isinstance(value, str) or not HEX.fullmatch(value):
        raise ValueError('a full lowercase SHA256 is required')
    return value


def finite(value, *, nonnegative=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (nonnegative and value < 0):
        raise ValueError('finite numeric value required')
    return float(value)


@dataclass(frozen=True)
class Context:
    task: str
    dataset_sha256: str
    target_sha256: str
    cut_id: str
    evaluator_sha256: str
    vocabulary_sha256: str
    protocol_sha256: str
    split_sha256: str
    metric: str
    units: str
    direction: str

    def __post_init__(self):
        for name in ('dataset_sha256', 'target_sha256', 'evaluator_sha256', 'vocabulary_sha256',
                     'protocol_sha256', 'split_sha256'):
            sha256(getattr(self, name))
        for name in ('task', 'cut_id', 'metric', 'units'):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError('invalid context field: ' + name)
        if self.direction not in ('minimize', 'maximize'):
            raise ValueError('explicit metric direction required')

    @property
    def identity(self):
        return digest(asdict(self))


def partition(parent_lineage_sha256, split_namespace):
    """Stable 5% probability per independent lineage; descendants inherit it.

This is not an exact 5% row guarantee. Existing protected roles take precedence;
callers must pass the authoritative original lineage, never a variant hash.
"""
    sha256(parent_lineage_sha256)
    sha256(split_namespace)
    h = digest({'schema': 'jope.flywheel-lineage-split/v1',
                'parent_lineage_sha256': parent_lineage_sha256,
                'namespace_sha256': split_namespace})
    return 'validation' if int(h, 16) * 20 < 1 << 256 else 'train'


def inherit_partition(parent, child):
    required = ('parent_lineage_sha256', 'split_namespace_sha256', 'partition')
    if any(parent.get(k) != child.get(k) for k in required):
        raise ValueError('synthetic child must inherit authoritative lineage and partition')
    want = partition(parent['parent_lineage_sha256'], parent['split_namespace_sha256'])
    if parent['partition'] != want:
        raise ValueError('role does not match this versioned split protocol')
    if child.get('parent_dataset_sha256') != parent.get('dataset_sha256'):
        raise ValueError('child parent dataset mismatch')
    sha256(child['dataset_sha256'])
    return want


def normalized_quality(rmse, baseline_rmse, epsilon):
    """Single-fold q=-log(max(RMSE,epsilon)/max(baseline,epsilon)).

Epsilon must be frozen from training data in the protocol. Return raw and clipped
optimization values separately; do not silently replace canonical S1 labels.
"""
    rmse, baseline_rmse = finite(rmse, nonnegative=True), finite(baseline_rmse, nonnegative=True)
    epsilon = finite(epsilon)
    if epsilon <= 0:
        raise ValueError('positive train-derived epsilon required')
    q = math.log(max(baseline_rmse, epsilon)) - math.log(max(rmse, epsilon))
    cap = math.log(1000)
    return {'q_raw': q, 'q_opt': max(-cap, min(cap, q)),
            'epsilon': epsilon, 'baseline_degenerate': baseline_rmse <= epsilon}


def freeze_opponents(context, pool, candidate_program_sha256, seed):
    """Freeze an actual-score pool before candidate evaluation; sample uniformly.

pool rows carry program_sha256, context_id, value, score_receipt_sha256. The
winner is deterministic on ties. A snapshot is retained even when no opponents
are available. Only completed, actual evaluations may be in this pool.
"""
    sha256(candidate_program_sha256)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError('uint64 seed required')
    rows = []
    seen = set()
    for r in pool:
        p = sha256(r['program_sha256'])
        if p in seen:
            raise ValueError('duplicate program in frozen pool')
        seen.add(p)
        if r.get('context_id') != context.identity or r.get('status') != 'completed' or r.get('measurement') != 'executed':
            raise ValueError('pool must contain matched executed scores')
        value = finite(r['value'])
        rows.append({'program_sha256': p, 'value': value,
                     'score_receipt_sha256': sha256(r['score_receipt_sha256'])})
    rows.sort(key=lambda r: r['program_sha256'])
    snapshot = {'context': asdict(context), 'scores': rows}
    available = [r for r in rows if r['program_sha256'] != candidate_program_sha256]
    selected = None
    if available:
        limit = (1 << 256) - ((1 << 256) % len(available))
        counter = 0
        while True:
            draw = int(digest({'seed': seed, 'pool_sha256': digest(snapshot),
                               'candidate': candidate_program_sha256, 'counter': counter}), 16)
            if draw < limit:
                selected = available[draw % len(available)]
                break
            counter += 1
    winner = None if not available else min(available, key=lambda r: (
        r['value'] if context.direction == 'minimize' else -r['value'], r['program_sha256']))
    return {'schema': 'jope.flywheel-opponents/v1', 'context_id': context.identity,
            'candidate_program_sha256': candidate_program_sha256, 'seed': seed,
            'sampler': 'sha256-rejection-uniform/v1',
            'pool_sha256': digest(snapshot), 'pool': snapshot,
            'random': selected, 'winner': winner}


def compare(context, candidate, opponent, *, absolute_tolerance=0.0, relative_tolerance=0.001):
    """Measured paired result. Failures count as attempted losses, never zero RMSE.

Missing/running evaluations remain pending and are excluded from completed wins.
A caller must preserve all attempts and explicitly report pending/failure counts.
"""
    absolute_tolerance, relative_tolerance = finite(absolute_tolerance, nonnegative=True), finite(relative_tolerance, nonnegative=True)
    if relative_tolerance > 1:
        raise ValueError('relative tolerance outside [0,1]')
    for row in (candidate, opponent):
        sha256(row['program_sha256'])
        if row.get('context_id') != context.identity:
            raise ValueError('different context/evaluator/protocol scores cannot be compared')
        if row.get('measurement') != 'executed':
            raise ValueError('predictions are not executed scores')
        sha256(row['score_receipt_sha256'])
    if candidate['program_sha256'] == opponent['program_sha256']:
        raise ValueError('self comparison is not evidence of improvement')
    if opponent.get('status') != 'completed':
        raise ValueError('opponent must have a completed actual score')
    other = finite(opponent['value'])
    status = candidate.get('status')
    if status == 'completed':
        value = finite(candidate['value'])
        delta = other - value if context.direction == 'minimize' else value - other
        tolerance = max(absolute_tolerance, relative_tolerance * abs(other))
        outcome = 'tie' if abs(delta) <= tolerance else ('win' if delta > 0 else 'loss')
    elif status in ('failed', 'timeout', 'invalid'):
        value, delta, tolerance, outcome = None, None, None, 'loss'
    elif status in ('pending', 'running'):
        value, delta, tolerance, outcome = None, None, None, 'pending'
    else:
        raise ValueError('unknown evaluation status')
    return {'schema': SCHEMA, 'context_id': context.identity,
            'candidate_program_sha256': candidate['program_sha256'],
            'opponent_program_sha256': opponent['program_sha256'],
            'candidate_receipt_sha256': candidate['score_receipt_sha256'],
            'opponent_receipt_sha256': opponent['score_receipt_sha256'],
            'candidate_status': status, 'candidate_value': value, 'opponent_value': other,
            'outcome': outcome, 'favorable_delta': delta, 'tie_tolerance': tolerance,
            'absolute_tolerance': absolute_tolerance, 'relative_tolerance': relative_tolerance}


def elo_update(candidate_rating, opponent_rating, outcome, k=16):
    """Descriptive sequential Elo only, not a significance test or HWM authority."""
    a, b, k = finite(candidate_rating), finite(opponent_rating), finite(k)
    if not 0 < k <= 128 or outcome not in ('win', 'tie', 'loss'):
        raise ValueError('invalid completed Elo match')
    z = max(-300, min(300, (b-a)/400))
    expected = 1/(1+10**z)
    observed = {'win': 1, 'tie': .5, 'loss': 0}[outcome]
    delta = k*(observed-expected)
    return a+delta, b-delta
