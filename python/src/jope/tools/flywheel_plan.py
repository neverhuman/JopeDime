"""Plan helpers: validation-best, 16-cut seed probe, seed library, Hyperion fail-closed."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from resume_expired_cycle import read, require, sha

RECOVERY_FROZEN = dict(sequence_nll=1.394498497, ranking_bt=0.533676382)


def validation_best(root, output):
    latest = read(Path(output) / 'latest.json')
    metrics = Path(output) / 'metrics.jsonl'
    if not metrics.is_file():
        return latest
    best = None
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        nll = (row.get('validation') or {}).get('sequence_nll')
        checkpoint = row.get('checkpoint')
        if nll is None or not checkpoint:
            continue
        if best is None or nll < best[0]:
            best = (nll, checkpoint)
    if best is None:
        return latest
    return dict(checkpoint=best[1], state_sha256=sha(Path(root) / best[1] / 'state.json'))


def quality_counts(report):
    """16 val-cut matched triplets: nseed<t and neural exclusive wins."""
    groups = {}
    for row in report.get('rows') or []:
        if row.get('partition') != 'meta_validation':
            continue
        if row.get('metric') not in (None, 'rmse', 'logloss', 'mase'):
            continue
        key = (row['dataset'], row['family'], row['cut'])
        groups.setdefault(key, {})[row['arm']] = row
    nseed = exclusive = triplets = 0
    for arms in groups.values():
        if set(arms) != {'neural', 'generic', 'shuffled'}:
            continue
        triplets += 1
        neural = arms['neural']
        teacher, seed = neural.get('best_retained_teacher'), neural.get('best_initial_neural_seed')
        if isinstance(teacher, (int, float)) and isinstance(seed, (int, float)) and seed < teacher:
            nseed += 1
        finals = [arms[a].get('best_final_gp') for a in ('neural', 'generic', 'shuffled')]
        if all(isinstance(v, (int, float)) for v in finals):
            if finals[0] < min(finals[1], finals[2]):
                exclusive += 1
    return dict(val_triplets=triplets, direct_seed_beats_teacher=nseed, neural_exclusive_wins=exclusive)


def probe_rejects(candidate_report, serving_report):
    if candidate_report is None:
        return ['diagnostic probe: candidate quality-report missing']
    if serving_report is None:
        return []
    cand, serve = quality_counts(candidate_report), quality_counts(serving_report)
    if cand['val_triplets'] < 8:
        return ['diagnostic probe: fewer than 8 validation triplets']
    reasons = []
    if cand['direct_seed_beats_teacher'] < serve['direct_seed_beats_teacher']:
        reasons.append('diagnostic probe: nseed<t fell')
    if cand['neural_exclusive_wins'] < serve['neural_exclusive_wins']:
        reasons.append('diagnostic probe: exclusive wins fell')
    return reasons


def latest_serving_quality(root, generations, serving_sha):
    for generation in reversed(generations):
        if generation.get('checkpoint_sha256') != serving_sha:
            continue
        path = Path(root) / generation['path'] / 'quality-report.json'
        if path.is_file():
            return read(path)
    return None


def extract_subtrees(genome):
    """Mine parenthesized subexpressions; do not invent grammar tokens."""
    found = []
    for match in re.finditer(r'\((?:[^()]|\([^()]*\))+\)', genome):
        text = match.group(0)
        if text.startswith('(pipe ') and text != genome:
            found.append(text)
    return found


def enrich_seed_population(population, family_champions, ranking_rows):
    """Replace random_exploration slots with cross-cut teachers and frequent subtrees."""
    candidates = list(population['candidates'])
    seen = {row['genome'] for row in candidates}
    extras = []
    for genome in family_champions:
        if genome not in seen:
            extras.append(dict(genome=genome, origin='same_family_cross_cut_teacher'))
            seen.add(genome)
    counts = Counter()
    for genome in list(seen):
        counts.update(extract_subtrees(genome))
    for genome, count in counts.most_common():
        if count < 2 or genome in seen:
            continue
        extras.append(dict(genome=genome, origin='component_seed_library'))
        seen.add(genome)
        if len(extras) >= 8:
            break
    if not extras:
        return population
    out = []
    extra_i = 0
    for row in candidates:
        if extra_i < len(extras) and row.get('origin') in ('random_exploration', 'retained_slot_fallback'):
            out.append(extras[extra_i])
            extra_i += 1
        else:
            out.append(row)
    population = dict(population)
    population['candidates'] = out
    return population


def selected_view(gate, prepared, cut):
    """Fail closed to raw unless a frozen CV gate selected hyperion_cell for this cut."""
    if not isinstance(gate, dict):
        return 'raw'
    key = f'{prepared}:{cut}'
    value = (gate.get('cuts') or {}).get(key) or gate.get(key)
    if value == 'hyperion_cell':
        return 'hyperion_cell'
    return 'raw'


def mix(seed, index):
    digest = hashlib.sha256(f'{seed}:{index}'.encode()).digest()
    return int.from_bytes(digest[:8], 'little')


def teachers_improved(timing, growth, n=1, m=1):
    """Train only when first-label programs were newly admitted.

    `rebuilt_groups` is a pack dirty-shard count. Under reference_v1 a group
    is rewritten whenever any new program lands, including losers, and every
    percentile/value in that shard is recomputed. That is not a teacher
    improvement and must not spend 3000 steps on 8 GPUs.
    """
    del timing, n
    added = 0 if growth is None else int(growth)
    return added >= m


DIAGNOSTIC_HOLD_AFTER = 3


def consecutive_diagnostic_hold(events, limit=DIAGNOSTIC_HOLD_AFTER):
    """Stop automatic 3000-step trains after repeated frozen-guard rejects."""
    n = 0
    for event in reversed(events or []):
        name = event.get('event') if isinstance(event, dict) else event
        if name == 'diagnostic_rejected':
            n += 1
            if n >= limit:
                return True
        elif name in ('diagnostic_accepted', 'diagnostic_passed_pending_full_qualification'):
            return False
    return False


def ranking_bt_stable(bt):
    return isinstance(bt, (int, float)) and bt <= RECOVERY_FROZEN['ranking_bt'] * 1.01


def search_beating_training_allowed(bt):
    """SEARCH_BEATING_SAMPLER_ID stays opt-in until frozen BT is at recovery."""
    return ranking_bt_stable(bt)


def may_serve(diagnostic_ok, probe_ok, paper_ok, challenger_quality_report=False):
    """Diagnostic pass does not assign serving."""
    if not challenger_quality_report:
        return False
    return bool(diagnostic_ok and probe_ok and paper_ok)


def neural_seed_win_rate(rows):
    n = beats = 0
    for row in rows or []:
        teacher, seed = row.get('best_retained_teacher'), row.get('best_initial_neural_seed')
        if isinstance(teacher, (int, float)) and isinstance(seed, (int, float)):
            n += 1
            if seed < teacher:
                beats += 1
    return 0.0 if n == 0 else beats / n


def hyperion_view_report(jobs, gate):
    """Record the gate decision. Cell views must not teach until this exists."""
    rows = []
    for job in jobs or []:
        prepared, cut = job.get('prepared'), job.get('cut')
        selected = job.get('selected_view') or 'raw'
        gated = selected_view(gate, prepared, cut)
        rows.append(dict(prepared=prepared, cut=cut, selected_view=selected,
                         gate_would_select=gated, teaches_cell=selected == 'hyperion_cell'))
    return dict(schema='parallel.hyperion_view_report.v1', jobs=len(rows),
                cell_jobs=sum(r['teaches_cell'] for r in rows),
                would_select_cell=sum(r['gate_would_select'] == 'hyperion_cell' for r in rows),
                rows=rows)


def prefix_success(observations, winner, limit=200):
    if winner is None:
        return None
    for index, row in enumerate(observations or []):
        if row is winner:
            return index < limit
    return None


def search_beats_seed_and_teacher(final_rmse, seed_rmse, teacher_rmse):
    values = (final_rmse, seed_rmse, teacher_rmse)
    if not all(isinstance(v, (int, float)) for v in values):
        return False
    return final_rmse < seed_rmse and final_rmse < teacher_rmse
