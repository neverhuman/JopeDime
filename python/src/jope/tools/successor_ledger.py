#!/usr/bin/env python3
"""Successor ledger: episode roles, deterministic controls, replicate panel, seed values, bundles.

Wraps Codex's seed-dispatch Dispatcher (recovery/eight-a100-20260912/experiments/seed-dispatch,
imported by path) and extends its coverage.sqlite schema in successor/ledger.sqlite:
episodes gain arm, role, replicate, generation, context_key, parent_program_sha256, seed_origin;
tables seed_values, economics, bundles are added. Dispatcher.reconcile/transition/pending_bundle
are reused so the same process/result receipts (dispatch-process.json with deadline_at =
started_at + 36, job-binding.json, search/search.json) complete an episode.

Per meta_train context and generation (plan D1): candidate = every unique-byte neural emission
(<= 64), control_generic = 8 generic_comparison seeds chosen by sha256(prepared:cut:g:'generic':
program) order, control_shuffled = 8 shuffled-arm seeds the same way, gp_final = each arm's
search.json best_program, replicate_panel = 8 candidates re-run with rng_seed_1 =
sha256(context_key:g:parent_sha:'1')[:8] as u64. Identity protocol v3 keeps evaluations == 2,
candidate_seconds 2, seconds 6 and resource_protocol single_candidate_2s_8gib_v1.

Recorded per episode: r0 (row-0 own cut-normalized RMSE), r_child (row 1) or its failure,
flip = 1[r_child < r0], S_rmse = min(r0, r_child) (failed child -> r0). Parent labels are never
overwritten; children enter the corpus as their own rows through bundles that list only episodes
which passed verify_episode (evaluator/continuation/partition/resource-protocol/row-0 checks).

Admission identity (plan D4): the evaluator is manifest['bindings'][manifest['cpu_executable']]
(cross-checked against the binary on disk when present); the resource protocol is pinned to
single_candidate_2s_8gib_v1 unless an explicit, journaled override is given. Continuation episodes
run by the dual-identity GP supervisor are evaluated by the CONTINUATION evaluator: the manifest's
continuation_source_admission.configuration (the seed driver's admission config, hash-checked)
names evaluator/evaluator_sha256 and the manifest must bind that path to that digest; identities
and rows carrying a continuation_protocol are verified against it (an explicit
--continuation-evaluator must equal it, or, without a derivable one, be a manifest-bound digest),
ordinary identities keep bindings[cpu_executable]; without any continuation binding the two
coincide. Row-0 evidence is
mandatory (F5): an episode is verified only when its seed's admitted RMSE is known (from the arm's
search observations at plan time, or --seed-rmse at ingest) and row 0 reproduces it bit-exactly;
episodes without that evidence stay unverified ('row0_label_unverified') and never reach a bundle.
next_outcome(generation) joins generation g's candidate seeds to generation g+1 outcomes for F4(b).

Tier-B (protocol bounded_seed_lineage_v3, cap variants _cap2s / _cap4s, never mixed inside a comparison group):
gen-0 is the seed alone (row 0; F5 row-0 equality), every later fit is a mutation of a lineage member or a crossover
of two evaluated lineage members (no immigrants, no unevaluated donors), budget 24 attempted fits (seed + 23; failures
consume budget), identity evaluations 24 / candidate_seconds cap / seconds 24*cap+6 / single_candidate_{cap}s_8gib_v1.
verify_episode checks row 0 is the seed, every descendant's genetic_parents are program_sha256 values of earlier rows
of the same episode (else 'lineage_violation': excluded from bundles and credit), and attempted fits <= 24. Credit is
S_lineage = min finite RMSE over seed + descendants (S_rmse of the export row); every descendant keeps its own row.
termination_reason comes from result.json ({budget_exhausted, episode_timeout, seed_failed, lineage_exhausted}); an
episode ended by the outer timeout, or without a known termination, is status 'incomplete' (retained, never resumed,
credited only for what it measured). plan_generation(protocol=...) plans the top-3 admitted seeds per arm on the
rotating 32-context subset (smallest sha256(prepared:cut:g:'tierb')), matched arms under the same rng derivation.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import experiment_runtime as runtime
from successor_targets import spearman

HERE = Path(__file__).resolve().parent
# The seed-dispatch lane modules (Codex's dispatch.py / run_episode.py byte-for-byte, continuation.py with the additive
# arm/role/seed_origin/replicate/protocol/generation/select plan() keywords) are vendored under tools/seed_dispatch so the
# ledger and its tests run from a checkout; JOPE_SEED_DISPATCH_LANE points at another lane copy (provenance in PROVENANCE.md).
LANE = Path(os.environ.get('JOPE_SEED_DISPATCH_LANE', HERE / 'seed_dispatch'))
HEX64 = re.compile(r'^[0-9a-f]{64}$')

PROTOCOL_V2 = 'parallel.seed_continuation.qualification.v2'
PROTOCOL_V3 = 'parallel.seed_continuation.qualification.v3'
ONESTEP_PROTOCOLS = (PROTOCOL_V2, PROTOCOL_V3)
TIERB_BASE = 'bounded_seed_lineage_v3'
TIERB_CAP_SECONDS = {'bounded_seed_lineage_v3_cap2s': 2, 'bounded_seed_lineage_v3_cap4s': 4}
TIERB_PROTOCOLS = tuple(TIERB_CAP_SECONDS)
PROTOCOLS = ONESTEP_PROTOCOLS + TIERB_PROTOCOLS
TIERB_BUDGET = 24
TIERB_PARENT_POOL = 4
TIERB_TERMINATIONS = ('budget_exhausted', 'episode_timeout', 'seed_failed', 'lineage_exhausted')
TIERB_SUBSET = 32
TIERB_TOP_SEEDS = 3
TIERB_SALT = 'tierb'
LINEAGE_VIOLATION = 'lineage_violation'
RESOURCE_PROTOCOL = 'single_candidate_2s_8gib_v1'
EVALUATOR_PROTOCOL = 'featmath_training_fold_normalized_v1'
CREDIT_PROTOCOL = 'onestep_forced_mutation_v2'
CONTINUATION_PROTOCOL = 'forced_first_mutation_v2'
ARMS = ('neural', 'generic', 'shuffled')
SEED_ORIGIN_BY_ARM = dict(neural='neural_policy', generic='generic_comparison', shuffled='neural_policy')
PRIORITY = ('candidate', 'control_generic', 'control_shuffled', 'gp_final', 'replicate_panel', 'reporting')
TERMINAL = ('completed', 'failed', 'deadline', 'superseded')
ROW_SCHEMA = 'parallel.seed_value.v1'
PLAN_SCHEMA = 'parallel.successor_plan.v1'
PLAN_SUMMARY_SCHEMA = 'parallel.successor_plan_summary.v1'
VERIFY_SCHEMA = 'parallel.successor_episode_verification.v1'
BUNDLE_SCHEMA = 'parallel.successor_bundle.v1'
BUNDLES_SCHEMA = 'parallel.successor_bundles.v1'
SOURCES_SCHEMA = 'parallel.successor_sources.v1'
ECONOMICS_SCHEMA = 'parallel.successor_economics.v1'
INFO_SCHEMA = 'parallel.successor_info.v1'
SUPERSEDE_SCHEMA = 'parallel.successor_supersede.v1'
SEARCH_SCHEMA = 'parallel.search.v1'
QUALITY_SCHEMA = 'parallel.continuous_quality.v1'
NEXT_OUTCOME_SCHEMA = 'parallel.successor_next_outcome.v1'
NEXT_OUTCOMES = ('best_initial', 'child_beats_incumbent')
ROW0_MISMATCH = 'row0_label_mismatch'
EXACT_TIE = 1e-12
IDENTITY_KEYS = ('program_sha256', 'family', 'partition', 'prepared_sha256', 'cut', 'training_sha256', 'validation_sha256',
                 'evaluator_sha256', 'evaluator_protocol', 'preprocessing_sha256', 'target_scale', 'metric_units',
                 'vocabulary_sha256', 'feature_view_sha256', 'evaluation_seed', 'resource_protocol')
DEFAULTS = dict(candidate_seconds=2, seconds=6, evaluations=2, protocol=PROTOCOL_V3, controls=8, replicates=8, candidate_cap=64)
SCHEMA = """CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS episodes(id TEXT PRIMARY KEY,payload TEXT NOT NULL,status TEXT NOT NULL,owner TEXT,deadline REAL,result TEXT,
    arm TEXT,role TEXT,replicate INTEGER,generation INTEGER,context_key TEXT,parent_program_sha256 TEXT,seed_origin TEXT);
CREATE TABLE IF NOT EXISTS emissions(generation INTEGER,population_sha TEXT,slot INTEGER,episode TEXT,status TEXT,PRIMARY KEY(generation,population_sha,slot));
CREATE TABLE IF NOT EXISTS events(sequence INTEGER PRIMARY KEY AUTOINCREMENT,episode TEXT,status TEXT,at REAL,detail TEXT);
CREATE TABLE IF NOT EXISTS seed_values(generation INTEGER,context_key TEXT,arm TEXT,role TEXT,parent_program_sha256 TEXT,replicate INTEGER,
    protocol TEXT NOT NULL DEFAULT 'onestep_forced_mutation_v2',
    own_rmse REAL,child_program_sha256 TEXT,child_rmse REAL,child_failure TEXT,flip INTEGER,S_rmse REAL,
    attempted_descendants INTEGER,successful_descendants INTEGER,termination_reason TEXT,lineage_size INTEGER,
    seed_beats_incumbent INTEGER,child_beats_incumbent INTEGER,episode TEXT,verified INTEGER,detail TEXT,
    PRIMARY KEY(generation,context_key,arm,parent_program_sha256,replicate,protocol));
CREATE TABLE IF NOT EXISTS economics(generation INTEGER PRIMARY KEY,tier TEXT,reserved_core_seconds REAL,attempted_fits INTEGER,
    successful_fits INTEGER,unique_fits_per_core_second REAL,timeouts INTEGER,failures INTEGER,superseded INTEGER,
    completed_within_generation INTEGER,receipt TEXT);
CREATE TABLE IF NOT EXISTS bundles(generation INTEGER,context_key TEXT,path TEXT,observations_sha256 TEXT,search_sha256 TEXT,
    episodes INTEGER,admitted INTEGER,detail TEXT,PRIMARY KEY(generation,context_key));"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def lane_module(name, lane=LANE):
    """Import one of Codex's seed-dispatch modules by path (they import each other by bare name)."""
    if name in sys.modules and getattr(sys.modules[name], '__file__', None) == str(lane / (name + '.py')):
        return sys.modules[name]
    require((lane / (name + '.py')).is_file(), 'seed-dispatch lane module missing: ' + str(lane / (name + '.py')))
    spec = importlib.util.spec_from_file_location(name, lane / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


continuation = lane_module('continuation')
run_episode = lane_module('run_episode')
dispatch = lane_module('dispatch')
Dispatcher, immutable, within, sha = dispatch.Dispatcher, dispatch.immutable, dispatch.within, dispatch.sha
plan, encoded, digest = continuation.plan, continuation.encoded, continuation.digest
read_rows, summarize = run_episode.read_rows, run_episode.summarize


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def selection_key(prepared_sha256, cut, generation, salt, program_sha256):
    """Deterministic control/panel order: sha256(prepared:cut:generation:salt:program)."""
    return hashlib.sha256(f'{prepared_sha256}:{cut}:{generation}:{salt}:{program_sha256}'.encode()).hexdigest()


def replicate_rng(context_key, generation, parent_sha256, replicate=1):
    """rng_seed of a replicate: sha256(context_key:generation:parent_sha:replicate)[:8] as little-endian u64."""
    return int.from_bytes(hashlib.sha256(f'{context_key}:{generation}:{parent_sha256}:{replicate}'.encode()).digest()[:8], 'little')


def is_tierb(protocol):
    return protocol in TIERB_CAP_SECONDS


def credit_protocol_of(identity_protocol):
    """Credit protocol of an export row: the Tier-B cap variant itself, else the one-step credit protocol."""
    return identity_protocol if is_tierb(identity_protocol) else CREDIT_PROTOCOL


def credited_statuses(protocol):
    """Tier-B episodes ended by the outer timeout are 'incomplete' and credited for what they measured."""
    return ('measured', 'incomplete') if is_tierb(protocol) else ('measured',)


def tierb_budget(protocol):
    """Identity budget of a Tier-B cap variant: 24 attempted fits, candidate cap, episode ceiling 24*cap + 6 s."""
    cap = TIERB_CAP_SECONDS[protocol]
    return dict(evaluations=TIERB_BUDGET, candidate_seconds=cap, seconds=TIERB_BUDGET * cap + 6,
                resource_protocol='single_candidate_%ds_8gib_v1' % cap)


def tierb_context_rank(prepared_sha256, cut, generation):
    """Rotation key of a context: sha256(prepared:cut:generation:'tierb'); the subset is the 32 smallest."""
    return hashlib.sha256(f'{prepared_sha256}:{cut}:{generation}:{TIERB_SALT}'.encode()).hexdigest()


def tierb_subset(context_keys, generation, size=TIERB_SUBSET):
    """The rotating Tier-B subset of a generation: `size` context keys (prepared_sha256:cut) of smallest rank."""
    keyed = sorted((tierb_context_rank(*key.rsplit(':', 1), generation), key) for key in dict.fromkeys(context_keys))
    return [key for _, key in keyed[:size]]


def rebind_tierb(episodes, protocol):
    """Codex's continuation.plan() pins candidate_seconds 2 and the 2 s resource protocol; a Tier-B identity rebinds
    exactly those two fields to its cap variant and recomputes the episode id (identity bytes -> id)."""
    budget = tierb_budget(protocol)
    out = []
    for episode in episodes:
        identity = dict(episode['identity'])
        require(identity['protocol'] == protocol and identity['evaluations'] == budget['evaluations'] and identity['seconds'] == budget['seconds'],
                'Tier-B identity budget drift')
        identity.update(candidate_seconds=budget['candidate_seconds'], resource_protocol=budget['resource_protocol'])
        out.append(dict(episode, identity=identity, episode_id=digest(encoded(identity))))
    return out


TIERB_OPERATORS = {1: 'mutation', 2: 'crossover'}
TIERB_CROSSOVER_AFTER = 2


def tierb_parent_pool(evaluated):
    """The top-4 finite-evaluated lineage members (by RMSE, ties by program sha256) a new fit may draw its parents from."""
    ranked = sorted(evaluated.items(), key=lambda item: (item[1], item[0]))
    return [program for program, _ in ranked[:TIERB_PARENT_POOL]]


def verify_lineage(identity, rows):
    """Tier-B lineage rule over one episode's rows; returns verification reasons (empty = valid).

    Row 0 is the seed (program == parent, no genetic parents, initial_seed_slot 0). Every later row is a `mutation` (one
    parent) or a `crossover` (two parents): its parents are program_sha256 values of EARLIER rows of the same episode
    that were finite-evaluated and lie in the top-4 of those by RMSE at the time of the fit (else 'lineage_violation' /
    'parent_outside_top4'), the row's genetic_operator names the operation of its parent count ('operator_mismatch'), and a
    crossover is legal only once at least two descendants have been finite-evaluated ('crossover_before_two_descendants').
    Programs never repeat; attempted fits (failures included) never exceed the 24 budget.
    """
    reasons, seen, evaluated = [], set(), {}
    parent = identity.get('parent_program_sha256')
    evaluated_descendants = 0
    for index, row in enumerate(rows):
        prefix = 'row%d_' % index
        program, parents = row.get('program_sha256'), row.get('genetic_parents')
        if not isinstance(row.get('genome'), str) or digest(row['genome'].encode()) != program:
            reasons.append(prefix + 'program_bytes_mismatch')
        if program in seen:
            reasons.append(prefix + 'repeated_fit')
        if row.get('evaluation_index') != index:
            reasons.append(prefix + 'evaluation_order')
        if index == 0:
            if program != parent or parents != [] or row.get('initial_seed_slot') != 0:
                reasons.append('row0_not_the_seed: program %s parents %s initial_seed_slot %s' % (program, parents, row.get('initial_seed_slot')))
        elif not isinstance(parents, list) or len(parents) not in (1, 2) or any(p not in seen for p in parents):
            reasons.append('%s: row %d parents %s are not all earlier lineage members' % (LINEAGE_VIOLATION, index, parents))
        else:
            pool = tierb_parent_pool(evaluated)
            outside = [p for p in parents if p not in pool]
            if outside:
                reasons.append('%s: row %d parents %s are not in the top-%d finite-evaluated members %s'
                               % ('parent_outside_top4', index, outside, TIERB_PARENT_POOL, pool))
            operator = row.get('genetic_operator')
            if operator != TIERB_OPERATORS[len(parents)]:
                reasons.append('operator_mismatch: row %d has %d parent(s) but genetic_operator %r (expected %r)'
                               % (index, len(parents), operator, TIERB_OPERATORS[len(parents)]))
            if len(parents) == 2 and evaluated_descendants < TIERB_CROSSOVER_AFTER:
                reasons.append('crossover_before_two_descendants: row %d is a crossover after %d finite-evaluated descendant(s)' % (index, evaluated_descendants))
        score = rmse_of(row)
        if score is not None and program not in evaluated:
            evaluated[program] = score
            if index > 0:
                evaluated_descendants += 1
        seen.add(program)
    if len(rows) > TIERB_BUDGET:
        reasons.append('budget_exceeded: %d attempted fits > %d' % (len(rows), TIERB_BUDGET))
    return reasons


def seed_genomes(doc, origin):
    """Unique-byte genomes of one seed origin in slot order."""
    return list(dict.fromkeys(c['genome'] for c in doc['candidates'] if c.get('origin') == origin and isinstance(c.get('genome'), str) and c['genome']))


def rmse_of(row):
    if not row or row.get('failure') is not None or not isinstance(row.get('metrics'), dict):
        return None
    score = row['metrics'].get('rmse')
    return score if finite(score) and score >= 0 else None


def admitted_rmse(rows):
    """program_sha256 -> admitted cut-normalized RMSE from a search ledger's rows (first measured row wins)."""
    result = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        score, program = rmse_of(row), row.get('program_sha256')
        if score is not None and isinstance(program, str) and program not in result:
            result[program] = score
    return result


def load_seed_rmse(root, path):
    """--seed-rmse input: a {program_sha256: rmse} object or a search observations.json row list."""
    document = json.loads(within(root, path).read_bytes())
    if isinstance(document, list):
        return admitted_rmse(document)
    require(isinstance(document, dict) and all(HEX64.match(str(k)) and finite(v) for k, v in document.items()),
            'seed-rmse file must map program_sha256 -> finite rmse or be a search observations.json row list')
    return dict(document)


def continuation_evaluator_binding(root, manifest, bindings, explicit=None):
    """The continuation evaluator bound by the manifest: (sha256, source) or (None, None) when the manifest carries none.

    Derived from manifest['continuation_source_admission']['configuration'] (the seed driver's admission config within
    root, hash-checked against configuration_sha256 when declared): its evaluator/evaluator_sha256 must be bound by the
    manifest's bindings (and by the binary on disk when present). An explicit 64-hex must equal the derived digest, or,
    when the manifest declares no readable admission configuration, be one of the manifest's bound digests.
    """
    require(explicit is None or (isinstance(explicit, str) and HEX64.match(explicit)), 'continuation evaluator must be a 64-hex sha256')
    admission = manifest.get('continuation_source_admission')
    configuration = admission.get('configuration') if isinstance(admission, dict) else None
    if isinstance(configuration, str) and configuration and within(root, configuration).is_file():
        config_path = within(root, configuration)
        declared = admission.get('configuration_sha256')
        require(declared is None or sha(config_path) == declared,
                'continuation admission configuration differs from the manifest binding: ' + configuration)
        config = json.loads(config_path.read_bytes())
        executable, bound = config.get('evaluator'), config.get('evaluator_sha256')
        require(isinstance(executable, str) and isinstance(bound, str) and HEX64.match(bound) and bindings.get(executable) == bound,
                'continuation admission configuration evaluator is not bound by the manifest: ' + str(executable))
        binary = within(root, executable)
        if binary.is_file():
            require(sha(binary) == bound, 'continuation evaluator on disk differs from the manifest binding: ' + executable)
        require(explicit is None or explicit == bound, 'explicit continuation evaluator contradicts the manifest-derived binding ' + bound)
        return bound, dict(kind='continuation_source_admission', configuration=configuration, configuration_sha256=sha(config_path),
                           evaluator=executable, binary_present=binary.is_file())
    if explicit is not None:
        executable = next((k for k, v in bindings.items() if v == explicit), None)
        require(executable is not None, 'explicit continuation evaluator is not a manifest-bound digest: ' + explicit)
        return explicit, dict(kind='explicit', configuration=None, configuration_sha256=None, evaluator=executable, binary_present=None)
    return None, None


def manifest_evaluator(root, manifest_path, continuation_evaluator=None):
    """bindings[cpu_executable] of the manifest, cross-checked against the binary on disk when present (gp_queue_gc pattern).

    continuation_evaluator_sha256 is the manifest's continuation evaluator (continuation_evaluator_binding), defaulting to
    bindings[cpu_executable] when the manifest binds none; continuation_evaluator_source says which.
    """
    manifest_path = within(root, manifest_path)
    manifest = json.loads(manifest_path.read_bytes())
    executable = manifest.get('cpu_executable')
    require(isinstance(executable, str) and executable and not executable.startswith('/') and '..' not in executable.split('/'),
            'manifest cpu_executable missing or unsafe')
    bindings = manifest.get('bindings')
    bound = bindings.get(executable) if isinstance(bindings, dict) else None
    require(isinstance(bound, str) and HEX64.match(bound), 'manifest does not bind cpu_executable: ' + executable)
    binary = root / executable
    on_disk = binary.is_file()
    if on_disk:
        require(sha(binary) == bound, 'cpu_executable on disk differs from the manifest binding: ' + executable)
    continuation, source = continuation_evaluator_binding(root, manifest, bindings, continuation_evaluator)
    return dict(manifest_path=str(manifest_path.relative_to(root)), manifest_sha256=sha(manifest_path), cpu_executable=executable,
                evaluator_sha256=bound, binary_present=on_disk, vocabulary_sha256=manifest.get('vocabulary_sha256'),
                bound_digests=sorted({v for v in bindings.values() if isinstance(v, str) and HEX64.match(v)}),
                continuation_evaluator_sha256=continuation if continuation is not None else bound,
                continuation_evaluator_source=source if source is not None else dict(kind='cpu_executable', configuration=None,
                                                                                     configuration_sha256=None, evaluator=executable,
                                                                                     binary_present=on_disk))


def seed_value(identity, rows, summary=None):
    """Per-episode credit; never rewrites a label.

    One-step: r0, the forced child, flip = 1[r_child < r0], S = min(r0, r_child) (failed/absent child -> r0).
    Tier-B: `child` is the best measured lineage descendant, flip = 1[best < r0], S = S_lineage = min finite RMSE over
    seed + descendants; attempted/successful descendants are counted from the rows (failures consume budget);
    termination_reason comes from result.json (`summary`), else 'unknown'; an episode ended by the outer timeout or
    without a known termination is status 'incomplete' (credited only for what it measured).
    """
    protocol = credit_protocol_of(identity.get('protocol'))
    tierb = is_tierb(protocol)
    row0 = rows[0] if rows else None
    descendants = list(rows[1:])
    scored = [(rmse_of(r), i, r) for i, r in enumerate(descendants) if rmse_of(r) is not None]
    if tierb:
        best = min(scored, key=lambda s: (s[0], s[1])) if scored else None
        row1 = best[2] if best else None
        failure = None if best else ('no_descendant_measured' if descendants else 'child_unstarted')
    else:
        row1 = descendants[0] if descendants else None
        failure = row1.get('failure') if row1 else 'child_unstarted'
    child = dict(program_sha256=row1.get('program_sha256') if row1 else None, rmse=rmse_of(row1), failure=failure,
                 genetic_operator=row1.get('genetic_operator') if row1 else None)
    if row1 and child['rmse'] is None and child['failure'] is None:
        child['failure'] = 'child_unmeasured'
    r0 = rmse_of(row0)
    termination = None
    if tierb:
        declared = (summary or {}).get('termination_reason') if isinstance(summary, dict) else None
        termination = declared if declared in TIERB_TERMINATIONS else 'unknown'
    binding = dict(protocol=protocol, attempted_descendants=len(descendants), successful_descendants=len(scored),
                   lineage_size=int(r0 is not None) + len(scored), termination_reason=termination, parent=identity['parent_program_sha256'])
    if r0 is None:
        return dict(r0=None, child=child, flip=None, S_rmse=None, status='failed',
                    failure=(row0.get('failure') if row0 else None) or 'seed_unmeasured', **binding)
    flip = 1 if child['rmse'] is not None and child['rmse'] < r0 else 0
    status = 'incomplete' if tierb and termination in ('episode_timeout', 'unknown') else 'measured'
    return dict(r0=r0, child=child, flip=flip, S_rmse=min(r0, child['rmse']) if child['rmse'] is not None else r0,
                status=status, failure=None, **binding)


def verify_episode(episode_dir, expected):
    """Self-contained admission check of one episode directory; collects every reason instead of raising.

    expected: evaluator_sha256, resource_protocol, protocols, partition ('meta_train'), optional
    vocabulary_sha256, seed_rmse ({parent_program_sha256: admitted row-0 rmse} for F5) and
    continuation_evaluator_sha256 (the evaluator expected of receipts/rows that carry a continuation_protocol;
    absent or None -> evaluator_sha256). expected_evaluator_sha256 in the result names the digest applied.
    """
    directory = Path(episode_dir)
    reasons = []
    out = dict(schema=VERIFY_SCHEMA, episode_dir=str(directory), verified=False, reasons=reasons, rows=None, episode_id=None,
               context_key=None, observations_sha256=None, search_sha256=None, completed_at=None, driver_sha256=None,
               row0_label_verified=None, admitted_seed_rmse=None, seed_unmeasured=False, evaluator_sha256=None,
               resource_protocol=None, protocol=None, termination_reason=None, expected_evaluator_sha256=None)
    try:
        identity_bytes = (directory / 'identity.json').read_bytes()
        identity = json.loads(identity_bytes)
    except (OSError, ValueError) as error:
        reasons.append('identity_unreadable: ' + str(error))
        return out
    episode_id = digest(identity_bytes)
    out.update(episode_id=episode_id, context_key=f"{identity.get('prepared_sha256')}:{identity.get('cut')}", protocol=identity.get('protocol'))
    search = directory / 'search'
    if not (search / 'search.json').is_file():
        reasons.append('search_receipt_absent')
        return out
    try:
        receipt = json.loads((search / 'search.json').read_bytes())
        rows = read_rows(search)
    except (OSError, ValueError, KeyError, TypeError) as error:
        reasons.append('observations_unverifiable: ' + str(error))
        return out
    out.update(rows=rows, observations_sha256=receipt.get('observations_sha256'), search_sha256=sha(search / 'search.json'),
               completed_at=receipt.get('completed_at'), driver_sha256=receipt.get('search_driver_sha256'),
               evaluator_sha256=receipt.get('evaluator_sha256'), resource_protocol=rows[0].get('resource_protocol') if rows else None)
    # Receipts/rows carrying a continuation_protocol were produced by the continuation evaluator when the manifest binds
    # one (ordinary identities keep bindings[cpu_executable]); without a continuation binding the two coincide.
    continuation_bound = receipt.get('continuation_protocol') is not None or any(r.get('continuation_protocol') is not None for r in rows)
    evaluator = (expected.get('continuation_evaluator_sha256') or expected['evaluator_sha256']) if continuation_bound else expected['evaluator_sha256']
    out['expected_evaluator_sha256'] = evaluator
    partition = expected.get('partition', 'meta_train')
    protocol = identity.get('protocol')
    tierb = is_tierb(protocol)
    if tierb:
        # A cap variant binds its own budget and resource protocol; rows may carry the cap variant or the base rule id.
        budget = tierb_budget(protocol)
        resource = budget['resource_protocol']
        budget_ok = all(identity.get(k) == v for k, v in budget.items() if k != 'resource_protocol')
        budget_reason = 'identity budget must pin evaluations %d / candidate_seconds %d / seconds %d' % (
            budget['evaluations'], budget['candidate_seconds'], budget['seconds'])
        identity_resource_ok = identity.get('resource_protocol') == resource
        continuation_protocols = (protocol,)   # rows carry the exact cap variant, never the base rule id
    else:
        resource = expected['resource_protocol']
        budget_ok = identity.get('evaluations') == 2 and identity.get('candidate_seconds') == 2
        budget_reason = 'identity budget must pin evaluations 2 / candidate_seconds 2'
        # v2 identities (seed-continuation-v2) carry no resource_protocol; v3 binds it. Rows are always checked below.
        identity_resource_ok = 'resource_protocol' not in identity or identity['resource_protocol'] == resource
        continuation_protocols = (CONTINUATION_PROTOCOL,)
    checks = [
        (receipt.get('schema') == SEARCH_SCHEMA, 'search_schema: ' + str(receipt.get('schema'))),
        (protocol in expected.get('protocols', PROTOCOLS), 'identity_protocol: ' + str(protocol)),
        (identity.get('final_tests_opened') is False and receipt.get('final_tests_opened') is False, 'final_tests_opened must be false'),
        (identity.get('partition') == partition, 'identity_partition: ' + str(identity.get('partition'))),
        (identity.get('evaluator_sha256') == evaluator, 'identity_evaluator_mismatch: ' + str(identity.get('evaluator_sha256'))),
        # A Tier-B identity may name the ordinary evaluator whose admitted row-0 label the seed reproduces; it must be that one.
        (identity.get('row0_source_evaluator_sha256') in (None, expected['evaluator_sha256']),
         'identity_row0_source_evaluator_mismatch: ' + str(identity.get('row0_source_evaluator_sha256'))),
        (receipt.get('evaluator_sha256') == evaluator, 'evaluator_mismatch: ' + str(receipt.get('evaluator_sha256'))),
        (receipt.get('evaluator_protocol') == EVALUATOR_PROTOCOL, 'evaluator_protocol: ' + str(receipt.get('evaluator_protocol'))),
        (receipt.get('continuation_sha256') == episode_id, 'receipt_continuation_mismatch'),
        (finite(receipt.get('completed_at')), 'completed_at absent or nonfinite'),
        (budget_ok, budget_reason),
        (identity_resource_ok, 'identity_resource_protocol: ' + str(identity.get('resource_protocol'))),
        (bool(rows), 'no observation rows'),
    ]
    if expected.get('vocabulary_sha256') is not None:
        checks.append((identity.get('vocabulary_sha256') == expected['vocabulary_sha256'], 'identity_vocabulary_mismatch'))
    for index, row in enumerate(rows):
        prefix = 'row%d_' % index
        checks += [
            (row.get('evaluator_sha256') == evaluator, prefix + 'evaluator_mismatch: ' + str(row.get('evaluator_sha256'))),
            (row.get('resource_protocol') == resource, prefix + 'resource_protocol: ' + str(row.get('resource_protocol'))),
            (row.get('partition') == partition, prefix + 'partition: ' + str(row.get('partition'))),
            (row.get('continuation_sha256') == episode_id, prefix + 'continuation_mismatch'),
            (row.get('continuation_protocol') in continuation_protocols, prefix + 'continuation_protocol: ' + str(row.get('continuation_protocol'))),
            (row.get('metric_units') == 'cut_normalized_target', prefix + 'metric_units'),
            (row.get('evaluator_protocol') == EVALUATOR_PROTOCOL, prefix + 'evaluator_protocol'),
            ('failure' in row, prefix + 'explicit outcome absent'),
        ]
        if row.get('failure') is None:
            checks.append((isinstance(row.get('metrics'), dict) and row['metrics'].get('units') == 'cut_normalized_target' and finite(row['metrics'].get('rmse')),
                           prefix + 'metrics missing, nonfinite or wrong units'))
        try:
            checks.append((digest(canonical({k: row[k] for k in IDENTITY_KEYS})) == row.get('evaluation_identity_sha256'),
                           prefix + 'evaluation_identity_digest_changed'))
        except (KeyError, ValueError, TypeError):
            checks.append((False, prefix + 'evaluation_identity_keys_missing'))
    for ok, reason in checks:
        if not ok:
            reasons.append(reason)
    if tierb:
        reasons.extend(verify_lineage(identity, rows))
        # termination_reason is bound by result.json when present; absence is not a verification failure (the episode
        # is then 'incomplete'), an unknown or inconsistent value is.
        if (directory / 'result.json').is_file():
            try:
                result = json.loads((directory / 'result.json').read_bytes())
                termination = result.get('termination_reason')
            except (OSError, ValueError, AttributeError) as error:
                reasons.append('result_unreadable: ' + str(error))
            else:
                out['termination_reason'] = termination
                if termination not in TIERB_TERMINATIONS:
                    reasons.append('termination_reason_invalid: ' + str(termination))
                elif termination == 'budget_exhausted' and len(rows) != TIERB_BUDGET:
                    reasons.append('termination_inconsistent: budget_exhausted with %d attempted fits' % len(rows))
                elif termination == 'seed_failed' and rmse_of(rows[0] if rows else None) is not None:
                    reasons.append('termination_inconsistent: seed_failed with a measured seed')
                # The declared counts must equal what the rows show: attempted = rows - 1, successful = finite descendants.
                attempted, successful = max(len(rows) - 1, 0), sum(1 for row in rows[1:] if rmse_of(row) is not None)
                declared = (result.get('attempted_descendants'), result.get('successful_descendants'))
                if declared != (attempted, successful):
                    reasons.append('result_counts_mismatch: declared attempted/successful %s/%s, rows show %d/%d'
                                   % (declared[0], declared[1], attempted, successful))
    else:
        try:
            summarize(identity, episode_id, rows)
        except (ValueError, KeyError, TypeError) as error:
            reasons.append('lineage: ' + str(error))
    # F5 (mandatory): row 0 must reproduce the seed's admitted RMSE bit-exactly. True = equal, False = measured
    # and different (a mismatch), None = not verifiable (no admitted value supplied, or seed row unmeasured);
    # anything but True leaves the episode unverified.
    parent = identity.get('parent_program_sha256')
    admitted = (expected.get('seed_rmse') or {}).get(parent)
    own = rmse_of(rows[0]) if rows else None
    out['admitted_seed_rmse'] = admitted
    if own is None:
        out['seed_unmeasured'] = True
        reasons.append('row0_label_unverifiable: seed row unmeasured')
    elif admitted is None:
        reasons.append('row0_label_unverified: no admitted seed RMSE supplied for parent ' + str(parent))
    elif own == admitted and type(own) is type(admitted):
        out['row0_label_verified'] = True
    else:
        out['row0_label_verified'] = False
        reasons.append('%s: %r != admitted %r' % (ROW0_MISMATCH, own, admitted))
    out['verified'] = not reasons
    return out


def describe(rows, tie=EXACT_TIE):
    """Descriptive statistics of export rows (candidate role, replicate 0) per protocol; exact-tie rule, no eps.

    contexts/pooled keep the one-step protocol (unchanged keys); by_protocol carries every protocol present, Tier-B
    included (credited rows are status measured or incomplete; child = the best lineage descendant).
    """
    grouped = {}
    for row in rows:
        if row.get('role') == 'candidate' and row.get('replicate', 0) == 0:
            grouped.setdefault(row.get('protocol') or CREDIT_PROTOCOL, {}).setdefault(row['context_key'], []).append(row)
    by_protocol = {protocol: describe_protocol(protocol, contexts, tie) for protocol, contexts in sorted(grouped.items())}
    onestep = by_protocol.get(CREDIT_PROTOCOL) or describe_protocol(CREDIT_PROTOCOL, {}, tie)
    return dict(schema=INFO_SCHEMA, tie_rule='exact: |d| <= %g (no replicate floor)' % tie, protocol=CREDIT_PROTOCOL,
                contexts=onestep['contexts'], pooled=onestep['pooled'], by_protocol=by_protocol,
                improvement_definition='ln(r0 / r_child) over finite parent/child pairs (Tier-B: r_child = best lineage descendant)',
                final_tests_opened=False)


def describe_protocol(protocol, contexts, tie=EXACT_TIE):
    statuses = credited_statuses(protocol)
    pooled = dict(candidates=0, measured=0, incomplete=0, finite_children=0, better=0, tied=0, worse=0, flips=0,
                  p_flip_flips=0, p_flip_pairs=0, p_flip_finite_flips=0, p_flip_finite_pairs=0)
    report = {}
    for context_key, group in sorted(contexts.items()):
        measured = [r for r in group if r.get('status') in statuses and finite(r.get('r0'))]
        finite_rows = [r for r in measured if finite(r['child'].get('rmse'))]
        better = sum(r['child']['rmse'] < r['r0'] for r in finite_rows)
        tied = sum(r['child']['rmse'] == r['r0'] for r in finite_rows)
        worse = len(finite_rows) - better - tied
        flips_all, pairs_all = flip_pairs(measured, tie)
        flips_finite, pairs_finite = flip_pairs(finite_rows, tie)
        best_parent = min((r['r0'] for r in measured), default=None)
        best_child = min((r['child']['rmse'] for r in finite_rows), default=None)
        entry = dict(candidates=len(group), measured=len(measured), incomplete=sum(r.get('status') == 'incomplete' for r in measured),
                     finite_children=len(finite_rows), better=better, tied=tied, worse=worse,
                     flips=sum(r.get('flip') or 0 for r in measured),
                     spearman_r0_log_improvement=spearman([r['r0'] for r in finite_rows], [math.log(r['r0'] / r['child']['rmse']) for r in finite_rows]),
                     p_flip=(flips_all / pairs_all if pairs_all else None), p_flip_flips=flips_all, p_flip_pairs=pairs_all,
                     p_flip_finite=(flips_finite / pairs_finite if pairs_finite else None), p_flip_finite_flips=flips_finite, p_flip_finite_pairs=pairs_finite,
                     best_parent=best_parent, best_child=best_child,
                     child_beats_best_parent=(best_child is not None and best_parent is not None and best_child < best_parent))
        report[context_key] = entry
        for key in pooled:
            pooled[key] += entry[key]
    pooled['p_flip'] = pooled['p_flip_flips'] / pooled['p_flip_pairs'] if pooled['p_flip_pairs'] else None
    pooled['p_flip_finite'] = pooled['p_flip_finite_flips'] / pooled['p_flip_finite_pairs'] if pooled['p_flip_finite_pairs'] else None
    return dict(protocol=protocol, credited_statuses=list(statuses), contexts=report, pooled=pooled)


def flip_pairs(rows, tie):
    flips = pairs = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            a, b = rows[i], rows[j]
            if abs(a['S_rmse'] - b['S_rmse']) <= tie or abs(a['r0'] - b['r0']) <= tie:
                continue
            pairs += 1
            flips += (a['S_rmse'] < b['S_rmse']) != (a['r0'] < b['r0'])
    return flips, pairs


class Ledger(Dispatcher):
    """Dispatcher over successor/ledger.sqlite with roles, controls, replicates, seed values and bundles."""

    def __init__(self, root, directory, run_identity, run_output, manifest, *, vocabulary=None,
                 allow_resource_protocol=None, protocols=PROTOCOLS, continuation_evaluator=None):
        # Mirrors Dispatcher.__init__ (dispatch.py) but opens ledger.sqlite with the D2 schema. The evaluator
        # identity is manifest bindings[cpu_executable] (D4), never an operator-typed digest; `vocabulary` is a
        # path within root (hashed) or a 64-hex identity, defaulting to the manifest's vocabulary_sha256.
        # Continuation episodes verify against the manifest's continuation evaluator (continuation_evaluator_binding);
        # `continuation_evaluator` is only ever a manifest-bound digest. The ledger identity names it only when it
        # differs from bindings[cpu_executable], so ledgers of a single-evaluator manifest keep their identity bytes.
        self.root = Path(root).resolve()
        self.directory = within(self.root, directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.run_output = within(self.root, run_output)
        self.evaluator_binding = manifest_evaluator(self.root, manifest, continuation_evaluator=continuation_evaluator)
        evaluator_sha = self.evaluator_binding['evaluator_sha256']
        self.continuation_evaluator_sha256 = self.evaluator_binding['continuation_evaluator_sha256']
        self.evaluator = self.root / self.evaluator_binding['cpu_executable'] if self.evaluator_binding['binary_present'] else None
        self.vocabulary, vocabulary_sha = self.binding(vocabulary if vocabulary is not None else self.evaluator_binding['vocabulary_sha256'])
        require(vocabulary_sha is not None, 'vocabulary identity required: manifest vocabulary_sha256 or an explicit path/digest')
        if allow_resource_protocol is None:
            resource_protocol = RESOURCE_PROTOCOL
        else:
            require(isinstance(allow_resource_protocol, str) and allow_resource_protocol, 'resource protocol override must be a non-empty string')
            resource_protocol = allow_resource_protocol
        self.resource_protocol = resource_protocol
        self.protocols = tuple(protocols)
        self.db = sqlite3.connect(self.directory / 'ledger.sqlite', timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript(SCHEMA)
        identity = dict(run_identity=run_identity, run_output=str(self.run_output), vocabulary_sha256=vocabulary_sha,
                        evaluator_sha256=evaluator_sha, resource_protocol=resource_protocol, protocols=list(self.protocols),
                        ledger='parallel.successor_ledger.v1')
        if self.continuation_evaluator_sha256 != evaluator_sha:
            identity['continuation_evaluator_sha256'] = self.continuation_evaluator_sha256
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO metadata VALUES('identity',?)", (json.dumps(identity, sort_keys=True),))
            existing = self.db.execute("SELECT value FROM metadata WHERE key='identity'").fetchone()
            if json.loads(existing[0]) != identity:
                raise ValueError('Ledger run/evaluator/vocabulary/protocol identity changed')
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES('evaluator_binding',?)",
                            (json.dumps(dict(self.evaluator_binding, opened_at=time.time()), sort_keys=True),))
            if resource_protocol != RESOURCE_PROTOCOL:
                # The override is journaled, never silent: the identity carries the protocol and the metadata the decision.
                self.db.execute("INSERT OR REPLACE INTO metadata VALUES('resource_protocol_override',?)",
                                (json.dumps(dict(resource_protocol=resource_protocol, pinned=RESOURCE_PROTOCOL, at=time.time(),
                                                 argv=list(sys.argv)), sort_keys=True),))
        self.identity = identity
        if resource_protocol != RESOURCE_PROTOCOL:
            self.event(None, 'resource_protocol_override', dict(resource_protocol=resource_protocol, pinned=RESOURCE_PROTOCOL))

    def resource_protocol_override(self):
        row = self.db.execute("SELECT value FROM metadata WHERE key='resource_protocol_override'").fetchone()
        return json.loads(row[0]) if row else None

    def binding(self, value):
        if isinstance(value, str) and HEX64.match(value):
            return None, value
        path = within(self.root, value)
        return path, sha(path)

    def relative(self, path):
        return str(Path(path).resolve().relative_to(self.root))

    def sources(self, admitted_evaluator_sha256):
        require(self.evaluator is not None and self.vocabulary is not None,
                'sources() requires evaluator/vocabulary binaries; successor admission uses bundle() and admitted-sources.json')
        return super().sources(admitted_evaluator_sha256)

    def event(self, episode, status, detail=None):
        with self.db:
            self.db.execute('INSERT INTO events(episode,status,at,detail) VALUES(?,?,?,?)',
                            (episode, status, time.time(), json.dumps(detail, sort_keys=True) if detail is not None else None))

    def transition(self, episode, status, *, owner=None, deadline=None, result=None):
        current = self.db.execute('SELECT status FROM episodes WHERE id=?', (episode,)).fetchone()
        if current and current[0] == 'superseded' and status != 'superseded':
            self.event(episode, 'transition_refused_superseded', dict(requested=status))
            return
        super().transition(episode, status, owner=owner, deadline=deadline, result=result)

    def pending_bundle(self, limit=104, generation=None):
        """Pending jobs in D8 priority order: candidates, controls, gp_final, replicates, reporting."""
        require(0 <= limit <= 104, 'Bundle cannot exceed shared 104-slot allocation')
        order = 'CASE role ' + ' '.join("WHEN '%s' THEN %d" % (role, i) for i, role in enumerate(PRIORITY)) + ' ELSE 99 END'
        where = "status='pending'" + ('' if generation is None else ' AND generation=%d' % int(generation))
        rows = self.db.execute('SELECT payload FROM episodes WHERE %s ORDER BY generation, %s, id LIMIT ?' % (where, order), (limit,))
        vocabulary = self.relative(self.vocabulary) if self.vocabulary is not None else None
        return dict(schema='parallel.gp_jobs.v1', qualification=True, vocabulary=vocabulary, jobs=[json.loads(r[0])['job'] for r in rows])

    def episodes_of(self, generation):
        order = 'CASE role ' + ' '.join("WHEN '%s' THEN %d" % (role, i) for i, role in enumerate(PRIORITY)) + ' ELSE 99 END'
        return list(self.db.execute('SELECT * FROM episodes WHERE generation=? ORDER BY context_key, %s, id' % order, (generation,)))

    # ----------------------------------------------------------------------------- planning
    def plan_generation(self, generation, neural_seeds_json, generic_seeds_json, shuffled_seeds_json, search_best_by_arm,
                        prepared, generic_background, identity_defaults, protocol=None, tierb_contexts=None, execution_evaluator=None):
        """Plan every role for one (generation, context); writes immutable episode inputs and pending rows.

        protocol None plans the one-step v3 roles (candidate/controls/gp_final/replicate_panel). A Tier-B cap variant
        plans instead the top-3 admitted seeds per arm under bounded_seed_lineage_v3 when the context lies in the
        rotating 32-context subset of `tierb_contexts` (every context key of the generation); its receipt is written
        beside the one-step receipt as <context_key>.<protocol>.json. A Tier-B plan binds `execution_evaluator` (the
        digest of the binary that will run the episode; must be bound by the manifest) as identity.evaluator_sha256 and
        the ledger's ordinary evaluator as identity.row0_source_evaluator_sha256 (the label the F5 row-0 check reproduces).
        """
        require(type(generation) is int and generation >= 0, 'generation must be a nonnegative integer')
        require(protocol is None or is_tierb(protocol), 'protocol must be None (one-step v3) or one of ' + ', '.join(TIERB_PROTOCOLS))
        require(execution_evaluator is None or protocol is not None, 'execution_evaluator applies to Tier-B plans only')
        if protocol is not None:
            require(isinstance(execution_evaluator, str) and HEX64.match(execution_evaluator),
                    'a Tier-B plan requires execution_evaluator: the 64-hex digest of the binary that will execute the episodes')
            require(execution_evaluator in self.evaluator_binding['bound_digests'],
                    'execution_evaluator %s... is not bound by the manifest %s' % (execution_evaluator[:12], self.evaluator_binding['manifest_path']))
        defaults = dict(DEFAULTS)
        defaults.update(identity_defaults)
        for key in ('excluded_families', 'rng_seed', 'parent_checkpoint', 'parent_checkpoint_sha256'):
            require(key in defaults, 'identity_defaults requires ' + key)
        if protocol is None:
            require(defaults['evaluations'] == 2 and defaults['candidate_seconds'] == 2 and defaults['seconds'] == 6,
                    'protocol pins evaluations 2, candidate_seconds 2, seconds 6')
        else:
            budget = tierb_budget(protocol)
            for key, value in budget.items():
                require(identity_defaults.get(key, value) == value, 'Tier-B identity_defaults contradict the protocol budget on ' + key)
            defaults.update(budget, protocol=protocol)
        prepared_dir = within(self.root, prepared)
        prepared_bytes = (prepared_dir / 'prepared.json').read_bytes()
        prepared_sha, prepared_doc = digest(prepared_bytes), json.loads(prepared_bytes)
        background = generic_background if isinstance(generic_background, list) else json.loads(within(self.root, generic_background).read_bytes())
        background = list(dict.fromkeys(background))
        require(len(background) >= 80 and all(isinstance(g, str) and g for g in background), 'generic background requires >= 80 distinct genomes')
        populations = {}
        for arm, path in (('neural', neural_seeds_json), ('generic', generic_seeds_json), ('shuffled', shuffled_seeds_json)):
            if path is None:
                continue
            file = within(self.root, path)
            doc = json.loads(file.read_bytes())
            require(doc.get('schema') == 'parallel.seed_population.v1' and doc.get('prepared_sha256') == prepared_sha,
                    arm + ' seeds are not a seed population of this prepared dataset')
            populations[arm] = dict(path=self.relative(file), sha=sha(file), doc=doc)
        require('neural' in populations, 'neural arm seeds required')
        # F5 evidence: each arm's search ledger (observations.json beside search.json) carries the seed's admitted
        # row-0 RMSE; it is stored with the episode so ingest can verify bit-equality without an operator-supplied map.
        admitted_by_arm, admitted_sources = {}, {}
        for arm in ARMS:
            receipt_path = (search_best_by_arm or {}).get(arm)
            if receipt_path is None:
                continue
            observations = within(self.root, receipt_path).parent / 'observations.json'
            if observations.is_file():
                admitted_by_arm[arm] = admitted_rmse(json.loads(observations.read_bytes()))
                admitted_sources[arm] = self.relative(observations)

        def evidence(arm, parent):
            score = admitted_by_arm.get(arm, {}).get(parent)
            return dict(admitted_rmse=score, admitted_source=admitted_sources.get(arm) if score is not None else None)
        cut = populations['neural']['doc']['cut']
        require(all(p['doc']['cut'] == cut for p in populations.values()), 'arm populations disagree on the cut')
        context_key = f'{prepared_sha}:{cut}'
        common = dict(prepared_sha=prepared_sha, generic_sha=digest(encoded(background)), evaluator_sha=self.identity['evaluator_sha256'],
                      vocabulary_sha=self.identity['vocabulary_sha256'], excluded_families=defaults['excluded_families'],
                      evaluations=defaults['evaluations'], seconds=defaults['seconds'], candidate_seconds=defaults['candidate_seconds'],
                      protocol=defaults['protocol'], generation=generation)
        if protocol is not None:
            return self.plan_tierb(generation, protocol, tierb_contexts, dict(
                populations=populations, prepared_dir=prepared_dir, prepared_doc=prepared_doc, prepared_sha=prepared_sha, cut=cut,
                context_key=context_key, background=background, defaults=defaults,
                common=dict(common, candidate_seconds=2, evaluator_sha=execution_evaluator, row0_source_evaluator_sha=self.identity['evaluator_sha256']),
                execution_evaluator=execution_evaluator, admitted_by_arm=admitted_by_arm, admitted_sources=admitted_sources, evidence=evidence))
        planned, skipped, selection = [], [], {}
        genomes = seed_genomes
        # candidates: every unique-byte neural emission, capped in slot order
        neural = populations['neural']
        unique = genomes(neural['doc'], 'neural_policy')
        capped = unique[:defaults['candidate_cap']]
        if len(unique) > len(capped):
            skipped.append(dict(role='candidate', reason='candidate_cap_exceeded', unique=len(unique), cap=defaults['candidate_cap']))
        candidates = plan(neural['doc'], prepared_doc, background, population_sha=neural['sha'], rng_seed=defaults['rng_seed'],
                          arm='neural', role='candidate', seed_origin='neural_policy', replicate=0,
                          select=capped if len(unique) > len(capped) else None, **common)
        candidate_shas = {e['identity']['parent_program_sha256'] for e in candidates['episodes']}
        for episode in candidates['episodes']:
            planned.append(dict(episode=episode, role='candidate', arm='neural', replicate=0, seed_origin='neural_policy',
                                source=neural['path'], source_sha256=neural['sha'],
                                extra=evidence('neural', episode['identity']['parent_program_sha256'])))
        emissions = dict(population_sha=neural['sha'], rows=candidates['emissions'])
        # matched controls: 8 seeds per arm by deterministic sha order, disjoint from candidates
        for arm, role in (('generic', 'control_generic'), ('shuffled', 'control_shuffled')):
            origin = SEED_ORIGIN_BY_ARM[arm]
            population = populations.get(arm)
            if population is None:
                skipped.append(dict(role=role, reason=arm + ' seeds absent'))
                continue
            pool = [g for g in genomes(population['doc'], origin) if digest(g.encode()) not in candidate_shas]
            keyed = sorted(((selection_key(prepared_sha, cut, generation, arm, digest(g.encode())), g) for g in pool))
            chosen = [g for _, g in keyed[:defaults['controls']]]
            selection[role] = dict(pool=len(pool), chosen=[digest(g.encode()) for g in chosen], salt=arm)
            if not chosen:
                skipped.append(dict(role=role, reason='no eligible ' + origin + ' seeds'))
                continue
            result = plan(population['doc'], prepared_doc, background, population_sha=population['sha'], rng_seed=defaults['rng_seed'],
                          arm=arm, role=role, seed_origin=origin, replicate=0, select=chosen, **common)
            for episode in result['episodes']:
                planned.append(dict(episode=episode, role=role, arm=arm, replicate=0, seed_origin=origin,
                                    source=population['path'], source_sha256=population['sha'],
                                    extra=evidence(arm, episode['identity']['parent_program_sha256'])))
        # gp_final: each arm's search.json best_program
        for arm in ARMS:
            receipt_path = (search_best_by_arm or {}).get(arm)
            if receipt_path is None:
                skipped.append(dict(role='gp_final', arm=arm, reason='search receipt absent'))
                continue
            file = within(self.root, receipt_path)
            receipt = json.loads(file.read_bytes())
            require(receipt.get('schema') == SEARCH_SCHEMA and receipt.get('prepared_sha256') == prepared_sha and receipt.get('cut') == cut
                    and receipt.get('final_tests_opened') is False, arm + ' search receipt is not a parallel.search.v1 receipt of this context')
            best = receipt.get('best_program')
            require(isinstance(best, str) and best, arm + ' search receipt has no best_program')
            origin = SEED_ORIGIN_BY_ARM[arm]
            header = populations.get(arm) or neural
            synthetic = {k: header['doc'][k] for k in ('schema', 'prepared_sha256', 'cut', 'training_sha256', 'context_sha256', 'checkpoint_sha256')}
            synthetic['candidates'] = [dict(genome=best, origin=origin)] + [dict(genome=g, origin='generic_bootstrap') for g in background if g != best][:79]
            require(len(synthetic['candidates']) == 80, 'gp_final population requires 79 background genomes')
            best_row = None
            observations = file.parent / 'observations.json'
            if observations.is_file():
                best_row = next((dict(origin=r.get('origin'), rmse=rmse_of(r), program_sha256=r.get('program_sha256'), initial_seed_slot=r.get('initial_seed_slot'))
                                 for r in json.loads(observations.read_bytes()) if r.get('genome') == best), None)
            result = plan(synthetic, prepared_doc, background, population_sha=sha(file), rng_seed=defaults['rng_seed'],
                          arm=arm, role='gp_final', seed_origin=origin, replicate=0, **common)
            for episode in result['episodes']:
                planned.append(dict(episode=episode, role='gp_final', arm=arm, replicate=0, seed_origin=origin,
                                    source=self.relative(file), source_sha256=sha(file),
                                    extra=dict(search_best=best_row, **evidence(arm, episode['identity']['parent_program_sha256']))))
        # replicate panel: 8 candidates by sha order re-run with the derived second rng
        keyed = sorted((selection_key(prepared_sha, cut, generation, 'replicate', e['identity']['parent_program_sha256']), e) for e in candidates['episodes'])
        panel = [e for _, e in keyed[:defaults['replicates']]]
        selection['replicate_panel'] = dict(pool=len(candidates['episodes']), chosen=[e['identity']['parent_program_sha256'] for e in panel], salt='replicate')
        for episode in panel:
            parent = episode['identity']['parent_program_sha256']
            genome = episode['seeds']['candidates'][0]['genome']
            result = plan(neural['doc'], prepared_doc, background, population_sha=neural['sha'],
                          rng_seed=replicate_rng(context_key, generation, parent, 1), arm='neural', role='replicate_panel',
                          seed_origin='neural_policy', replicate=1, select=[genome], **common)
            for replicate in result['episodes']:
                planned.append(dict(episode=replicate, role='replicate_panel', arm='neural', replicate=1, seed_origin='neural_policy',
                                    source=neural['path'], source_sha256=neural['sha'],
                                    extra=dict(replicate_of=episode['episode_id'], **evidence('neural', parent))))
        inserted = self.insert(generation, context_key, prepared_dir, prepared_sha, cut, defaults, planned, emissions)
        counts, without_evidence = {}, 0
        for item in planned:
            counts[item['role']] = counts.get(item['role'], 0) + 1
            without_evidence += (item['extra'] or {}).get('admitted_rmse') is None
        if without_evidence:
            skipped.append(dict(role='*', reason='row0_evidence_absent', episodes=without_evidence,
                                detail='no admitted seed RMSE in the arm search observations; these episodes cannot verify (F5) unless --seed-rmse supplies it'))
        receipt = dict(schema=PLAN_SCHEMA, generation=generation, context_key=context_key, prepared_sha256=prepared_sha, cut=cut,
                       dataset=prepared_doc['dataset'], family=prepared_doc['family'], partition=prepared_doc['meta_partition'],
                       protocol=defaults['protocol'], credit_protocol=CREDIT_PROTOCOL, resource_protocol=RESOURCE_PROTOCOL,
                       evaluator_sha256=self.identity['evaluator_sha256'], producing_checkpoint_sha256=neural['doc']['checkpoint_sha256'],
                       episodes=[dict(episode_id=i['episode']['episode_id'], role=i['role'], arm=i['arm'], replicate=i['replicate'],
                                      seed_origin=i['seed_origin'], parent_program_sha256=i['episode']['identity']['parent_program_sha256'],
                                      rng_seed=i['episode']['identity']['rng_seed'], priority=PRIORITY.index(i['role'])) for i in planned],
                       counts=counts, selection=selection, skipped=skipped, sources={arm: dict(path=p['path'], sha256=p['sha']) for arm, p in populations.items()},
                       row0_evidence=dict(sources=admitted_sources, episodes_with_evidence=len(planned) - without_evidence,
                                          episodes_without_evidence=without_evidence),
                       final_tests_opened=False)
        plans = self.directory / 'plans' / ('generation-%d' % generation)
        receipt_sha = immutable(plans / (context_key + '.json'), receipt)
        return dict(receipt, receipt_sha256=receipt_sha, inserted=inserted)

    def plan_tierb(self, generation, protocol, tierb_contexts, state):
        """Tier-B roles of one context: the top-3 seeds per arm by admitted row-0 RMSE (neural -> candidate, generic ->
        control_generic, shuffled -> control_shuffled), one bounded_seed_lineage episode each under the one-step rng
        derivation, only when the context lies in the generation's rotating 32-context subset."""
        require(tierb_contexts is not None, 'Tier-B planning requires tierb_contexts: every context key (prepared_sha256:cut) of the generation')
        universe = tierb_contexts if isinstance(tierb_contexts, list) else json.loads(within(self.root, tierb_contexts).read_bytes())
        universe = list(dict.fromkeys(universe))
        context_key, populations, defaults, evidence = state['context_key'], state['populations'], state['defaults'], state['evidence']
        require(all(isinstance(k, str) and k.count(':') == 1 for k in universe) and context_key in universe,
                'tierb_contexts must list context keys prepared_sha256:cut and include ' + context_key)
        budget = tierb_budget(protocol)
        subset = tierb_subset(universe, generation)
        selected = context_key in subset
        planned, skipped, selection = [], [], {}
        if not selected:
            skipped.append(dict(role='*', reason='tierb_subset_excluded', rank=tierb_context_rank(state['prepared_sha'], state['cut'], generation)))
        for arm, role in (('neural', 'candidate'), ('generic', 'control_generic'), ('shuffled', 'control_shuffled')):
            if not selected:
                break
            population = populations.get(arm)
            if population is None:
                skipped.append(dict(role=role, reason=arm + ' seeds absent'))
                continue
            origin = SEED_ORIGIN_BY_ARM[arm]
            admitted = state['admitted_by_arm'].get(arm, {})
            pool = seed_genomes(population['doc'], origin)
            ranked = sorted((admitted[digest(g.encode())], digest(g.encode()), g) for g in pool if digest(g.encode()) in admitted)
            chosen = [g for _, _, g in ranked[:TIERB_TOP_SEEDS]]
            selection[role] = dict(pool=len(pool), with_evidence=len(ranked), chosen=[digest(g.encode()) for g in chosen],
                                   rule='top-%d by admitted row-0 RMSE, ties by program sha256' % TIERB_TOP_SEEDS)
            if not chosen:
                skipped.append(dict(role=role, reason='no ' + origin + ' seed with admitted row-0 RMSE'))
                continue
            result = plan(population['doc'], state['prepared_doc'], state['background'], population_sha=population['sha'], rng_seed=defaults['rng_seed'],
                          arm=arm, role=role, seed_origin=origin, replicate=0, select=chosen, **state['common'])
            for episode in rebind_tierb(result['episodes'], protocol):
                planned.append(dict(episode=episode, role=role, arm=arm, replicate=0, seed_origin=origin,
                                    source=population['path'], source_sha256=population['sha'],
                                    extra=evidence(arm, episode['identity']['parent_program_sha256'])))
        inserted = self.insert(generation, context_key, state['prepared_dir'], state['prepared_sha'], state['cut'], defaults, planned,
                               dict(population_sha=populations['neural']['sha'], rows=[]))
        counts = {}
        for item in planned:
            counts[item['role']] = counts.get(item['role'], 0) + 1
        prepared_doc = state['prepared_doc']
        receipt = dict(schema=PLAN_SCHEMA, generation=generation, context_key=context_key, prepared_sha256=state['prepared_sha'], cut=state['cut'],
                       dataset=prepared_doc['dataset'], family=prepared_doc['family'], partition=prepared_doc['meta_partition'],
                       protocol=protocol, credit_protocol=protocol, resource_protocol=budget['resource_protocol'],
                       evaluator_sha256=state['execution_evaluator'], row0_source_evaluator_sha256=self.identity['evaluator_sha256'],
                       evaluator_rule='identity.evaluator_sha256 = the manifest-bound digest of the binary executing the episode; '
                                      'identity.row0_source_evaluator_sha256 = the ordinary evaluator whose admitted row-0 label the seed reproduces (F5)',
                       producing_checkpoint_sha256=populations['neural']['doc']['checkpoint_sha256'],
                       episodes=[dict(episode_id=i['episode']['episode_id'], role=i['role'], arm=i['arm'], replicate=i['replicate'],
                                      seed_origin=i['seed_origin'], parent_program_sha256=i['episode']['identity']['parent_program_sha256'],
                                      rng_seed=i['episode']['identity']['rng_seed'], priority=PRIORITY.index(i['role']), protocol=protocol) for i in planned],
                       counts=counts, selection=selection, skipped=skipped,
                       sources={arm: dict(path=p['path'], sha256=p['sha']) for arm, p in populations.items()},
                       row0_evidence=dict(sources=state['admitted_sources'], episodes_with_evidence=len(planned), episodes_without_evidence=0),
                       tierb=dict(base=TIERB_BASE, cap_seconds=budget['candidate_seconds'], budget=dict(budget), parent_pool=TIERB_PARENT_POOL,
                                  terminations=list(TIERB_TERMINATIONS), top_seeds=TIERB_TOP_SEEDS, subset_size=TIERB_SUBSET,
                                  subset=subset, selected=selected, universe=len(universe), universe_sha256=digest(encoded(sorted(universe))),
                                  rotation_rule='the %d context keys of the generation with the smallest sha256(prepared_sha256:cut:generation:%r); '
                                                'the generation term rotates the subset' % (TIERB_SUBSET, TIERB_SALT),
                                  seed_rule='top-%d seeds per arm by admitted row-0 RMSE (arm search observations); seeds without '
                                            'admitted evidence are never ranked' % TIERB_TOP_SEEDS,
                                  rng_rule='identity_defaults rng_seed for every arm (same derivation as the one-step arms)'),
                       final_tests_opened=False)
        plans = self.directory / 'plans' / ('generation-%d' % generation)
        receipt_sha = immutable(plans / ('%s.%s.json' % (context_key, protocol)), receipt)
        return dict(receipt, receipt_sha256=receipt_sha, inserted=inserted)

    def plan_from_jobs(self, generation, jobs_path, excluded_families, protocol=None, limit=None, partition='meta_train', execution_evaluator=None):
        """Plan every context of one live generation from its jobs.json (the recovery run's parallel.gp_jobs layout).

        Each job row names its arm by the seeds path segment (/neural/, /generic/, /shuffled/), the prepared dataset,
        cut, search output, seeds population, rng_seed and parent checkpoint. Per context (prepared_sha256:cut) the
        three arms' seeds and search.json receipts feed plan_generation; identity defaults come from the neural job;
        the generic background is the union of generic_comparison genomes over the generic-arm seeds of that prepared
        dataset (all cuts, Codex's qualify_dispatch rule). Only contexts of `partition` are planned; for a Tier-B
        protocol only the generation's rotating subset (tierb_subset over that partition) is visited, in subset order;
        `limit` truncates that order. Writes plans/generation-<g>/summary.<protocol>.json (immutable).
        """
        require(type(generation) is int and generation >= 0, 'generation must be a nonnegative integer')
        require(protocol is None or is_tierb(protocol), 'protocol must be None (one-step v3) or one of ' + ', '.join(TIERB_PROTOCOLS))
        require(limit is None or (type(limit) is int and limit >= 1), 'limit must be a positive integer')
        # The execution evaluator is a generation-level input: validated here so a missing or unbound digest refuses the plan
        # outright instead of surfacing as a per-context skip.
        require(execution_evaluator is None or protocol is not None, 'execution_evaluator applies to Tier-B plans only')
        if protocol is not None:
            require(isinstance(execution_evaluator, str) and HEX64.match(execution_evaluator),
                    'a Tier-B plan requires execution_evaluator: the 64-hex digest of the binary that will execute the episodes')
            require(execution_evaluator in self.evaluator_binding['bound_digests'],
                    'execution_evaluator %s... is not bound by the manifest %s' % (execution_evaluator[:12], self.evaluator_binding['manifest_path']))
        require(isinstance(excluded_families, list) and all(isinstance(f, str) for f in excluded_families), 'excluded_families must be a list of strings')
        jobs_file = within(self.root, jobs_path)
        doc = json.loads(jobs_file.read_bytes())
        rows = doc.get('jobs') if isinstance(doc, dict) else None
        require(isinstance(rows, list) and rows, 'jobs.json lists no jobs: ' + self.relative(jobs_file))
        contexts = {}
        for index, job in enumerate(rows):
            where = '%s jobs[%d]' % (self.relative(jobs_file), index)
            require(job.get('generation') == generation, where + ': generation %r is not %d' % (job.get('generation'), generation))
            for key in ('prepared', 'prepared_sha256', 'seeds', 'output', 'parent_checkpoint', 'parent_checkpoint_sha256'):
                require(isinstance(job.get(key), str) and job[key], where + ': %s missing' % key)
            require(type(job.get('cut')) is int and type(job.get('rng_seed')) is int, where + ': cut / rng_seed are not integers')
            arms = [arm for arm in ARMS if '/%s/' % arm in job['seeds']]
            require(len(arms) == 1, where + ': seeds path names no single arm: ' + job['seeds'])
            key = '%s:%d' % (job['prepared_sha256'], job['cut'])
            entry = contexts.setdefault(key, dict(prepared=job['prepared'], prepared_sha256=job['prepared_sha256'], cut=job['cut'], arms={}))
            require(entry['prepared'] == job['prepared'], where + ': prepared path differs from an earlier job of ' + key)
            require(arms[0] not in entry['arms'], where + ': second %s job for context %s' % (arms[0], key))
            entry['arms'][arms[0]] = job
        # The rotating subset is drawn over ADMISSIBLE contexts only (the lane's plan() admission: regression task, the
        # requested partition, family not excluded, clean_evaluation true); the rest are listed, never silently dropped.
        partitions, universe, inadmissible = {}, [], []
        normalize = lambda family: family.removeprefix('rdatasets-')
        for key in sorted(contexts):
            entry = contexts[key]
            prepared_bytes = (within(self.root, entry['prepared']) / 'prepared.json').read_bytes()
            require(digest(prepared_bytes) == entry['prepared_sha256'], 'prepared.json of %s does not hash to the jobs.json prepared_sha256' % key)
            prepared_doc = json.loads(prepared_bytes)
            partitions[key] = prepared_doc.get('meta_partition')
            reasons = []
            if prepared_doc.get('task') != 'regression':
                reasons.append('task %r is not regression' % prepared_doc.get('task'))
            if partition is not None and partitions[key] != partition:
                reasons.append('partition %r is not %s' % (partitions[key], partition))
            if any(normalize(prepared_doc.get('family', '')) == normalize(f) for f in excluded_families):
                reasons.append('family %r is excluded' % prepared_doc.get('family'))
            if not prepared_doc.get('clean_evaluation'):
                reasons.append('clean_evaluation is not true')
            if reasons:
                inadmissible.append(dict(context_key=key, dataset=prepared_doc.get('dataset'), family=prepared_doc.get('family'), reasons=reasons))
            else:
                universe.append(key)
        backgrounds = {}
        for key, entry in contexts.items():
            generic_job = entry['arms'].get('generic')
            if generic_job is None:
                continue
            seeds = json.loads(within(self.root, generic_job['seeds']).read_bytes())
            backgrounds.setdefault(entry['prepared_sha256'], []).extend(c['genome'] for c in seeds.get('candidates', []) if c.get('origin') == 'generic_comparison')
        subset = tierb_subset(universe, generation) if protocol is not None and universe else None
        order = list(subset) if subset is not None else list(universe)
        selected = order[:limit] if limit is not None else order
        receipts, skipped, counts, inserted = [], [], {}, {}
        for key in selected:
            entry = contexts[key]
            neural = entry['arms'].get('neural')
            if neural is None:
                skipped.append(dict(context_key=key, reason='neural job absent'))
                continue
            background = list(dict.fromkeys(backgrounds.get(entry['prepared_sha256'], [])))
            if len(background) < 80:
                skipped.append(dict(context_key=key, reason='generic background below 80 distinct genomes', genomes=len(background)))
                continue
            defaults = dict(excluded_families=list(excluded_families), rng_seed=neural['rng_seed'], parent_checkpoint=neural['parent_checkpoint'],
                            parent_checkpoint_sha256=neural['parent_checkpoint_sha256'])
            searches = {arm: self.relative(within(self.root, job['output']) / 'search.json') for arm, job in entry['arms'].items()
                        if (within(self.root, job['output']) / 'search.json').is_file()}
            try:
                result = self.plan_generation(generation, neural['seeds'], (entry['arms'].get('generic') or {}).get('seeds'),
                                              (entry['arms'].get('shuffled') or {}).get('seeds'), searches, entry['prepared'], background, defaults,
                                              protocol=protocol, tierb_contexts=universe if protocol is not None else None,
                                              execution_evaluator=execution_evaluator)
            except ValueError as refusal:   # a lane/plan refusal of one context is recorded, never a generation-wide abort
                skipped.append(dict(context_key=key, reason='plan_refused', detail=str(refusal)))
                continue
            for role, n in result['counts'].items():
                counts[role] = counts.get(role, 0) + n
            inserted[key] = result['inserted']
            receipts.append(dict(context_key=key, dataset=result['dataset'], family=result['family'], partition=result['partition'],
                                 receipt_sha256=result['receipt_sha256'], counts=result['counts'], skipped=result['skipped'], searches=sorted(searches)))
        # The summary is a pure function of the inputs (immutable, reproducible); how many rows this call inserted is returned beside it.
        summary = dict(schema=PLAN_SUMMARY_SCHEMA, generation=generation, jobs=dict(path=self.relative(jobs_file), sha256=sha(jobs_file), rows=len(rows)),
                       protocol=protocol if protocol is not None else DEFAULTS['protocol'], partition=partition, excluded_families=sorted(excluded_families),
                       evaluator_sha256=self.identity['evaluator_sha256'], continuation_evaluator_sha256=self.continuation_evaluator_sha256,
                       execution_evaluator_sha256=execution_evaluator,
                       contexts=dict(total=len(contexts), admissible=len(universe), visited=len(selected), planned=len(receipts),
                                     by_partition={p: sum(1 for k in contexts if partitions[k] == p) for p in sorted(set(partitions.values()), key=str)}),
                       inadmissible=inadmissible,
                       tierb=None if subset is None else dict(subset=subset, subset_size=TIERB_SUBSET, universe=len(universe),
                                                              universe_sha256=digest(encoded(sorted(universe))), limit=limit),
                       counts=counts, episodes_planned=sum(counts.values()), plans=receipts, skipped=skipped, final_tests_opened=False)
        plans = self.directory / 'plans' / ('generation-%d' % generation)
        summary_sha = immutable(plans / ('summary.%s.json' % (protocol if protocol is not None else 'onestep')), summary)
        return dict(summary, summary_sha256=summary_sha, episodes_inserted=sum(inserted.values()), inserted_by_context=inserted)

    def insert(self, generation, context_key, prepared_dir, prepared_sha, cut, defaults, planned, emissions):
        inserted = 0
        with self.db:
            for item in planned:
                episode = item['episode']
                identity = episode['identity']
                directory = self.directory / 'episodes' / episode['episode_id']
                identity_sha = immutable(directory / 'identity.json', identity)
                require(identity_sha == episode['episode_id'], 'episode identity bytes do not reproduce the episode id')
                seeds_sha = immutable(directory / 'seeds.json', episode['seeds'])
                immutable(directory / 'manifest.json', dict(schema='parallel.seed_experiment.v2', continuation_identity_sha256=identity_sha, final_tests_opened=False))
                job = dict(prepared=self.relative(prepared_dir), prepared_sha256=prepared_sha, cut=cut, output=self.relative(directory / 'search'),
                           seeds=self.relative(directory / 'seeds.json'), seeds_sha256=seeds_sha, rng_seed=identity['rng_seed'], generation=generation,
                           parent_checkpoint=defaults['parent_checkpoint'], parent_checkpoint_sha256=defaults['parent_checkpoint_sha256'],
                           continuation=dict(identity=self.relative(directory / 'identity.json'), identity_sha256=identity_sha))
                payload = dict(identity=identity, job=job, source_population=item['source'], source_population_sha256=item['source_sha256'],
                               role=item['role'], arm=item['arm'], replicate=item['replicate'], seed_origin=item['seed_origin'],
                               priority=PRIORITY.index(item['role']), context_key=context_key, extra=item['extra'])
                old = self.db.execute('SELECT payload FROM episodes WHERE id=?', (episode['episode_id'],)).fetchone()
                if old and json.loads(old[0])['identity'] != identity:
                    raise ValueError('Episode identity collision')
                cursor = self.db.execute(
                    "INSERT OR IGNORE INTO episodes(id,payload,status,arm,role,replicate,generation,context_key,parent_program_sha256,seed_origin) "
                    "VALUES(?,?,'pending',?,?,?,?,?,?,?)",
                    (episode['episode_id'], json.dumps(payload, sort_keys=True), item['arm'], item['role'], item['replicate'], generation,
                     context_key, identity['parent_program_sha256'], item['seed_origin']))
                inserted += cursor.rowcount
            for emission in emissions['rows']:
                self.db.execute('INSERT OR IGNORE INTO emissions VALUES(?,?,?,?,?)',
                                (generation, emissions['population_sha'], emission['slot'], emission.get('episode_id'), emission['status']))
        return inserted

    # ----------------------------------------------------------------------------- lifecycle
    def supersede(self, generation, started_generation):
        """Episodes of `generation` still pending/running become terminal once generation + 2 starts."""
        require(type(started_generation) is int and started_generation >= generation + 2,
                'generation %s is superseded only when generation %s starts (given %s)' % (generation, generation + 2, started_generation))
        rows = list(self.db.execute("SELECT id,status FROM episodes WHERE generation=? AND status IN ('pending','running')", (generation,)))
        for row in rows:
            self.transition(row['id'], 'superseded', result=dict(reason='superseded: generation %d started' % started_generation,
                                                                  previous_status=row['status'], started_generation=started_generation))
        return dict(schema=SUPERSEDE_SCHEMA, generation=generation, started_generation=started_generation,
                    superseded=[r['id'] for r in rows], count=len(rows), final_tests_opened=False)

    def start_generation(self, generation):
        older = [r[0] for r in self.db.execute("SELECT DISTINCT generation FROM episodes WHERE generation<=? AND status IN ('pending','running')", (generation - 2,))]
        return [self.supersede(g, generation) for g in sorted(older)]

    def ingest_completed(self, episode_dir, *, generation=None, arm=None, role=None, replicate=None, seed_origin=None,
                         seed_rmse=None, incumbent_rmse=None):
        """Read one episode directory, verify it, and record r0 / r_child / flip / S_rmse (never a label)."""
        directory = within(self.root, episode_dir)
        identity_bytes = (directory / 'identity.json').read_bytes()
        identity = json.loads(identity_bytes)
        episode_id = digest(identity_bytes)
        record = self.db.execute('SELECT * FROM episodes WHERE id=?', (episode_id,)).fetchone()
        context_key = f"{identity['prepared_sha256']}:{identity['cut']}"
        if record is not None:
            binding = dict(generation=record['generation'], arm=record['arm'], role=record['role'], replicate=record['replicate'], seed_origin=record['seed_origin'])
            for key, given in (('generation', generation), ('arm', arm), ('role', role), ('replicate', replicate), ('seed_origin', seed_origin)):
                require(given is None or given == binding[key], 'episode binding conflict on ' + key)
            if record['status'] == 'superseded':
                self.event(episode_id, 'ingest_refused_superseded')
                return dict(episode_id=episode_id, status='superseded', context_key=context_key, ingested=False, **binding)
            payload = json.loads(record['payload'])
        else:
            binding = dict(generation=generation if generation is not None else identity.get('generation'),
                           arm=arm or identity.get('seed_arm'), role=role or identity.get('role'),
                           replicate=replicate if replicate is not None else identity.get('replicate', 0),
                           seed_origin=seed_origin or identity.get('seed_origin', 'neural_policy'))
            require(all(v is not None for v in binding.values()), 'adopting an unplanned episode requires generation/arm/role/replicate/seed_origin bindings')
            require(binding['arm'] in ARMS and binding['role'] in PRIORITY and binding['replicate'] in (0, 1), 'invalid episode binding')
            job = dict(prepared=None, prepared_sha256=identity['prepared_sha256'], cut=identity['cut'], output=self.relative(directory / 'search'),
                       seeds=self.relative(directory / 'seeds.json'), seeds_sha256=identity['seeds_sha256'], rng_seed=identity['rng_seed'],
                       generation=binding['generation'], parent_checkpoint=None, parent_checkpoint_sha256=identity['producing_checkpoint_sha256'],
                       continuation=dict(identity=self.relative(directory / 'identity.json'), identity_sha256=episode_id))
            payload = dict(identity=identity, job=job, adopted=True, source_population=None, source_population_sha256=identity.get('source_population_sha256'),
                           priority=PRIORITY.index(binding['role']), context_key=context_key, extra=None, **binding)
            with self.db:
                self.db.execute(
                    "INSERT INTO episodes(id,payload,status,arm,role,replicate,generation,context_key,parent_program_sha256,seed_origin) "
                    "VALUES(?,?,'pending',?,?,?,?,?,?,?)",
                    (episode_id, json.dumps(payload, sort_keys=True), binding['arm'], binding['role'], binding['replicate'], binding['generation'],
                     context_key, identity['parent_program_sha256'], binding['seed_origin']))
            self.event(episode_id, 'adopted', binding)
        # Row-0 evidence: the admitted RMSE stored at plan time (arm search observations) plus any --seed-rmse map;
        # an explicit map wins for its parents. Without either, verification refuses ('row0_label_unverified').
        planned_evidence = (payload.get('extra') or {}).get('admitted_rmse') if isinstance(payload.get('extra'), dict) else None
        evidence = {identity['parent_program_sha256']: planned_evidence} if planned_evidence is not None else {}
        evidence.update(seed_rmse or {})
        expected = dict(evaluator_sha256=self.identity['evaluator_sha256'], resource_protocol=self.resource_protocol, protocols=self.protocols,
                        partition='meta_train', vocabulary_sha256=self.identity['vocabulary_sha256'], seed_rmse=evidence,
                        continuation_evaluator_sha256=self.continuation_evaluator_sha256)
        verification = verify_episode(directory, expected)
        rows = verification.pop('rows')
        summary = json.loads((directory / 'result.json').read_bytes()) if (directory / 'result.json').is_file() else None
        measurement = seed_value(identity, rows or [], summary)
        fits = dict(attempted=len(rows or []), successful=sum(rmse_of(r) is not None for r in rows or []),
                    timeouts=sum((r.get('failure') or '') == 'candidate_or_view_timeout' for r in rows or []),
                    other_failures=sum(r.get('failure') is not None and r.get('failure') != 'candidate_or_view_timeout' for r in rows or []),
                    programs=[r.get('program_sha256') for r in rows or [] if rmse_of(r) is not None])
        successor = dict(verification=verification, measurement=measurement, fits=fits,
                         elapsed_seconds=(summary or {}).get('elapsed_seconds'), runner_status=(summary or {}).get('status'),
                         seed_genome=None, incumbent_rmse=incumbent_rmse)
        try:
            successor['seed_genome'] = json.loads((directory / 'seeds.json').read_bytes())['candidates'][0]['genome']
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            successor['seed_genome'] = None
        previous = json.loads(record['result']) if record is not None and record['result'] else {}
        if not isinstance(previous, dict):
            previous = dict(previous_result=previous)
        previous['successor'] = successor
        status = 'completed' if rows is not None else ('failed' if record is None or record['status'] not in TERMINAL else record['status'])
        with self.db:
            self.db.execute('UPDATE episodes SET status=?,result=? WHERE id=?', (status, json.dumps(previous, sort_keys=True), episode_id))
        self.event(episode_id, 'ingested', dict(status=status, verified=verification['verified'], reasons=verification['reasons'][:8]))
        detail = dict(episode_id=episode_id, role=binding['role'], verified=verification['verified'], reasons=verification['reasons'],
                      observations_sha256=verification['observations_sha256'], search_sha256=verification['search_sha256'],
                      completed_at=verification['completed_at'], driver_sha256=verification['driver_sha256'],
                      row0_label_verified=verification['row0_label_verified'], rows=len(rows or []), elapsed_seconds=successor['elapsed_seconds'])
        protocol = measurement['protocol']
        key = (binding['generation'], context_key, binding['arm'], identity['parent_program_sha256'], binding['replicate'], protocol)
        existing = self.db.execute('SELECT episode FROM seed_values WHERE generation=? AND context_key=? AND arm=? AND parent_program_sha256=? '
                                   'AND replicate=? AND protocol=?', key).fetchone()
        if existing is not None and existing[0] != episode_id:
            self.event(episode_id, 'seed_value_duplicate_role', dict(kept=existing[0]))
        else:
            with self.db:
                self.db.execute(
                    'INSERT OR REPLACE INTO seed_values(generation,context_key,arm,role,parent_program_sha256,replicate,protocol,'
                    'own_rmse,child_program_sha256,child_rmse,child_failure,flip,S_rmse,'
                    'attempted_descendants,successful_descendants,termination_reason,lineage_size,'
                    'seed_beats_incumbent,child_beats_incumbent,episode,verified,detail) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    key[:3] + (binding['role'],) + key[3:] + (
                        measurement['r0'], measurement['child']['program_sha256'], measurement['child']['rmse'], measurement['child']['failure'],
                        measurement['flip'], measurement['S_rmse'],
                        measurement['attempted_descendants'], measurement['successful_descendants'], measurement['termination_reason'],
                        measurement['lineage_size'],
                        None if incumbent_rmse is None or measurement['r0'] is None else int(measurement['r0'] < incumbent_rmse),
                        None if incumbent_rmse is None or measurement['child']['rmse'] is None else int(measurement['child']['rmse'] < incumbent_rmse),
                        episode_id, int(verification['verified']), json.dumps(detail, sort_keys=True)))
        return dict(episode_id=episode_id, status=status, context_key=context_key, ingested=True, verified=verification['verified'],
                    reasons=verification['reasons'], r0=measurement['r0'], child=measurement['child'], flip=measurement['flip'],
                    S_rmse=measurement['S_rmse'], measurement_status=measurement['status'], completed_at=verification['completed_at'],
                    observations_sha256=verification['observations_sha256'], search_sha256=verification['search_sha256'],
                    row0_label_verified=verification['row0_label_verified'], protocol=protocol,
                    attempted_descendants=measurement['attempted_descendants'], successful_descendants=measurement['successful_descendants'],
                    termination_reason=measurement['termination_reason'], lineage_size=measurement['lineage_size'], **binding)

    # ----------------------------------------------------------------------------- outputs
    def export_rows(self, generation):
        episodes = self.episodes_of(generation)
        require(episodes, 'no episodes recorded for generation %s' % generation)
        rows = []
        for record in episodes:
            payload = json.loads(record['payload'])
            identity = payload['identity']
            result = json.loads(record['result']) if record['result'] else {}
            successor = result.get('successor') if isinstance(result, dict) else None
            measurement = (successor or {}).get('measurement') or {}
            verification = (successor or {}).get('verification') or {}
            if measurement.get('status') in ('measured', 'incomplete'):
                status = measurement['status']
            elif record['status'] in ('deadline', 'superseded'):
                status = record['status']
            else:
                status = 'failed'
            protocol = measurement.get('protocol') or credit_protocol_of(identity.get('protocol'))
            rows.append(dict(schema=ROW_SCHEMA, generation=generation, dataset=identity['dataset'], family=identity['family'],
                             prepared_sha256=identity['prepared_sha256'], cut=identity['cut'], context_key=record['context_key'],
                             episode_id=record['id'], identity_sha256=record['id'], protocol=protocol, role=record['role'], arm=record['arm'],
                             replicate=record['replicate'], seed_origin=record['seed_origin'], seed_program_sha256=identity['parent_program_sha256'],
                             seed_genome=(successor or {}).get('seed_genome'), producing_checkpoint_sha256=identity['producing_checkpoint_sha256'],
                             evaluator_sha256=identity['evaluator_sha256'], driver_sha256=verification.get('driver_sha256'),
                             r0=measurement.get('r0'), child=measurement.get('child') or dict(program_sha256=None, rmse=None, failure=None, genetic_operator=None),
                             flip=measurement.get('flip'), S_rmse=measurement.get('S_rmse'), replicate_pair=None, status=status,
                             attempted_descendants=measurement.get('attempted_descendants'), successful_descendants=measurement.get('successful_descendants'),
                             termination_reason=measurement.get('termination_reason'), lineage_size=measurement.get('lineage_size'),
                             episode_status=record['status'], failure=measurement.get('failure'), verified=verification.get('verified'),
                             row0_label_verified=verification.get('row0_label_verified'),
                             admitted_seed_rmse=verification.get('admitted_seed_rmse'),
                             verification_reasons=list(verification.get('reasons') or [])[:8],
                             partition=identity['partition'], completed_at=verification.get('completed_at')))
        pairs = {}
        for row in rows:
            pairs.setdefault((row['context_key'], row['arm'], row['seed_program_sha256'], row['protocol']), {})[row['replicate']] = row
        for members in pairs.values():
            if 0 in members and 1 in members:
                pair = dict(child_rmse_rep0=members[0]['child']['rmse'], child_rmse_rep1=members[1]['child']['rmse'])
                members[0]['replicate_pair'] = dict(pair)
                members[1]['replicate_pair'] = dict(pair)
        return rows

    def export(self, generation):
        """Immutable successor/seed-values-<generation>.json once every episode of the generation is terminal."""
        episodes = self.episodes_of(generation)
        require(episodes, 'no episodes recorded for generation %s' % generation)
        open_episodes = [r['id'] for r in episodes if r['status'] not in TERMINAL]
        require(not open_episodes, 'export requires every episode of generation %s to be terminal; %d pending/running (supersede first)'
                % (generation, len(open_episodes)))
        rows = self.export_rows(generation)
        info = describe(rows)
        counts = dict(episodes=len(rows), by_status={}, by_role={})
        for row in rows:
            counts['by_status'][row['status']] = counts['by_status'].get(row['status'], 0) + 1
            counts['by_role'][row['role']] = counts['by_role'].get(row['role'], 0) + 1
        counts.update({k: info['pooled'][k] for k in ('measured', 'finite_children', 'better', 'tied', 'worse', 'flips')})
        counts['by_protocol'] = {p: {k: r['pooled'][k] for k in ('candidates', 'measured', 'incomplete', 'finite_children', 'better', 'tied', 'worse', 'flips')}
                                 for p, r in info['by_protocol'].items()}
        document = dict(schema=ROW_SCHEMA, generation=generation, protocol=CREDIT_PROTOCOL, protocols=sorted({r['protocol'] for r in rows}),
                        evaluator_sha256=self.identity['evaluator_sha256'], continuation_evaluator_sha256=self.continuation_evaluator_sha256,
                        resource_protocol=self.resource_protocol, contexts=sorted({r['context_key'] for r in rows}), rows=rows, counts=counts,
                        final_tests_opened=False)
        path = self.directory / ('seed-values-%d.json' % generation)
        return dict(path=str(path), sha256=immutable(path, document), counts=counts)

    def info_report(self, generation):
        return describe(self.export_rows(generation))

    def next_outcome(self, generation, quality_report=None, outcome='best_initial'):
        """F4(b) producer: generation g candidate seeds joined to generation g+1 outcomes (parallel.successor_next_outcome.v1).

        outcome='best_initial': the neural arm's seed-only best of generation g+1 in the same context, from that
        generation's quality report (higher outcome = better, so -best_initial_neural_seed); every seed of a context
        shares it. outcome='child_beats_incumbent': the seed's own generation g+1 episode (recursion through the loop)
        from this ledger's seed_values. Units are never mixed within one file.
        """
        require(outcome in NEXT_OUTCOMES, 'outcome must be one of ' + ', '.join(NEXT_OUTCOMES))
        following = generation + 1
        seeds = sorted({(r['context_key'], r['seed_program_sha256'], r['prepared_sha256'], r['cut']) for r in self.export_rows(generation)
                        if r['role'] == 'candidate' and r['replicate'] == 0 and r['status'] == 'measured' and r['protocol'] == CREDIT_PROTOCOL})
        require(seeds, 'generation %s has no measured candidate seeds' % generation)
        records, source = [], None
        if outcome == 'best_initial':
            require(quality_report is not None, 'best_initial outcome requires the generation %d quality report' % following)
            report_path = within(self.root, quality_report)
            report = json.loads(report_path.read_bytes())
            require(report.get('schema') == QUALITY_SCHEMA and report.get('final_tests_opened') is False and isinstance(report.get('rows'), list),
                    'not a parallel.continuous_quality.v1 report with final_tests_opened false: ' + str(quality_report))
            stamped = {r.get('generation') for r in report['rows'] if isinstance(r, dict) and type(r.get('generation')) is int}
            require(not stamped or stamped == {following}, 'quality report rows are stamped generation %s, expected %d' % (sorted(stamped), following))
            by_context = {}
            for row in report['rows']:
                if isinstance(row, dict) and row.get('arm') == 'neural' and finite(row.get('best_initial_neural_seed')):
                    by_context.setdefault((row.get('prepared_sha256'), row.get('cut')), row['best_initial_neural_seed'])
            for context_key, seed, prepared, cut in seeds:
                best = by_context.get((prepared, cut))
                if best is not None:
                    records.append(dict(context_key=context_key, seed_program_sha256=seed, outcome=-best, next_generation=following))
            source = dict(kind='quality_report', path=self.relative(report_path), sha256=sha(report_path), contexts=len(by_context),
                          definition='-best_initial_neural_seed of the neural arm at generation %d (higher = better), shared by every seed of the context' % following)
        else:
            rows = self.db.execute('SELECT context_key,parent_program_sha256,child_beats_incumbent FROM seed_values '
                                   'WHERE generation=? AND replicate=0 AND protocol=? AND child_beats_incumbent IS NOT NULL', (following, CREDIT_PROTOCOL))
            by_seed = {(r[0], r[1]): float(r[2]) for r in rows}
            for context_key, seed, _, _ in seeds:
                if (context_key, seed) in by_seed:
                    records.append(dict(context_key=context_key, seed_program_sha256=seed, outcome=by_seed[(context_key, seed)], next_generation=following))
            source = dict(kind='ledger_seed_values', generation=following, rows=len(by_seed),
                          definition='1[child_rmse < incumbent_rmse] of the seed\'s own generation %d episode (replicate 0)' % following)
        document = dict(schema=NEXT_OUTCOME_SCHEMA, generation=generation, next_generation=following, outcome=outcome, source=source,
                        rows=records, counts=dict(seeds=len(seeds), joined=len(records), unjoined=len(seeds) - len(records),
                                                  contexts=len({s[0] for s in seeds}), joined_contexts=len({r['context_key'] for r in records})),
                        credit_protocol=CREDIT_PROTOCOL, evaluator_sha256=self.identity['evaluator_sha256'],
                        continuation_evaluator_sha256=self.continuation_evaluator_sha256, final_tests_opened=False)
        name = 'next-outcome-%d.json' % generation if outcome == 'best_initial' else 'next-outcome-%d-%s.json' % (generation, outcome)
        path = self.directory / name
        return dict(document, path=str(path), sha256=immutable(path, document))

    def bundle(self, generation):
        """One ledger per (generation, context) from verified episodes; admitted-sources lists only fully verified bundles."""
        episodes = [r for r in self.episodes_of(generation) if r['status'] == 'completed']
        by_context = {}
        for record in episodes:
            result = json.loads(record['result']) if record['result'] else {}
            successor = result.get('successor') if isinstance(result, dict) else None
            require(successor is not None, 'completed episode %s lacks successor ingest; call ingest_completed first' % record['id'])
            by_context.setdefault(record['context_key'], []).append((record, successor))
        receipts = []
        unmeasured_only = ['row0_label_unverifiable: seed row unmeasured']
        for context_key in sorted(by_context):
            items = by_context[context_key]
            # verified implies row0_label_verified True (verify_episode); the explicit check keeps admission fail-closed.
            # An episode whose seed fit never measured (timeout) carries no label and cannot contradict F5: it is
            # listed as failed, its rows never enter the bundle, and it does not block the context's other episodes.
            verified = [(r, s) for r, s in items if s['verification']['verified'] and s['verification']['row0_label_verified'] is True]
            failed = [r['id'] for r, s in items if s['verification'].get('seed_unmeasured') and s['verification']['reasons'] == unmeasured_only]
            unverified = [dict(episode_id=r['id'], reasons=(s['verification']['reasons'] or ['row0_label_unverified: row-0 evidence absent'])[:8],
                               row0_label_verified=s['verification']['row0_label_verified'])
                          for r, s in items if not (s['verification']['verified'] and s['verification']['row0_label_verified'] is True)
                          and r['id'] not in failed]
            reasons = sorted({reason.split(':', 1)[0] for u in unverified for reason in u['reasons']})
            entry = dict(generation=generation, context_key=context_key, episodes=len(items), verified=len(verified), unverified=unverified,
                         failed_unmeasured=failed, admitted=bool(verified) and not unverified, path=None, observations_sha256=None, search_sha256=None,
                         row0_label_verified=bool(verified) and all(s['verification']['row0_label_verified'] is True for _, s in verified),
                         row0_mismatches=sum(1 for _, s in items if s['verification']['row0_label_verified'] is False),
                         reasons=reasons, reason=None if not unverified else '%d unverified episode(s): %s' % (len(unverified), ', '.join(reasons)))
            if verified:
                rows, provenance, identity = [], [], None
                for record, successor in verified:
                    payload = json.loads(record['payload'])
                    identity = payload['identity']
                    output = within(self.root, payload['job']['output'])
                    require(sha(output / 'observations.json') == successor['verification']['observations_sha256']
                            and sha(output / 'search.json') == successor['verification']['search_sha256'],
                            'episode source changed after ingest: ' + record['id'])
                    episode_rows = read_rows(output)
                    rows.extend(episode_rows)
                    provenance.append(dict(path=self.relative(output), observations_sha256=successor['verification']['observations_sha256'],
                                           search_sha256=successor['verification']['search_sha256'], continuation_sha256=record['id'],
                                           role=record['role'], arm=record['arm'], protocol=credit_protocol_of(identity.get('protocol')),
                                           rows=len(episode_rows)))
                measured = [(r['metrics']['rmse'] * r['target_scale'], r['genome']) for r in rows if rmse_of(r) is not None and finite(r.get('target_scale'))]
                best = min(measured, default=None)
                directory = self.directory / 'bundles' / ('generation-%d' % generation) / context_key
                observations_sha = immutable(directory / 'observations.json', rows)
                receipt = dict(schema=SEARCH_SCHEMA, observations_sha256=observations_sha, evaluator_sha256=self.identity['evaluator_sha256'],
                               evaluator_protocol=EVALUATOR_PROTOCOL, completed_evaluations=len(rows),
                               completed_at=max(s['verification']['completed_at'] for _, s in verified),
                               prepared_sha256=identity['prepared_sha256'], cut=identity['cut'], dataset=identity['dataset'], family=identity['family'],
                               partition=identity['partition'], vocabulary_sha256=identity['vocabulary_sha256'],
                               best_program=None if best is None else best[1], best_validation_rmse=None if best is None else best[0],
                               unique_programs=len({r['program_sha256'] for r in rows}), resource_protocol=self.resource_protocol,
                               final_tests_opened=False,
                               bundle=dict(schema=BUNDLE_SCHEMA, generation=generation, context_key=context_key, episodes=provenance,
                                           unverified_episodes=len(unverified), failed_unmeasured_episodes=len(failed),
                                           row0_label_verified=True, credit_protocol=CREDIT_PROTOCOL,
                                           protocols=sorted({p['protocol'] for p in provenance})))
                entry.update(path=self.relative(directory / 'observations.json'), observations_sha256=observations_sha,
                             search_sha256=immutable(directory / 'search.json', receipt), rows=len(rows))
            with self.db:
                self.db.execute('INSERT OR REPLACE INTO bundles VALUES(?,?,?,?,?,?,?,?)',
                                (generation, context_key, entry['path'], entry['observations_sha256'], entry['search_sha256'], len(items),
                                 int(entry['admitted']), json.dumps(entry, sort_keys=True)))
            receipts.append(entry)
        summary = dict(schema=BUNDLES_SCHEMA, generation=generation, bundles=receipts, admitted=sum(e['admitted'] for e in receipts),
                       contexts=len(receipts), final_tests_opened=False)
        if receipts:
            immutable(self.directory / 'bundles' / ('generation-%d' % generation) / 'bundle-receipt.json', summary)
        summary['admitted_sources'] = self.admitted_sources()
        return summary

    def admitted_sources(self):
        """Rewrite successor/admitted-sources.json: sources only from bundles whose every episode verified with row0_label_verified true."""
        rows = list(self.db.execute('SELECT * FROM bundles ORDER BY generation, context_key'))
        details = [json.loads(r['detail']) for r in rows]
        admitted, excluded = [], []
        for row, detail in zip(rows, details):
            if row['admitted'] and detail.get('row0_label_verified') is True:
                admitted.append(row)
                continue
            reasons = list(detail.get('reasons') or [])
            if row['admitted'] and detail.get('row0_label_verified') is not True:
                reasons.append('row0_unverified')
            excluded.append(dict(generation=detail['generation'], context_key=detail['context_key'], unverified=detail['unverified'],
                                 verified=detail['verified'], reasons=reasons, row0_mismatches=detail.get('row0_mismatches', 0),
                                 reason=detail.get('reason') or ('row0_unverified' if 'row0_unverified' in reasons else 'no verified episode')))
        by_generation = {}
        for row in admitted:
            by_generation.setdefault(str(row['generation']), []).append(row['path'])
        document = dict(schema=SOURCES_SCHEMA, evaluator_sha256=self.identity['evaluator_sha256'], resource_protocol=self.resource_protocol,
                        resource_protocol_override=self.resource_protocol_override(), evaluator_binding=self.evaluator_binding,
                        generation=max((r['generation'] for r in rows), default=None), sources=[r['path'] for r in admitted], by_generation=by_generation,
                        row0_label_verified=bool(admitted), row0_unverified_bundles=sum(1 for e in excluded if 'row0_unverified' in e['reasons']),
                        row0_mismatches=sum(d.get('row0_mismatches', 0) for d in details),
                        row0_rule='sources list only bundles whose every episode verified with row0_label_verified true (F5)',
                        meta_validation_excluded=True, excluded=excluded, final_tests_opened=False)
        runtime.atomic_bytes(self.directory / 'admitted-sources.json', encoded(document))
        return document

    def economics(self, generation, *, reserved_core_seconds=None, generation_end=None, tier='A'):
        """Economics receipt per D8; never invents a denominator (reserved seconds come from the runner or result.json)."""
        episodes = self.episodes_of(generation)
        require(episodes, 'no episodes recorded for generation %s' % generation)
        statuses, attempted, successful, timeouts, failures, elapsed, elapsed_known, within_generation, programs = {}, 0, 0, 0, 0, 0.0, True, 0, set()
        for record in episodes:
            statuses[record['status']] = statuses.get(record['status'], 0) + 1
            result = json.loads(record['result']) if record['result'] else {}
            successor = result.get('successor') if isinstance(result, dict) else None
            if record['status'] != 'completed' or successor is None:
                failures += record['status'] in ('failed', 'deadline')
                continue
            fits = successor['fits']
            attempted += fits['attempted']
            successful += fits['successful']
            timeouts += fits['timeouts']
            failures += fits['other_failures']
            programs.update(fits['programs'])
            if successor.get('elapsed_seconds') is None:
                elapsed_known = False
            else:
                elapsed += successor['elapsed_seconds']
            completed_at = successor['verification'].get('completed_at')
            if generation_end is not None and finite(completed_at) and completed_at <= generation_end:
                within_generation += 1
        if reserved_core_seconds is not None:
            reserved, source = reserved_core_seconds, 'supplied'
        elif elapsed_known and statuses.get('completed'):
            reserved, source = elapsed, 'sum_of_result_elapsed_seconds'
        else:
            reserved, source = None, None
        receipt = dict(schema=ECONOMICS_SCHEMA, generation=generation, tier=tier, credit_protocol=CREDIT_PROTOCOL, reserved_core_seconds=reserved,
                       reserved_source=source, attempted_fits=attempted, successful_fits=successful, unique_successful_fits=len(programs),
                       unique_fits_per_core_second=(len(programs) / reserved if reserved else None), timeouts=timeouts, failures=failures,
                       superseded=statuses.get('superseded', 0), completed=statuses.get('completed', 0),
                       completed_within_generation=(within_generation if generation_end is not None else None), generation_end=generation_end,
                       episodes=statuses, final_tests_opened=False)
        with self.db:
            self.db.execute('INSERT OR REPLACE INTO economics VALUES(?,?,?,?,?,?,?,?,?,?,?)',
                            (generation, tier, reserved, attempted, successful, receipt['unique_fits_per_core_second'], timeouts, failures,
                             receipt['superseded'], receipt['completed_within_generation'], json.dumps(receipt, sort_keys=True)))
        path = self.directory / 'economics' / ('generation-%d.json' % generation)
        return dict(receipt, receipt_sha256=immutable(path, receipt), path=str(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--directory', required=True, help='ledger directory within root (e.g. runs/<run>/successor)')
    parser.add_argument('--run-identity', required=True)
    parser.add_argument('--run-output', required=True)
    parser.add_argument('--manifest', required=True, help='continuous manifest within root; evaluator = bindings[cpu_executable]')
    parser.add_argument('--vocabulary', help='path within root or 64-hex sha (default: manifest vocabulary_sha256)')
    parser.add_argument('--allow-resource-protocol', metavar='PROTOCOL',
                        help='explicit, journaled override of the pinned ' + RESOURCE_PROTOCOL + ' resource protocol')
    parser.add_argument('--continuation-evaluator', metavar='SHA256',
                        help='continuation evaluator digest; must equal the manifest-derived continuation binding '
                             '(continuation_source_admission configuration) or, without one, be a manifest-bound digest')
    parser.add_argument('--generation', type=int, required=True)
    parser.add_argument('action', choices=('ingest', 'export', 'bundle', 'economics', 'info', 'supersede', 'pending', 'next-outcome', 'plan-generation'))
    parser.add_argument('--jobs', help='plan-generation: the generation jobs.json within root (default <run-output>/generation-%06d/jobs.json)')
    parser.add_argument('--protocol', choices=TIERB_PROTOCOLS, help='plan-generation: Tier-B cap variant; default plans the one-step v3 roles')
    parser.add_argument('--limit', type=int, help='plan-generation: plan at most this many contexts (subset order)')
    parser.add_argument('--partition', default='meta_train', help='plan-generation: meta partition of the contexts to plan (default meta_train)')
    parser.add_argument('--excluded-families', help='plan-generation: comma-separated families; default the manifest withdrawn_families')
    parser.add_argument('--execution-evaluator', metavar='SHA256',
                        help='plan-generation (Tier-B, required): manifest-bound digest of the binary that will execute the episodes')
    parser.add_argument('--episode', action='append', default=[], help='ingest: episode directory within root (repeatable)')
    parser.add_argument('--arm', choices=ARMS, help='ingest: binding for episodes planned outside this ledger (v2 identities)')
    parser.add_argument('--role', choices=PRIORITY)
    parser.add_argument('--replicate', type=int, choices=(0, 1))
    parser.add_argument('--seed-origin', choices=('neural_policy', 'generic_comparison'))
    parser.add_argument('--seed-rmse', help='ingest: {program_sha256: admitted rmse} json or the arm search observations.json (F5 evidence)')
    parser.add_argument('--started-generation', type=int)
    parser.add_argument('--reserved-core-seconds', type=float)
    parser.add_argument('--generation-end', type=float)
    parser.add_argument('--quality-report', help='next-outcome: generation g+1 quality-report.json within root')
    parser.add_argument('--outcome', choices=NEXT_OUTCOMES, default='best_initial')
    args = parser.parse_args()
    ledger = Ledger(args.root, args.directory, args.run_identity, args.run_output, args.manifest, vocabulary=args.vocabulary,
                    allow_resource_protocol=args.allow_resource_protocol, continuation_evaluator=args.continuation_evaluator)
    if args.action == 'ingest':
        require(args.episode, 'ingest requires at least one --episode')
        seed_rmse = load_seed_rmse(ledger.root, args.seed_rmse) if args.seed_rmse else None
        result = [ledger.ingest_completed(episode, generation=args.generation, arm=args.arm, role=args.role, replicate=args.replicate,
                                          seed_origin=args.seed_origin, seed_rmse=seed_rmse) for episode in args.episode]
    elif args.action == 'export':
        result = ledger.export(args.generation)
    elif args.action == 'plan-generation':
        jobs = args.jobs if args.jobs else str(Path(args.run_output) / ('generation-%06d' % args.generation) / 'jobs.json')
        if args.excluded_families is not None:
            excluded = [f for f in args.excluded_families.split(',') if f]
        else:
            excluded = json.loads(within(ledger.root, args.manifest).read_bytes()).get('withdrawn_families') or []
        result = ledger.plan_from_jobs(args.generation, jobs, list(excluded), protocol=args.protocol, limit=args.limit, partition=args.partition,
                                       execution_evaluator=args.execution_evaluator)
    elif args.action == 'next-outcome':
        result = ledger.next_outcome(args.generation, quality_report=args.quality_report, outcome=args.outcome)
    elif args.action == 'bundle':
        result = ledger.bundle(args.generation)
    elif args.action == 'economics':
        result = ledger.economics(args.generation, reserved_core_seconds=args.reserved_core_seconds, generation_end=args.generation_end)
    elif args.action == 'info':
        result = ledger.info_report(args.generation)
    elif args.action == 'supersede':
        require(args.started_generation is not None, 'supersede requires --started-generation')
        result = ledger.supersede(args.generation, args.started_generation)
    else:
        result = ledger.pending_bundle(generation=args.generation)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
