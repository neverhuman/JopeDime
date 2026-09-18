"""Validate the exact scope of the development probes; never qualify deployment."""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path


def identity(domain, payload):
    digest = hashlib.sha256()
    for value in (b'feat-math.identity.v1', domain.encode(), payload):
        digest.update(len(value).to_bytes(8, 'little'))
        digest.update(value)
    return digest.hexdigest()


def screening(path, manifest_path, reference_path):
    doc = json.loads(Path(path).read_text())
    manifest_bytes = Path(manifest_path).read_bytes()
    manifest = json.loads(manifest_bytes)
    reference = json.loads(Path(reference_path).read_text())
    if (doc.get('schema') != 'kernel.screening-probe.v1' or doc.get('manifest') != manifest
            or doc.get('manifest_identity') != identity('screening-manifest', manifest_bytes)
            or doc.get('qualified_for_deployment') is not False):
        raise ValueError('Probe context/identity mismatch')

    def indexed(value):
        result = {}
        cases = value['cases']
        expected = {(seed, mechanism, genome) for seed in manifest['seeds']
                    for mechanism in manifest['mechanisms'] for genome in manifest['genomes']}
        if len(cases) != len(manifest['seeds']) * len(manifest['mechanisms']):
            raise ValueError('Missing or duplicate probe case')
        for case in cases:
            if len(case['models']) != len(manifest['genomes']):
                raise ValueError('Missing or duplicate learner')
            for model in case['models']:
                key = case['seed'], case['mechanism'], model['genome']
                loss = model['mse']
                if key in result or not isinstance(loss, (float, int)) or not math.isfinite(loss) or loss < 0:
                    raise ValueError('Duplicate or nonfinite probe result')
                result[key] = float(loss).hex()
        if set(result) != expected:
            raise ValueError('Probe coverage mismatch')
        return result

    actual = indexed(doc)
    if actual != indexed(reference):
        raise ValueError('Deterministic CPU probe loss changed')
    wins = {mechanism: {g: 0 for g in manifest['genomes']} for mechanism in manifest['mechanisms']}
    for case in doc['cases']:
        winner = min(case['models'], key=lambda m: m['mse'])['genome']
        wins[case['mechanism']][winner] += 1
    return {'gate': 'actual-learner-counterexample', 'results_compared': len(actual),
            'exact_loss_replay': True, 'wins': wins,
            'scope': 'Synthetic development fixture, not campaign false-rejection frequency'}


def assumptions(path, reference_path):
    doc = json.loads(Path(path).read_text())
    if doc != json.loads(Path(reference_path).read_text()):
        raise ValueError('Exact-rational assumption fixture changed')
    cases = {c['name']: c for c in doc['cases']}
    a = cases['small_omission_well_conditioned']
    b = cases['same_omission_sensitive_direction']
    c = cases['bound_unavailable_does_not_mean_inaccurate']
    if (a['raw_omitted_norm'] != b['raw_omitted_norm'] or not a['contract_accepts']
            or b['contract_accepts'] or Fraction(b['actual_parameter_error']) != 999
            or c['bound_available'] or not c['within_error_budget']):
        raise ValueError('Sensitivity/abstention counterexample failed')
    return {'gate': 'exact-assumption-counterexamples', 'cases': len(cases),
            'scope': 'Exact diagonal rational fixtures; not floating-point certification or speedup'}
