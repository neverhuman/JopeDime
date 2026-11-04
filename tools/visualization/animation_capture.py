#!/usr/bin/env python3
"""Prepare and capture ten provenance-bound synthetic animation examples.

Six existing diverse cases are retained byte-for-byte. Four additions continue
the predeclared 24-pool max-min learned-embedding selection, without looking at
model quality, generated source or holdout outcomes. Native capture, five-fold
inference and source proofs are separate, receipted stages; no model rerolls.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
EVIDENCE = Path('/Users/bentaylor/Documents/ChatGPT/JopeDime/blender-render-20261103')
TOOLCHAIN = Path('/Users/bentaylor/.rustup/toolchains/1.97.1-aarch64-apple-darwin/bin')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    path = Path(path)
    require(path.stat().st_size <= 512 * 1024 * 1024, 'JSON archive exceeds bound')
    return json.loads(path.read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def git(*args):
    return subprocess.check_output(['rtk', 'proxy', 'git', *args], cwd=WORKTREE, text=True).strip()


def command(args, output, name, *, env=None, timeout=900):
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter_ns()
    receipt = {'argv': [str(v) for v in args], 'cwd': str(WORKTREE),
               'started_utc': datetime.now(timezone.utc).isoformat(),
               'scope': 'whole proof/build process; never a source-generation timing'}
    with (output / (name + '.stdout.log')).open('xb') as stdout, (output / (name + '.stderr.log')).open('xb') as stderr:
        result = subprocess.run([str(v) for v in args], cwd=WORKTREE, stdout=stdout, stderr=stderr,
                                env={**os.environ, **(env or {})}, timeout=timeout, check=False)
    receipt.update(exit_code=result.returncode, process_wall_ns=time.perf_counter_ns()-started,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    write(output / (name + '.receipt.json'), receipt)
    require(result.returncode == 0, 'stage failed; preserve logs and do not reroll: ' + name)
    return receipt


def prepare(args):
    import select_examples as selection
    root = args.output_root
    require(not root.exists(), 'animation output already exists')
    catalog_path = args.pool / 'catalog.json'
    selected_path = args.pool / 'selection.json'
    catalog, prior = read(catalog_path), read(selected_path)
    require(sha(catalog_path) == prior['catalog_sha256'] and prior['all_24_candidates_observed']
            and prior['holdout_quality_used_for_selection'] is False, 'pool/selection identity differs')
    traces = [Path(t['path']) for t in prior['traces']]
    require(all(sha(path) == pin['sha256'] for path, pin in zip(traces, prior['traces'], strict=True)), 'pool trace changed')
    embeddings, observations, trace_ids, _assets = selection.load_embeddings(catalog, traces)
    ids = sorted(embeddings)
    matrix = selection.distances([embeddings[cid] for cid in ids])
    require(ids == prior['pool_ids'] and matrix == prior['pool_pairwise_cosine_distances'], 'pool distances changed')
    chosen = [item['id'] for item in prior['selected']]
    original = read(args.original_native / 'trace.json')
    require(len(chosen) == len(original['cases']) == 6, 'expected original six cases')
    by_id = {item['id']: item for item in catalog['candidates']}
    steps = []
    while len(chosen) < 10:
        next_id = min((cid for cid in ids if cid not in chosen),
                      key=lambda cid: (-min(matrix[ids.index(cid)][ids.index(other)] for other in chosen), cid))
        steps.append({'pool_id': next_id, 'minimum_distance_to_previous': min(matrix[ids.index(next_id)][ids.index(other)] for other in chosen)})
        chosen.append(next_id)
    root.mkdir(parents=True)
    cases = []
    for index, pool_id in enumerate(chosen):
        item = by_id[pool_id]
        cid = 'case-' + chr(97 + index)
        source = Path(item['csv_path'])
        require(sha(source) == item['dataset_sha256'], 'pool CSV changed')
        if index < 6:
            native = original['cases'][index]
            require(native['id'] == cid and native['dataset_sha256'] == item['dataset_sha256'], 'original six identity/order changed')
        else:
            target = root / 'new-inputs' / (cid + '.csv')
            target.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, target)
        family = item['description']
        label = family.split(';')[0].replace(' inputs', '').replace('unchanged prior six-case ', '')
        cases.append({'case_id': cid, 'display_name': 'Synthetic · ' + label.capitalize(),
            'rows': item['rows'], 'features': item['features'], 'origin': 'synthetic', 'family': family,
            'dataset_sha256': item['dataset_sha256'], 'pool_id': pool_id, 'source_csv': str(source),
            'pool_observation': observations[pool_id], 'new_native_capture': index >= 6})
    write(root / 'dataset-catalog.json', {'schema': 'jain.animation-dataset-catalog/v1', 'status': 'PREPARED',
        'cases': cases, 'selection': 'fixed original six, then deterministic farthest point by max-min learned DOPE256 cosine distance; lexical ID tie',
        'selection_used_quality_or_source': False, 'pool_catalog': identity(catalog_path),
        'original_selection': identity(selected_path), 'pool_native_traces': trace_ids,
        'extension_steps': steps, 'origin': 'synthetic', 'source': identity(__file__)})
    print(json.dumps({'status': 'PREPARED', 'catalog': str(root / 'dataset-catalog.json'),
                      'new': [(c['case_id'], c['pool_id'], c['family']) for c in cases[6:]]}), flush=True)


def build_native(args):
    root = args.output_root
    require(git('rev-parse', 'HEAD') == args.expected_head, 'HEAD changed before native build')
    require(not git('status', '--porcelain', '--untracked-files=normal', '--', 'crates/jope-service'), 'service source is dirty')
    history = root / 'native-history'
    history.mkdir()
    for name in ('visual_run', 'visual_fit_dsl', 'visual_inference_metric'):
        old = WORKTREE / 'target/release/examples' / name
        if old.exists():
            shutil.copy2(old, history / name)
    write(history / 'receipt.json', {'status': 'PRESERVED', 'binaries': [identity(p) for p in history.iterdir() if p.is_file()]})
    environment = {'CARGO_INCREMENTAL': '0', 'CARGO_PROFILE_DEV_DEBUG': 'line-tables-only',
        'CARGO_TARGET_DIR': str(WORKTREE / 'target'), 'RUSTC': str(TOOLCHAIN / 'rustc'),
        'JOPE_BUILD_SOURCE_SHA': args.expected_head}
    cargo = ['rtk', 'proxy', str(TOOLCHAIN / 'cargo')]
    common = ['--release', '--locked', '-j4', '-p', 'jope-service', '--features', 'visual-trace']
    command([*cargo, 'build', *common, '--example', 'visual_run', '--example', 'visual_fit_dsl', '--example', 'visual_inference_metric'],
            root / 'build', 'native-build', env=environment)
    for name in ('visual_inference_metric', 'visual_fit_dsl'):
        command([*cargo, 'test', *common, '--example', name], root / 'build', name + '-tests', env=environment)
    require(git('rev-parse', 'HEAD') == args.expected_head and not git('status', '--porcelain', '--untracked-files=normal', '--', 'crates/jope-service'),
            'source changed during native build')
    write(root / 'build/executables.receipt.json', {'status': 'PASS', 'source_commit': args.expected_head,
        'binaries': [identity(WORKTREE / 'target/release/examples' / name) for name in ('visual_run', 'visual_fit_dsl', 'visual_inference_metric')]})


def capture_native(args):
    root = args.output_root
    require(git('rev-parse', 'HEAD') == args.expected_head and not git('status', '--porcelain', '--untracked-files=normal', '--', 'crates/jope-service'),
            'native capture needs unchanged clean service source')
    built = read(root / 'build/executables.receipt.json')
    require(built['source_commit'] == args.expected_head and all(sha(b['path']) == b['sha256'] for b in built['binaries']), 'native executable/source freeze differs')
    invocation = read(args.original_native / 'invocation-receipt.json')
    binary = WORKTREE / 'target/release/examples/visual_run'
    environment = {name: '1' for name in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    command(['rtk', 'proxy', binary, invocation['argv'][1], invocation['argv'][2], root / 'new-inputs', root / 'new-native'],
            root / 'native-command', 'four-native-runs', env=environment)
    captured = read(root / 'new-native/trace.json')
    require(captured['source_identity_verified'] is True and captured['source_commit'] == args.expected_head
            and len(captured['cases']) == 4, 'new native runs lack exact source identity or cases')
    write(root / 'native-command/verified.receipt.json', {'status': 'PASS', 'trace': identity(root / 'new-native/trace.json'),
        'source_commit': args.expected_head, 'service_clean': not bool(git('status', '--porcelain', '--untracked-files=normal', '--', 'crates/jope-service'))})
    print('PASS four actual native runs: ' + str(root / 'new-native/trace.json'), flush=True)


def assemble(args):
    """Retain each true native archive; aggregate only the catalog and contracts."""
    root = args.output_root
    dataset_catalog = read(root / 'dataset-catalog.json')
    archives = [args.original_native.resolve(), (root / 'new-native').resolve()]
    merged, models, fit_cases, feature_cases, archive_refs = [], [], [], [], []
    asset_identity = None
    native_root = root / 'native-merged'
    native_root.mkdir()
    for archive in archives:
        trace_path = archive / 'trace.json'
        trace = read(trace_path)
        require(trace['source_identity_verified'] is True and trace['source_commit'], 'source-unverified native archive')
        require(asset_identity is None or trace['assets'] == asset_identity, 'authenticated bundle changed between native archives')
        asset_identity = trace['assets']
        archive_refs.append(identity(trace_path))
        model_rows = read(archive / 'model-paths.json')
        require(isinstance(model_rows, list), 'native model index is not a list')
        models.extend(model_rows)
        for case in trace['cases']:
            cid = case['id']
            declared = next(c for c in dataset_catalog['cases'] if c['case_id'] == cid)
            csv_path = archive / (cid + '.csv')
            require(sha(csv_path) == declared['dataset_sha256'] == case['dataset_sha256']
                    and case['input_rows'] == declared['rows'] == 96 and case['features'] == 4,
                    'animation dataset/shape differs from actual native run')
            shutil.copyfile(csv_path, native_root / (cid + '.csv'))
            proof_dir = native_root / cid / 'proof'
            proof_dir.mkdir(parents=True)
            for filename in ('split.json', 'summary.json'):
                shutil.copyfile(archive / cid / 'proof' / filename, proof_dir / filename)
            journal_path = archive / case['native_journal_path']
            require(sha(journal_path) == case['native_journal_sha256'], 'native journal hash changed')
            journal = read(journal_path)
            require(journal['run_id'] == case['run_id'], 'journal/run identity differs')
            condition = next(call for call in journal['calls'] if call['identity']['phase'] == 'fit_conditioning')
            slate = next(call for call in journal['calls'] if call['identity']['phase'] == 'ranker_admission')
            programs = slate['identity']['revealed_programs'] + slate['identity']['proposal_programs']
            require(len(programs) == 12 and len(set(programs)) == 12, 'initial actual native slate differs from12candidate contract')
            hidden = next(a for a in condition['trace']['affine'] if a['name'] == 'dope_model.router.task_latent_head')['input']
            provenance = {'native_trace_path': str(trace_path), 'native_trace_sha256': sha(trace_path),
                'native_journal_path': str(journal_path), 'native_journal_sha256': sha(journal_path),
                'native_source_commit': trace['source_commit'], 'fit_matrix_sha256': condition['identity']['fit_matrix_sha256'],
                'source_run_id': case['run_id']}
            fit_cases.append({'id': cid, 'dataset_sha256': case['dataset_sha256'], 'dataset_file': str(csv_path),
                'rows': len(case['fit_row_ids']), 'features': case['features'], 'fit_row_ids': case['fit_row_ids'],
                'holdout_row_ids': case['holdout_row_ids'], 'input_header': case['input_header'],
                'table_preview': case['table_preview'], 'v256': condition['outputs']['v256'],
                'col64x17': condition['outputs']['col'], 'reference_dope629': condition['outputs']['dope629'],
                'reference_hidden256': hidden, 'provenance': provenance})
            feature_cases.append({'id': cid, 'dataset_sha256': case['dataset_sha256'], 'native_call_id': slate['call_id'],
                'programs': programs, 'provenance': {**provenance, 'native_call_id': slate['call_id'],
                    'program_source': 'first actual native ranker admission; four reference rows plus eight proposal rows'}})
            merged.append(copy.deepcopy(case))
    require([c['id'] for c in merged] == ['case-' + chr(97+i) for i in range(10)], 'ten-case order/inventory differs')
    write(native_root / 'model-paths.json', models)
    # A catalog aggregate is not a new native run or one source commit. All G
    # input provenance points to the immutable original per-run trace/journal.
    write(native_root / 'trace.json', {'schema': 'jain.visual-run-trace/v1', 'assets': asset_identity,
        'cases': merged, 'source_identity_verified': False, 'source_commit': None,
        'capture_authority': 'aggregate catalog of ten separately verified actual CPU native runs; see original_archives',
        'aggregate_only': True, 'original_archives': archive_refs})
    write(root / 'inputs/native-fit-inputs.json', {'schema': 'jain.h256-fit-inputs/v1',
        'scope': 'exact original FIT-only descriptors from ten explicitly synthetic tables', 'cases': fit_cases})
    request = {'schema': 'jain.h256-program-slate-request/v1', 'cases': feature_cases}
    write(root / 'inputs/feature-request.json', request)
    exports = []
    fit_binary = WORKTREE / 'target/release/examples/visual_fit_dsl'
    for index, group in enumerate((feature_cases[:6], feature_cases[6:])):
        req = root / f'inputs/feature-request-{index}.json'
        out = root / f'inputs/native-features-{index}.json'
        write(req, {'schema': request['schema'], 'cases': group})
        command(['rtk', 'proxy', fit_binary, '--features', req, out], root / 'feature-commands', f'export-{index}')
        exports.append(read(out))
    for key in ('schema', 'helper_source_sha256', 'feature_recipe_sha256', 'feature_names_sha256'):
        require(exports[0][key] == exports[1][key], 'native feature recipe/source differs between chunks')
    aggregate = {key: exports[0][key] for key in ('schema', 'helper_source_sha256', 'feature_recipe_sha256', 'feature_names_sha256')}
    aggregate.update(cases=exports[0]['cases'] + exports[1]['cases'],
        aggregate_request_sha256=sha(root / 'inputs/feature-request.json'),
        chunk_exports=[identity(root / f'inputs/native-features-{i}.json') for i in range(2)],
        scope='ordered aggregation of two actual native DSL-only raw412/proj16 exports; no fitting or quality selection')
    write(root / 'inputs/native-program-features.json', aggregate)
    write(root / 'assembly.receipt.json', {'status': 'PASS', 'original_native_archives': archive_refs,
        'native_catalog': identity(native_root / 'trace.json'), 'fit_inputs': identity(root / 'inputs/native-fit-inputs.json'),
        'native_features': identity(root / 'inputs/native-program-features.json'), 'case_count': len(merged),
        'original_six_preserved': True, 'controller': identity(__file__)})
    print('PASS ten-case native provenance/inputs/features: ' + str(root), flush=True)


def neural(args):
    root, h256 = args.output_root, EVIDENCE / 'h256'
    environment = {name: '1' for name in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    command(['rtk', 'proxy', sys.executable, HERE / 'h256_stage2_inputs.py',
        '--critic', h256 / 'stage2-custody-20261103T1911Z/assets/critic.pt',
        '--source', h256 / 'stage2-custody-20261103T1911Z/source',
        '--native-features', root / 'inputs/native-program-features.json', '--fit-inputs', root / 'inputs/native-fit-inputs.json',
        '--round882', h256 / 'assets/round-882.pt', '--output-dir', root / 'bridge'], root / 'neural-commands', 'fresh-critics', env=environment)
    command(['rtk', 'proxy', sys.executable, HERE / 'h256_stage2_inference.py', '--root', h256,
        '--stage2-custody', h256 / 'stage2-custody-20261103T1911Z', '--inputs', root / 'inputs/native-fit-inputs.json',
        '--native-features', root / 'inputs/native-program-features.json', '--bridge-inputs', root / 'bridge/bridge-inputs.json',
        '--output-dir', root / 'raw-neural', '--limit', '10'], root / 'neural-commands', 'ten-saved-g-capture', env=environment)
    command(['rtk', 'proxy', sys.executable, HERE / 'h256_stage2_validate.py', root / 'raw-neural/trace.json',
        '--receipt', root / 'raw-neural/validation.receipt.json'], root / 'neural-commands', 'strict-neural-validation', env=environment)
    print('PASS ten genuine saved-fivefold CPU inference cases', flush=True)


def sources(args):
    root = args.output_root
    environment = {name: '1' for name in ('RAYON_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS')}
    command(['rtk', 'proxy', sys.executable, HERE / 'h256_gp_source.py', '--trace', root / 'raw-neural/trace.json',
        '--native-proof', root / 'native-merged', '--fit-binary', WORKTREE / 'target/release/examples/visual_fit_dsl',
        '--output-dir', root / 'gp-source', '--code-pages-dir', root / 'proof-source-pages'],
        root / 'source-commands', 'actual-selected-gp-source', env=environment, timeout=1800)
    command(['rtk', 'proxy', sys.executable, HERE / 'h256_stage2_validate.py', root / 'gp-source/trace.json',
        '--gp-root', root / 'gp-source', '--receipt', root / 'gp-source/validation.receipt.json'],
        root / 'source-commands', 'strict-source-validation', env=environment)
    write(root / 'capture.receipt.json', {'status': 'PASS', 'dataset_catalog': identity(root / 'dataset-catalog.json'),
        'raw_neural': identity(root / 'raw-neural/trace.json'), 'source_trace': identity(root / 'gp-source/trace.json'),
        'source_controller_receipt': identity(root / 'gp-source/controller.receipt.json'),
        'strict_validation': identity(root / 'gp-source/validation.receipt.json'), 'controller': identity(__file__),
        'case_count': 10, 'origin': 'synthetic', 'training': False, 'gpu_inference': False})
    print('PASS ten actual G-selected native GP proofs and complete Rust exports', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'build-native', 'capture-native', 'assemble', 'neural', 'sources'))
    parser.add_argument('--output-root', type=Path, default=EVIDENCE / 'h256/animation-ten-v1')
    parser.add_argument('--pool', type=Path, default=EVIDENCE / 'embedding-selection')
    parser.add_argument('--original-native', type=Path, default=EVIDENCE / 'single-run-final')
    parser.add_argument('--expected-head')
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    if args.stage in ('build-native', 'capture-native'):
        require(args.expected_head and len(args.expected_head) == 40, 'exact expected source HEAD is required')
    {'prepare': prepare, 'build-native': build_native, 'capture-native': capture_native,
     'assemble': assemble, 'neural': neural, 'sources': sources}[args.stage](args)


if __name__ == '__main__':
    main()
