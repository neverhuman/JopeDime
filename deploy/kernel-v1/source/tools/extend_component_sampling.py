#!/usr/bin/env python3
"""Extend frozen sampling recipes; weights and exported CModules are never rewritten.

--prepare-only uses CPU sampler references. A later explicit --qualify-cuda run
checks just the new lengths/shapes against the original frozen model source.
"""
import argparse
import experiment_runtime as runtime
import hashlib
import json
import os
from pathlib import Path
import sys


def load_registries(root, kind, paths):
    """Union compatible, hash-verified registries without changing their files."""
    merged = None
    combined_parity = None
    anchor = None
    records = []
    tests = []
    seen_tests = set()
    for path in paths:
        path = root / path
        spec = json.loads(path.read_text())
        for name in ['module', 'sampling', 'parity']:
            if not (sha(root / spec[name]) == spec[name + '_sha256']):
                raise ValueError('sampling validation failed at original line 27')
        parity = json.loads((root / spec['parity']).read_text())
        if not (parity['passed'] and parity['module_sha256'] == spec['module_sha256']):
            raise ValueError('sampling validation failed at original line 29')
        if not (parity['sampling_sha256'] == spec['sampling_sha256']):
            raise ValueError('sampling validation failed at original line 30')
        sampler = json.loads((root / spec['sampling']).read_text())
        if anchor is None:
            anchor = spec
            merged = dict(sampler, recipes=dict(sampler['recipes']))
            combined_parity = dict(parity)
        else:
            if not (spec['module_sha256'] == anchor['module_sha256']):
                raise ValueError(f'{kind} frozen modules differ')
            if not ({k:v for k,v in sampler.items() if k != 'recipes'} == {k:v for k,v in merged.items() if k != 'recipes'}):
                raise ValueError(f'{kind} sampler configurations differ')
            for key, recipe in sampler['recipes'].items():
                if key in merged['recipes']:
                    if not (merged['recipes'][key] == recipe):
                        raise ValueError(f'{kind} conflicting existing recipe {key}')
                else:
                    merged['recipes'][key] = recipe
        for test in parity.get('tests', []) + parity.get('extension_tests', []):
            identity = json.dumps(test, sort_keys=True)
            if identity not in seen_tests:
                seen_tests.add(identity)
                tests.append(test)
        records.append(dict(path=str(path.relative_to(root)), spec_sha256=sha(path), spec=spec, recipes=len(sampler['recipes'])))
    if not (anchor is not None):
        raise ValueError(f'{kind} registry list is empty')
    combined_parity['tests'] = tests
    combined_parity['source_parity_sha256'] = [record['spec']['parity_sha256'] for record in records]
    return anchor, merged, combined_parity, records


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2) + '\n'
    if path.exists():
        if not (json.loads(path.read_text()) == value):
            raise ValueError(f'Immutable artifact changed: {path}')
    else:
        runtime.write(path, value)


def validate_coverage(request, target_tests, foundation_tests):
    expected_target = {(task,n,kind) for task,n in request['target_new_lengths'] for kind in ['random','constant','endpoints']}
    actual_target = [(v['task'],v['length'],v['kind']) for v in target_tests]
    expected_foundation = {tuple(v) for v in request['foundation_new_shapes']}
    actual_foundation = [(v['context'],v['query'],v['width'],v['batch']) for v in foundation_tests]
    if not (len(actual_target) == len(set(actual_target)) and set(actual_target) == expected_target):
        raise ValueError('Target parity coverage missing or duplicated')
    if not (len(actual_foundation) == len(set(actual_foundation)) and set(actual_foundation) == expected_foundation):
        raise ValueError('Foundation parity coverage missing or duplicated')
    if not (all(v['passed'] for v in target_tests + foundation_tests)):
        raise ValueError('Frozen module parity failed; no runtime specs published')


def publish(root, output, request, specs, parities, world_size):
    target_tests, foundation_tests, shards = [], [], []
    for rank in range(world_size):
        path = output/f'qualification-rank-{rank}.json'
        shard = json.loads(path.read_text())
        if not (shard['rank'] == rank and shard['world_size'] == world_size):
            raise ValueError('sampling validation failed at original line 86')
        if not (shard['request_sha256'] == sha(output/'request.json')):
            raise ValueError('sampling validation failed at original line 87')
        if not (shard['passed']):
            raise ValueError(f'Qualification shard {rank} failed')
        target_tests.extend(shard['target_tests'])
        foundation_tests.extend(shard['foundation_tests'])
        shards.append(dict(path=path.name, sha256=sha(path)))
    validate_coverage(request, target_tests, foundation_tests)
    for kind, tests in [('target', target_tests), ('foundation', foundation_tests)]:
        spec = specs[kind]
        parity = dict(parities[kind])
        parity.update(schema=f'parallel.{kind}_sampling_extension_parity.v2', sampling_sha256=request['sampling_sha256'][kind], base_parity_sha256=spec['parity_sha256'], extension_request_sha256=sha(output/'request.json'), extension_tests=tests, passed=True, weights_reexported=False, numerical_backend='LibTorch 2.7 CUDA FP32; existing immutable module', qualification_shards=shards)
        parity['tests'] = parity['tests'] + tests
        save(output/kind/'parity.json', parity)
        extended = dict(spec)
        extended.update(sampling=str((output/kind/'sampling.json').relative_to(root)), sampling_sha256=sha(output/kind/'sampling.json'), parity=str((output/kind/'parity.json').relative_to(root)), parity_sha256=sha(output/kind/'parity.json'))
        save(output/f'{kind}-spec.json', extended)
    print(json.dumps(dict(passed=True, target_tests=len(target_tests), foundation_tests=len(foundation_tests), output=str(output), shards=world_size)), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--jobs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--target-spec', type=Path, action='append', help='Repeat to union previously qualified sampler registries')
    p.add_argument('--foundation-spec', type=Path, action='append', help='Repeat to union previously qualified sampler registries')
    p.add_argument('--rank', type=int, default=0)
    p.add_argument('--world-size', type=int, default=1)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--qualify-cuda', action='store_true')
    mode.add_argument('--merge-qualification', action='store_true', help='Verify all shards and publish specs; no CUDA work')
    a = p.parse_args()
    r = a.root.resolve()
    if not (a.output.is_absolute() and a.output.is_relative_to(r)):
        raise ValueError('sampling validation failed at original line 120')
    if not (a.world_size > 0 and 0 <= a.rank < a.world_size):
        raise ValueError('sampling validation failed at original line 121')
    if (a.qualify_cuda and a.world_size > 1) or a.merge_qualification:
        if not ((a.output/'request.json').exists()):
            raise ValueError('Run --prepare-only once before parallel qualification')
    import numpy as np
    source = r / 'assets/source/target_maker'
    sys.path.insert(0, str(source))
    from target_maker.data import stable_seed, sample_indices
    from target_maker.encode import _window_indices

    jobs = json.loads(a.jobs.read_text())
    lengths = set()
    prepared = {}
    widths = {}
    for job in jobs['jobs']:
        key = job['prepared']
        if key not in prepared:
            path = r / key / 'prepared.json'
            if not (sha(path) == job['prepared_sha256']):
                raise ValueError('sampling validation failed at original line 138')
            prepared[key] = json.loads(path.read_text())
        m = prepared[key]
        c = next(c for c in m['cuts'] if c['index'] == job['cut'])
        if not (m['task'] in ['binary', 'regression']):
            raise ValueError('sampling validation failed at original line 142')
        n = len(c['train_indices'])
        lengths.add((0 if m['task'] == 'binary' else 1, n))
        widths.setdefault(n, set()).add(len(m['storage']['feature_columns']))

    specs = {}
    samplers = {}
    parities = {}
    source_registries = {}
    for kind, paths in [('target', a.target_spec or [Path('manifests/target-spec.json')]), ('foundation', a.foundation_spec or [Path('manifests/foundation-spec.json')])]:
        specs[kind], samplers[kind], parities[kind], source_registries[kind] = load_registries(r, kind, paths)
    target_new = []
    foundation_new = []
    for task, n in sorted(lengths):
        key = f'{task}:{n}'
        recipe = dict(classical=[sample_indices(n, 32, np.random.default_rng(stable_seed('fusion-signature-inference', 17, task, n, i, 32)))[0].tolist() for i in range(2)], neural=[v.tolist() for v in _window_indices(n, 32, 32, 2, 17)])
        recipe['neural'] = (recipe['neural'] + recipe['neural'])[:2]
        if key in samplers['target']['recipes']:
            if not (samplers['target']['recipes'][key] == recipe):
                raise ValueError(f'Target sampler reference changed for {key}')
        else:
            target_new.append((task, n))
            samplers['target']['recipes'][key] = recipe
    shapes = set()
    for n, feature_widths in sorted(widths.items()):
        indices = np.random.default_rng(20260707).permutation(n)
        nc = min(768, max(8, 3*n//4))
        nq = min(256, n-nc)
        recipe = dict(context=indices[:nc].tolist(), query=(indices[max(0, n-8):] if nq < 4 else indices[nc:nc+nq]).tolist())
        if str(n) in samplers['foundation']['recipes']:
            if not (samplers['foundation']['recipes'][str(n)] == recipe):
                raise ValueError(f'Foundation sampler reference changed for {n}')
        else:
            foundation_new.append(n)
            samplers['foundation']['recipes'][str(n)] = recipe
        shapes.update((len(recipe['context']), len(recipe['query']), width, batch) for width in feature_widths for batch in [1, 2])
    qualified = {(v['context'], v['query'], v['width'], v['batch']) for v in parities['foundation']['tests'] if v['passed']}
    new_shapes = sorted(shapes - qualified)
    for kind, sampler in samplers.items():
        save(a.output / kind / 'sampling.json', sampler)
    request = dict(schema='parallel.component_sampling_extension.v2', jobs_sha256=sha(a.jobs), jobs=len(jobs['jobs']), source_specs=specs, source_registries=source_registries, target_new_lengths=[list(v) for v in target_new], foundation_new_lengths=foundation_new, foundation_new_shapes=[list(v) for v in new_shapes], target_sampler_reference='target_maker.data.sample_indices + stable_seed; target_maker.encode._window_indices', foundation_sampler_reference='numpy PCG64 seed 20260707 permutation; existing context/query rule', sampler_reference_passed=True, numpy_version=np.__version__, module_weights_reused=True, copied_vector_bytes=0, sampling_sha256={k:sha(a.output/k/'sampling.json') for k in samplers}, source_sha256={str(p.relative_to(r)):sha(p) for p in [source/'target_maker/data.py',source/'target_maker/encode.py',Path(__file__).resolve()]})
    save(a.output / 'request.json', request)
    print(json.dumps(dict(request=str(a.output/'request.json'), jobs=request['jobs'], target_new_lengths=len(target_new), foundation_new_lengths=len(foundation_new), foundation_new_shapes=len(new_shapes), rank=a.rank, world_size=a.world_size)), flush=True)
    if a.prepare_only:
        return
    if a.merge_qualification:
        publish(r, a.output, request, specs, parities, a.world_size)
        return

    target_new = [v for i,v in enumerate(target_new) if i % a.world_size == a.rank]
    new_shapes = [v for i,v in enumerate(new_shapes) if i % a.world_size == a.rank]
    import time
    target_tests, foundation_tests = [], []
    def progress(phase):
        value = dict(schema='parallel.component_parity_progress.v1',timestamp=int(time.time()),
            request_sha256=sha(a.output/'request.json'),rank=a.rank,world_size=a.world_size,
            phase=phase,target_completed=len(target_tests),target_total=len(target_new)*3,
            foundation_completed=len(foundation_tests),foundation_total=len(new_shapes))
        dest=a.output/f'progress-rank-{a.rank}.json'
        tmp=dest.with_suffix('.tmp');tmp.write_text(json.dumps(value)+'\n');tmp.replace(dest)
    progress('initializing frozen reference modules')
    if not target_new and not new_shapes:
        save(a.output/f'qualification-rank-{a.rank}.json', dict(schema='parallel.component_parity_shard.v1', rank=a.rank, world_size=a.world_size, request_sha256=sha(a.output/'request.json'), passed=True, target_tests=[], foundation_tests=[], cuda_work_required=False))
        if a.world_size == 1:
            publish(r, a.output, request, specs, parities, 1)
        return

    import torch
    torch.set_num_threads(2)
    torch._C._set_graph_executor_optimize(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not (torch.cuda.is_available()):
        raise ValueError('sampling validation failed at original line 212')
    from target_maker.fused import FusedTargetEncoder
    encoder = FusedTargetEncoder(r/'assets/models/target-fused', device='cuda')
    module = torch.jit.load(str(r/specs['target']['module']), map_location='cuda').eval()
    with torch.no_grad():
        for task, n in target_new:
            recipe = samplers['target']['recipes'][f'{task}:{n}']
            for kind in ['random', 'constant', 'endpoints']:
                y = np.random.default_rng(n).random(n).astype('float32') if kind == 'random' else np.full(n, .5, dtype='float32') if kind == 'constant' else (np.arange(n)%2).astype('float32')
                if task == 0:
                    y = (y >= .5).astype('float32')
                progress('target reference: CPU sampler and CUDA frozen encoder')
                expected = encoder.encode_values(y, task)
                progress('target CUDA exported-module comparison')
                actual = module(torch.tensor(y[recipe['classical']], device='cuda')[None], torch.tensor(y[recipe['neural']], device='cuda')[None], torch.tensor([n], dtype=torch.float32, device='cuda'), torch.tensor([task], device='cuda'))[0].cpu().numpy()
                error = float(np.abs(actual-expected).max())
                target_tests.append(dict(length=n, task=task, kind=kind, max_abs_error=error, passed=error<3e-4))
                progress('target comparison complete')
    del module, encoder
    torch.cuda.empty_cache()
    source = r/'assets/source/fable_god'
    progress('loading frozen foundation reference on CUDA')
    sys.path.insert(0, str(source))
    os.environ['FABLE_REGRESSOR_SOURCE'] = str(source/'provenance/regressor_source')
    from python.elo_finetune.model_io import load_regressor
    from python.elo_finetune.paths import import_regressor_train
    model = load_regressor(r/'assets/models/starlight/starlight_core_regressor.safetensors', r/'assets/models/starlight/starlight_config.json', device='cuda')
    for m in model.modules():
        if hasattr(m, 'num_kv_heads_test'):
            m.num_kv_heads_test = None
    module = torch.jit.load(str(r/specs['foundation']['module']), map_location='cuda').eval()
    torch.manual_seed(42)
    with torch.no_grad():
        for nc, nq, width, batch in new_shapes:
            progress('foundation CUDA reference and exported-module comparison')
            x = torch.rand(nc+nq, batch, width, device='cuda')
            y = torch.rand(nc, batch, device='cuda')
            hidden = model(x, y, only_return_standard_out=False, performance_options=import_regressor_train().PerformanceOptions(use_chunkwise_inference=False))['test_embeddings'].float().transpose(0, 1)
            expected = torch.cat([hidden.mean(1), hidden.std(1, unbiased=False), torch.quantile(hidden, .1, dim=1), torch.quantile(hidden, .9, dim=1)], 1)
            actual = module(x, y)
            error = float((actual-expected).abs().max())
            foundation_tests.append(dict(context=nc, query=nq, width=width, batch=batch, max_abs_error=error, passed=error<3e-4))
            progress('foundation comparison complete')
    passed = all(v['passed'] for v in target_tests + foundation_tests)
    save(a.output/f'qualification-rank-{a.rank}.json', dict(schema='parallel.component_parity_shard.v1', rank=a.rank, world_size=a.world_size, request_sha256=sha(a.output/'request.json'), passed=passed, target_tests=target_tests, foundation_tests=foundation_tests, cuda_work_required=True, torch_version=torch.__version__, cuda_version=torch.version.cuda, device_name=torch.cuda.get_device_name(0)))
    if not (passed):
        raise ValueError('Frozen module parity failed; no runtime specs published')
    progress('passed')
    if a.world_size == 1:
        publish(r, a.output, request, specs, parities, 1)
    else:
        print(json.dumps(dict(passed=True, rank=a.rank, target_tests=len(target_tests), foundation_tests=len(foundation_tests))), flush=True)


if __name__ == '__main__':
    main()
