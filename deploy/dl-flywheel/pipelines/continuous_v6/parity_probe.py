"""Exact NEXT-update parity on v1's UNCHANGED control, before R14 (v2build r2; answers Codex F1).

Scratch only.  It never opens the live v1 root, never reserves on a live planner, and never runs on a live v2 root:
every side refuses unless the campaign genesis is a ``probe`` root (pre-R14: v1 control unchanged, no R14 event).

  bstar    v1's own next batch B*: a BYTE COPY of the sealed v1 planner backup (sha-checked) gets the bounded-verify
           index (secondary index only), then the frozen planner, opened exactly as v1 (active manifest 56640c46, v1
           seed, batch size and initial checkpoint), reserves at v1's final checkpoint.  Records the batch identity,
           epoch/index, ordered ids, the open-epoch state before the reservation and the copy's sha.
  side A   the frozen v1 step primitives, transcribed from pipelines/prepared/continuous_runner.py (745be43): restore,
           [CUDA: gpu_preflight], restore, v1 learning rates; prepare_microbatches(B*, epoch=E-1, index); decode the two
           v1 proposals (training, validation) with v1's seeds at v1's local step; run_microbatches with the optimizer;
           Recovery.save.  Same process start as a v1 resume (cold start; one update).
  side B1  runner_v6.run with emissions_per_step=1 (the live setting) and a planner shim whose reserve() returns B* in
           runner_v6's lineage numbering and whose commit() records but never writes.  One update; the closure then
           REFUSES (an emission ACK and a proof without a planner commit) — recorded as an expected counterexample.
  side B2  the same with emissions_per_step=2 (v1's emission count).
  compare  bitwise comparison of every post-update tensor (48 model, 144 AdamW, RNG) and the state file bytes:
           A vs B1 and A vs B2.  Writes ``jope.v2-next-update-parity/v1``; genesis requires it before R14.

Each side is its own cold-start process (``side``), launched under a capped scope; ``compare`` runs last.
"""
import argparse
import hashlib
import json
import shutil
import sqlite3
import sys
import time
import types
from pathlib import Path

SCHEMA = 'jope.v2-next-update-parity/v1'
# The shim never commits, so custody must refuse the closure: an ACK without a commit (checked first) or an orphan proof.
CUSTODY_REFUSALS = ('no ACK without a commit', 'proof inventory is exactly the committed steps')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cli_v6 as cli   # noqa: E402  (sibling module; imports its pinned source only after check_source)


def _sha(b): return hashlib.sha256(b).hexdigest()


def _canonical(v): return json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()


def _write(path, value):
    with Path(path).open('xb') as f:
        f.write(_canonical(value)); f.flush()
    return {'path': str(path), 'sha256': _sha(Path(path).read_bytes())}


def _imports(execution):
    execution_raw = Path(execution).read_bytes(); ex = json.loads(execution_raw)
    cli.check_source(ex); sys.path[:0] = cli.search_path(ex['source_root'])
    return ex


def bstar(args):
    ex = _imports(args.execution)
    from neural_native_stream import BatchStream
    from jope_flywheel.protocol import digest
    cap = Path(args.capture); body = json.loads((cap / 'final-capture.json').read_bytes()); pl = body['planner']
    scratch = Path(args.scratch); scratch.mkdir(); copy = scratch / 'v1-planner-copy.sqlite'
    shutil.copyfile(cap / pl['backup']['file'], copy)
    assert _sha(copy.read_bytes()) == pl['backup']['sha256'], 'byte copy of the sealed v1 planner backup'
    db = sqlite3.connect(copy); db.execute('CREATE INDEX IF NOT EXISTS v2_members_batch ON members(batch)'); db.commit(); db.close()
    manifest = (cap / pl['active_manifest']['file']).read_bytes(); assert _sha(manifest) == pl['active_manifest']['sha256']
    cfg = pl['config']; final = body['checkpoint']['parameter_sha256']
    planner = BatchStream(copy.resolve(), manifest, _sha(manifest), seed=cfg['seed'], batch_size=cfg['batch_size'],
                          initial_checkpoint_sha256=cfg['initial_checkpoint_sha256'])
    try:
        assert planner.pending() is None and planner._get('checkpoint') == final
        local_steps = planner._get('optimizer_step'); epoch = planner._get('epoch')
        left = [r[0] for r in planner.db.execute('SELECT run FROM members WHERE epoch=? AND batch IS NULL ORDER BY run', (epoch,))]
        assert epoch == pl['current_epoch']['id'] and len(left) == pl['current_epoch']['remaining'] and \
            digest(left) == pl['current_epoch']['remaining_ids_sha256'], 'the captured open v1 epoch'
        b = planner.reserve(final)
    finally:
        planner.close()
    out = {'schema': 'jope.v2-parity-bstar/v1', 'source': 'v1 planner backup (byte copy)', 'backup_sha256': pl['backup']['sha256'],
           'capture_sha256': _sha((cap / 'final-capture.json').read_bytes()), 'v1_local_steps': local_steps,
           'v1_absolute_next_step': body['ancestry']['optimizer_steps'] + 1 if 'ancestry' in body else None,
           'v1_ancestry_completed_epochs': pl['completed_epochs_lineage'] - pl['completed_epochs_local'],
           'batch_id': b.batch_id, 'fingerprint': b.fingerprint, 'manifest_sha256': b.manifest_sha256, 'v1_epoch_local': b.epoch,
           'index': b.index, 'checkpoint_before': b.checkpoint_before, 'ordered_data_run_ids': list(b.ordered_data_run_ids),
           'rows_sha256': digest([dict(r) for r in b.rows]), 'open_epoch_before': pl['current_epoch'],
           'note': 'reserved on a scratch byte copy only; the sealed backup and the live v1 root are never opened for write'}
    print(json.dumps(_write(args.out, out)))


def _groups(state_path):
    import torch
    from safetensors.torch import load
    tensors = load(Path(state_path).read_bytes()); groups = {'model': {}, 'adamw': {}, 'rng': {}}
    for k, v in tensors.items():
        g = 'model' if k.startswith('model.') else 'adamw' if k.startswith('optimizer.') else 'rng'
        groups[g][k] = {'dtype': str(v.dtype), 'shape': list(v.shape), 'sha256': _sha(v.contiguous().view(-1).view(torch.uint8).numpy().tobytes())}
    return groups


def _side_a(x, b):
    """Transcription of the frozen v1 step (pipelines/prepared/continuous_runner.py at 745be43, run() lines 60-117 and
    160-175): restore → [cuda preflight] → restore → v1 rates → one update.  No outbox/planner writes (they touch no
    tensor or RNG); every tensor-touching call and its order is v1's."""
    import numpy as np
    import torch
    from dltrain import model as M
    from jope_flywheel.protocol import canonical, digest
    from jope_flywheel import timing
    from neural_native_train import prepare_microbatches, run_microbatches, gpu_preflight
    from neural_decode import BatchedTrie
    from continuous_state import restore, normalizer, Recovery
    n, resume, snapshot, native, deadline = x.n, x.resume, x.snapshot, x.native, x.deadline
    device = torch.device(n['device'])
    x.fresh(snapshot); M.configure_determinism()
    net, opt, origin = restore(resume['checkpoint_manifest'], resume['checkpoint_state'],
                               expected_parameter_sha256=resume['control']['checkpoint_parameter_sha256'], device=device)
    norm = normalizer(x.pre, native)
    (x.work / 'side-a').mkdir(); recovery = Recovery(x.work / 'side-a')
    native_train = sorted(c for c, r in native.contexts.items() if r['partition'] == 'train')
    native_dev = sorted(c for c, r in native.contexts.items() if r['partition'] == 'validation')
    train_programs = {native.winners[c].program_sha256 for c in native_train}
    catalog = [p for p in native.proposal_programs if p['program_sha256'] in train_programs]
    trie = BatchedTrie(catalog, native.vocab, native.max_seq)
    train_panel = sorted(native_train, key=lambda q: digest({'seed': n['seed'], 'panel': q}))[:n['evaluation_panel_contexts']]
    dev_panel = sorted(native_dev, key=lambda q: digest({'seed': n['seed'], 'panel': q}))[:n['evaluation_panel_contexts']]
    {role: prepare_microbatches(native, ids, epoch=0, batch_index=0, normalizer=norm, max_padded_tokens=n['max_padded_tokens'])[:2]
     for role, ids in [('training', train_panel), ('validation', dev_panel)]}
    if device.type == 'cuda':
        gpu_preflight(native, net, norm, n, deadline); torch.cuda.reset_peak_memory_stats(device)
    net, opt, origin = restore(resume['checkpoint_manifest'], resume['checkpoint_state'],
                               expected_parameter_sha256=resume['control']['checkpoint_parameter_sha256'], device=device)
    for group, rate in zip(opt.param_groups, resume['control']['group_learning_rates']): group['lr'] = rate
    steps = b['v1_local_steps']; ids = list(b['ordered_data_run_ids'])
    absolute = 430 + steps if b['v1_ancestry_completed_epochs'] == 1 else None
    assert absolute == origin['step'] + 1, 'v1 absolute step of B*'
    epoch = b['v1_ancestry_completed_epochs'] + b['v1_epoch_local']
    x.fresh(snapshot)
    batches, total, selection = prepare_microbatches(snapshot, ids, epoch=epoch - 1, batch_index=b['index'], normalizer=norm,
                                                     max_padded_tokens=n['max_padded_tokens'])
    dev = sorted(c for c, r in snapshot.contexts.items() if r['partition'] == 'validation'); emit = [ids[steps % len(ids)], dev[steps % len(dev)]]
    seeds = [int(digest({'seed': n['seed'], 'step': absolute, 'phase': phase, 'context': cid})[:15], 16)
             for phase, cid in zip(['training', 'validation'], emit)]
    with timing.CUDAForwards(net):
        z = torch.tensor(np.stack([norm(c, snapshot.encoding(c)) for c in emit]), device=device)
        decoded, _ = trie.decode_many(net, z, seeds=seeds, temperature=n['decode_temperature'], deadline=deadline)
    before = M.parameter_sha256(net); assert before == b['checkpoint_before']
    with timing.CUDAForwards(net):
        objective = run_microbatches(net, batches, total, optimizer=opt, deadline=deadline)
    after = M.parameter_sha256(net); assert after != before
    path, cp = recovery.save(net, opt, absolute, x.c['campaign_sha256'])
    return {'state': str(path / 'state.safetensors'), 'checkpoint': cp, 'parameter_before': before, 'parameter_after': after,
            'loss': objective['loss'], 'selection_sha256': _sha(canonical(selection)), 'absolute_step': absolute, 'epoch': epoch,
            'decoded_program_sha256': [r['program']['program_sha256'] for r in decoded], 'emissions': 2}


def _side_b(x, b, k):
    from neural_native_stream import BatchStream, Batch, decode
    from jope_flywheel.protocol import canonical
    commits = []

    class Shim(BatchStream):
        """Returns B* once (lineage-renumbered so runner_v6 computes v1's epoch); commit records, never writes."""
        def reserve(self, checkpoint_sha256):
            assert self._get('checkpoint') == checkpoint_sha256 == b['checkpoint_before'] and not commits
            epoch = b['v1_ancestry_completed_epochs'] + b['v1_epoch_local'] - x.ancestry['completed_epochs']
            ids = tuple(b['ordered_data_run_ids'])
            return Batch(b['batch_id'], b['fingerprint'], self.manifest_sha256, epoch, b['index'], checkpoint_sha256, ids,
                         tuple(decode(canonical(self.rows[i])) for i in ids))

        def commit(self, batch_id, **kw):
            commits.append({'batch_id': batch_id, **{k2: v for k2, v in kw.items() if k2 != 'ordered_data_run_ids'}})
            return None

    assert b['manifest_sha256'] == x.g['active_manifest_sha256'], 'v1 and the probe planner share the active manifest'
    try:
        cli.run_prepared(x, planner_factory=Shim)
        raise AssertionError('the probe closure must refuse (no planner commit)')
    except RuntimeError as error:
        refusal = str(error)
    cf = json.loads((x.t / 'closure-failure.json').read_bytes())
    absolute = x.ancestry['optimizer_steps'] + 1; slot = x.t / 'rolling' / f'{absolute:09d}'
    proof = json.loads((x.t / f'batch-{absolute:09d}.json').read_bytes())
    assert len(commits) == 1 and not (x.t / 'result.json').exists() and any(r in cf['error'] for r in CUSTODY_REFUSALS), cf['error']
    return {'state': str(slot / 'state.safetensors'), 'checkpoint': json.loads((slot / 'manifest.json').read_bytes()),
            'parameter_before': commits[0]['checkpoint_before'], 'parameter_after': commits[0]['checkpoint_after'],
            'loss': proof['objective']['loss'], 'selection_sha256': _sha(canonical(proof['selection'])), 'absolute_step': absolute,
            'epoch': x.ancestry['completed_epochs'] + proof['batch']['epoch'], 'emissions': k,
            'decoded_program_sha256': [json.loads(p.read_bytes())['program']['program_sha256'] for p in sorted(x.t.glob(f'decoded-{absolute:09d}-*.json'))],
            'expected_closure_refusal': {'closure_failure': cf['error'][:300], 'raised': refusal[:300], 'result_json_written': False}}


def side(args):
    b = json.loads(Path(args.bstar).read_bytes())
    k = {'A': 2, 'B1': 1, 'B2': 2}[args.side]
    ns = cli.parser().parse_args(['--execution', str(args.execution), '--execution-sha', args.execution_sha, '--dry-run', str(args.dry_run),
                                  '--dry-updates', '1', '--dry-device', args.dry_device])
    x = cli.prepare(ns, n_overrides={'emissions_per_step': k})
    from jope_flywheel.program_io import need, sha
    from jope_flywheel.protocol import canonical
    need(x.g['probe'] is True and 'control-event-r14.json' not in x.g['refs'] and 'r14_event' not in x.g['refs'],
         'parity sides run only on a pre-R14 probe root')
    v1_control = json.loads((x.t / 'v1-final-optimizer-control.json').read_bytes())
    need(x.resume['control'] == v1_control, 'v1 control unchanged (pre-R14)')
    began = time.time()
    out = _side_a(x, b) if args.side == 'A' else _side_b(x, b, k)
    out.update({'side': args.side, 'device': x.n['device'], 'v1_control_sha256': sha(canonical(v1_control)),
                'state_file_sha256': _sha(Path(out['state']).read_bytes()), 'groups': _groups(out['state']),
                'genesis_sha256': x.genesis_sha256, 'execution_sha256': x.execution_sha256, 'wall_seconds': time.time() - began,
                'source_manifest': json.loads(Path(args.execution).read_bytes())['source_manifest']})
    print(json.dumps(_write(args.out, out)))


def compare(args):
    b_raw = Path(args.bstar).read_bytes(); b = json.loads(b_raw)
    sides = {Path(p).stem.split('-')[-1]: json.loads(Path(p).read_bytes()) for p in args.sides}
    a = sides['A']; comparisons = {}
    for name in ('B1', 'B2'):
        s = sides[name]; groups = {}
        for g in ('model', 'adamw', 'rng'):
            diff = sorted(k for k in set(a['groups'][g]) | set(s['groups'][g]) if a['groups'][g].get(k) != s['groups'][g].get(k))
            groups[g] = {'tensors': len(a['groups'][g]), 'mismatched': diff}
        equal = (a['state_file_sha256'] == s['state_file_sha256'] and a['parameter_after'] == s['parameter_after'] and
                 a['selection_sha256'] == s['selection_sha256'] and a['loss'] == s['loss'] and
                 a['absolute_step'] == s['absolute_step'] and a['epoch'] == s['epoch'] and all(not v['mismatched'] for v in groups.values()))
        comparisons['A_vs_' + name] = {'bitwise_equal': equal, 'state_file_equal': a['state_file_sha256'] == s['state_file_sha256'],
                                       'groups': groups, 'parameter_after': [a['parameter_after'], s['parameter_after']],
                                       'loss': [a['loss'], s['loss']]}
    same = {k: len({sides[s][k] for s in sides}) == 1 for k in ('device', 'v1_control_sha256', 'genesis_sha256', 'execution_sha256')}
    same['source_manifest'] = len({_canonical(sides[s]['source_manifest']) for s in sides}) == 1
    receipt = {'schema': SCHEMA, 'passed': all(c['bitwise_equal'] for c in comparisons.values()) and all(same.values()),
               'capture_sha256': b['capture_sha256'], 'v1_control_sha256': a['v1_control_sha256'],
               'control_applied': 'v1 unchanged (pre-R14)', 'device': a['device'],
               'bstar': {'source': b['source'], 'sha256': _sha(b_raw), 'batch_id': b['batch_id'], 'v1_epoch_local': b['v1_epoch_local'],
                         'lineage_epoch': a['epoch'], 'index': b['index'], 'absolute_step': a['absolute_step'],
                         'ordered_data_run_ids_sha256': _sha(_canonical(b['ordered_data_run_ids']))},
               'sides': {k: {'sha256': _sha(Path(p).read_bytes()), 'path': str(p)} for k, p in
                         zip([Path(p).stem.split('-')[-1] for p in args.sides], args.sides)},
               'comparisons': comparisons, 'same_inputs': same, 'probe_source_manifest': a['source_manifest'],
               'scope': 'One isolated, never-committed update from v1\'s final checkpoint on v1\'s own next batch B*, v1 control '
                        'unchanged.  Side A = frozen v1 step primitives (transcribed); B1/B2 = runner_v6 (1 and 2 emissions). '
                        'Not a statement that v2 schedules B* (v2 opens a fresh epoch; genesis.epoch_accounting).'}
    print(json.dumps(_write(args.out, receipt))); return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('bstar'); p.add_argument('--execution', type=Path, required=True); p.add_argument('--capture', required=True)
    p.add_argument('--scratch', required=True); p.add_argument('--out', required=True)
    p = sub.add_parser('side'); p.add_argument('--side', choices=('A', 'B1', 'B2'), required=True)
    p.add_argument('--execution', type=Path, required=True); p.add_argument('--execution-sha', required=True)
    p.add_argument('--bstar', required=True); p.add_argument('--dry-run', type=Path, required=True)
    p.add_argument('--dry-device', choices=('cpu', 'cuda'), default='cpu'); p.add_argument('--out', required=True)
    p = sub.add_parser('compare'); p.add_argument('--bstar', required=True); p.add_argument('--sides', nargs=3, required=True)
    p.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    {'bstar': bstar, 'side': side, 'compare': compare}[args.cmd](args)


if __name__ == '__main__':
    main()
