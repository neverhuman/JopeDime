"""W10 final_capture_v1 fixtures on real CPU terminal worlds (tests/world.py); every capture reads a private copy.

Each refusal test asserts the specific refusal message, so disabling that one check (fault injection) fails it.
r2: r1 fixtures adapted mechanically only (capture schema v2, terminal 'class', owner_ruling None on the clean class);
the runtime_limit_stopped_idle (R17) fixtures live in test_runtime_limit.py.
"""
import copy, hashlib, json, os, shutil, sqlite3, tempfile, unittest
from pathlib import Path
import torch
from safetensors.torch import load as load_tensors, save as save_tensors
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import sha
from neural_native_stream import BatchStream
from pipelines.final_capture_v1 import capture as C, probe as P
from pipelines.final_capture_v1.tests.world import readback, world

ANCESTRY_KEYS = {'planner_ref', 'head', 'checkpoint_sha256', 'optimizer_steps', 'completed_epochs', 'fingerprints',
                 'exposed_context_ids', 'presentations'}


def tree(path):
    """Identity of every regular file (relative path + bytes) under path."""
    h = hashlib.sha256(); path = Path(path)
    for p in sorted(path.rglob('*')):
        if p.is_file(): h.update(str(p.relative_to(path)).encode() + b'\0' + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def rewrite(path, edit):
    """Fixture tamper of one canonical JSON file inside a private copy."""
    value = json.loads(Path(path).read_bytes()); edit(value); Path(path).write_bytes(canonical(value))


def active_manifest(planner_path):
    db = sqlite3.connect(planner_path)
    try:
        active = json.loads(db.execute("SELECT value FROM state WHERE key='manifest'").fetchone()[0])
        return bytes(db.execute('SELECT body FROM manifests WHERE sha=?', (active,)).fetchone()[0])
    finally: db.close()


class FinalCaptureTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='w10-test-'); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve(); self.n = 0

    def copy(self, kind='killed'):
        self.n += 1; w = world(kind); return w, w.private_copy(self.base / f'copy{self.n}')

    def refuse(self, w, root, pattern, **overrides):
        self.n += 1; out = self.base / f'out{self.n}'
        with self.assertRaisesRegex(ValueError, pattern): w.capture(root, out, **overrides)
        self.assertFalse(out.exists()); self.assertEqual(list(self.base.glob('.final-capture-*')), [])

    def final_slot(self, root):
        p = json.loads((root / 'training' / 'progress.json').read_bytes())
        return root / 'training' / 'rolling' / f"{p['optimizer_steps']:09d}", p

    def test_a_missing_result_is_captured_and_fresh_journal_inherits(self):
        w, root = self.copy('killed')
        self.assertFalse((root / 'training' / 'result.json').exists())
        self.assertGreater((w.world / 'training' / 'batches.sqlite-wal').stat().st_size, 0, 'killed planner left -wal frames')
        before = tree(root); out = self.base / 'out'
        body, receipt = w.capture(root, out)
        self.assertEqual(tree(root), before, 'private copy is never written')
        progress = json.loads((root / 'training' / 'progress.json').read_bytes())
        self.assertEqual(body['schema'], 'jope.continuous-generation-final-capture/v2')
        self.assertEqual(body['terminal'], {'class': 'clean_terminal', 'state': 'global_deadline_reached', 'closure': 'proven_without_result_json',
                                            'result_json': None, 'failure_markers': [], 'interrupted_publications': []})
        self.assertIsNone(body['owner_ruling'])
        pl = body['planner']; steps = pl['local_optimizer_steps']
        self.assertEqual((steps, pl['absolute_optimizer_steps'], pl['head']['unresolved_batches']), (5, 434, 0))
        self.assertEqual(pl['own_fingerprints']['count'], pl['head']['committed_batches'])
        self.assertEqual(len(pl['manifest_chain']), 2, 'prepared manifest adoption is in the chain')
        self.assertEqual(body['owner']['readback']['ActiveState'], 'failed'); self.assertEqual(body['owner']['readback']['Result'], 'timeout')
        self.assertEqual(body['checkpoint']['cpu_restore']['parameter_sha256'], progress['checkpoint_sha256'])
        self.assertEqual(body['checkpoint']['tensor_keys']['rng'], ['rng.torch_cpu'])
        self.assertEqual(body['outbox']['acks']['count'], steps)
        loaded, joins, anc = C.load_capture(out)
        self.assertEqual(loaded, body); self.assertEqual([j['absolute_step'] for j in joins], list(range(430, 435)))
        self.assertEqual(set(anc), ANCESTRY_KEYS); self.assertEqual(digest(anc), body['ancestry']['ancestry_sha256'])
        self.assertEqual(anc['optimizer_steps'], 429 + steps); self.assertEqual(anc['checkpoint_sha256'], progress['checkpoint_sha256'])
        self.assertEqual(anc['completed_epochs'], progress['completed_epochs'])
        self.assertEqual(len(anc['exposed_context_ids']), progress['unique_train_data_runs_seen'])
        self.assertEqual(anc['presentations'], progress['total_train_presentations'])
        self.assertEqual(len(anc['fingerprints']), 429 + steps); self.assertEqual(anc['head']['committed_batches'], 429 + steps)
        self.assertEqual([g['head']['committed_batches'] for g in anc['head']['generations']], [429, steps])
        for name, ref in receipt['files'].items():
            self.assertEqual((out / name).stat().st_mode & 0o777, 0o444, name); self.assertEqual(sha((out / name).read_bytes()), ref['sha256'], name)
        self.assertEqual(sha((out / 'final-capture.json').read_bytes()), receipt['capture_sha256'])
        manifest = (out / 'active-manifest.json').read_bytes(); cfg = pl['config']
        fresh = BatchStream(self.base / 'fresh.sqlite', manifest, sha(manifest), seed=cfg['seed'], batch_size=cfg['batch_size'],
                            initial_checkpoint_sha256=anc['checkpoint_sha256'])
        try:
            fresh.inherit(anc); fresh.inherit(anc)
            self.assertEqual(fresh.verify(), body['ancestry']['inherit_dry_run']['fresh_head'])
            changed = copy.deepcopy(anc); changed['presentations'] += 1
            with self.assertRaisesRegex(ValueError, 'ancestry cannot change'): fresh.inherit(changed)
            batch = fresh.reserve(anc['checkpoint_sha256']); self.assertNotIn(batch.fingerprint, set(anc['fingerprints']))
        finally: fresh.close()

    def test_a_present_result_must_join_exactly(self):
        w, root = self.copy('clean')
        body, _ = w.capture(root, self.base / 'out'); raw = (root / 'training' / 'result.json').read_bytes()
        self.assertEqual(body['terminal']['closure'], 'result_json_joined')
        self.assertEqual(body['terminal']['result_json'], {'sha256': sha(raw), 'bytes': len(raw)})
        self.assertEqual(body['owner']['readback']['ActiveState'], 'inactive')
        for i, edit in enumerate([lambda r: r.update(current_parameter_sha256=digest({'other': 1})), lambda r: r.update(state='failed'),
                                  lambda r: r['proof_tail'].pop(), lambda r: r['optimizer_control']['plateau'].update(bad=0, reductions=99)]):
            with self.subTest(edit=i):
                _, bad = self.copy('clean'); rewrite(bad / 'training' / 'result.json', edit)
                self.refuse(w, bad, 'result.json disagrees with the proven final closure')

    def test_b_failed_or_uncertain_terminal_state_refused(self):
        w, root = self.copy('failed')
        self.assertEqual(json.loads((root / 'training' / 'progress.json').read_bytes())['state'], 'failed')
        self.refuse(w, root, 'failed or uncertain terminal state refused: failure.json')
        (root / 'training' / 'failure.json').unlink()
        self.refuse(w, root, 'non-terminal or failed progress state refused: failed')
        k, root = self.copy('killed'); (root / 'training' / 'uncertain-physical-state').mkdir()
        self.refuse(k, root, 'failed or uncertain terminal state refused: uncertain-physical-state')
        k, root = self.copy('killed'); rewrite(root / 'training' / 'progress.json', lambda p: p.update(state='running'))
        self.refuse(k, root, 'non-terminal or failed progress state refused: running')

    def test_b_pending_reservation_or_orphan_state_is_a_hold(self):
        for state in ('reserved', 'needs_reconcile'):
            with self.subTest(state=state):
                w, root = self.copy('killed'); path = root / 'training' / 'batches.sqlite'; raw = active_manifest(path)
                p = BatchStream(path, raw, sha(raw), seed=w.config['neural']['seed'], batch_size=w.config['neural']['batch_size'],
                                initial_checkpoint_sha256=w.config['parameter_sha256'])
                try:
                    b = p.reserve(p._get('checkpoint'))
                    if state == 'needs_reconcile': p.mark_uncertain(b.batch_id, 'fixture: optimizer outcome unknown')
                finally: p.close()
                self.refuse(w, root, 'reserved/needs_reconcile planner batch: uncertain optimizer outcome is a HOLD')
        w, root = self.copy('killed'); slot, progress = self.final_slot(root)
        nxt = root / 'training' / f"batch-{progress['optimizer_steps'] + 1:09d}.json"; nxt.write_bytes(canonical({'fixture': 'orphan proof'}))
        self.refuse(w, root, 'orphan proof beyond the last commit')
        w, root = self.copy('killed'); slot, progress = self.final_slot(root)
        later = slot.parent / f"{progress['optimizer_steps'] + 1:09d}"; later.mkdir()
        self.refuse(w, root, 'final rolling slot is the last committed update')

    def test_b_parameter_sha_disagreement_refused(self):
        w, root = self.copy('killed'); slot, progress = self.final_slot(root)
        prev = json.loads((slot.parent / f"{progress['optimizer_steps'] - 1:09d}" / 'manifest.json').read_bytes())
        stale_progress = ('progress.json', lambda p: p.update(checkpoint_sha256=prev['parameter_sha256']))
        stale_control = ('optimizer-control.json', lambda c: c.update(checkpoint_parameter_sha256=prev['parameter_sha256'],
                                                                     checkpoint_state_sha256=prev['state_sha256'], absolute_optimizer_step=prev['step']))
        for name, edits in (('progress', [stale_progress]), ('control', [stale_control]), ('planner', [stale_progress, stale_control])):
            with self.subTest(stale=name):
                _, bad = self.copy('killed')
                for file, edit in edits: rewrite(bad / 'training' / file, edit)
                self.refuse(w, bad, 'parameter sha disagrees across planner, progress and optimizer-control')

    def test_b_checkpoint_bytes_tamper_refused(self):
        for rehash in (False, True):
            with self.subTest(rehashed_manifest=rehash):
                w, root = self.copy('killed'); slot, _ = self.final_slot(root)
                state = bytearray((slot / 'state.safetensors').read_bytes()); state[-1] ^= 1; (slot / 'state.safetensors').write_bytes(bytes(state))
                if rehash: rewrite(slot / 'manifest.json', lambda m: m.update(state_sha256=sha(bytes(state))))
                self.refuse(w, root, 'final checkpoint closure')

    def test_b_owner_active_or_cgroup_nonempty_refused(self):
        w, root = self.copy('killed'); u = w.unit; cg = '/sys/fs/cgroup/fixture.slice/' + u
        for i, rb in enumerate([readback(u, active='active', main_pid=4242, cgroup_exists=True, procs=[4242], populated=True, control_group=cg),
                                readback(u, active='deactivating', main_pid=4242, cgroup_exists=True, procs=[4242], populated=True, control_group=cg),
                                readback(u, main_pid=4242), readback(u, cgroup_exists=True, procs=[77], populated=True),
                                readback(u, cgroup_exists=True, populated=True), readback('another.service')]):
            with self.subTest(readback=i): self.refuse(w, root, 'owner still active or its cgroup is not empty', owner_readback=rb)
        self.refuse(w, root, 'closed owner readback', owner_readback=lambda: {**readback(u)(), 'extra': 1})
        calls = []
        def turns_active():
            calls.append(1); return readback(u)() if len(calls) == 1 else readback(u, active='active', main_pid=1)()
        self.refuse(w, root, 'owner still active', owner_readback=turns_active)
        calls = []
        def changes():
            calls.append(1); return readback(u)() if len(calls) == 1 else readback(u, result='timeout')()
        self.refuse(w, root, 'owner readback changed during capture', owner_readback=changes)
        body, _ = w.capture(root, self.base / 'ok', owner_readback=readback(u, cgroup_exists=True, populated=False))
        self.assertEqual(body['owner']['readback']['cgroup']['exists'], True)

    def test_b_proof_or_ack_evidence_mismatch_refused(self):
        w, root = self.copy('killed')
        rewrite(root / 'training' / 'batch-000000431.json', lambda d: d['objective'].update(loss=d['objective']['loss'] + 1.0))
        self.refuse(w, root, 'committed proof does not join planner commit evidence')
        w, root = self.copy('killed'); (root / 'training' / 'batch-000000432.json').unlink()
        self.refuse(w, root, 'committed proof missing: batch-000000432.json')
        w, root = self.copy('killed')
        def tampered():
            value = w.reader(root, self.base)(); value['v1']['acks_sha256'] = digest({'other': 1}); return value
        self.refuse(w, root, 'outbox ACK join', outbox_reader=tampered)
        def dropped():
            value = w.reader(root, self.base)(); value['v1']['batches'] -= 1; value['v1']['ack_count'] -= 1; return value
        self.refuse(w, root, 'outbox ACK join', outbox_reader=dropped)
        def unclosed():
            value = w.reader(root, self.base)(); del value['v1']['ack_count']; return value
        self.refuse(w, root, 'closed injected outbox summary', outbox_reader=unclosed)

    def test_c_capture_bytes_are_deterministic(self):
        w = world('killed'); a = w.private_copy(self.base / 'a'); b = w.private_copy(self.base / 'b')
        _, ra = w.capture(a, self.base / 'out-a'); _, rb = w.capture(b, self.base / 'out-b'); _, rc = w.capture(a, self.base / 'out-c')
        raw = (self.base / 'out-a' / 'final-capture.json').read_bytes()
        self.assertEqual(raw, (self.base / 'out-b' / 'final-capture.json').read_bytes())
        self.assertEqual(raw, (self.base / 'out-c' / 'final-capture.json').read_bytes())
        self.assertEqual(ra['files'], rb['files']); self.assertEqual(ra['files'], rc['files'])
        self.assertEqual({k for k in ra if ra[k] != rb[k]}, {'captured_at_unix', 'campaign_root', 'output'})
        self.assertNotIn(str(a), raw.decode()); self.assertNotIn('captured_at', raw.decode())

    def test_d_next_update_parity_on_isolated_scratch(self):
        w = world('killed'); root = w.private_copy(self.base / 'copy')
        world_tree, copy_tree = tree(w.world), tree(root)
        w.capture(root, self.base / 'out'); scratch = self.base / 'scratch'; scratch.mkdir()
        cap = P.captured_inputs(self.base / 'out'); fresh = P.fresh_resume_inputs(root, scratch)
        self.assertEqual(cap['ancestry'], fresh['ancestry']); self.assertEqual(cap['manifest_raw'], fresh['manifest_raw'])
        self.assertEqual(cap['learning_rates'], fresh['learning_rates'])
        kw = dict(native=w.native, snapshot=w.active, preprocessing_raw=w.pre_raw, max_padded_tokens=w.config['neural']['max_padded_tokens'],
                  campaign_sha256=w.campaign)
        x = P.first_update(scratch, cap, **kw); y = P.first_update(scratch, fresh, **kw)
        self.assertEqual(x['batch'], y['batch']); self.assertEqual(x['lineage_epoch'], cap['ancestry']['completed_epochs'] + 1)
        self.assertNotIn(x['batch']['fingerprint'], set(cap['ancestry']['fingerprints']))
        self.assertEqual(x['parameter_before'], cap['ancestry']['checkpoint_sha256'])
        self.assertEqual((x['parameter_sha256'], x['state_sha256']), (y['parameter_sha256'], y['state_sha256']))
        tx, ty = load_tensors(x['state']), load_tensors(y['state']); self.assertEqual(set(tx), set(ty))
        self.assertIn('rng.torch_cpu', tx); self.assertTrue(any(k.endswith('.exp_avg_sq') for k in tx))
        for key in tx: self.assertTrue(torch.equal(tx[key], ty[key]), key)
        control = json.loads((root / 'training' / 'optimizer-control.json').read_bytes())
        self.assertEqual(x['learning_rates'], control['group_learning_rates'])
        rates = [r / 2 for r in cap['learning_rates']]; other = P.first_update(scratch, {**cap, 'learning_rates': rates}, **kw)
        self.assertEqual(other['learning_rates'], rates); self.assertEqual(other['batch'], x['batch'])
        self.assertNotEqual(other['parameter_sha256'], x['parameter_sha256'], 'the carried LR groups change the update')
        reseeded = P.first_update(scratch, cap, before_update=lambda net, opt: torch.manual_seed(0), **kw)
        self.assertEqual(reseeded['batch'], x['batch'])
        self.assertNotEqual(reseeded['parameter_sha256'], x['parameter_sha256'], 'an unrestored dropout RNG changes the update')
        self.assertEqual(tree(w.world), world_tree, 'original world planner/outbox untouched')
        self.assertEqual(tree(root), copy_tree, 'private copy untouched'); self.assertEqual(list(scratch.iterdir()), [])

    def test_e_private_copy_only_pinned_runtime_and_create_only_output(self):
        w, root = self.copy('killed')
        self.refuse(w, Path(w.config['owned_root']), 'never the owned live root')
        with self.assertRaisesRegex(ValueError, 'new output directory outside the private copy'): w.capture(root, root / 'out')
        self.refuse(w, root, 'campaign config sha mismatch', campaign_sha256=digest({'other': 1}))
        for pattern, edit in (('verifier sources must equal the campaign-owned sources', lambda c: c['owned_sources'].update({'neural_native_stream.py': digest({'other': 1})})),
                              ('exact pinned numeric runtime required', lambda c: c['runtime_versions'].update(torch='0.0.0'))):
            with self.subTest(pattern=pattern):
                _, bad = self.copy('killed'); rewrite(bad / 'config.json', edit)
                self.refuse(w, bad, pattern, campaign_sha256=sha((bad / 'config.json').read_bytes()))
        before = tree(root)
        self.refuse(w, root, 'owner still active', owner_readback=readback(w.unit, active='active', main_pid=1))
        w.capture(root, self.base / 'ok'); self.assertEqual(tree(root), before)
        with self.assertRaisesRegex(ValueError, 'new output directory'): w.capture(root, self.base / 'ok')

    def test_g_chunked_parts_boundaries_tamper_missing_and_reorder(self):
        d = self.base / 'unit'; d.mkdir()
        line = lambda i, n: f'{i:02d}' + 'a' * (n - 5)  # canonical JSON string line of exactly n bytes (quotes + newline)
        exact = [line(i, 32) for i in range(4)]; chain = C.write_parts(d, 'exact', exact, part_max=64)
        self.assertEqual([(p['count'], p['bytes']) for p in chain['parts']], [(2, 64), (2, 64)]); self.assertEqual(C.read_parts(d, chain), exact)
        spill = [line(0, 32), line(1, 33), line(2, 32)]; chain = C.write_parts(d, 'spill', spill, part_max=64)
        self.assertEqual([(p['count'], p['bytes']) for p in chain['parts']], [(1, 32), (1, 33), (1, 32)]); self.assertEqual(C.read_parts(d, chain), spill)
        chain = C.write_parts(d, 'full', [line(0, 64)], part_max=64); self.assertEqual([p['bytes'] for p in chain['parts']], [64])
        chain = C.write_parts(d, 'empty', [], part_max=64); self.assertEqual((chain['count'], chain['parts'][0]['bytes']), (0, 0)); self.assertEqual(C.read_parts(d, chain), [])
        with self.assertRaisesRegex(ValueError, 'one record must fit one part'): C.write_parts(d, 'over', [line(0, 65)], part_max=64)
        for bad in (63, C.PART_MAX + 1):
            with self.assertRaisesRegex(ValueError, 'bounded part kind/size'): C.write_parts(d, 'bound', [], part_max=bad)
        w, root = self.copy('killed')
        big, _ = w.capture(root, self.base / 'big'); small, _ = w.capture(root, self.base / 'small', part_max=1024)
        self.assertEqual((len(big['planner']['joins']['parts']), len(big['ancestry']['fingerprints']['parts'])), (1, 1))
        chains = [small['planner']['joins'], small['ancestry']['fingerprints'], small['ancestry']['exposed_context_ids']]
        self.assertGreaterEqual(len(chains[0]['parts']), 3); self.assertGreaterEqual(len(chains[1]['parts']), 20)
        self.assertTrue(all(p['bytes'] <= 1024 for c in chains for p in c['parts']))
        self.assertEqual(C.load_capture(self.base / 'big')[1:], C.load_capture(self.base / 'small')[1:], 'part boundaries never change records')

        def variant(name, edit):
            v = self.base / name; shutil.copytree(self.base / 'small', v)
            for p in v.iterdir():
                if p.is_file(): os.chmod(p, 0o644)
            edit(v); return v

        def flip(v, name, at=10):
            raw = bytearray((v / name).read_bytes()); raw[at] ^= 1; (v / name).write_bytes(bytes(raw))

        def swap(v, a, b):
            (v / a).rename(v / 'tmp-swap'); (v / b).rename(v / a); (v / 'tmp-swap').rename(v / b)

        def rechain(v, section, reorder):
            body = json.loads((v / 'final-capture.json').read_bytes()); chain = body['planner']['joins'] if section == 'joins' else body['ancestry'][section]
            records = C.read_parts(v, chain)
            for p in chain['parts']: (v / p['file']).unlink()
            new = C.write_parts(v, chain['kind'], reorder(records), part_max=chain['part_max'])
            if section == 'joins': body['planner']['joins'] = new
            else: body['ancestry'][section] = new
            raw = canonical(body); (v / 'final-capture.json').write_bytes(raw)
            receipt = json.loads((v / 'capture-receipt.json').read_bytes()); receipt['capture_sha256'] = sha(raw)
            (v / 'capture-receipt.json').write_bytes(canonical(receipt))

        cases = [('tampered joins part', lambda v: flip(v, 'joins-000002.jsonl'), 'capture part tampered, reordered or missing: joins-000002.jsonl'),
                 ('tampered fingerprints part', lambda v: flip(v, 'fingerprints-000003.jsonl', 5), 'capture part tampered, reordered or missing: fingerprints-000003.jsonl'),
                 ('missing joins part', lambda v: (v / 'joins-000002.jsonl').unlink(), 'capture part missing: joins-000002.jsonl'),
                 ('missing last fingerprints part', lambda v: (v / chains[1]['parts'][-1]['file']).unlink(), 'capture part missing: ' + chains[1]['parts'][-1]['file']),
                 ('swapped part files', lambda v: swap(v, 'joins-000001.jsonl', 'joins-000002.jsonl'), 'capture part tampered, reordered or missing: joins-000001.jsonl'),
                 ('re-chained reversed joins', lambda v: rechain(v, 'joins', lambda r: r[::-1]), 'joins: contiguous steps'),
                 ('re-chained reversed fingerprints', lambda v: rechain(v, 'fingerprints', lambda r: r[::-1]), 'ancestry lists strictly sorted'),
                 ('re-chained duplicated exposed id', lambda v: rechain(v, 'exposed_context_ids', lambda r: r + r[-1:]), 'ancestry lists strictly sorted')]
        for name, edit, pattern in cases:
            with self.subTest(case=name):
                v = variant(name.replace(' ', '-'), edit)
                with self.assertRaisesRegex(ValueError, pattern): C.load_capture(v)
        reordered = copy.deepcopy(chains[0]); reordered['parts'][0], reordered['parts'][1] = reordered['parts'][1], reordered['parts'][0]
        with self.assertRaisesRegex(ValueError, 'ordered part list'): C.read_parts(self.base / 'small', reordered)
        renumbered = copy.deepcopy(reordered)
        for i, p in enumerate(renumbered['parts'], 1): p['file'], p['ordinal'] = f'joins-{i:06d}.jsonl', i
        v = variant('renumbered', lambda v: swap(v, 'joins-000001.jsonl', 'joins-000002.jsonl'))
        with self.assertRaisesRegex(ValueError, 'capture part tampered, reordered or missing: joins-000001.jsonl'): C.read_parts(v, renumbered)
        truncated = copy.deepcopy(chains[0]); truncated['parts'].pop()
        with self.assertRaisesRegex(ValueError, 'capture part chain head and count'): C.read_parts(self.base / 'small', truncated)

    def test_h_capacity_at_v1_and_v2_scale(self):
        """Synthetic records of live field widths through the real part writer/reader (see measure.py)."""
        from pipelines.final_capture_v1 import measure
        for name, joins, fps in (('v1', 17773, 17773 + 429), ('v2cap', 80000, 17773 + 80000)):
            with self.subTest(scale=name):
                d = self.base / name; d.mkdir(); r = measure.profile(d, joins=joins, lineage_fingerprints=fps, exposed=7248)
                for kind in ('joins', 'fingerprints', 'exposed'): self.assertLessEqual(r[kind]['max_part_bytes'], C.PART_MAX)
                self.assertGreater(r['joins']['total_bytes'], C.PART_MAX if name == 'v2cap' else 0)
                self.assertGreaterEqual(r['joins']['parts'], 5 if name == 'v2cap' else 1)
                self.assertGreater(r['inherit_event']['headroom_bytes'], 0, 'canonical ancestry fits the inherit() event bound')

    def test_f_tensor_key_classes_synthetic_headers(self):
        """Synthetic safetensors headers (real serializer, toy tensors): the CUDA RNG branch the CPU worlds cannot reach."""
        z = lambda: torch.zeros(2); rng = lambda: torch.zeros(4, dtype=torch.uint8)
        good = {'model.w': z(), 'optimizer.w.step': torch.tensor(1.), 'optimizer.w.exp_avg': z(), 'optimizer.w.exp_avg_sq': z(), 'rng.torch_cpu': rng()}
        self.assertEqual(C.tensor_classes(save_tensors(good))['rng'], ['rng.torch_cpu'])
        self.assertEqual(C.tensor_classes(save_tensors({**good, 'rng.torch_cuda.0': rng()}))['rng'], ['rng.torch_cpu', 'rng.torch_cuda.0'])
        for i, bad in enumerate([{**good, 'rng.torch_cuda.x': rng()}, {k: v for k, v in good.items() if k != 'rng.torch_cpu'},
                                 {**good, 'other.w': z()}, {k: v for k, v in good.items() if k != 'optimizer.w.exp_avg_sq'},
                                 {**good, 'rng.torch_cuda.1': torch.zeros(4)}, {**good, 'optimizer.v.exp_avg': z()}]):
            with self.subTest(bad=i), self.assertRaises(ValueError): C.tensor_classes(save_tensors(bad))


if __name__ == '__main__':
    unittest.main()
