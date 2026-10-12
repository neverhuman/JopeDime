"""W10 r2 fixtures: the runtime_limit_stopped_idle terminal class (owner ruling R17) on real CPU worlds.

The 'idle' world (tests/world.py) reproduces the live v1 end: the pinned runner itself publishes progress
'waiting_for_mandatory_gp_capacity' with no reservation, systemd's RuntimeMaxSec kill (failed/timeout) lands after the
in-loop deadline, no status.json/result.json exists and every write precedes the in-loop deadline. Each counterexample
tampers a private copy inside frozen_times(), i.e. models a live root that reached that state before the deadline,
so exactly the guard under test refuses; every refusal asserts its message, so disabling that guard (fault
injection) fails the fixture.
"""
import contextlib, hashlib, json, os, re, shutil, tempfile, time, types, unittest
from pathlib import Path
from unittest.mock import patch
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import sha
from neural_native_stream import BatchStream
from pipelines.final_capture_v1 import capture as C
from pipelines.final_capture_v1.tests.test_final_capture import active_manifest, rewrite, tree
from pipelines.final_capture_v1.tests.world import R17_RECEIPT, entries, readback, world

KINDS = 'campaign-root walk admits only regular files and directories (symlinks are never followed): '


@contextlib.contextmanager
def frozen_times(root):
    """FIXTURE tamper made before the in-loop deadline: existing entries keep their mtimes, new ones get the newest."""
    before = {p: os.lstat(p).st_mtime_ns for p in entries(root)}
    yield
    newest = max(before.values())
    for p in sorted(entries(root), key=lambda p: -len(p.parts)):
        os.utime(p, ns=(newest, before.get(p, newest)), follow_symlinks=False)


class RuntimeLimitTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='w10r2-test-'); self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve(); self.n = 0; self.w = world('idle')

    def copy(self):
        self.n += 1; return self.w.private_copy(self.base / f'copy{self.n}')

    def refuse(self, root, pattern, w=None, **overrides):
        self.n += 1; out = self.base / f'out{self.n}'
        with self.assertRaisesRegex(ValueError, pattern): (w or self.w).capture(root, out, **overrides)
        self.assertFalse(out.exists()); self.assertEqual(list(self.base.glob('.final-capture-*')), [])

    def edit(self, root, rel, change):
        with frozen_times(root): rewrite(root / rel, change)

    def timeout(self, **kw):
        return readback(self.w.unit, **{'active': 'failed', 'sub_state': 'failed', 'result': 'timeout', **kw})

    def final_slot(self, root):
        p = json.loads((root / 'training' / 'progress.json').read_bytes())
        return root / 'training' / 'rolling' / f"{p['optimizer_steps']:09d}", p

    def test_r_live_shape_runtime_limit_stopped_idle_is_captured(self):
        root = self.copy(); t = root / 'training'; progress = json.loads((t / 'progress.json').read_bytes())
        self.assertEqual(progress['state'], 'waiting_for_mandatory_gp_capacity')
        self.assertGreater(progress["unresolved_programs"] + 2, self.w.config["neural"]["max_pending_emissions"])
        self.assertFalse((t / 'result.json').exists()); self.assertFalse((root / 'status.json').exists())
        records, summary = C.walk(root)
        published = {r['path']: r['mtime_ns'] for r in records if r['kind'] == 'file' and not r['path'].endswith('-shm')}
        self.assertEqual(max(published.values()), published['training/progress.json'], 'the waiting progress is the last publication')
        self.assertLess(summary['max_mtime_ns'], self.w.cutoff_ns); self.assertLess(self.w.cutoff_ns, time.time_ns())
        before = tree(root); out = self.base / 'out'
        body, receipt = self.w.capture(root, out)
        self.assertEqual(tree(root), before, 'private copy is never written')
        raw = R17_RECEIPT.read_bytes(); self.assertEqual(sha(raw), C.R17['receipt_sha256'])
        self.assertEqual(body['schema'], 'jope.continuous-generation-final-capture/v2')
        self.assertEqual(body['owner_ruling'], {'id': 'R17', 'receipt_sha256': sha(raw)})
        term = body['terminal']
        self.assertEqual({k: v for k, v in term.items() if k != 'runtime_limit'},
                         {'class': 'runtime_limit_stopped_idle', 'state': 'waiting_for_mandatory_gp_capacity', 'closure': 'r17_runtime_limit_stopped_idle',
                          'result_json': 'absent', 'status_json': 'absent', 'failure_markers': [], 'interrupted_publications': []})
        rl = term['runtime_limit']; proofs = rl['r17_proofs']; absolute = body['planner']['absolute_optimizer_steps']
        self.assertEqual(rl['deadline']['in_loop_deadline_ns'], self.w.cutoff_ns)
        self.assertEqual(rl['deadline']['deadline_unix'] - rl['deadline']['closure_reserve_seconds'], self.w.config['global_deadline_unix'] - 180)
        self.assertEqual({k: v for k, v in rl['walk'].items() if k not in ('parts', 'live_root_walk_equal')}, summary)
        self.assertEqual(proofs['unit_stopped'], {'ActiveState': 'failed', 'Result': 'timeout', 'MainPID': 0, 'NRestarts': 0, 'cgroup_empty': True})
        self.assertEqual(proofs['no_write_after_in_loop_deadline']['max_mtime_ns'], summary['max_mtime_ns'])
        pl = proofs['planner']
        self.assertEqual((pl['unresolved_batches'], pl['pending_batches']), (0, 0))
        self.assertEqual((pl['last_committed_absolute_step'], pl['progress_optimizer_steps'], pl['rolling_step']), (absolute,) * 3)
        self.assertEqual(sorted(p.name for p in (t / 'rolling').iterdir())[-1], f'{absolute:09d}')
        self.assertEqual({pl['checkpoint_after'], pl['rolling_parameter_sha256'], pl['progress_parameter_sha256'],
                          proofs['cpu_restore']['parameter_sha256'], progress['checkpoint_sha256']}, {progress['checkpoint_sha256']})
        loaded, joins, anc = C.load_capture(out)
        self.assertEqual(loaded, body); self.assertEqual(joins[-1]['absolute_step'], absolute)
        self.assertEqual((anc['optimizer_steps'], anc['checkpoint_sha256']), (absolute, progress['checkpoint_sha256']))
        self.assertEqual(anc['completed_epochs'], progress['completed_epochs'])
        manifest = (out / 'active-manifest.json').read_bytes(); cfg = body['planner']['config']
        fresh = BatchStream(self.base / 'fresh.sqlite', manifest, sha(manifest), seed=cfg['seed'], batch_size=cfg['batch_size'],
                            initial_checkpoint_sha256=anc['checkpoint_sha256'])
        try:
            fresh.inherit(anc); fresh.inherit(anc); self.assertEqual(fresh.verify(), body['ancestry']['inherit_dry_run']['fresh_head'])
        finally: fresh.close()
        other = self.copy(); _, rb = self.w.capture(other, self.base / 'out-b')
        self.assertEqual((out / 'final-capture.json').read_bytes(), (self.base / 'out-b' / 'final-capture.json').read_bytes(), 'deterministic')
        self.assertNotIn(str(root), (out / 'final-capture.json').read_text())
        body, _ = self.w.capture(other, self.base / 'out-inactive', owner_readback=readback(self.w.unit, active='inactive', result='timeout'))
        self.assertEqual(body['terminal']['runtime_limit']['r17_proofs']['unit_stopped']['ActiveState'], 'inactive')
        edge = self.copy(); os.utime(edge / 'training' / 'loss-history.jsonl', ns=(self.w.cutoff_ns, self.w.cutoff_ns))
        body, _ = self.w.capture(edge, self.base / 'out-edge')
        self.assertEqual(body['terminal']['runtime_limit']['walk']['max_mtime_ns'], self.w.cutoff_ns, 'an mtime exactly at the in-loop deadline is admitted')

    def test_r_a_timeout_with_unresolved_reservation_is_a_hold(self):
        n = self.w.config['neural']
        for state in ('reserved', 'needs_reconcile'):
            with self.subTest(state=state):
                root = self.copy(); path = root / 'training' / 'batches.sqlite'
                with frozen_times(root):
                    raw = active_manifest(path)
                    p = BatchStream(path, raw, sha(raw), seed=n['seed'], batch_size=n['batch_size'], initial_checkpoint_sha256=self.w.config['parameter_sha256'])
                    try:
                        b = p.reserve(p._get('checkpoint'))
                        if state == 'needs_reconcile': p.mark_uncertain(b.batch_id, 'fixture: killed mid-step')
                    finally: p.close()
                self.refuse(root, 'reserved/needs_reconcile planner batch: uncertain optimizer outcome is a HOLD')

    def test_r_b_write_after_in_loop_deadline_refused(self):
        late = self.w.cutoff_ns + 1
        def touch(rel): return lambda root: os.utime(root / rel, ns=(late, late))
        def create(root):
            with frozen_times(root): (root / 'training' / 'late-write.json').write_bytes(b'{}')
            os.utime(root / 'training' / 'late-write.json', ns=(late, late))
        for name, change, where in (('one file 1 ns after the deadline', touch('training/loss-history.jsonl'), 'training/loss-history.jsonl'),
                                    ('planner -wal', touch('training/batches.sqlite-wal'), 'training/batches.sqlite-wal'),
                                    ('directory entry added/removed (rolling)', touch('training/rolling'), 'training/rolling'),
                                    ('new file', create, 'training/late-write.json'),
                                    ('campaign config', touch('config.json'), 'config.json')):
            with self.subTest(case=name):
                root = self.copy(); change(root)
                self.refuse(root, re.escape('campaign root written after the in-loop deadline: ' + where) + '$')

    def test_r_b_in_loop_deadline_comes_from_window_and_config(self):
        deadline = self.w.config['global_deadline_unix']
        for name, change in (('window of another config', lambda w: w.update(config_sha256=digest({'other': 1}))),
                             ('window deadline differs from the config', lambda w: w.update(deadline_unix=deadline + 1.0)),
                             ('window without a deadline', lambda w: w.pop('deadline_unix'))):
            with self.subTest(case=name):
                root = self.copy(); self.edit(root, 'window.json', change)
                self.refuse(root, re.escape('in-loop deadline: window.json must join config.json'))
        root = self.copy()
        with frozen_times(root): (root / 'window.json').unlink()
        self.refuse(root, 'required regular private-copy file: window.json')

    def test_r_b_in_loop_deadline_must_have_passed(self):
        root = self.copy()
        for name, now in (('at the deadline', self.w.cutoff_ns), ('before it', self.w.cutoff_ns - 10**9)):
            with self.subTest(clock=name), patch.object(C, 'time', types.SimpleNamespace(time=time.time, time_ns=lambda: now)):
                self.refuse(root, 'in-loop deadline has not passed')

    def test_r_c_owner_cgroup_or_pid_refused(self):
        root = self.copy()
        for i, rb in enumerate([self.timeout(cgroup_exists=True, procs=[77], populated=True), self.timeout(main_pid=4242),
                                self.timeout(cgroup_exists=True, populated=True)]):
            with self.subTest(readback=i): self.refuse(root, 'owner still active or its cgroup is not empty', owner_readback=rb)

    def test_r_c_restarted_unit_refused(self):
        root = self.copy()
        for restarts in (1, 3):
            with self.subTest(restarts=restarts):
                self.refuse(root, 'runtime_limit_stopped_idle requires NRestarts=0: ' + str(restarts), owner_readback=self.timeout(nrestarts=restarts))

    def test_r_d_systemd_result_other_than_timeout_refused(self):
        root = self.copy()
        for result in ('exit-code', 'signal', 'core-dump', 'oom-kill', 'watchdog', 'success', 'start-limit-hit'):
            with self.subTest(result=result):
                self.refuse(root, 'runtime_limit_stopped_idle requires systemd Result=timeout: ' + result + '$', owner_readback=self.timeout(result=result))
        self.refuse(root, 'requires systemd Result=timeout: exit-code', owner_readback=readback(self.w.unit, active='inactive', result='exit-code'))

    def test_r_e_running_failed_or_unknown_state_refused(self):
        for state in ('running', 'failed', 'uncertain_optimizer_outcome', 'waiting_for_operator', 'global_deadline'):
            with self.subTest(state=state):
                root = self.copy(); self.edit(root, 'training/progress.json', lambda p: p.update(state=state))
                self.refuse(root, 'non-terminal or failed progress state refused: ' + state + '$')
        root = self.copy()
        with frozen_times(root): (root / 'training' / 'uncertain-physical-state').mkdir()
        self.refuse(root, 'failed or uncertain terminal state refused: uncertain-physical-state')

    def test_r_f_last_commit_progress_and_rolling_must_agree(self):
        slot, progress = self.final_slot(self.w.world)
        prev = json.loads((slot.parent / f"{progress['optimizer_steps'] - 1:09d}" / 'manifest.json').read_bytes())
        for name, rel, change in (('progress one step behind the last commit', 'training/progress.json', lambda p: p.update(optimizer_steps=p['optimizer_steps'] - 1)),
                                  ('progress parameter sha of the previous slot', 'training/progress.json', lambda p: p.update(checkpoint_sha256=prev['parameter_sha256'])),
                                  ('rolling parameter sha differs from checkpoint_after', f'training/rolling/{slot.name}/manifest.json',
                                   lambda m: m.update(parameter_sha256=prev['parameter_sha256']))):
            with self.subTest(case=name):
                root = self.copy(); self.edit(root, rel, change)
                self.refuse(root, 'R17 closure: last committed step, progress optimizer_steps and final rolling slot')

    def test_r_g_missing_or_forged_r17_receipt_refused(self):
        root = self.copy(); raw = R17_RECEIPT.read_bytes()
        for name, value in (('missing', None), ('empty', b''), ('text, not bytes', raw.decode()), ('trailing newline', raw + b'\n'),
                            ('another ruling id', raw.replace(b'"R17"', b'"R18"', 1)),
                            ('another decision', raw.replace(b'"Accept as clean stop (Recommended)"', b'"HOLD for Codex review"', 1))):
            with self.subTest(receipt=name):
                self.refuse(root, 'runtime_limit_stopped_idle requires the sealed owner R17 ruling receipt bytes', owner_ruling=value)

    def test_r_g_clean_class_refuses_r17_inputs(self):
        k = world('killed'); root = k.private_copy(self.base / 'killed-copy'); raw = R17_RECEIPT.read_bytes()
        live = lambda: C.walk(root)[1]
        for name, extra in (('receipt', {'owner_ruling': raw}), ('live walk', {'live_walk': live}), ('both', {'owner_ruling': raw, 'live_walk': live})):
            with self.subTest(inputs=name):
                self.refuse(root, 'apply only to the runtime_limit_stopped_idle class', w=k, **extra)

    def test_r_h_symlink_or_special_file_in_root_refused(self):
        outside = self.base / 'outside.json'; outside.write_bytes(b'{}')
        for name, rel, make in (('symlink escaping the root (directory)', 'training/escape', lambda p: os.symlink('/etc', p)),
                                ('symlink escaping the root (file)', 'outside-link.json', lambda p: os.symlink(str(outside), p)),
                                ('symlink inside the root (still never followed)', 'training/progress-link.json', lambda p: os.symlink('progress.json', p)),
                                ('fifo', 'training/pipe', os.mkfifo)):
            with self.subTest(case=name):
                root = self.copy()
                with frozen_times(root): make(root / rel)
                self.refuse(root, re.escape(KINDS + rel) + '$')

    def test_r_h_walk_entry_and_byte_bounds_refused(self):
        root = self.copy(); records, summary = C.walk(root); live = lambda: summary
        record_bytes = sum(len(canonical(r)) + 1 for r in records)
        for name, attr, value in (('one entry below the count', 'WALK_MAX_ENTRIES', summary['entries'] - 1),
                                  ('one byte below the record bytes', 'WALK_MAX_BYTES', record_bytes - 1)):
            with self.subTest(bound=name), patch.object(C, attr, value):
                self.refuse(root, 'campaign-root walk exceeds its entry/byte bounds', live_walk=live)
        with patch.object(C, 'WALK_MAX_ENTRIES', summary['entries']), patch.object(C, 'WALK_MAX_BYTES', record_bytes):
            self.w.capture(root, self.base / 'exactly-at-bounds', live_walk=live)

    def test_r_i_live_walk_must_equal_private_copy_walk(self):
        pristine = C.walk(self.w.world)[1]; message = re.escape('private copy is not a metadata-faithful copy of the live campaign root')
        self.w.capture(self.copy(), self.base / 'faithful', live_walk=lambda: pristine)
        def extra(r):
            with frozen_times(r): (r / 'training' / 'copy-only.json').write_bytes(b'{}')
        def lost(r):
            victim = sorted((r / 'training').glob('decoded-*.json'))[0]
            with frozen_times(r): victim.unlink()
        def retimed(r):
            p = r / 'training' / 'loss-history.jsonl'; t = os.lstat(p).st_mtime_ns - 10**9; os.utime(p, ns=(t, t))
        for name, change, live in (('no live walk', None, None), ('copy has an extra file', extra, lambda: pristine),
                                   ('copy lost a file', lost, lambda: pristine), ('copy changed an mtime', retimed, lambda: pristine)):
            with self.subTest(case=name):
                root = self.copy()
                if change: change(root)
                self.refuse(root, message, live_walk=live)

    def test_r_i_live_walk_changed_during_capture_refused(self):
        root = self.copy(); calls = []
        def moving():
            calls.append(1); s = C.walk(root)[1]
            return s if len(calls) == 1 else {**s, 'max_mtime_ns': s['max_mtime_ns'] + 1}
        self.refuse(root, 'live campaign-root walk changed during capture', live_walk=moving)
        self.assertEqual(len(calls), 2)

    def test_r_j_status_or_result_json_present_refused(self):
        for name, rel, value in (('status.json at the campaign root', 'status.json',
                                  {'schema': 'jope.continuous-neural-campaign-status/v1', 'state': 'global_deadline_reached'}),
                                 ('stale result.json', 'training/result.json',
                                  {'schema': 'jope.continuous-neural-result/v1', 'state': 'global_deadline_reached', 'error': None})):
            with self.subTest(case=name):
                root = self.copy()
                with frozen_times(root): (root / rel).write_bytes(canonical(value))
                self.refuse(root, re.escape('runtime_limit_stopped_idle: status.json and training/result.json must be absent'))

    def test_r_k_walk_parts_reader_refuses_tamper(self):
        root = self.copy(); out = self.base / 'out'; body, _ = self.w.capture(root, out, part_max=1024)
        self.assertGreater(len(body['terminal']['runtime_limit']['walk']['parts']['parts']), 1)
        late = self.w.cutoff_ns + 1

        def variant(name, change, refresh):
            v = self.base / name; shutil.copytree(out, v)
            for p in v.iterdir():
                if p.is_file(): os.chmod(p, 0o644)
            doc = json.loads((v / 'final-capture.json').read_bytes()); walk = doc['terminal']['runtime_limit']['walk']
            records = change(C.read_parts(v, walk['parts']))
            for p in walk['parts']['parts']: (v / p['file']).unlink()
            walk['parts'] = C.write_parts(v, 'walk', records, part_max=walk['parts']['part_max'])
            if refresh:
                h = hashlib.sha256()
                for r in records: h.update(canonical(r) + b'\n')
                walk.update(records_sha256=h.hexdigest(), entries=len(records))
            raw = canonical(doc); (v / 'final-capture.json').write_bytes(raw)
            receipt = json.loads((v / 'capture-receipt.json').read_bytes()); receipt['capture_sha256'] = sha(raw)
            (v / 'capture-receipt.json').write_bytes(canonical(receipt)); return v

        bump = lambda rs: [dict(r, mtime_ns=late) if r['path'] == 'training/progress.json' else r for r in rs]
        for name, change, refresh in (('re-chained record after the deadline', bump, False), ('re-chained and re-digested record after the deadline', bump, True),
                                      ('re-chained reversed records', lambda rs: rs[::-1], True), ('re-chained dropped record', lambda rs: rs[1:], False)):
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, 'walk records: sorted, pinned by their digest and never after the in-loop deadline'):
                    C.load_capture(variant(name.replace(' ', '-'), change, refresh))


if __name__ == '__main__':
    unittest.main()
