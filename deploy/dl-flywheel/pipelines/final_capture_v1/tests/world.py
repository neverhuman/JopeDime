"""Private CPU fixture worlds written by the base runners themselves (tests only; never live evidence).

A world mirrors the v1 lineage shape: fixture parent checkpoint 429 with a labelled SYNTHETIC parent ancestry ->
baseline_runner.run (continuous genesis, two updates; its result.json publication is killed, as the live ancestor
was retired by a kill) -> reconcile.capture of the frozen owner -> continuous_runner.run (transfer runner, resume),
which adopts a prepared manifest at the first closed epoch and leaves its loop at an injected in-loop deadline after
three more updates. Kinds:
  'killed' - the finally block publishes progress (global_deadline_reached) and optimizer-control, then the unit is
             killed before result.json (systemd RuntimeMaxSec); the world is copied while the planner/outbox
             connections are still open, so it keeps their -wal/-shm exactly as a killed process leaves them.
  'clean'  - the finally block completes; result.json is present.
  'failed' - an authority check raises inside the loop: state failed, failure.json, result.json state failed.
  'idle'   - r2, the live v1 end under owner ruling R17. The shared unresolved cap (max_pending_emissions 11) is reached
             after the same three updates, so the next iteration publishes progress 'waiting_for_mandatory_gp_capacity'
             and sleeps with no reservation; the in-loop deadline then fires and systemd's RuntimeMaxSec kill lands on
             the closure's first read, so nothing after the waiting progress is published (no result.json, no
             status.json). The world is copied while the connections are open. Like the live root its config carries
             global_deadline_unix/closure_reserve_seconds and the campaign root a window.json. A fixture clock then
             shifts every mtime uniformly so the newest entry lies 2 s before the in-loop deadline, which is itself in
             the past: written before the deadline, captured after it.
Each world is frozen once (``world``); tests capture only private copies of it.
"""
import contextlib, copy, gc, json, os, shutil, sys, tempfile, time, types
from pathlib import Path
from unittest.mock import patch
import numpy, safetensors, torch
from dltrain import model as M
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import sha
from neural_native_train import fit_preprocessing, prepare_microbatches, run_microbatches
from neural_storage import checkpoint
from neural_outbox import Outbox
from fixture_continuous import ContinuousTests
import baseline_runner, continuous_runner, reconcile
from pipelines.final_capture_v1 import capture as C, probe as P

ORIGIN = 429
PHASE2_UPDATES = 3
IDLE_PENDING_CAP = 11   # phase 1 + 3 updates leave 10 unresolved emissions; 10 + 2 > 11 is the GP-capacity wait
R17_RECEIPT = Path(__file__).resolve().parent / 'owner-askuserquestion-receipt-R17-v1.json'


class Killed(BaseException):
    """Stands in for systemd's RuntimeMaxSec kill at one publication point; nothing after it runs."""


def killer(real):
    def publish(root, name, obj):
        if name == 'result.json': raise Killed('fixture: unit killed before result.json')
        return real(root, name, obj)
    return publish


def killed_in_closure(real, fired):
    """The kill lands on the closure's first parent-outbox read after the in-loop deadline (before any publication)."""
    def readonly_outbox(path):
        if fired: raise Killed('fixture: RuntimeMaxSec kill in the closure, before progress/result publication')
        return real(path)
    return readonly_outbox


def deadline_after(k, fired=None):
    """Loop-head tick that lets k iterations pass, then reports the in-loop global deadline."""
    calls = [0]
    def tick(deadline):
        calls[0] += 1
        if calls[0] > k:
            if fired is not None: fired.append(True)
            raise TimeoutError('fixture in-loop global deadline')
    return tick


def readback(unit, *, active='inactive', result='success', main_pid=0, cgroup_exists=False, procs=(), populated=None, control_group='',
             nrestarts=0, sub_state=None):
    """FIXTURE owner readback in the closed shape capture() requires (the live one is the owner's systemd readback)."""
    value = {'unit': unit, 'LoadState': 'loaded', 'ActiveState': active,
             'SubState': sub_state or ('dead' if active in C.STOPPED else 'running'), 'MainPID': main_pid, 'Result': result,
             'NRestarts': nrestarts, 'ControlGroup': control_group,
             'cgroup': {'path': '/sys/fs/cgroup/fixture.slice/' + unit, 'exists': cgroup_exists, 'procs': list(procs), 'populated': populated}}
    return lambda: copy.deepcopy(value)


def entries(root):
    """Every entry under root (root first), never following a symlink."""
    root = Path(root); out = [root]
    for base, dirs, files in os.walk(root, followlinks=False): out += [Path(base) / name for name in dirs + files]
    return out


def written_before(root, cutoff_ns, margin_ns=2 * 10**9):
    """FIXTURE clock: shift every mtime uniformly so the newest entry lies margin_ns before the in-loop deadline."""
    items = [(p, os.lstat(p)) for p in entries(root)]
    delta = max(s.st_mtime_ns for _, s in items) - (cutoff_ns - margin_ns)
    for p, s in sorted(items, key=lambda x: -len(x[0].parts)):
        os.utime(p, ns=(s.st_atime_ns, s.st_mtime_ns - delta), follow_symlinks=False)


class World:
    def __init__(self, base, kind):
        assert kind in ('killed', 'clean', 'failed', 'idle')
        self.kind = kind; base = Path(base); base.mkdir(parents=True); self.base = base; idle = kind == 'idle'
        snapshot, n = ContinuousTests().fixture()
        n.update(max_updates=50, evaluation_every_batches=1, max_pending_emissions=IDLE_PENDING_CAP if idle else 100,
                 output_cap_bytes=24 * 1024**3, plateau_patience=1, learning_rate_floor=1e-6, learning_rate=3e-4)
        torch.manual_seed(4)
        net = M.PolicyLM(M.PolicyConfig(282, 2051, 32, 1, 4, 64, 128, dropout=.2))
        opt = torch.optim.AdamW(M.param_groups(net, .01), lr=3e-4)
        norm, pre = fit_preprocessing(snapshot); ids = sorted(c for c, r in snapshot.contexts.items() if r['partition'] == 'train')
        packs, total, _ = prepare_microbatches(snapshot, ids[:2], epoch=0, batch_index=0, normalizer=norm, max_padded_tokens=4096)
        run_microbatches(net, packs, total, optimizer=opt, deadline=time.time() + 60)
        parent = checkpoint(net, opt, base / 'parent', step=ORIGIN, config_sha256=digest({'fixture-parent-config': 1}))
        Outbox(base / 'parent-outbox.sqlite', model_origin_sha256=parent['parameter_sha256']).close()
        self.pre_raw = canonical(pre); (base / 'preprocessing.json').write_bytes(self.pre_raw)
        ref = lambda p: {'path': str(p), 'sha256': sha(p.read_bytes()), 'bytes': len(p.read_bytes())}
        continuous = Path(C.neural_native_stream.__file__).parent
        build = base / 'build'; build.mkdir()
        self.config = {'schema': 'jope.continuous-neural-campaign/v1', 'scope': 'PRIVATE CPU FIXTURE WORLD; not a live campaign',
                       'owned_root': str(build), 'host': 'fixture', 'neural': n, 'parameter_sha256': parent['parameter_sha256'],
                       'checkpoint_manifest': ref(base / 'parent' / 'manifest.json'), 'checkpoint_state': ref(base / 'parent' / 'state.safetensors'),
                       'parent_outbox': str(base / 'parent-outbox.sqlite'), 'training_config': {'sha256': digest({'fixture-parent-training-config': 1})},
                       'preprocessing': ref(base / 'preprocessing.json'), 'global_deadline_unix': 4102444800.0,
                       'runtime_versions': {'python': sys.version.split()[0], 'numpy': str(numpy.__version__),
                                            'torch': str(torch.__version__), 'safetensors': str(safetensors.__version__)},
                       'owned_sources': {rel: sha((continuous / rel).read_bytes()) for rel in C.VERIFIER_SOURCES}}
        if idle: self.config.update(global_deadline_unix=time.time() - 600, closure_reserve_seconds=180)
        raw = canonical(self.config); (build / 'config.json').write_bytes(raw); self.campaign = sha(raw)
        if idle:
            deadline = self.config['global_deadline_unix']
            (build / 'window.json').write_bytes(canonical({'config_sha256': self.campaign, 'created_unix': deadline - 59468.0, 'deadline_unix': deadline}))
            self.cutoff_ns = C.in_loop_deadline_ns(deadline, self.config['closure_reserve_seconds'])
        training = build / 'training'
        ancestry = {'planner_ref': 'fixture-parent-planner (synthetic; not live)',
                    'head': {'events': 861, 'committed_batches': ORIGIN, 'unresolved_batches': 0, 'head_sha256': digest({'fixture-parent-head': ORIGIN})},
                    'checkpoint_sha256': parent['parameter_sha256'], 'optimizer_steps': ORIGIN, 'completed_epochs': 1,
                    'fingerprints': [digest({'oldbatch': i}) for i in range(ORIGIN)], 'exposed_context_ids': ids, 'presentations': len(ids)}
        phase1 = copy.deepcopy(self.config); phase1['neural']['max_updates'] = 2
        try:
            with patch.object(baseline_runner, 'publish', killer(baseline_runner.publish)):
                baseline_runner.run(snapshot, snapshot, phase1, root=training, campaign_sha=self.campaign, ancestry=ancestry,
                                    preprocessing=self.pre_raw, deadline=time.time() + 600, authority_check=lambda s: None,
                                    adoption_check=lambda s, r: None)
            raise AssertionError('phase 1 must stop at its killed result publication')
        except Killed: pass
        gc.collect()
        resume = reconcile.capture(training, campaign_sha256=self.campaign, frozen=True, max_updates=n['max_updates'],
                                   plateau_patience=n['plateau_patience'], learning_rate_floor=n['learning_rate_floor'], require_closed_epoch=False)
        resume['execution_config_sha256'] = digest({'fixture-execution': kind})
        snapshot.extension_raw = canonical({'tasks': []})
        replacement = copy.copy(snapshot); replacement.identity = digest({'fixture-prepared-snapshot': 1})
        replacement.batch_manifest = copy.deepcopy(snapshot.batch_manifest); replacement.batch_manifest['snapshot_sha256'] = replacement.identity
        adopted = []
        def adoption(s, r):
            if adopted: return None
            adopted.append(True); return replacement
        def authority(s):
            if kind == 'failed' and len(list(training.glob('batch-*.json'))) >= 4: raise ValueError('fixture authority drift inside the loop')
        publish = killer(continuous_runner.publish) if kind == 'killed' else continuous_runner.publish
        self.world = base / 'world'; fired = []
        sleepless = types.SimpleNamespace(time=time.time, monotonic=time.monotonic, sleep=lambda seconds: None)
        idle_patches = (patch.object(continuous_runner, 'readonly_outbox', killed_in_closure(continuous_runner.readonly_outbox, fired)),
                        patch.object(continuous_runner, 'time', sleepless)) if idle else ()
        try:
            with patch.object(continuous_runner, 'tick', deadline_after(PHASE2_UPDATES + idle, fired)), \
                    patch.object(continuous_runner, 'publish', publish), contextlib.ExitStack() as stack:
                for p in idle_patches: stack.enter_context(p)
                continuous_runner.run(snapshot, snapshot, self.config, root=training, campaign_sha=self.campaign, ancestry=ancestry,
                                      preprocessing=self.pre_raw, deadline=time.time() + 600, authority_check=authority,
                                      adoption_check=adoption, resume=resume)
            assert kind == 'clean', 'only the clean world returns normally'
            shutil.copytree(build, self.world)
        except Killed:
            assert kind in ('killed', 'idle')
            shutil.copytree(build, self.world)  # connections still open: -wal/-shm as a killed process leaves them
        except ValueError as error:
            if kind != 'failed' or 'fixture authority drift' not in str(error): raise
            shutil.copytree(build, self.world)
        gc.collect()
        assert adopted, 'prepared manifest adopted at a closed epoch'
        if idle:
            state = json.loads((self.world / 'training' / 'progress.json').read_bytes())['state']
            assert state == 'waiting_for_mandatory_gp_capacity' and fired and not (self.world / 'training' / 'result.json').exists(), state
            written_before(self.world, self.cutoff_ns)
        self.native, self.active = snapshot, replacement
        self.unit = f'w10-fixture-{kind}.service'
        self.readback = {'killed': readback(self.unit, active='failed', result='timeout'),
                         'clean': readback(self.unit), 'failed': readback(self.unit, active='failed', result='exit-code'),
                         'idle': readback(self.unit, active='failed', sub_state='failed', result='timeout')}[kind]

    def private_copy(self, dest):
        shutil.copytree(self.world, dest); return Path(dest)

    def reader(self, copy_root, scratch):
        return P.staged_outbox_reader(Path(copy_root) / 'training' / 'outbox.sqlite', self.config['parent_outbox'], scratch)

    def capture(self, copy_root, out, **overrides):
        args = {'campaign_sha256': self.campaign, 'unit': self.unit, 'owner_readback': self.readback,
                'outbox_reader': self.reader(copy_root, Path(out).parent)}
        if self.kind == 'idle':
            # FIXTURE: each private copy stands for a faithful copy of a live root in exactly that state, so the
            # injected live-root walk is the copy's own; the binding fixtures override it with the pristine world.
            args.update(owner_ruling=R17_RECEIPT.read_bytes(), live_walk=lambda: C.walk(copy_root)[1])
        args.update(overrides)
        return C.capture(copy_root, out, **args)


_ROOT = []; _WORLDS = {}


def world(kind):
    """Build each kind once per process inside one private temporary directory."""
    if kind not in _WORLDS:
        if not _ROOT: _ROOT.append(tempfile.TemporaryDirectory(prefix='w10-worlds-'))
        _WORLDS[kind] = World(Path(_ROOT[0].name).resolve() / kind, kind)
    return _WORLDS[kind]
