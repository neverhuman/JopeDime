"""cli_v6: the v2.0.0 learner entry point (derived from pipelines/prepared/continuous_recovery_cli.py at 745be43).

    python3 -B cli_v6.py --execution EXECUTION_JSON --execution-sha SHA256            # live unit (owner-approved only)
    python3 -B cli_v6.py --execution EXECUTION_JSON --execution-sha SHA256 \\
        --dry-run SCRATCH_DIR --dry-updates N                                           # CPU qualification on a copy

The execution config (``jope.generation-execution-config/v1``) pins the immutable source snapshot (every file's
sha256), the v2 campaign config (``jope.flywheel-campaign/v2``), the genesis receipt and the prepared genesis
descriptor.  Live mode checks host, cgroup caps, GPU/BLAS/temp boundaries and the exact numeric runtime like v1, fixes
the R11 window ONCE at first launch and never resets it, hydrates the genesis snapshot from the pinned prepared bytes
(no S3 at startup), rebuilds the active snapshot from the inbox adoption chain, and runs runner_v6.  Dry-run mode
copies the campaign root into SCRATCH_DIR and runs the same path for N updates there; the real root is never written.

r2 (Codex F4, F6, F2):
  * Before any data or model work the CLI joins the capture back to genesis and the campaign: the final-capture and
    capture-receipt sha256 equal the genesis pins (and campaign.identity.parents), ``digest(ancestry)`` equals
    genesis.ancestry_sha256, and the planner's stored ancestry equals the loaded ancestry.
  * ``window.json`` (``jope.v2-window/v2``) pins the boot id and PAIRED wall/boot/monotonic clocks at first launch.
    Every start derives the remaining lifetime from the boot clock and fails closed on a different boot (a reboot
    ends restartability until an owner ruling) or when wall and boot elapsed time disagree beyond NTP slew.
  * Live mode proves physical ownership: the process is inside exactly ``<unit>.service`` under the inclusive owner
    slice whose caps are within the envelope; GPU0 is the pinned physical device with no compute process on it; and
    every ``execution.stopped_owners`` unit is inactive with MainPID 0 and an empty or absent cgroup.
  * A fatal error writes a create-only ``fatal-<ns>.json`` into the campaign root (never overwritten); every later
    start refuses while one exists.  The first start uses genesis only if no segment ever started; otherwise only a
    verified closure resumes (resume_v6).
"""
import argparse, hashlib, json, os, shutil, socket, subprocess, sys, time, types
from pathlib import Path
sys.dont_write_bytecode = True

EXECUTION_SCHEMA = 'jope.generation-execution-config/v1'
EXECUTION_KEYS = {'schema', 'campaign_root', 'campaign_config', 'source_root', 'source_manifest', 'genesis_receipt',
                  'prepared_genesis_descriptor', 'stopped_owners'}
CAMPAIGN_SCHEMA = 'jope.flywheel-campaign/v2'
WINDOW_SCHEMA = 'jope.v2-window/v2'
FATAL_DIR = None
ANCESTRY_SUMMARY = ('checkpoint_sha256', 'optimizer_steps', 'completed_epochs', 'presentations')


def _need(ok, msg):
    if not ok: raise ValueError(msg)


def clocks():
    return {'wall_ns': time.time_ns(), 'boot_ns': time.clock_gettime_ns(time.CLOCK_BOOTTIME), 'mono_ns': time.monotonic_ns(),
            'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def slew_seconds(elapsed):
    return 60.0 + 5e-4 * elapsed       # NTP slews at most 500 ppm; 60 s covers one step correction


def window_remaining(window, now):
    """Seconds left in the fixed lifetime, from the paired boot clock; fails closed on a reboot or wall disagreement."""
    c0 = window['created']
    _need(now['boot_id'] == c0['boot_id'], 'same boot as the window start: a reboot ends restartability (owner ruling required)')
    eb = (now['boot_ns'] - c0['boot_ns']) / 1e9; ew = (now['wall_ns'] - c0['wall_ns']) / 1e9
    _need(eb >= 0 and abs(ew - eb) <= slew_seconds(eb), 'paired wall/boot clocks agree (no wall rollback or jump)')
    left = window['window_seconds'] - eb
    _need(left > 0, 'inside the fixed v2 window')
    return left, {'elapsed_boot_seconds': eb, 'elapsed_wall_seconds': ew}


def join_capture(c, g, load_capture):
    """F4: the capture on disk is exactly the one genesis and the campaign identity pinned, and so is its ancestry."""
    cap = Path(c['capture_dir'])
    _need(_sha((cap / 'final-capture.json').read_bytes()) == g['capture']['final_capture_sha256'] ==
          c['identity']['parents']['final_capture_sha256'] and
          _sha((cap / 'capture-receipt.json').read_bytes()) == g['capture']['capture_receipt_sha256'], 'pinned capture and capture receipt')
    _, _, ancestry = load_capture(cap)
    from jope_flywheel.protocol import digest
    _need(digest(ancestry) == g['ancestry_sha256'] and {k: ancestry[k] for k in ANCESTRY_SUMMARY} == g['ancestry_summary'],
          'loaded ancestry is the genesis-pinned ancestry')
    return ancestry


def own_cgroup():
    rel = next(x[3:] for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
    return Path('/sys/fs/cgroup') / rel.lstrip('/')


def check_unit(group, c):
    env = c['envelope']
    _need(group.name == c['unit'] + '.service', 'process runs inside exactly the campaign unit ' + c['unit'])
    owner = next((q for q in group.parents if q.name == env['slice']), None)
    _need(owner is not None, 'campaign unit inside the inclusive owner slice ' + env['slice'])
    mem = (owner / 'memory.max').read_text().strip(); quota, period = (owner / 'cpu.max').read_text().split()
    _need(mem != 'max' and int(mem) <= env['inclusive_memory_bytes'] and quota != 'max' and
          int(quota) <= env['inclusive_cpus'] * int(period), 'inclusive owner allocation within the R1 envelope')
    return {'unit_cgroup': str(group), 'owner_slice': str(owner), 'owner_memory_max': int(mem), 'owner_cpu_max': [int(quota), int(period)]}


def check_gpu(expected_uuid, smi='nvidia-smi'):
    """GPU0 is the pinned physical device and carries no compute process (run before CUDA is initialised)."""
    uuid = subprocess.run([smi, '--query-gpu=uuid', '--format=csv,noheader', '-i', '0'], capture_output=True, text=True,
                          check=True, timeout=30).stdout.strip()
    apps = subprocess.run([smi, '--query-compute-apps=pid,gpu_uuid', '--format=csv,noheader'], capture_output=True, text=True,
                          check=True, timeout=30).stdout.split('\n')
    users = [a for a in apps if a.strip() and a.split(',')[1].strip() == uuid]
    _need(uuid == expected_uuid and not users, 'pinned GPU0 with no other compute process')
    return {'gpu_uuid': uuid, 'compute_apps_on_gpu': 0}


def check_stopped_owners(units, systemctl=('systemctl', '--user'), cgroup_root=Path('/sys/fs/cgroup')):
    proofs = []
    for unit in units:
        out = subprocess.run([*systemctl, 'show', unit, '-p', 'ActiveState,SubState,MainPID,ControlGroup,LoadState,InvocationID'],
                             capture_output=True, text=True, check=True, timeout=30).stdout
        props = dict(line.split('=', 1) for line in out.splitlines() if '=' in line)
        group = Path(cgroup_root) / props.get('ControlGroup', '').lstrip('/') if props.get('ControlGroup') else None
        procs = (group / 'cgroup.procs').read_text().split() if group is not None and (group / 'cgroup.procs').exists() else []
        _need(props.get('ActiveState') in ('inactive', 'failed') and props.get('MainPID') == '0' and not procs,
              'stopped owner is closed: ' + unit)
        proofs.append({'unit': unit, **{k: props.get(k) for k in ('ActiveState', 'SubState', 'MainPID', 'LoadState', 'InvocationID')},
                       'cgroup_procs': 0})
    return proofs


def _sha(b): return hashlib.sha256(b).hexdigest()


def search_path(src):
    src = Path(src)
    return [str(src / 'pipelines/prepared'), str(src / 'pipelines/continuous'), str(src / 'pipelines/continuous/consumer'),
            str(src / 'pipelines/continuous/runtime'), str(src), str(src / 'pipelines/native'),
            str(src / 'pipelines/hpc_verified_v5')]


def check_source(execution):
    """Every file of the immutable source snapshot is byte-pinned by the source manifest (no ambient source)."""
    src = Path(execution['source_root']); raw = Path(execution['source_manifest']['path']).read_bytes()
    if _sha(raw) != execution['source_manifest']['sha256']: raise ValueError('pinned source manifest')
    m = json.loads(raw)
    if m.get('schema') != 'jope.v2-source-snapshot/v1': raise ValueError('source manifest schema')
    for rel, h in m['files'].items():
        p = src / rel
        if Path(rel).is_absolute() or '..' in Path(rel).parts or p.is_symlink() or _sha(p.read_bytes()) != h:
            raise ValueError('pinned source file: ' + rel)
    return m


def hydrate_genesis(descriptor, native, *, campaign_sha256, preprocessing_sha256, registry_sha256, authority_check):
    """prepared_admission.candidate's descriptor/config/status/owned-source/adapter closure for an ALREADY-ADOPTED
    prepared root, minus only the predecessor-prefix comparison (there is no predecessor at v2 genesis)."""
    import prepared_snapshot as P
    from jope_flywheel import native_variant_snapshot as V
    from jope_flywheel.program_io import bound, need, read, sha
    need(set(descriptor) == {'schema', 'campaign_sha256', 'approved_for_bounded_research', 'preparation_config',
                             'preparation_status', 'prepared'} and descriptor['schema'] == 'jope.prepared-epoch-candidate/v1' and
         descriptor['approved_for_bounded_research'] is True, 'closed prepared genesis descriptor')
    config_raw = bound(descriptor['preparation_config']); status_raw = bound(descriptor['preparation_status'])
    c = json.loads(config_raw); receipt = json.loads(status_raw); root = Path(c['owned_root'])
    need(root.is_absolute() and c['schema'] == 'jope.prepared-mixed-snapshot-config/v1', 'owned CPU preparation schema')
    for ref, name in ((descriptor['preparation_config'], 'config.json'), (descriptor['preparation_status'], 'status.json'),
                      (descriptor['prepared'], 'prepared.json')):
        need(Path(ref['path']) == root / name, 'one immutable preparation root')
    need(receipt['config_sha256'] == sha(config_raw) and receipt['prepared'] == descriptor['prepared'] and
         receipt['extension_sha256'] == c['extension_sha256'], 'pinned successful receipt and configuration closure')
    need(all(c[k] == receipt[k] == 0 for k in ('network_gets', 's3_writes', 'optimizer_updates', 'planner_mutations')),
         'preparation cannot mutate optimizer or admission')
    for rel, pin in c['owned_sources'].items():
        p = Path(rel); need(not p.is_absolute() and '..' not in p.parts and sha(read(root / p, 32 * 1024**2)) == pin,
                            'exact preparation source closure')
    need(receipt['prepared_source_sha256'] == c['owned_sources']['prepared_snapshot.py'] == sha(read(Path(P.__file__).resolve())),
         'qualified hydration implementation')
    need(receipt['source_adapter_sha256'] == sha(read(Path(V.__file__).resolve())), 'qualified measured record adapter')
    need(sha(read(root / 'extension.json', 32 * 1024**2)) == c['extension_sha256'], 'pinned extension')
    prepared = bound(descriptor['prepared'], 32 * 1024**2)
    return P.load(prepared, native, receipt=receipt, expected_extension_sha256=c['extension_sha256'],
                  expected_preprocessing_sha256=preprocessing_sha256, expected_parent_registry_sha256=registry_sha256,
                  authority_check=authority_check)


def prepare(args, *, n_overrides=None):
    """Everything up to (not including) runner_v6.run: the same path for live, dry-run and the parity probe."""
    global FATAL_DIR
    execution_raw = args.execution.read_bytes(); execution = json.loads(execution_raw)
    if _sha(execution_raw) != args.execution_sha or execution.get('schema') != EXECUTION_SCHEMA or set(execution) != EXECUTION_KEYS:
        raise ValueError('immutable closed execution configuration')
    check_source(execution)
    sys.path[:0] = search_path(execution['source_root'])
    from jope_flywheel.program_io import read, bound, sha, need, create, bind_campaign_deadline
    from jope_flywheel.protocol import canonical, digest
    from jope_flywheel import training_snapshot_native as N, native_variant_snapshot as V, timing
    import native_corpus_authority as authority
    from neural_native_train import publish, OBJECTIVE
    from neural_native_stream import BatchStream
    from pipelines.boundary_admission_v2 import boundary as Bd
    from pipelines.native_result_overlay_v1 import loader as OL, packet as PK
    from pipelines.final_capture_v1 import capture as FC
    from pipelines.continuous_v6 import genesis as G, resume_v6 as RV, runner_v6 as R6
    dry = args.dry_run is not None
    campaign_root = Path(execution['campaign_root'])
    raw = bound(execution['campaign_config']); c = json.loads(raw); pin = sha(raw)
    need(c['schema'] == CAMPAIGN_SCHEMA and c['generation'] == 'v2.0.0' and Path(execution['campaign_config']['path']) == campaign_root / 'config.json',
         'v2 campaign config')
    g_raw = bound(execution['genesis_receipt']); g = json.loads(g_raw)
    need(Path(execution['genesis_receipt']['path']) == campaign_root / 'genesis.json' and g['campaign_sha256'] == c['campaign_sha256'],
         'genesis receipt of this campaign')
    need(dry or g['probe'] is False, 'a pre-R14 probe root never runs live')
    n = c['neural']
    need(n['objective'] == OBJECTIVE and n['max_new_updates'] == c['max_new_updates'] and n['max_padded_tokens'] == 8192 and
         n['max_seq'] == 640 and n['emissions_per_step'] in (1, 2) and n['closure_budget_seconds'] < c['closure_reserve_seconds'],
         'reviewed v2 numerical scope; closure budget inside the closure reserve')
    ancestry = join_capture(c, g, FC.load_capture)   # F4: before any data or model work
    if dry:
        need(not args.dry_run.exists() and args.dry_run.parent.is_dir() and 0 < args.dry_updates <= 16, 'new dry-run scratch')
        args.dry_run.mkdir(); work = args.dry_run / 'campaign'; FATAL_DIR = work
        shutil.copytree(campaign_root, work, symlinks=True)
        n = {**n, 'device': 'cuda:0' if args.dry_device == 'cuda' else 'cpu', 'max_new_updates': args.dry_updates,
             'step_budget_floor_seconds': 1.0, **(n_overrides or {})}
        now = clocks(); window = {'schema': WINDOW_SCHEMA, 'config_sha256': pin, 'created': now, 'window_seconds': 6 * 3600,
                                  'deadline_unix': now['wall_ns'] / 1e9 + 6 * 3600, 'dry_run': True}
        physical = {'dry_run': True}
        if args.dry_device == 'cuda': physical['gpu'] = check_gpu(c['gpu_uuid'])
    else:
        work = campaign_root; FATAL_DIR = work
        need(c['host'] == socket.gethostname().split('.')[0], 'campaign host')
        need(not list(work.glob('fatal*.json')), 'no unreviewed fatal record')
        wpath = campaign_root / 'window.json'
        if not wpath.exists():
            need(g['resume']['local_step'] == 0 and not (work / 'training' / 'segment.json').exists(), 'the window is fixed once, at the first launch')
            now = clocks()
            create(wpath, canonical({'schema': WINDOW_SCHEMA, 'config_sha256': pin, 'created': now, 'window_seconds': 3600 * c['window_hours'],
                                     'deadline_unix': now['wall_ns'] / 1e9 + 3600 * c['window_hours'], 'ruling': c['owner_ruling_refs']['R11']}))
        window = json.loads(read(wpath))
        need(window['schema'] == WINDOW_SCHEMA and window['config_sha256'] == pin, 'fixed paired-clock v2 window')
        group = own_cgroup()
        quota, period = (group / 'cpu.max').read_text().split()
        env = c['envelope']
        need(quota != 'max' and int(quota) <= env['cpus'] * int(period) and int((group / 'memory.max').read_text()) <= env['memory_bytes'] and
             int((group / 'memory.swap.max').read_text()) == 0 and int((group / 'pids.max').read_text()) <= env['tasks'], 'owned cgroup caps')
        need(os.getpriority(os.PRIO_PROCESS, 0) >= 10 and os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and
             os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and os.environ.get('TMPDIR') == str(campaign_root / 'tmp') and
             all(os.environ.get(k) == '1' for k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')),
             'owned GPU/BLAS/temp boundary')
        physical = {'unit': check_unit(group, c), 'gpu': check_gpu(c['gpu_uuid']),
                    'stopped_owners': check_stopped_owners(execution['stopped_owners'])}
    now = clocks(); left, elapsed = window_remaining(window, now)
    deadline = now['wall_ns'] / 1e9 + left - c['closure_reserve_seconds']; bind_campaign_deadline(deadline)
    create(work / f'window-start-{now["wall_ns"]}.json', canonical({'window_sha256': digest(window), 'clocks': now, 'remaining_seconds': left,
                                                                   **elapsed, 'deadline_unix': deadline, 'physical': physical}))
    import numpy as np, torch, safetensors
    actual = {'python': sys.version.split()[0], 'numpy': str(np.__version__), 'torch': str(torch.__version__), 'safetensors': str(safetensors.__version__)}
    need(actual == c['runtime_versions'], 'exact numeric runtime')
    timing.ACTIVE = timing.Timing(work / f'stage-timing-{time.time_ns()}.jsonl', None if dry else group, config_sha256=sha(execution_raw),
                                  origin_checkpoint_sha256=g['ancestry_summary']['checkpoint_sha256'])   # one create-only file per launch
    pre = bound(c['preprocessing']); registry = bound(c['parent_registry']); parents = {p['parent_lineage_sha256']: p for p in map(N.document, registry.splitlines())}
    specraw = bound(c['snapshot']); spec = json.loads(specraw)

    def fresh(snapshot=None):
        p = canonical(authority.check(spec)); doc = json.loads(p)
        need(doc['passed'] is True and doc['source_pins'] == spec['authority_pins'] and 0 <= time.time() - doc['checked_at_unix'] <= 60,
             'fresh exact native/current role proof')
        if snapshot is not None: V.check_authority(json.loads(snapshot.extension_raw), spec, p, parents, time.time())
        publish(work, 'latest-role-proof.json', doc); return p

    with timing.span('input.native_snapshot_load'):
        native = N.load_snapshot(specraw, c['snapshot']['sha256'], N.LocalStore(c['read_roots']), authority_bytes=fresh())
    descriptor = json.loads(bound(execution['prepared_genesis_descriptor']))
    with timing.span('input.genesis_snapshot_hydration'):
        genesis = hydrate_genesis(descriptor, native, campaign_sha256=descriptor['campaign_sha256'],
                                  preprocessing_sha256=c['preprocessing']['sha256'], registry_sha256=c['parent_registry']['sha256'],
                                  authority_check=lambda ext: fresh(types.SimpleNamespace(extension_raw=ext)))
    genesis.extension_raw = read(Path(json.loads(bound(descriptor['preparation_config']))['owned_root']) / 'extension.json', 32 * 1024**2)
    need(genesis.identity == g['genesis_snapshot_sha256'] and sha(canonical(genesis.batch_manifest)) == g['active_manifest_sha256'],
         'hydrated genesis snapshot is the captured active snapshot')
    overlay = types.SimpleNamespace(validate_chain=PK.validate_chain, head_from_sha=PK.head_from_sha, overlay_snapshot=OL.overlay_snapshot)
    inbox = Bd.Inbox(work / 'inbox', overlay=overlay, max_padded_tokens=n['max_padded_tokens'])
    t = work / 'training'; manifest_raw, active = Bd.active_manifest(t / 'batches.sqlite')
    planner = BatchStream(t / 'batches.sqlite', manifest_raw, active, seed=n['seed'], batch_size=n['batch_size'],
                          initial_checkpoint_sha256=g['ancestry_summary']['checkpoint_sha256'])
    try:
        need(planner._get('ancestry') == ancestry, 'planner stored ancestry is the loaded, genesis-pinned ancestry')
        state = inbox.reconcile(planner); need(not state['holds'], 'no inbox HOLD at start (review required)')
    finally:
        planner.close()
    check = lambda raw_ext, proof: V.check_authority(json.loads(raw_ext), spec, proof, parents, time.time())
    snapshot = genesis
    if state['records']:
        snapshot = Bd.load_adopted(inbox, state['records'][-1], native, genesis=genesis, preprocessing_sha256=c['preprocessing']['sha256'],
                                   registry_sha256=c['parent_registry']['sha256'], authority_check=lambda ext: check(ext, fresh()))
    resume = (RV.v2_resume(work, execution_config_sha256=sha(execution_raw)) if (t / 'segment.json').exists() else
              G.genesis_resume(work, execution_config_sha256=sha(execution_raw)))
    boundary = R6.BoundaryDriver(inbox, native, root=t, campaign_sha256=c['campaign_sha256'],
                                 preprocessing_sha256=c['preprocessing']['sha256'], registry_sha256=c['parent_registry']['sha256'],
                                 fresh_authority=fresh, authority_check=check)
    if not dry or args.dry_device == 'cuda': need(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one available CUDA device')
    return types.SimpleNamespace(native=native, snapshot=snapshot, n=n, c=c, g=g, genesis_sha256=sha(g_raw), work=work, t=t, pin=pin,
                                 window=window, dry=dry, ancestry=ancestry, pre=pre, deadline=deadline, fresh=fresh,
                                 boundary=boundary, resume=resume, R6=R6, publish=publish, execution_sha256=sha(execution_raw))


def run_prepared(x, **run_kw):
    result = x.R6.run(x.native, x.snapshot, {'neural': x.n, 'inherited_backlog': x.g['inherited_backlog']}, root=x.t,
                      campaign_sha=x.c['campaign_sha256'], ancestry=x.ancestry, preprocessing=x.pre, deadline=x.deadline,
                      authority_check=x.fresh, boundary=x.boundary, resume=x.resume, genesis_sha256=x.genesis_sha256, **run_kw)
    x.publish(x.work, 'status.json', {'schema': 'jope.v2-campaign-status/v1', 'state': result['state'], 'config_sha256': x.pin,
                                      'at': time.time(), 'result': result, 'window': x.window, 'dry_run': x.dry, 'training_root': str(x.t)})
    return result


def parser():
    ap = argparse.ArgumentParser(); ap.add_argument('--execution', type=Path, required=True)
    ap.add_argument('--execution-sha', required=True); ap.add_argument('--dry-run', type=Path)
    ap.add_argument('--dry-updates', type=int, default=2); ap.add_argument('--dry-device', choices=('cpu', 'cuda'), default='cpu')
    return ap


def main(argv=None):
    return run_prepared(prepare(parser().parse_args(argv)))


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        try:
            with ((FATAL_DIR or Path.cwd()) / f'fatal-{time.time_ns()}.json').open('xb') as f:
                f.write(json.dumps({'schema': 'jope.v2-campaign-fatal/v1', 'at': time.time(), 'error': type(error).__name__ + ': ' + str(error)[:4000],
                                    'automatic_retry': False}, sort_keys=True).encode()); f.flush(); os.fsync(f.fileno())
        except BaseException: pass
        raise
