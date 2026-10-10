"""jope.continuous-generation-final-capture/v2: read-only final capture of one terminal continuous generation.

Closed terminal classification (r2); every other combination refuses:
- clean_terminal (r1, unchanged): progress.state is global_deadline_reached or maximum_updates_reached. It takes no
  owner ruling and no live-root walk.
- runtime_limit_stopped_idle (owner ruling R17): progress.state is one of the pinned runner's NON-RESERVING waiting
  states (IDLE_WAITING; each is published before any reservation of its loop iteration) and systemd stopped the unit
  by its runtime limit. It additionally requires: owner Result=timeout and NRestarts=0 on top of the r1 stopped-owner
  check; the sealed R17 receipt bytes (pinned sha256); window.json joined to config.json, the in-loop deadline being
  window.deadline_unix - config.closure_reserve_seconds (the runner's own; nothing else subtracted), already passed;
  a bounded lstat-only walk of the private copy that admits only regular files and directories, every mtime <= that
  deadline, whose summary equals the injected live-root walk before the first and after the last input read; the r1
  planner/proof/checkpoint/restore closure plus an explicit R17 closure (last commit step == progress optimizer_steps
  == final rolling slot, and commit checkpoint_after == rolling parameter sha == progress parameter sha); and
  status.json and training/result.json absent, recorded 'absent' and never reconstructed.

Invariants (each is a refusal, never a repair; a refusal leaves no output directory):
- Runs only on a PRIVATE COPY of a terminal campaign root, never the owned live root. SQLite never opens a file of
  the copy: the planner (+ -wal) is staged byte-for-byte into scratch, then copied by one single-step online backup
  (one read transaction) into a sealed rollback-journal file that every later planner read uses (immutable=1).
- The owner is stopped: the injected readback is closed, ActiveState is inactive|failed (a RuntimeMaxSec kill is
  failed/timeout), MainPID is 0 and the recorded cgroup is absent or empty; it is read before the first input
  read and after the last, and both readbacks must be identical.
- Terminal state is one of the two classes above. failure.json(.pending), uncertain-physical-state,
  closure-errors.jsonl, any other progress/result state, any reserved/needs_reconcile batch, an orphan proof or a
  rolling slot beyond the last commit refuse: an uncertain optimizer outcome is a HOLD.
- Every committed batch joins its immutable proof (proof sha == commit evidence event_sha256), its absolute and
  local step, intent, checkpoint chain and pre-optimizer ACK; the ordered ACK digest equals the injected outbox
  summary. The verifier modules are byte-identical to the campaign's owned sources; the runtime is the pinned one.
- One parameter sha joins planner, final proof, rolling manifest, progress, optimizer-control and (if present)
  result.json; the state bytes match; a throwaway CPU continuous_state.restore reproduces it and restores the
  caller's RNG. LR groups and the plateau controller are carried exactly, never reset.
- result.json is optional: absent, the closure above is the proof; present, it must agree exactly.
- The emitted ancestry is the closed dict BatchStream.inherit accepts; a fresh dry-run journal accepts it twice.
- Output is a new directory of create-only (O_EXCL|O_NOFOLLOW) 0444 fsynced files. final-capture.json is
  deterministic (no clock, no private path) and at most DOC_MAX bytes; capture-receipt.json carries the timestamp
  and private paths. Unbounded sections (per-batch proof joins, lineage fingerprints, exposed context ids) are
  ordered canonical-JSONL parts of at most PART_MAX bytes, each pinned by sha256/bytes/count and a link that binds
  (kind, ordinal, previous link, file, sha256, bytes, count); the document pins each ordered part list and its chain
  head. load_capture re-verifies every part, the order of records and the reassembled ancestry digest (and, for the
  R17 class, the walk records) before the capture is published, and is the only supported reader.
"""
import hashlib, json, math, os, shutil, sqlite3, stat, struct, sys, tempfile, time
from fractions import Fraction
from pathlib import Path
import numpy, safetensors, torch
from dltrain import model as M
from jope_flywheel import program_io, protocol
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import need, read, sha
import continuous_state, neural_native_stream, neural_storage
from continuous_state import restore
from neural_native_stream import BatchStream, decode, validate_manifest

SCHEMA = 'jope.continuous-generation-final-capture/v2'
RECEIPT = 'jope.continuous-generation-final-capture-receipt/v2'
LINEAGE_HEAD = 'jope.continuous-lineage-planner-head/v1'
TERMINAL = ('global_deadline_reached', 'maximum_updates_reached')
# The pinned runner (prepared/continuous_runner.py, sha 10cbf4ea) publishes these before reserving in that iteration;
# waiting_for_unique_batch follows a reserve() that returned Waiting (no batch row). 'running' is published after a
# commit, so the next iteration may hold a reservation: it is never idle.
IDLE_WAITING = ('waiting_for_mandatory_gp_capacity', 'waiting_for_output_capacity', 'waiting_for_unique_batch')
CLEAN, RUNTIME_LIMIT = 'clean_terminal', 'runtime_limit_stopped_idle'
R17 = {'id': 'R17', 'receipt_sha256': '2a1b0a5e96b54cf48d4f7f8a2bf57c86938814d92846d4c30a7e642711d7b534'}
STOPPED = ('inactive', 'failed')
FAILURE_MARKERS = ('failure.json', 'failure.json.pending', 'uncertain-physical-state', 'closure-errors.jsonl')
READBACK_KEYS = {'unit', 'LoadState', 'ActiveState', 'SubState', 'MainPID', 'Result', 'NRestarts', 'ControlGroup', 'cgroup'}
CGROUP_KEYS = {'path', 'exists', 'procs', 'populated'}
PARENT_OUTBOX_KEYS = {'schema', 'observations', 'batches', 'states', 'unresolved', 'numeric_scores_synthesized'}
V1_OUTBOX_KEYS = PARENT_OUTBOX_KEYS | {'events', 'head_sha256', 'acks_sha256', 'ack_count'}
VERIFIER_SOURCES = {'neural_native_stream.py': neural_native_stream, 'continuous_state.py': continuous_state,
                    'neural_storage.py': neural_storage, 'consumer/dltrain/model.py': M,
                    'runtime/jope_flywheel/protocol.py': protocol, 'runtime/jope_flywheel/program_io.py': program_io}
RUNTIME_KEYS = ('python', 'numpy', 'torch', 'safetensors')
MIB = 1024**2
PART_MAX = 8 * MIB
DOC_MAX = 1 * MIB
CHAIN = 'jope.final-capture-part-chain/v1'
ANCESTRY_KEYS = ('planner_ref', 'head', 'checkpoint_sha256', 'optimizer_steps', 'completed_epochs', 'fingerprints',
                 'exposed_context_ids', 'presentations')
JOIN_KEYS = {'absolute_step', 'local_step', 'batch_id', 'epoch', 'index', 'fingerprint', 'proof_sha256',
             'checkpoint_before', 'parameter_sha256', 'ack_seq'}
WALK = 'jope.final-capture-root-walk/v1'
WALK_KEYS = {'path', 'kind', 'bytes', 'mtime_ns'}
WALK_MAX_ENTRIES = 1_000_000     # live v1 root: 52,329 entries (60 dirs, 52,269 files), depth 4
WALK_MAX_BYTES = 256 * MIB       # canonical JSONL walk records (~110 B each)


def ref(raw): return {'sha256': sha(raw), 'bytes': len(raw)}


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def seal(path, body):
    """Create-only O_EXCL|O_NOFOLLOW 0444 file, fully written and fsynced."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    try:
        view = memoryview(body)
        while view: view = view[os.write(fd, view):]
        os.fsync(fd)
    finally: os.close(fd)
    return ref(body)


def file_sha(path):
    h = hashlib.sha256(); size = 0
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(MIB), b''): h.update(block); size += len(block)
    return {'sha256': h.hexdigest(), 'bytes': size}


def load(path, cap=32 * MIB):
    need(path.is_file() and not path.is_symlink(), 'required regular private-copy file: ' + path.name)
    return read(path, cap)


def owner(readback, unit):
    """Closed readback of a stopped owner: inactive|failed, MainPID 0, recorded cgroup absent or empty. NRestarts is
    part of the closed shape; the R17 class additionally requires Result=timeout and NRestarts=0 (runtime_limit)."""
    r = readback()
    need(type(r) is dict and set(r) == READBACK_KEYS and type(r['cgroup']) is dict and set(r['cgroup']) == CGROUP_KEYS and
         type(r['NRestarts']) is int and r['NRestarts'] >= 0, 'closed owner readback')
    g = r['cgroup']
    need(r['unit'] == unit and r['ActiveState'] in STOPPED and type(r['MainPID']) is int and r['MainPID'] == 0 and
         type(g['path']) is str and g['path'] and r['ControlGroup'] in ('', g['path']) and type(g['procs']) is list and
         g['procs'] == [] and ((g['exists'] is False and g['populated'] is None) or (g['exists'] is True and g['populated'] is False)),
         'owner still active or its cgroup is not empty; capture refused')
    return r


def walk(root):
    """Bounded, non-following metadata walk of a campaign root: lstat and directory listings only, no file is opened,
    so it is safe on the live owned root. Only regular files and directories are admitted: any symlink (escaping the
    root or not; never followed) or special file refuses, as does more than WALK_MAX_ENTRIES entries or WALK_MAX_BYTES
    of canonical records. Returns (records, summary): one record {path, kind, bytes, mtime_ns} per entry, sorted by
    path ('.' is the root itself; bytes only for files), and the closed summary whose records_sha256 is the sha256 of
    the records' canonical JSONL."""
    records = []; pending = [('.', Path(root))]; count = [1, 0]  # entries discovered, canonical record bytes

    def bounded():
        need(count[0] <= WALK_MAX_ENTRIES and count[1] <= WALK_MAX_BYTES, 'campaign-root walk exceeds its entry/byte bounds')

    while pending:
        rel, path = pending.pop(); st = os.lstat(path); mode = st.st_mode
        kind = 'file' if stat.S_ISREG(mode) else 'dir' if stat.S_ISDIR(mode) else 'symlink' if stat.S_ISLNK(mode) else 'other'
        need(kind in ('file', 'dir'), 'campaign-root walk admits only regular files and directories (symlinks are never followed): ' + rel)
        record = {'path': rel, 'kind': kind, 'bytes': st.st_size if kind == 'file' else None, 'mtime_ns': st.st_mtime_ns}
        records.append(record); count[1] += len(canonical(record)) + 1; bounded()
        if kind == 'dir':
            with os.scandir(path) as listing:
                for entry in listing:
                    pending.append((entry.name if rel == '.' else rel + '/' + entry.name, path / entry.name)); count[0] += 1; bounded()
    records.sort(key=lambda r: r['path']); h = hashlib.sha256()
    for r in records: h.update(canonical(r) + b'\n')
    files = [r for r in records if r['kind'] == 'file']; newest = max(files, key=lambda r: (r['mtime_ns'], r['path']), default=None)
    return records, {'schema': WALK, 'entries': len(records), 'files': len(files), 'dirs': len(records) - len(files),
                     'file_bytes': sum(r['bytes'] for r in files), 'max_mtime_ns': max(r['mtime_ns'] for r in records),
                     'newest_file': None if newest is None else {'path': newest['path'], 'mtime_ns': newest['mtime_ns']},
                     'records_sha256': h.hexdigest()}


def in_loop_deadline_ns(deadline_unix, closure_reserve_seconds):
    """The runner's in-loop deadline, window.deadline_unix - config.closure_reserve_seconds, exact and floored to ns."""
    return math.floor((Fraction(deadline_unix) - Fraction(closure_reserve_seconds)) * 10**9)


def runtime_limit(root, c, campaign, readback, owner_ruling, live, stage, part_max):
    """R17 gates that need no planner: unit stopped by its runtime limit, the sealed ruling, the in-loop deadline from
    the campaign's own window/config, and the no-write walk bound to the live root. Returns the terminal section."""
    need(readback['Result'] == 'timeout', 'runtime_limit_stopped_idle requires systemd Result=timeout: ' + str(readback['Result']))
    need(readback['NRestarts'] == 0, 'runtime_limit_stopped_idle requires NRestarts=0: ' + str(readback['NRestarts']))
    need(type(owner_ruling) is bytes and sha(owner_ruling) == R17['receipt_sha256'],
         'runtime_limit_stopped_idle requires the sealed owner R17 ruling receipt bytes')
    window_raw = load(root / 'window.json'); w = json.loads(window_raw); reserve = c.get('closure_reserve_seconds')
    need(type(w) is dict and w.get('config_sha256') == campaign and type(w.get('deadline_unix')) in (int, float) and
         math.isfinite(w['deadline_unix']) and w['deadline_unix'] == c.get('global_deadline_unix') and
         type(reserve) in (int, float) and math.isfinite(reserve) and reserve >= 0,
         'in-loop deadline: window.json must join config.json (config_sha256, deadline_unix == global_deadline_unix, closure_reserve_seconds)')
    cutoff = in_loop_deadline_ns(w['deadline_unix'], reserve)
    need(time.time_ns() > cutoff, 'in-loop deadline has not passed; a no-write proof before it would be vacuous')
    records, summary = walk(root)
    late = [r['path'] for r in records if r['mtime_ns'] > cutoff]
    need(not late, 'campaign root written after the in-loop deadline: ' + ','.join(late[:8]))
    need(type(live) is dict and live == summary, 'private copy is not a metadata-faithful copy of the live campaign root (walk summaries differ)')
    return {'deadline': {'window': ref(window_raw), 'deadline_unix': w['deadline_unix'], 'closure_reserve_seconds': reserve,
                         'in_loop_deadline_ns': cutoff,
                         'rule': 'window.deadline_unix - config.closure_reserve_seconds (the runner in-loop deadline); every walked mtime_ns <= it'},
            'walk': {**summary, 'live_root_walk_equal': True, 'parts': write_parts(stage, 'walk', records, part_max=part_max)}}


def backup(src, dst, scratch):
    """Stage the private copy's planner (+ -wal) byte-for-byte, then one single-step online backup (one read
    transaction) into dst, converted to a rollback-journal file and sealed 0444. The copy itself is never opened."""
    stage = Path(tempfile.mkdtemp(prefix='planner-stage-', dir=scratch))
    try:
        for suffix in ('', '-wal'):
            p = Path(str(src) + suffix)
            if suffix and not (p.exists() or p.is_symlink()): continue
            need(p.is_file() and not p.is_symlink(), 'regular private planner file: ' + p.name)
            before = os.stat(p); shutil.copyfile(p, stage / ('planner.sqlite' + suffix), follow_symlinks=False)
            after = os.stat(p); need((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'private planner changed during staging')
        s = sqlite3.connect(stage / 'planner.sqlite', isolation_level=None)
        try:
            s.execute('PRAGMA query_only=ON'); d = sqlite3.connect(dst, isolation_level=None)
            try: s.backup(d, pages=-1); d.execute('PRAGMA journal_mode=DELETE')
            finally: d.close()
        finally: s.close()
    finally: shutil.rmtree(stage)
    os.chmod(dst, 0o444); fd = os.open(dst, os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return file_sha(dst)


def tensor_classes(raw):
    """Classify a safetensors header without loading tensors: model.*, optimizer.<param>.<step|exp_avg|exp_avg_sq>,
    rng.torch_cpu and, where present, rng.torch_cuda.<index>; any other name refuses."""
    need(type(raw) is bytes and len(raw) >= 8, 'safetensors header')
    size = struct.unpack('<Q', raw[:8])[0]; need(0 < size <= len(raw) - 8, 'safetensors header bound')
    header = json.loads(raw[8:8 + size]); header.pop('__metadata__', None)
    model = {k[6:] for k in header if k.startswith('model.')}; adam = {}; rng = []; other = []
    for k, v in header.items():
        if k.startswith('model.'): continue
        if k.startswith('optimizer.'):
            name, _, slot = k[10:].rpartition('.')
            if name in model and slot in ('step', 'exp_avg', 'exp_avg_sq'): adam.setdefault(name, set()).add(slot); continue
        if k == 'rng.torch_cpu' or (k.startswith('rng.torch_cuda.') and k[15:].isdigit()):
            need(v['dtype'] == 'U8' and len(v['shape']) == 1, 'byte RNG state'); rng.append(k); continue
        other.append(k)
    need(not other and model and adam and 'rng.torch_cpu' in rng and
         all(s == {'step', 'exp_avg', 'exp_avg_sq'} for s in adam.values()),
         'checkpoint tensor key set: model tensors, AdamW state and rng.torch_cpu (CUDA RNG where present)')
    return {'model_tensors': len(model), 'adamw_parameters': len(adam), 'adamw_tensors': 3 * len(adam),
            'rng': sorted(rng), 'inventory_sha256': digest(sorted([k, v['dtype'], v['shape']] for k, v in header.items()))}


def link(kind, ordinal, previous, part):
    return digest({'schema': CHAIN, 'kind': kind, 'ordinal': ordinal, 'previous': previous, 'file': part['file'],
                   'sha256': part['sha256'], 'bytes': part['bytes'], 'count': part['count']})


def write_parts(directory, kind, records, *, part_max=PART_MAX):
    """Seal records as ordered canonical-JSONL parts <kind>-NNNNNN.jsonl of at most part_max bytes (a record never
    spans parts; at least one part, possibly empty) and return the pinned, hash-chained part list."""
    need(type(kind) is str and kind.isalpha() and type(part_max) is int and 64 <= part_max <= PART_MAX, 'bounded part kind/size')
    parts = []; lines = []; size = 0; previous = '0' * 64; total = 0
    def flush():
        nonlocal lines, size, previous
        raw = b''.join(lines); name = f'{kind}-{len(parts) + 1:06d}.jsonl'
        part = {'file': name, 'ordinal': len(parts) + 1, **seal(Path(directory) / name, raw), 'count': len(lines)}
        part['link'] = previous = link(kind, part['ordinal'], previous, part); parts.append(part); lines = []; size = 0
    for record in records:
        line = canonical(record) + b'\n'; need(len(line) <= part_max, 'one record must fit one part')
        if lines and size + len(line) > part_max: flush()
        lines.append(line); size += len(line); total += 1
    if lines or not parts: flush()
    return {'schema': CHAIN, 'kind': kind, 'part_max': part_max, 'count': total, 'parts': parts, 'head': previous}


def read_parts(directory, chain):
    """Re-verify an ordered part list: names/ordinals, presence, size cap, sha256/bytes/count, canonical lines,
    every link and the chain head; returns the records in order."""
    need(type(chain) is dict and chain.get('schema') == CHAIN and type(chain.get('kind')) is str and chain['kind'].isalpha() and
         type(chain.get('parts')) is list and chain['parts'] and all(type(p) is dict for p in chain['parts']) and
         type(chain.get('part_max')) is int and 64 <= chain['part_max'] <= PART_MAX, 'closed part chain')
    previous = '0' * 64; records = []
    for ordinal, part in enumerate(chain['parts'], 1):
        name = f"{chain['kind']}-{ordinal:06d}.jsonl"
        need(part.get('file') == name and part.get('ordinal') == ordinal, 'ordered part list: ' + str(part.get('file')))
        path = Path(directory) / name; need(path.is_file() and not path.is_symlink(), 'capture part missing: ' + name)
        raw = read(path, chain['part_max']); lines = raw.split(b'\n')
        need(sha(raw) == part['sha256'] and len(raw) == part['bytes'] and lines[-1] == b'' and len(lines) - 1 == part['count'] and
             link(chain['kind'], ordinal, previous, part) == part['link'], 'capture part tampered, reordered or missing: ' + name)
        for line in lines[:-1]:
            value = json.loads(line); need(canonical(value) == line, 'canonical part record'); records.append(value)
        previous = part['link']
    need(previous == chain['head'] and len(records) == chain['count'], 'capture part chain head and count')
    return records


def load_capture(capture_dir):
    """The only supported reader: receipt binding, document cap, every chained part, record order, own fingerprint
    digest and the reassembled ancestry digest; for runtime_limit_stopped_idle also the R17 binding and the walk
    records (order, digest, deadline). Returns (body, joins, ancestry) with ancestry ready for inherit()."""
    d = Path(capture_dir); raw = read(d / 'final-capture.json', DOC_MAX); body = json.loads(raw)
    receipt = json.loads(read(d / 'capture-receipt.json', DOC_MAX))
    need(body['schema'] == SCHEMA and receipt['schema'] == RECEIPT and receipt['capture_sha256'] == sha(raw), 'capture receipt binds the capture document')
    pl = body['planner']; origin = pl['origin_absolute_step']; steps = pl['local_optimizer_steps']
    joins = read_parts(d, pl['joins'])
    need(joins and all(type(j) is dict and set(j) == JOIN_KEYS for j in joins) and
         [j['absolute_step'] for j in joins] == list(range(origin + 1, origin + steps + 1)) and
         [j['local_step'] for j in joins] == list(range(1, steps + 1)) and joins[0]['checkpoint_before'] == pl['config']['initial_checkpoint_sha256'] and
         all(a['parameter_sha256'] == b['checkpoint_before'] for a, b in zip(joins, joins[1:])) and
         joins[-1]['parameter_sha256'] == body['checkpoint']['parameter_sha256'] and
         digest(sorted(j['fingerprint'] for j in joins)) == pl['own_fingerprints']['sha256'],
         'joins: contiguous steps, unbroken checkpoint chain and own fingerprints')
    a = body['ancestry']; lists = {k: read_parts(d, a[k]) for k in ('fingerprints', 'exposed_context_ids')}
    need(all(all(type(x) is str for x in v) and all(x < y for x, y in zip(v, v[1:])) for v in lists.values()),
         'ancestry lists strictly sorted and unique')
    ancestry = {k: lists[k] if k in lists else a[k] for k in ANCESTRY_KEYS}
    need(digest(ancestry) == a['ancestry_sha256'] and len(lists['fingerprints']) == a['head']['committed_batches'] == a['optimizer_steps'],
         'reassembled ancestry equals the pinned digest')
    if body['terminal']['class'] == RUNTIME_LIMIT:
        rl = body['terminal']['runtime_limit']; walked = read_parts(d, rl['walk']['parts']); h = hashlib.sha256()
        for r in walked: h.update(canonical(r) + b'\n')
        need(body['owner_ruling'] == R17 and walked and all(type(r) is dict and set(r) == WALK_KEYS for r in walked) and
             all(x['path'] < y['path'] for x, y in zip(walked, walked[1:])) and h.hexdigest() == rl['walk']['records_sha256'] and
             len(walked) == rl['walk']['entries'] and all(r['mtime_ns'] <= rl['deadline']['in_loop_deadline_ns'] for r in walked),
             'walk records: sorted, pinned by their digest and never after the in-loop deadline')
    return body, joins, ancestry


def capture(campaign_root, out_dir, *, campaign_sha256, unit, owner_readback, outbox_reader, part_max=PART_MAX,
            owner_ruling=None, live_walk=None):
    """Capture a stopped terminal generation from a private copy into a new directory; returns (body, receipt).
    ``part_max`` (64..PART_MAX bytes) only moves part boundaries; the records and the ancestry are unchanged.
    ``owner_ruling`` (the sealed R17 receipt bytes) and ``live_walk`` (a callable returning walk(LIVE_ROOT)[1], the
    lstat-only summary of the live owned campaign root) are required by, and only accepted for, the
    runtime_limit_stopped_idle class; live_walk is read before the first and after the last input read."""
    root = Path(campaign_root); out = Path(out_dir)
    need(root.is_absolute() and out.is_absolute() and '..' not in root.parts + out.parts and
         all(not p.is_symlink() for p in (root, *root.parents, *out.parents)), 'absolute non-symlink private copy and output')
    need(not out.exists() and out.parent.is_dir() and root not in (out, *out.parents) and out not in root.parents,
         'new output directory outside the private copy')
    before = owner(owner_readback, unit); live = live_walk() if live_walk is not None else None
    stage = Path(tempfile.mkdtemp(prefix='.final-capture-', dir=out.parent))
    try:
        body, ancestry, joins = _capture(root, stage, campaign_sha256, unit, before, outbox_reader, part_max, owner_ruling, live)
        need(owner(owner_readback, unit) == before, 'owner readback changed during capture; refused')
        need(live_walk is None or live_walk() == live, 'live campaign-root walk changed during capture; refused')
        raw = canonical(body); need(len(raw) <= DOC_MAX, 'capture document cap')
        files = {'final-capture.json': seal(stage / 'final-capture.json', raw)}
        for p in sorted(stage.rglob('*')):
            if p.is_file() and p.name != 'final-capture.json': files[str(p.relative_to(stage))] = file_sha(p)
        receipt = {'schema': RECEIPT, 'capture_sha256': sha(raw), 'capture_bytes': len(raw), 'captured_at_unix': time.time(),
                   'campaign_root': str(root), 'output': str(out), 'files': files,
                   'scope': 'Timestamp and private paths live here only; final-capture.json is deterministic.'}
        receipt_raw = canonical(receipt); need(len(receipt_raw) <= DOC_MAX, 'capture receipt cap')
        seal(stage / 'capture-receipt.json', receipt_raw)
        need(load_capture(stage)[1:] == (joins, ancestry), 'sealed capture reads back exactly')
        for d in sorted({p.parent for p in stage.rglob('*')} | {stage}, key=lambda p: -len(p.parts)): fsync_dir(d)
        os.rename(stage, out); fsync_dir(out.parent)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True); raise
    return body, receipt


def _capture(root, stage, campaign, unit, readback, outbox_reader, part_max, owner_ruling, live):
    training = root / 'training'
    config_raw = load(root / 'config.json'); need(sha(config_raw) == campaign, 'campaign config sha mismatch')
    c = json.loads(config_raw); n = c['neural']
    need(Path(c['owned_root']) != root and Path(c['owned_root']) not in root.parents,
         'capture runs on a private copy, never the owned live root')
    versions = {'python': sys.version.split()[0], 'numpy': str(numpy.__version__), 'torch': str(torch.__version__),
                'safetensors': str(safetensors.__version__)}
    need({k: c['runtime_versions'].get(k) for k in RUNTIME_KEYS} == versions, 'exact pinned numeric runtime required')
    sources = {rel: sha(Path(m.__file__).read_bytes()) for rel, m in VERIFIER_SOURCES.items()}
    need(all(c['owned_sources'].get(rel) == h for rel, h in sources.items()),
         'verifier sources must equal the campaign-owned sources that wrote the journal')
    markers = [m for m in FAILURE_MARKERS if (training / m).exists() or (training / m).is_symlink()]
    need(not markers, 'failed or uncertain terminal state refused: ' + ','.join(markers))
    progress_raw = load(training / 'progress.json'); progress = json.loads(progress_raw); state = progress['state']
    need(state in TERMINAL or state in IDLE_WAITING, 'non-terminal or failed progress state refused: ' + str(state))
    cls = CLEAN if state in TERMINAL else RUNTIME_LIMIT
    if cls == CLEAN:
        need(owner_ruling is None and live is None, 'owner ruling R17 and the live-root walk apply only to the runtime_limit_stopped_idle class')
        limit = None
    else: limit = runtime_limit(root, c, campaign, readback, owner_ruling, live, stage, part_max)
    backup_ref = backup(training / 'batches.sqlite', stage / 'planner-backup.sqlite', stage)
    db = sqlite3.connect((stage / 'planner-backup.sqlite').as_uri() + '?mode=ro&immutable=1', uri=True, isolation_level=None)
    db.row_factory = sqlite3.Row
    try:
        planner = BatchStream.__new__(BatchStream); planner.db = db; planner.config = planner._get('config')
        head = planner.verify(); db.execute('BEGIN')
        pending = db.execute("SELECT count(*) FROM batches WHERE state!='committed'").fetchone()[0]
        need(head['unresolved_batches'] == 0 and pending == 0, 'reserved/needs_reconcile planner batch: uncertain optimizer outcome is a HOLD')
        pc = planner.config
        need(pc == {'schema': 'jope.native-batch-stream/v1', 'seed': n['seed'], 'batch_size': n['batch_size'],
                    'initial_checkpoint_sha256': c['parameter_sha256']}, 'planner configuration joins the campaign')
        anc = planner._get('ancestry')
        need(type(anc) is dict and anc['checkpoint_sha256'] == pc['initial_checkpoint_sha256'] and anc['head']['unresolved_batches'] == 0 and
             len(anc['fingerprints']) == anc['head']['committed_batches'] == anc['optimizer_steps'] > 0,
             'inherited ancestry: one committed batch per absolute optimizer step')
        anc_raw = load(training / 'ancestry.json', 256 * MIB); need(anc_raw == canonical(anc), 'training ancestry.json equals the planner ancestry')
        origin = anc['optimizer_steps']; steps = planner._get('optimizer_step'); absolute = origin + steps
        need(steps == head['committed_batches'] > 0 and steps <= n['max_updates'] and
             (state == 'maximum_updates_reached') == (steps == n['max_updates']), 'committed update count joins the terminal state')
        chain = []
        for row in db.execute("SELECT kind,body FROM events WHERE kind IN('stream.created','manifest.adopted')ORDER BY seq"):
            b = decode(row['body']); chain.append(b['manifest_sha256'] if row['kind'] == 'stream.created' else b['new'])
        active = planner._get('manifest')
        need(chain and chain[-1] == active and len(set(chain)) == len(chain) == db.execute('SELECT count(*)FROM manifests').fetchone()[0],
             'manifest adoption chain ends at the active manifest')
        manifest_raw = bytes(db.execute('SELECT body FROM manifests WHERE sha=?', (active,)).fetchone()[0])
        _, rows = validate_manifest(manifest_raw, active)
        initial_raw = load(training / 'initial-batch-manifest.json', 64 * MIB)
        need(sha(initial_raw) == chain[0], 'initial batch manifest joins the planner genesis')
        planner.manifest_sha256 = active; planner.rows = rows; summary = planner.summary()
        epochs = [{'id': r['id'], 'manifest': r['manifest'], 'state': r['state']} for r in db.execute('SELECT * FROM epochs ORDER BY id')]
        local_completed = sum(e['state'] == 'complete' for e in epochs); current = planner._get('epoch')
        need([e['id'] for e in epochs] == list(range(1, len(epochs) + 1)) and epochs[-1]['id'] == current, 'contiguous epochs')
        remaining = [r[0] for r in db.execute('SELECT run FROM members WHERE epoch=? AND batch IS NULL ORDER BY run', (current,))]
        consumed = db.execute('SELECT count(*)FROM members WHERE epoch=? AND batch IS NOT NULL', (current,)).fetchone()[0]
        commits = sorted(((decode(r['commit_body']), decode(r['body']), r['id'], r['fingerprint'])
                          for r in db.execute("SELECT * FROM batches WHERE state='committed'")), key=lambda x: x[0]['optimizer_step'])
        need([x[0]['optimizer_step'] for x in commits] == list(range(1, steps + 1)), 'contiguous committed steps')
        previous = pc['initial_checkpoint_sha256']; last_seq = 0; index = []; acks = []; own = []
        exposed = set(anc['exposed_context_ids']); presentations = anc['presentations']; proof = None
        for commit, intent, bid, fp in commits:
            step = commit['optimizer_step']; path = training / f'batch-{origin + step:09d}.json'
            need(path.is_file() and not path.is_symlink(), 'committed proof missing: ' + path.name)
            raw = read(path, 16 * MIB); proof = json.loads(raw); ev = commit['evidence']; cp = proof['checkpoint']
            need(ev['origin'] == 'native_optimizer' and sha(raw) == ev['event_sha256'] and
                 proof['schema'] == 'jope.continuous-neural-batch-proof/v1' and proof['config_sha256'] == campaign and
                 proof['absolute_optimizer_step'] == origin + step and proof['local_optimizer_step'] == step and
                 commit['batch_id'] == bid == intent['batch_id'] and commit['ordered_data_run_ids'] == intent['ordered_data_run_ids'] and
                 set(proof['batch']) == set(intent) | {'rows'} and all(proof['batch'][k] == intent[k] for k in intent) and
                 commit['checkpoint_before'] == previous == intent['checkpoint_before'] and cp['step'] == origin + step and
                 cp['config_sha256'] == campaign and cp['parameter_sha256'] == commit['checkpoint_after'],
                 'committed proof does not join planner commit evidence, step or checkpoint chain: ' + path.name)
            ack = proof['emission_ack']
            need(type(ack) is dict and ack.get('batch_id') == bid and ack.get('global_step') == origin + step and
                 ack.get('checkpoint_sha256') == previous and type(ack.get('head')) is dict and type(ack['head'].get('seq')) is int and
                 ack['head']['seq'] > last_seq, 'pre-optimizer outbox ACK must precede every committed update: ' + path.name)
            last_seq = ack['head']['seq']; previous = commit['checkpoint_after']; acks.append(ack); own.append(fp)
            exposed.update(intent['ordered_data_run_ids']); presentations += len(intent['ordered_data_run_ids'])
            index.append({'absolute_step': origin + step, 'local_step': step, 'batch_id': bid, 'epoch': intent['epoch'],
                          'index': intent['index'], 'fingerprint': fp, 'proof_sha256': ev['event_sha256'], 'checkpoint_before': commit['checkpoint_before'],
                          'parameter_sha256': commit['checkpoint_after'], 'ack_seq': last_seq})
        planner_param = planner._get('checkpoint'); need(previous == planner_param, 'planner cursor joins the last commit')
        need(len(set(own)) == steps and not set(own) & set(anc['fingerprints']) and
             presentations - anc['presentations'] == db.execute('SELECT count(*)FROM members WHERE batch IS NOT NULL').fetchone()[0],
             'v1 membership: one fingerprint per committed batch, disjoint from ancestry; exposures match members')
    finally:
        if db.in_transaction: db.execute('ROLLBACK')
        db.close()
    need(not (training / f'batch-{absolute + 1:09d}.json').exists(), 'orphan proof beyond the last commit: uncertain optimizer outcome is a HOLD')
    heads = outbox_reader()
    need(type(heads) is dict and set(heads) == {'v1', 'parent'} and type(heads['v1']) is dict and set(heads['v1']) == V1_OUTBOX_KEYS and
         type(heads['parent']) is dict and set(heads['parent']) == PARENT_OUTBOX_KEYS, 'closed injected outbox summary')
    v1q, pq = heads['v1'], heads['parent']
    need(v1q['schema'] == pq['schema'] == 'jope.flywheel-neural-outbox-status/v1' and v1q['numeric_scores_synthesized'] == pq['numeric_scores_synthesized'] == 0 and
         v1q['ack_count'] == v1q['batches'] == steps and v1q['acks_sha256'] == digest(acks) and last_seq <= v1q['events'] and
         v1q['observations'] == sum(len(a['emission_ids']) for a in acks), 'outbox ACK join: every committed update has exactly its durable ACK')
    rolling = sorted(p.name for p in (training / 'rolling').iterdir()); slot = training / 'rolling' / f'{absolute:09d}'
    need(rolling and rolling[-1] == slot.name, 'final rolling slot is the last committed update; no later physical state')
    m_raw = load(slot / 'manifest.json'); m = json.loads(m_raw); s_raw = load(slot / 'state.safetensors', 64 * MIB)
    if cls == RUNTIME_LIMIT:
        need(index[-1]['absolute_step'] == absolute == progress['optimizer_steps'] == m['step'] == int(rolling[-1]) and
             index[-1]['parameter_sha256'] == planner_param == m['parameter_sha256'] == progress['checkpoint_sha256'],
             'R17 closure: last committed step, progress optimizer_steps and final rolling slot, and the commit checkpoint_after, '
             'rolling parameter sha and progress parameter sha, must agree')
    need(m == proof['checkpoint'] and m['parameter_sha256'] == planner_param and sha(s_raw) == m['state_sha256'] and
         len(s_raw) == m['state_bytes'] and m['step'] == absolute and m['optimizer'] == 'AdamW',
         'final checkpoint closure: rolling manifest, state bytes and final proof disagree')
    param = m['parameter_sha256']; keys = tensor_classes(s_raw)
    control_raw = load(training / 'optimizer-control.json'); control = json.loads(control_raw)
    need(planner_param == param == progress['checkpoint_sha256'] == control['checkpoint_parameter_sha256'],
         'parameter sha disagrees across planner, progress and optimizer-control')
    rates = control['group_learning_rates']; pl = control['plateau']
    need(control['schema'] == 'jope.continuous-optimizer-control/v1' and control['absolute_optimizer_step'] == absolute and
         control['checkpoint_state_sha256'] == m['state_sha256'] and type(rates) is list and rates and
         all(type(x) in (int, float) and math.isfinite(x) and n['learning_rate_floor'] <= x <= n['learning_rate'] for x in rates) and
         pl['patience'] == n['plateau_patience'] and pl['floor'] == n['learning_rate_floor'] and type(pl['bad']) is int and
         0 <= pl['bad'] < pl['patience'] and type(pl['reductions']) is int and pl['reductions'] >= 0 and
         type(pl['best']) in (int, float) and math.isfinite(pl['best']),
         'optimizer-control: LR groups and plateau controller must be preserved, never reset')
    lineage_epochs = anc['completed_epochs'] + local_completed; monitor = progress['plateau']
    need(progress['schema'] == 'jope.flywheel-neural-progress/v1' and progress['config_sha256'] == campaign and
         progress['run_id'] == n['run_id'] and progress['optimizer_steps'] == progress['completed_batches'] == absolute and
         progress['new_optimizer_steps'] == progress['new_completed_batches'] == steps and progress['completed_epochs'] == lineage_epochs and
         progress['unique_train_data_runs_seen'] == len(exposed) and progress['total_train_presentations'] == presentations and
         progress['fixed_panel_sha256'] == control['fixed_panel_sha256'] and progress['planner'] == summary and
         progress['resumed_from']['absolute_optimizer_step'] == origin and progress['resumed_from']['planner_head'] == anc['head'] and
         progress['resumed_from']['checkpoint_parameter_sha256'] == pc['initial_checkpoint_sha256'] and
         monitor['best_since_continuation_loss'] == pl['best'] and monitor['nonimproving_validations'] == pl['bad'] and
         monitor['reductions'] == pl['reductions'] and monitor['learning_rate'] == rates[0] and
         progress['outbox']['batches'] == steps and progress['outbox']['observations'] == v1q['observations'] and
         progress['parent_outbox']['batches'] == pq['batches'] and progress['parent_outbox']['observations'] == pq['observations'] and
         (local_completed == 0 or progress['latest_full_validation']['actual_epoch'] == lineage_epochs),
         'published progress does not reconcile the planner, proofs and control')
    tail = [{k: e[k] for k in ('absolute_step', 'proof_sha256', 'parameter_sha256')} for e in index[-2:]]
    result_path = training / 'result.json'; status_path = root / 'status.json'; result_ref = None
    if cls == RUNTIME_LIMIT:
        need(not any(p.exists() or p.is_symlink() for p in (status_path, result_path)),
             'runtime_limit_stopped_idle: status.json and training/result.json must be absent (recorded absent, never reconstructed)')
    if result_path.exists() or result_path.is_symlink():
        result_raw = load(result_path, 64 * MIB); r = json.loads(result_raw); result_ref = ref(result_raw)
        need(r['schema'] == 'jope.continuous-neural-result/v1' and r['state'] == state and r['error'] is None and
             r['config_sha256'] == campaign and r['new_optimizer_steps'] == steps and r['absolute_optimizer_steps'] == absolute and
             r['planner'] == summary and r['planner_verification'] == head and r['current_parameter_sha256'] == param and
             r['latest_checkpoint'] == m and Path(r['latest_checkpoint_path']).name == slot.name and r['proof_tail'] == tail and
             r['unique_train_contexts_seen'] == len(exposed) and r['total_train_presentations'] == presentations and
             r['fixed_panel_sha256'] == control['fixed_panel_sha256'] and r['plateau'] == monitor and r['optimizer_control'] == control and
             r['outbox']['batches'] == steps and r['outbox']['observations'] == v1q['observations'],
             'result.json disagrees with the proven final closure')
    pre_raw = load(training / 'preprocessing.json', 64 * MIB); need(sha(pre_raw) == c['preprocessing']['sha256'], 'fixed preprocessing bytes')
    (stage / 'checkpoint').mkdir()
    sealed_m = seal(stage / 'checkpoint' / 'manifest.json', m_raw); sealed_s = seal(stage / 'checkpoint' / 'state.safetensors', s_raw)
    rng = torch.get_rng_state()
    try:
        net, opt, _ = restore({'path': str(stage / 'checkpoint' / 'manifest.json'), **sealed_m},
                              {'path': str(stage / 'checkpoint' / 'state.safetensors'), **sealed_s},
                              expected_parameter_sha256=param, device='cpu')
        recomputed = M.parameter_sha256(net); groups = len(opt.param_groups); moments = sum(len(v) for v in opt.state.values()); del net, opt
    finally: torch.set_rng_state(rng)
    need(recomputed == param and groups == len(rates), 'throwaway CPU restore reproduces the parameter sha and optimizer groups')
    prior = anc['head']['generations'] if anc['head'].get('schema') == LINEAGE_HEAD else [{'planner_ref': anc['planner_ref'], 'head': anc['head']}]
    planner_ref = str(Path(c['owned_root']) / 'training' / 'batches.sqlite')
    generations = prior + [{'planner_ref': planner_ref, 'head': head}]; fps = sorted(set(anc['fingerprints']) | set(own))
    need(sum(g['head']['committed_batches'] for g in generations) == len(fps) == absolute and
         all(g['head']['unresolved_batches'] == 0 for g in generations), 'lineage head: one resolved batch per absolute optimizer step')
    ancestry = {'planner_ref': planner_ref,
                'head': {'schema': LINEAGE_HEAD, 'committed_batches': len(fps), 'unresolved_batches': 0, 'generations': generations},
                'checkpoint_sha256': param, 'optimizer_steps': absolute, 'completed_epochs': lineage_epochs, 'fingerprints': fps,
                'exposed_context_ids': sorted(exposed), 'presentations': presentations}
    dry = Path(tempfile.mkdtemp(prefix='inherit-dry-run-', dir=stage))
    try:
        p = BatchStream(dry / 'fresh.sqlite', manifest_raw, active, seed=pc['seed'], batch_size=pc['batch_size'], initial_checkpoint_sha256=param)
        try: p.inherit(ancestry); p.inherit(ancestry); fresh = p.verify(); kept = p._get('ancestry')
        finally: p.close()
    finally: shutil.rmtree(dry)
    need(kept == ancestry and fresh['committed_batches'] == fresh['unresolved_batches'] == 0, 'fresh journal accepts the exact ancestry')
    files = {'active-manifest.json': seal(stage / 'active-manifest.json', manifest_raw),
             'optimizer-control.json': seal(stage / 'optimizer-control.json', control_raw)}
    chains = {'joins': write_parts(stage, 'joins', index, part_max=part_max),
              'fingerprints': write_parts(stage, 'fingerprints', ancestry['fingerprints'], part_max=part_max),
              'exposed_context_ids': write_parts(stage, 'exposed', ancestry['exposed_context_ids'], part_max=part_max)}
    hist = {}
    for name in ('loss-history.jsonl', 'optimizer-history.jsonl'):
        raw = load(training / name, 512 * MIB); hist[name] = {**ref(raw), 'lines': raw.count(b'\n')}
    owned = Path(c['owned_root'])
    terminal = {'class': cls, 'state': state, 'closure': 'result_json_joined' if result_ref else 'proven_without_result_json',
                'result_json': result_ref, 'failure_markers': [], 'interrupted_publications': sorted(p.name for p in training.glob('*.pending'))}
    if cls == RUNTIME_LIMIT:
        g = readback['cgroup']
        terminal.update(closure='r17_runtime_limit_stopped_idle', result_json='absent', status_json='absent', runtime_limit={**limit, 'r17_proofs': {
            'unit_stopped': {**{k: readback[k] for k in ('ActiveState', 'Result', 'MainPID', 'NRestarts')}, 'cgroup_empty': g['procs'] == [] and g['populated'] is not True},
            'no_write_after_in_loop_deadline': {'in_loop_deadline_ns': limit['deadline']['in_loop_deadline_ns'], 'max_mtime_ns': limit['walk']['max_mtime_ns'],
                                                'entries': limit['walk']['entries'], 'live_root_walk_equal': True},
            'planner': {'unresolved_batches': head['unresolved_batches'], 'pending_batches': pending, 'last_committed_absolute_step': index[-1]['absolute_step'],
                        'progress_optimizer_steps': progress['optimizer_steps'], 'rolling_step': m['step'], 'checkpoint_after': index[-1]['parameter_sha256'],
                        'rolling_parameter_sha256': m['parameter_sha256'], 'progress_parameter_sha256': progress['checkpoint_sha256']},
            'cpu_restore': {'parameter_sha256': recomputed}}})
    body = {
        'schema': SCHEMA,
        'campaign': {'config_sha256': campaign, 'config_bytes': len(config_raw), 'owned_root': c['owned_root'], 'run_id': n['run_id'], 'unit': unit},
        'owner': {'readback': readback, 'stable_across_capture': True},
        'owner_ruling': {'id': R17['id'], 'receipt_sha256': sha(owner_ruling)} if cls == RUNTIME_LIMIT else None,
        'runtime': {'versions': versions, 'verifier_sources': sources},
        'terminal': terminal,
        'planner': {'backup': {'file': 'planner-backup.sqlite', **backup_ref,
                               'method': 'staged byte copy (db + -wal), sqlite3 online backup in one step (one read transaction), journal_mode=DELETE, sealed 0444'},
                    'config': pc, 'head': head, 'summary': summary, 'origin_absolute_step': origin, 'local_optimizer_steps': steps,
                    'absolute_optimizer_steps': absolute, 'manifest_chain': chain,
                    'active_manifest': {'file': 'active-manifest.json', **files['active-manifest.json']}, 'initial_batch_manifest': ref(initial_raw),
                    'epochs': epochs, 'completed_epochs_local': local_completed, 'completed_epochs_lineage': lineage_epochs,
                    'current_epoch': {'id': current, 'state': epochs[-1]['state'], 'consumed': consumed, 'remaining': len(remaining),
                                      'remaining_ids_sha256': digest(remaining)},
                    'inherited_ancestry': {**ref(anc_raw), 'planner_ref': anc['planner_ref'], 'head': anc['head'], 'optimizer_steps': origin,
                                           'completed_epochs': anc['completed_epochs'], 'fingerprints': len(anc['fingerprints']),
                                           'exposed_context_ids': len(anc['exposed_context_ids']), 'presentations': anc['presentations']},
                    'own_fingerprints': {'count': len(own), 'sha256': digest(sorted(own))},
                    'joins': chains['joins'], 'proof_tail': tail},
        'checkpoint': {'slot': f'training/rolling/{slot.name}', 'live_slot': str(owned / 'training' / 'rolling' / slot.name),
                       'parameter_sha256': param, 'manifest': {'file': 'checkpoint/manifest.json', **sealed_m, 'body': m},
                       'state': {'file': 'checkpoint/state.safetensors', **sealed_s}, 'tensor_keys': keys,
                       'cpu_restore': {'parameter_sha256': recomputed, 'optimizer_groups': groups, 'adamw_state_tensors': moments,
                                       'caller_rng_restored': True}},
        'control': {'file': 'optimizer-control.json', **files['optimizer-control.json'], 'body': control},
        'progress': {**ref(progress_raw), 'state': state, 'completed_epochs': progress['completed_epochs'],
                     'execution_config_sha256': progress.get('execution_config_sha256'), 'fixed_panel_sha256': progress['fixed_panel_sha256']},
        'outbox': {'v1': v1q, 'parent': pq, 'acks': {'count': len(acks), 'sha256': digest(acks), 'last_seq': last_seq}},
        'inputs': {'preprocessing': ref(pre_raw), **hist},
        'ancestry': {'ancestry_sha256': digest(ancestry), 'ancestry_canonical_bytes': len(canonical(ancestry)),
                     'planner_ref': planner_ref, 'head': ancestry['head'], 'checkpoint_sha256': param, 'optimizer_steps': absolute,
                     'completed_epochs': lineage_epochs, 'presentations': presentations,
                     'fingerprints': chains['fingerprints'], 'exposed_context_ids': chains['exposed_context_ids'],
                     'inherit_dry_run': {'accepted': True, 'idempotent': True, 'fresh_head': fresh}},
        'mutations': {'replays': 0, 'optimizer_updates': 0, 'journal_writes': 0, 'private_copy_writes': 0},
        'scope': 'Read-only capture of a stopped generation from a private copy. Not a transfer, admission or optimizer step.'}
    return body, ancestry, index
