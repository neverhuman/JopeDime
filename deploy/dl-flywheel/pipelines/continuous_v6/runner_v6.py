"""runner_v6: the v2.0.0 continual-policy learner (derived from pipelines/prepared/continuous_runner.py at 745be43).

One persistent model continues from v1's sealed final capture (owner rulings R9, R14, R17).  The supervised objective
uses measured records only (L6, unchanged); generated proposals are durable work, never predictions substituted for
reward; there is no automatic recovery.  Deltas from v1 (plan v2.0 section 5b; rulings R8, R11, R13, R19):

 1. Lineage origin is ``ancestry.optimizer_steps`` (v1's final absolute step), not the literal 429.
 2. The planner opens on its ACTIVE manifest (read-only lookup); there is no pinned initial-manifest file.
 3. No one-shot admission latch.  At startup (when the current epoch is closed) and after every closed epoch the
    boundary driver reconciles the C10b inbox and adopts at most one candidate (variant and/or native-overlay
    components).  A HOLD keeps training on the active snapshot and is published; a refused inbox stops the run.
 4. The emission cap counts v2's own unresolved rows only.  v1/parent backlogs are reported from the sealed capture;
    no foreign outbox is opened inside the loop.  ``emissions_per_step`` (1 or 2) keeps the outbox's 100,000-row
    lifetime cap above the update window: 1 emits the training proposal only (the frozen outbox requires a training
    slot in every batch; validation proposals are evaluation-only), 2 emits training + validation as v1 did.
 5. Memory gate between steps: below ``memory_floor_fraction`` MemAvailable the loop waits without a reservation;
    ``memory_exit_checks`` consecutive readings at or below ``memory_exit_fraction`` end the run cleanly.
 6. Batch proofs carry ``snapshot_sha256`` and ``adoption_seq`` (first-exposure joins).
 7. Tick-safe stop: a reservation is made only when the remaining in-loop time covers the bounded step budget, so a
    deadline never lands inside a step; closed-epoch boundary work (full validation, adoption, watermark) starts only
    when ``epoch_boundary_budget_seconds`` remain, otherwise the run ends cleanly with the boundary deferred.
 8. Terminal custody (r2, Codex F2): the closure publishes ``terminal-intent.json`` FIRST (cheap; an intent is not a
    clean stop), then joins the committed suffix from the last verified watermark (custody_v6, Codex F3) and runs the
    full indexed planner and outbox verifiers within ``closure_budget_seconds``, then publishes
    ``result-verification.json`` (binding intent, watermark, heads, control, checkpoint and execution) and only then
    ``result.json`` (binding both).  Any closure error publishes ``closure-failure.json`` and no clean result; resume
    refuses missing, stale or failed closures.  Every run segment is named by ``segment.json`` before any reservation.
 9. A verified watermark is persisted after every closed epoch (``verified-watermark.json``), so the closure joins a
    bounded suffix; the prefix is re-joined by its event-head sha (r2, Codex F3).
 10. The supplied ancestry must equal the planner's stored ancestry (r2, Codex F4).
 11. Learning guard (plan G32): at every closed epoch the latest fixed-panel validation loss is compared with the
    sealed v1 reference; ``consecutive_epochs`` closed epochs above ``ratio`` x reference trip the guard, which is
    sticky (``learning-guard.json``): adoptions are HELD (L6 training continues) and the alert needs owner review.
    The guard state is sealed in every verified closure and recovered only from it.  Every closed epoch's boundary
    work runs exactly once (``close_epoch``, recorded as ``last_checked_epoch``); work deferred at a deadline runs at
    the next start BEFORE any adoption or reservation, or is deferred again if time is still short.
"""
from dataclasses import asdict
from pathlib import Path
import json, os, time
import numpy as np
import torch
from dltrain import model as M
from jope_flywheel.protocol import canonical, digest
from jope_flywheel.program_io import need, create, sha, remaining
from jope_flywheel import timing
from neural_native_train import prepare_microbatches, run_microbatches, evaluate_full_validation, publish, OBJECTIVE, gpu_preflight
from neural_native_stream import BatchStream, Waiting
from neural_outbox import Outbox
from neural_decode import BatchedTrie
from continuous_state import restore, normalizer, Recovery, append
from pipelines.boundary_admission_v2 import boundary as Bd
from . import custody_v6 as CU

GENERATION = 'v2.0.0'
PLANNER_INDEXES = {'v2_members_batch': ('members', 'CREATE INDEX IF NOT EXISTS v2_members_batch ON members(batch)')}
OUTBOX_INDEXES = {'v2_emissions_batch': ('emissions', 'CREATE INDEX IF NOT EXISTS v2_emissions_batch ON emissions(batch_id, ordinal)'),
                  'v2_emissions_state': ('emissions', 'CREATE INDEX IF NOT EXISTS v2_emissions_state ON emissions(state)')}
OUTBOX_MAX_ROWS, OUTBOX_MAX_PAYLOAD = 100000, 256 * 1024**2
MARKERS = ('failure.json', 'closure-failure.json', 'uncertain-physical-state', 'closure-errors.jsonl')
TERMINAL_FILES = ('segment.json', 'terminal-intent.json', 'result-verification.json', 'result.json')


def has_indexes(db, indexes):
    return all(db.execute("SELECT 1 FROM sqlite_master WHERE type='index' AND name=? AND tbl_name=?", (name, table)).fetchone()
               is not None for name, (table, _) in indexes.items())


def ensure_indexes(db, indexes):
    """Genesis only: secondary indexes leave every frozen query result unchanged and bound verify()."""
    for _, (_, statement) in sorted(indexes.items()):
        db.execute(statement)


def mem_available_fraction():
    info = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines() if ':' in line)
    return int(info['MemAvailable'].split()[0]) / int(info['MemTotal'].split()[0])


class Plateau:
    """v1's fixed-panel plateau control, unchanged."""
    def __init__(self, patience, floor):
        self.patience = patience; self.floor = floor; self.best = None; self.bad = 0; self.reductions = 0

    def observe(self, value, opt):
        need(np.isfinite(value), 'finite fixed-panel monitor')
        improved = self.best is None or value < self.best
        if improved: self.best = value; self.bad = 0
        else: self.bad += 1
        reduced = False
        if self.bad >= self.patience:
            old = opt.param_groups[0]['lr']; new = max(self.floor, old * .5)
            for g in opt.param_groups: g['lr'] = new
            self.bad = 0; reduced = new < old; self.reductions += int(reduced)
        return {'improved': improved, 'best_since_continuation_loss': self.best, 'nonimproving_validations': self.bad,
                'lr_reduced': reduced, 'learning_rate': opt.param_groups[0]['lr'], 'reductions': self.reductions,
                'policy': 'continue shuffled epochs; retain best; no guarantee of improvement'}


def bytes_used(root):
    seen = set(); total = 0
    for base, dirs, files in os.walk(root, followlinks=False):
        need(all(not (Path(base) / d).is_symlink() for d in dirs), 'owned output symlink')
        for name in files:
            p = Path(base) / name; need(not p.is_symlink(), 'owned output symlink'); s = p.stat(); key = (s.st_dev, s.st_ino)
            if key not in seen: seen.add(key); total += s.st_size
    return total


class BoundaryDriver:
    """Closed-epoch C10b inbox adoption with HOLD custody (no latch; at most one adoption per closed epoch)."""

    def __init__(self, inbox, native, *, root, campaign_sha256, preprocessing_sha256, registry_sha256, fresh_authority,
                 authority_check):
        self.inbox, self.native, self.root = inbox, native, Path(root)
        self.kw = dict(campaign_sha256=campaign_sha256, preprocessing_sha256=preprocessing_sha256,
                       registry_sha256=registry_sha256, fresh_authority=fresh_authority, authority_check=authority_check)
        self.seq = len(inbox.adoptions())
        self.status = {'schema': 'jope.v2-boundary-status/v1', 'state': 'idle', 'adoption_seq': self.seq, 'holds': [],
                       'last': None}

    def step(self, planner, snapshot, checkpoint_sha256, *, at_step):
        try:
            replacement = self.inbox.adopt_next(planner, snapshot, self.native, checkpoint_sha256=checkpoint_sha256, **self.kw)
        except Bd.Held as held:
            self.status.update(state='held', last={'at_step': at_step, 'reason': str(held)[:500], 'at': time.time()})
            self.status['holds'] = (self.status['holds'] + [self.status['last']])[-20:]
            publish(self.root, 'boundary-status.json', self.status)
            return snapshot
        if replacement is None:
            self.status.update(state='idle', last={'at_step': at_step, 'reason': 'no published candidate', 'at': time.time()})
        else:
            self.seq = len(self.inbox.adoptions())
            self.status.update(state='adopted', adoption_seq=self.seq,
                               last={'at_step': at_step, 'seq': self.seq, 'snapshot_sha256': replacement.identity, 'at': time.time()})
            snapshot = replacement
        publish(self.root, 'boundary-status.json', self.status)
        return snapshot


def planner_epoch_complete(planner):
    """True when the planner's current epoch is complete (its boundary work is due or done)."""
    epoch = planner._get('epoch')
    row = planner.db.execute('SELECT state FROM epochs WHERE id=?', (epoch,)).fetchone()
    return epoch > 0 and row is not None and row[0] == 'complete'


def boot_id():
    return Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def check_basis(root, resume, genesis_sha256):
    """The resume basis names exactly the files on disk: genesis (no segment ever started) or one verified closure."""
    need(not any((root / m).exists() for m in MARKERS), 'no failure, closure failure, uncertain state or closure error')
    basis = resume['basis']
    if basis['kind'] == 'genesis':
        need(basis['genesis_sha256'] == genesis_sha256 and not any((root / f).exists() for f in TERMINAL_FILES) and
             resume['local_step'] == 0 and resume['watermark']['anchor'] == 'genesis' and resume['watermark']['local_step'] == 0,
             'genesis start: no earlier segment, intent or result')
        return
    need(basis['kind'] == 'closure' and all((root / f).is_file() for f in TERMINAL_FILES), 'restart only from a complete closure')
    seg = json.loads((root / 'segment.json').read_bytes())
    need(seg['segment_id'] == basis['segment_id'] and sha((root / 'result.json').read_bytes()) == basis['result_sha256'] and
         sha((root / 'result-verification.json').read_bytes()) == basis['verification_sha256'] and
         resume['watermark']['local_step'] == resume['local_step'], 'restart basis is the latest verified closure')


def run(native, snapshot, config, *, root, campaign_sha, ancestry, preprocessing, deadline, authority_check, boundary,
        resume, genesis_sha256, memory=mem_available_fraction, planner_factory=BatchStream):
    root = Path(root); n = config['neural']; device = torch.device(n['device'])
    origin_step = ancestry['optimizer_steps']; start_steps = resume['local_step']
    need(type(origin_step) is int and origin_step > 0 and resume['origin_absolute_step'] == origin_step, 'lineage origin step')
    need(n['emissions_per_step'] in (1, 2) and n['step_budget_floor_seconds'] > 0, 'bounded emission and step budget policy')
    need(all(type(n[k]) in (int, float) and n[k] > 0 for k in ('epoch_boundary_budget_seconds', 'watermark_budget_seconds',
                                                              'closure_budget_seconds')), 'bounded boundary, watermark and closure budgets')
    check_basis(root, resume, genesis_sha256)
    rg = n['regression_guard']
    need(set(rg) == {'reference_loss', 'ratio', 'consecutive_epochs'} and rg['reference_loss'] > 0 and rg['ratio'] >= 1 and
         type(rg['consecutive_epochs']) is int and rg['consecutive_epochs'] >= 1, 'closed learning-guard policy')
    # The guard state is sealed in every verified closure and recovered ONLY from it (CX-REV 6b1ee581): genesis has no
    # sidecar; a restart requires the sidecar to be byte-identical to the closure-sealed state.  No reset path exists
    # here; an owner-authorized reset would be a distinct, recorded transition.
    gpath = root / 'learning-guard.json'
    if resume['basis']['kind'] == 'genesis':
        need(not gpath.exists() and resume.get('learning_guard') is None, 'genesis has no learning-guard state')
        guard = {'schema': 'jope.v2-learning-guard/v1', **rg, 'over': 0, 'tripped': False, 'history': []}
    else:
        sealed = resume.get('learning_guard')
        need(type(sealed) is dict and gpath.is_file() and not gpath.is_symlink() and
             gpath.read_bytes() == canonical(sealed), 'learning-guard state is exactly the closure-sealed state')
        guard = json.loads(gpath.read_bytes())
    need(guard['schema'] == 'jope.v2-learning-guard/v1' and all(guard[k] == rg[k] for k in rg) and
         type(guard['tripped']) is bool and type(guard['over']) is int and type(guard['history']) is list,
         'learning-guard policy unchanged across segments')
    authority_check(snapshot); M.configure_determinism()
    with timing.span('model.restore_checkpoint_and_optimizer'):
        net, opt, origin = restore(resume['checkpoint_manifest'], resume['checkpoint_state'],
                                   expected_parameter_sha256=resume['control']['checkpoint_parameter_sha256'], device=device)
    need(origin['step'] == origin_step + start_steps, 'physical step is the lineage origin plus committed local steps')
    norm = normalizer(preprocessing, native)
    need((root / 'preprocessing.json').read_bytes() == preprocessing, 'unchanged normalization bytes')
    recovery = Recovery.__new__(Recovery); recovery.root = root
    recovery.live = sorted((root / 'rolling').iterdir()); recovery.best = sorted((root / 'best').iterdir())
    need(0 < len(recovery.live) <= 2 and len(recovery.best) <= 2, 'existing bounded recovery retention')
    current_path = Path(resume['checkpoint_manifest']['path']).parent; cp = origin
    planner_path = (root / 'batches.sqlite').resolve()
    manifest_raw, active = Bd.active_manifest(planner_path)
    need(sha(manifest_raw) == active == sha(canonical(snapshot.batch_manifest)) and
         {c['context_id']: c for c in json.loads(manifest_raw)['contexts']} == snapshot.contexts,
         'snapshot is the planner ACTIVE manifest')
    planner = planner_factory(planner_path, manifest_raw, active, seed=n['seed'], batch_size=n['batch_size'],
                              initial_checkpoint_sha256=ancestry['checkpoint_sha256'])
    need(has_indexes(planner.db, PLANNER_INDEXES), 'bounded-verify planner indexes present (genesis)')
    need(planner._get('ancestry') == ancestry, 'supplied ancestry is exactly the planner\'s stored ancestry')
    head = planner.verify()
    need(head['unresolved_batches'] == 0 and planner.pending() is None and
         planner.summary()['checkpoint_sha256'] == resume['control']['checkpoint_parameter_sha256'], 'reconciled planner joins the checkpoint')
    outbox = Outbox((root / 'outbox.sqlite').resolve(), model_origin_sha256=ancestry['checkpoint_sha256'],
                    max_pending=n['max_pending_emissions'], max_rows=OUTBOX_MAX_ROWS, max_payload_bytes=OUTBOX_MAX_PAYLOAD)
    need(has_indexes(outbox.db, OUTBOX_INDEXES), 'bounded-verify outbox indexes present (genesis)')
    custody = dict(origin=origin_step, campaign_sha256=campaign_sha, execution_sha256=resume['execution_config_sha256'])
    watermark, _ = CU.extend(root, planner.db, outbox.db, resume['watermark'], budget_seconds=n['watermark_budget_seconds'],
                             expect_no_suffix=True, **custody)
    need(watermark['parameter_sha256'] == resume['control']['checkpoint_parameter_sha256'], 'verified watermark joins the resumed control')
    segment = {'schema': 'jope.v2-run-segment/v1', 'basis': resume['basis'], 'execution_config_sha256': resume['execution_config_sha256'],
               'watermark_sha256': digest(watermark), 'started_unix': time.time(), 'boot_id': boot_id(), 'pid': os.getpid()}
    segment_id = digest(segment); segment['segment_id'] = segment_id
    native_train = sorted(c for c, r in native.contexts.items() if r['partition'] == 'train')
    native_dev = sorted(c for c, r in native.contexts.items() if r['partition'] == 'validation')
    train_programs = {native.winners[c].program_sha256 for c in native_train}
    catalog = [p for p in native.proposal_programs if p['program_sha256'] in train_programs]
    trie = BatchedTrie(catalog, native.vocab, native.max_seq)
    train_panel = sorted(native_train, key=lambda x: digest({'seed': n['seed'], 'panel': x}))[:n['evaluation_panel_contexts']]
    dev_panel = sorted(native_dev, key=lambda x: digest({'seed': n['seed'], 'panel': x}))[:n['evaluation_panel_contexts']]
    panel_identity = digest({'native_snapshot': native.identity, 'train': train_panel, 'dev': dev_panel, 'epoch': 0,
                             'batch_index': 0, 'objective': OBJECTIVE})
    need(panel_identity == resume['control']['fixed_panel_sha256'], 'unchanged fixed evaluation panels')
    panels = {role: prepare_microbatches(native, ids, epoch=0, batch_index=0, normalizer=norm,
                                         max_padded_tokens=n['max_padded_tokens'])[:2]
              for role, ids in [('training', train_panel), ('validation', dev_panel)]}
    if device.type == 'cuda':
        with timing.span('model.gpu_resume_preflight'):
            create(root / f'gpu-resume-preflight-{origin_step + start_steps:09d}.json',
                   canonical(gpu_preflight(native, net, norm, n, deadline)))
        torch.cuda.reset_peak_memory_stats(device)
    # The preflight must not alter the resumed optimizer's dropout RNG stream: restore again.
    net, opt, origin = restore(resume['checkpoint_manifest'], resume['checkpoint_state'],
                               expected_parameter_sha256=resume['control']['checkpoint_parameter_sha256'], device=device)
    need(len(opt.param_groups) == len(resume['control']['group_learning_rates']), 'exact optimizer groups')
    for group, rate in zip(opt.param_groups, resume['control']['group_learning_rates']): group['lr'] = rate
    plateau = Plateau(n['plateau_patience'], n['learning_rate_floor'])
    need(resume['control']['plateau']['patience'] == plateau.patience and resume['control']['plateau']['floor'] == plateau.floor,
         'unchanged plateau patience and floor')
    for key in ('best', 'bad', 'reductions'): setattr(plateau, key, resume['control']['plateau'][key])
    steps = start_steps; checkpoint_sha = cp['parameter_sha256']; current = None
    seen = set(resume['unique_contexts']); presentations = resume['presentations']; newseen = set(resume['continuation_seen_ids'])
    curves = [json.loads(line) for line in (root / 'loss-history.jsonl').read_bytes().splitlines()][-1000:] if (root / 'loss-history.jsonl').exists() else []
    hist = [json.loads(line) for line in (root / 'optimizer-history.jsonl').read_bytes().splitlines()][-2048:] if (root / 'optimizer-history.jsonl').exists() else []
    proof_tail = list(resume['proof_tail']); state = 'running'; started = time.monotonic(); last_eval = steps
    last_full = planner.summary()['completed_epochs']; full_validation = resume.get('latest_full_validation'); extended_validation = None
    perf = {'input_seconds': 0., 'decode_seconds': 0., 'optimizer_seconds': 0.}; step_seconds = []
    used = bytes_used(root.parent); payload = outbox.db.execute('SELECT coalesce(sum(length(body)),0)FROM emissions').fetchone()[0]
    monitor = None; failure = None; low_memory = 0; mem = memory()

    def step_budget():
        recent = step_seconds[-200:]
        return max(n['step_budget_floor_seconds'], 2.0 * max(recent) if recent else 0.)

    def progress():
        s = planner.summary(); q = outbox.summary()
        return {'schema': 'jope.flywheel-neural-progress/v1', 'generation': GENERATION, 'run_id': n['run_id'],
                'config_sha256': campaign_sha, 'execution_config_sha256': resume['execution_config_sha256'], 'at': time.time(),
                'state': state, 'origin_absolute_step': origin_step,
                'optimizer_steps': origin_step + steps, 'new_optimizer_steps': steps, 'completed_batches': origin_step + steps,
                'new_completed_batches': steps, 'completed_epochs': ancestry['completed_epochs'] + s['completed_epochs'],
                'epoch_or_pass': ancestry['completed_epochs'] + max(1, s['epoch']),
                'unique_train_data_runs_seen': len(seen), 'total_train_presentations': presentations,
                'new_run_unique_train_contexts_seen': len(newseen), 'model_generated_programs': q['observations'],
                'published_results': q['states'].get('published_verified', 0), 'cache_hits': q['states'].get('cached_verified', 0),
                'failed_programs': q['states'].get('failure_custody_verified', 0), 'unresolved_programs': q['unresolved'],
                'outbox': q, 'inherited_backlog': config['inherited_backlog'], 'own_unresolved_cap': n['max_pending_emissions'],
                'loss_history': curves[-1000:], 'optimizer_history': hist[-2048:],
                'full_history_paths': {'loss': 'loss-history.jsonl', 'optimizer': 'optimizer-history.jsonl'},
                'prediction_outcomes_are_rewards': False,
                'performance': {**perf, 'optimizer_steps_per_wall_second': (steps - start_steps) / max(1e-9, time.monotonic() - started),
                                'step_budget_seconds': step_budget(), 'mem_available_fraction': mem,
                                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None},
                'latest_full_validation': full_validation, 'extended_validation': extended_validation,
                'fixed_panel_sha256': panel_identity, 'plateau': monitor, 'boundary': boundary.status,
                'planner': s, 'checkpoint_sha256': checkpoint_sha, 'active_snapshot_sha256': snapshot.identity,
                'segment_id': segment_id, 'verified_watermark': watermark, 'deferred_boundary': deferred,
                'learning_guard': guard,
                'storage': {'observed_unique_inode_bytes': used, 'cap_bytes': n['output_cap_bytes'], 'outbox_payload_bytes': payload},
                'proposal_catalog': {'sha256': trie.identity, 'programs': len(trie.programs), 'scope': 'fixed original training teachers only'},
                'scope': 'v2.0.0 continuation from the sealed v1 capture; L6 objective on measured records only; adopted verified '
                         'targets enter only through closed-epoch inbox adoption; pending generated outcomes never enter gradients.'}

    def control():
        value = {'schema': 'jope.continuous-optimizer-control/v1', 'checkpoint_parameter_sha256': checkpoint_sha,
                 'checkpoint_state_sha256': cp['state_sha256'], 'absolute_optimizer_step': origin_step + steps,
                 'group_learning_rates': [g['lr'] for g in opt.param_groups],
                 'plateau': {'best': plateau.best, 'bad': plateau.bad, 'reductions': plateau.reductions,
                             'patience': plateau.patience, 'floor': plateau.floor},
                 'fixed_panel_sha256': panel_identity,
                 'resume_requirement': 'Verify and apply this control state together with the physical checkpoint; no automatic resume'}
        publish(root, 'optimizer-control.json', value); return value

    def evaluate():
        nonlocal monitor, last_eval
        with timing.span('validation.fixed_native_panel', absolute_optimizer_step=origin_step + steps):
            metrics = {role: run_microbatches(net, *pack, deadline=deadline) for role, pack in panels.items()}
        monitor = plateau.observe(metrics['validation']['loss'], opt)
        row = {'batch': origin_step + steps, 'new_batch': steps, 'epoch': ancestry['completed_epochs'] + planner.summary()['epoch'],
               'unique_data_runs_seen': len(seen), 'train_loss': metrics['training']['loss'], 'validation_loss': metrics['validation']['loss'],
               'at': time.time(), 'dropout': False, 'objective': OBJECTIVE['name'], 'fixed_panel_sha256': panel_identity,
               'metrics': metrics, 'plateau': monitor}
        append(root / 'loss-history.jsonl', row); curves.append(row); last_eval = steps
        if monitor['improved']: recovery.retain(current_path, 'best', f'{origin_step + steps:09d}')
        control()

    def epoch_closed():
        return planner_epoch_complete(planner)

    def close_epoch(epoch):
        """All closed-epoch boundary work, exactly once per lineage epoch (in-loop or deferred to a later start):
        full validation, a fixed-panel evaluation AT this step, the epoch checkpoint, the G32 check (recorded as
        ``last_checked_epoch``), then adoption only if the guard is not tripped, then the verified watermark."""
        nonlocal full_validation, extended_validation, snapshot, watermark
        need(guard.get('last_checked_epoch') != epoch, 'closed epoch checked exactly once')
        with timing.span('validation.full_original_support', epoch=epoch):
            full = evaluate_full_validation(native, net, native_dev, epoch=0, normalizer=norm, config=n, deadline=deadline)
        full['actual_epoch'] = epoch; create(root / f'validation-original-epoch-{epoch:06d}.json', canonical(full))
        full_validation = {k: v for k, v in full.items() if k != 'program_windows'}
        dev = sorted(c for c, r in snapshot.contexts.items() if r['partition'] == 'validation')
        extra = sorted(set(dev) - set(native_dev))
        if extra:
            with timing.span('validation.expanded_measured_support', epoch=epoch):
                extended = evaluate_full_validation(snapshot, net, extra, epoch=0, normalizer=norm, config=n, deadline=deadline)
            extended['actual_epoch'] = epoch; extended['comparable_to_fixed_native_monitor'] = False
            create(root / f'validation-variant-epoch-{epoch:06d}.json', canonical(extended))
            extended_validation = {k: v for k, v in extended.items() if k != 'program_windows'}
        if not curves or curves[-1]['batch'] != origin_step + steps: evaluate()
        if not (root / 'epochs' / f'{epoch:06d}').exists(): recovery.retain(current_path, 'epochs', f'{epoch:06d}')
        latest = curves[-1]['validation_loss']
        guard['over'] = guard['over'] + 1 if latest > rg['ratio'] * rg['reference_loss'] else 0
        guard['history'] = (guard['history'] + [{'epoch': epoch, 'at_step': origin_step + steps, 'fixed_panel_loss': latest}])[-50:]
        guard['last_checked_epoch'] = epoch
        if guard['over'] >= rg['consecutive_epochs'] and not guard['tripped']:
            guard.update(tripped=True, tripped_at={'epoch': epoch, 'at_step': origin_step + steps, 'at': time.time()},
                         alert='fixed-panel validation regressed beyond the sealed reference; adoptions HELD; owner review required')
        publish(root, 'learning-guard.json', guard)
        if guard['tripped']:
            boundary.status.update(state='held_by_learning_guard', last={'at_step': origin_step + steps, 'reason': guard['alert']})
            publish(root, 'boundary-status.json', boundary.status)
        else:
            with timing.span('input.epoch_boundary_inbox_adoption', epoch=epoch):
                snapshot = boundary.step(planner, snapshot, checkpoint_sha, at_step=origin_step + steps)
        with timing.span('custody.epoch_watermark', epoch=epoch):
            watermark, report = CU.extend(root, planner.db, outbox.db, watermark,
                                          budget_seconds=n['watermark_budget_seconds'], **custody)
        publish(root, 'verified-watermark.json', {'segment_id': segment_id, 'watermark': watermark, 'report': report,
                                                  'source': 'closed epoch', 'epoch': epoch})

    publish(root, 'segment.json', segment)   # names this segment before any reservation (resume refuses a stale closure)
    deferred = None; closure_error = None
    try:
        start_deferred = False
        if epoch_closed():
            closed = ancestry['completed_epochs'] + planner._get('epoch')
            if guard.get('last_checked_epoch') != closed:
                # A closed epoch whose boundary work (G32 check included) was deferred: do it exactly once, BEFORE any
                # adoption or reservation; with too little time, defer again and end without a reservation.
                if remaining(deadline) <= n['epoch_boundary_budget_seconds']:
                    deferred = {'epoch': closed, 'at_step': origin_step + steps, 'reason': 'insufficient time for deferred boundary work at start'}
                    state = 'global_deadline_reached'; start_deferred = True
                else:
                    with timing.span('input.deferred_epoch_boundary_at_start', epoch=closed):
                        close_epoch(closed)
            elif not guard['tripped']:
                with timing.span('input.boundary_adoption_at_start'):
                    snapshot = boundary.step(planner, snapshot, checkpoint_sha, at_step=origin_step + steps)
        publish(root, 'progress.json', progress())
        while not start_deferred and steps < n['max_new_updates']:
            # Tick-safe stop: never reserve unless the remaining in-loop time covers the bounded step budget.
            if remaining(deadline) <= step_budget():
                state = 'global_deadline_reached'; break
            q = outbox.summary()
            if steps % 32 == 0:
                used = bytes_used(root.parent)
                payload = outbox.db.execute('SELECT coalesce(sum(length(body)),0)FROM emissions').fetchone()[0]
            mem = memory()
            low_memory = low_memory + 1 if mem <= n['memory_exit_fraction'] else 0
            if low_memory >= n['memory_exit_checks']:
                state = 'memory_floor_exit'; break
            k = n['emissions_per_step']
            if used >= n['output_cap_bytes'] - 64 * 1024**2: state = 'waiting_for_output_capacity'
            elif (q['observations'] + k > OUTBOX_MAX_ROWS or q['unresolved'] + k > n['max_pending_emissions'] or
                  payload + 256 * 1024 * (1 + steps % 32) > OUTBOX_MAX_PAYLOAD): state = 'waiting_for_mandatory_gp_capacity'
            elif mem < n['memory_floor_fraction']: state = 'waiting_for_memory'
            else: state = 'running'
            if state != 'running':
                publish(root, 'progress.json', progress()); time.sleep(min(10, max(0, remaining(deadline)))); continue
            authority_check(snapshot)
            current = planner.reserve(checkpoint_sha)
            if isinstance(current, Waiting):
                state = 'waiting_for_unique_batch'; current = None; publish(root, 'progress.json', progress())
                time.sleep(min(10, max(0, remaining(deadline)))); continue
            began = time.monotonic()
            ids = list(current.ordered_data_run_ids); absolute = origin_step + steps + 1; epoch = ancestry['completed_epochs'] + current.epoch
            stage = time.monotonic()
            with timing.span('input.training_batch', batch_id=current.batch_id, absolute_optimizer_step=absolute, epoch=epoch):
                batches, total, selection = prepare_microbatches(snapshot, ids, epoch=epoch - 1, batch_index=current.index,
                                                                 normalizer=norm, max_padded_tokens=n['max_padded_tokens'])
            perf['input_seconds'] += time.monotonic() - stage
            dev = sorted(c for c, r in snapshot.contexts.items() if r['partition'] == 'validation')
            pairs = [('training', ids[steps % len(ids)]), ('validation', dev[steps % len(dev)])]
            emit = pairs if k == 2 else pairs[:1]
            seeds = [int(digest({'seed': n['seed'], 'step': absolute, 'phase': phase, 'context': cid})[:15], 16) for phase, cid in emit]
            stage = time.monotonic()
            with timing.span('model.decode_proposals', absolute_optimizer_step=absolute), timing.CUDAForwards(net) as cuda:
                z = torch.tensor(np.stack([norm(c, snapshot.encoding(c)) for _, c in emit]), device=device)
                decoded, decode_info = trie.decode_many(net, z, seeds=seeds, temperature=n['decode_temperature'], deadline=deadline)
                decode_info['cuda_timing'] = cuda.receipt()
            perf['decode_seconds'] += time.monotonic() - stage; emissions = []
            for (phase, cid), row in zip(emit, decoded):
                raw = canonical(row); create(root / f'decoded-{absolute:09d}-{phase}.json', raw); ctx = snapshot.contexts[cid]
                emissions.append({'phase': phase, 'data_run_id': cid, 'context_id': cid, 'parent_lineage_sha256': ctx['parent_lineage_sha256'],
                                  'input_descriptor': ctx, 'raw_program_utf8': row['program']['dsl'],
                                  'raw_program_sha256': row['program']['program_sha256'], 'decoder_seed': row['decoder_seed'],
                                  'decoder_protocol_sha256': row['batch_protocol_sha256'], 'decode_artifact_sha256': sha(raw)})
            with timing.span('gp.durable_outbox_ack', absolute_optimizer_step=absolute):
                ack = outbox.record_batch({'batch_id': current.batch_id, 'campaign_sha256': campaign_sha, 'checkpoint_sha256': checkpoint_sha,
                                           'global_step': absolute, 'manifest_sha256': planner.manifest_sha256, 'train_data_run_ids': ids,
                                           'emitted_train_data_run_ids': [c for p, c in emit if p == 'training'],
                                           'validation_data_run_ids': [c for p, c in emit if p == 'validation']}, emissions)
            before = M.parameter_sha256(net); need(before == checkpoint_sha, 'actual pre-update checkpoint chain'); stage = time.monotonic()
            with timing.span('model.forward_backward_optimizer', absolute_optimizer_step=absolute), timing.CUDAForwards(net) as cuda:
                objective = run_microbatches(net, batches, total, optimizer=opt, deadline=deadline); objective['cuda_forward_timing'] = cuda.receipt()
            perf['optimizer_seconds'] += time.monotonic() - stage; after = M.parameter_sha256(net); need(after != before, 'actual parameter update')
            with timing.span('checkpoint.atomic_durable_physical_state', absolute_optimizer_step=absolute):
                newpath, newcp = recovery.save(net, opt, absolute, campaign_sha)
            proof = {'schema': 'jope.continuous-neural-batch-proof/v2', 'generation': GENERATION, 'config_sha256': campaign_sha,
                     'execution_config_sha256': resume['execution_config_sha256'], 'absolute_optimizer_step': absolute,
                     'local_optimizer_step': steps + 1, 'batch': asdict(current), 'selection': selection, 'objective': objective,
                     'decoder': decode_info, 'emission_ack': ack, 'checkpoint': newcp, 'snapshot_sha256': snapshot.identity,
                     'adoption_seq': boundary.seq,
                     'checkpoint_retention': 'rolling; retained only if among last2/best2/epoch checkpoints',
                     'fresh_generated_rewards_used': False}
            raw = canonical(proof); create(root / f'batch-{absolute:09d}.json', raw)
            with timing.span('checkpoint.planner_commit', absolute_optimizer_step=absolute):
                planner.commit(current.batch_id, checkpoint_before=before, checkpoint_after=after, optimizer_step=steps + 1,
                               ordered_data_run_ids=ids, evidence={'origin': 'native_optimizer', 'event_sha256': sha(raw)})
            current_path = newpath; cp = newcp; checkpoint_sha = after; steps += 1; seen.update(ids); newseen.update(ids)
            presentations += len(ids); current = None; step_seconds.append(time.monotonic() - began)
            recovery.committed(newpath); control()
            history = {'batch': absolute, 'new_batch': steps, 'epoch': epoch, 'context_count': len(ids), 'train_loss': objective['loss'],
                       'terms': objective['terms'], 'support': objective['support'], 'dropout': True, 'at': time.time()}
            append(root / 'optimizer-history.jsonl', history); hist.append(history); hist = hist[-2048:]
            proof_tail.append({'absolute_step': absolute, 'proof_sha256': sha(raw), 'parameter_sha256': after}); proof_tail = proof_tail[-2:]
            completed = planner.summary()['completed_epochs']
            if steps % n['evaluation_every_batches'] == 0: evaluate()
            if completed > last_full:
                if remaining(deadline) <= n['epoch_boundary_budget_seconds']:
                    # Tick-safe: never start boundary work that could cross the in-loop deadline; defer it to the
                    # next segment's start (adoption runs there; this epoch's full validation is recorded as deferred).
                    deferred = {'epoch': epoch, 'at_step': origin_step + steps, 'reason': 'insufficient in-loop time for boundary work'}
                    state = 'global_deadline_reached'; break
                close_epoch(epoch); last_full = completed
            publish(root, 'progress.json', progress())
        else:
            if not start_deferred: state = 'maximum_updates_reached'
    except BaseException as error:
        state = 'failed'; failure = type(error).__name__ + ': ' + str(error)[:4000]
        if current is not None:
            # An optimizer may have completed between an exception and commit: preserve, leave intent unresolved.
            try:
                from neural_storage import checkpoint
                checkpoint(net, opt, root / 'uncertain-physical-state', step=origin_step + steps + 1, config_sha256=campaign_sha)
                planner.mark_uncertain(current.batch_id, failure)
            except BaseException as nested:
                append(root / 'closure-errors.jsonl', {'error': str(nested)[:2000]})
        publish(root, 'failure.json', {'error': failure, 'at': time.time(), 'pending_intent': planner.pending(), 'automatic_retry': False})
        raise
    finally:
        # Terminal INTENT first (cheap; not a clean stop), verified custody after, the clean result only last.
        closing = time.monotonic()
        try:
            final_control = control()
            publish(root, 'progress.json', progress())
            intent = {'schema': 'jope.continuous-neural-terminal-intent/v1', 'generation': GENERATION, 'segment_id': segment_id,
                      'state': state, 'error': failure, 'config_sha256': campaign_sha, 'deferred_boundary': deferred,
                      'execution_config_sha256': resume['execution_config_sha256'],
                      'resumed_from': {k: origin[k] for k in ('step', 'parameter_sha256', 'state_sha256') if k in origin},
                      'origin_absolute_step': origin_step, 'new_optimizer_steps': steps, 'absolute_optimizer_steps': origin_step + steps,
                      'planner': planner.summary(), 'outbox': outbox.summary(), 'current_parameter_sha256': M.parameter_sha256(net),
                      'latest_checkpoint': cp, 'latest_checkpoint_path': str(current_path), 'proof_tail': proof_tail,
                      'unique_train_contexts_seen': len(seen), 'total_train_presentations': presentations,
                      'fixed_panel_sha256': panel_identity, 'plateau': monitor, 'preprocessing_sha256': sha(preprocessing),
                      'snapshot_sha256': snapshot.identity, 'adoption_seq': boundary.seq, 'boundary': boundary.status,
                      'optimizer_control': final_control, 'verified': False}
            publish(root, 'terminal-intent.json', intent); intent_sha = sha(canonical(intent))
            publish(root, 'learning-guard.json', guard)   # sealed below; the next segment recovers it only from the seal
            t0 = time.monotonic()
            final_mark, report = CU.extend(root, planner.db, outbox.db, watermark, budget_seconds=n['closure_budget_seconds'], **custody)
            prefix = CU.prefix_integrity(root, planner.db, origin_step, final_mark['local_step'],
                                         budget_seconds=n['closure_budget_seconds'])
            t1 = time.monotonic(); planner_v = planner.verify(); t2 = time.monotonic(); outbox_v = outbox.verify(); t3 = time.monotonic()
            need(planner_v['events'] == final_mark['planner']['events'] and planner_v['head_sha256'] == final_mark['planner']['head_sha256'] and
                 planner_v['committed_batches'] == final_mark['local_step'] == steps and planner_v['unresolved_batches'] == 0,
                 'full planner verification is the joined watermark head with no unresolved reservation')
            need(outbox_v['events'] >= final_mark['outbox']['events'] and outbox_v['batches'] == final_mark['outbox']['batches'] and
                 outbox_v['observations'] == final_mark['outbox']['observations'], 'full outbox verification continues the joined watermark')
            need(final_mark['parameter_sha256'] == checkpoint_sha == final_control['checkpoint_parameter_sha256'] == cp['parameter_sha256'] and
                 final_mark['absolute_step'] == final_control['absolute_optimizer_step'], 'watermark, control and physical checkpoint agree')
            need(time.monotonic() - closing <= n['closure_budget_seconds'], 'closure stayed within its qualified budget')
            verification = {'schema': 'jope.continuous-neural-result-verification/v2', 'generation': GENERATION, 'segment_id': segment_id,
                            'state': state, 'passed': True, 'intent_sha256': intent_sha, 'watermark': final_mark, 'watermark_report': report,
                            'planner_verification': planner_v, 'outbox_verification': outbox_v, 'prefix_integrity': prefix,
                            'control_sha256': sha(canonical(final_control)), 'checkpoint': cp, 'checkpoint_path': str(current_path),
                            'learning_guard': guard, 'learning_guard_sha256': sha(canonical(guard)),
                            'execution_config_sha256': resume['execution_config_sha256'], 'campaign_sha256': campaign_sha,
                            'seconds': {'suffix_joins': t1 - t0, 'planner_verify': t2 - t1, 'outbox_verify': t3 - t2,
                                        'closure': time.monotonic() - closing}}
            publish(root, 'result-verification.json', verification); verification_sha = sha(canonical(verification))
            publish(root, 'result.json', {'schema': 'jope.continuous-neural-result/v3', 'generation': GENERATION, 'segment_id': segment_id,
                                          'state': state, 'error': failure, 'verified': True, 'intent_sha256': intent_sha,
                                          'verification_sha256': verification_sha, 'config_sha256': campaign_sha,
                                          'origin_absolute_step': origin_step, 'new_optimizer_steps': steps,
                                          'absolute_optimizer_steps': origin_step + steps, 'current_parameter_sha256': checkpoint_sha,
                                          'latest_checkpoint': cp, 'proof_tail': proof_tail, 'deferred_boundary': deferred,
                                          'snapshot_sha256': snapshot.identity, 'adoption_seq': boundary.seq})
        except BaseException as error:
            closure_error = type(error).__name__ + ': ' + str(error)[:4000]
            try:
                publish(root, 'closure-failure.json', {'schema': 'jope.continuous-neural-closure-failure/v1', 'segment_id': segment_id,
                                                       'state': state, 'error': closure_error, 'at': time.time(),
                                                       'automatic_retry': False})
            except BaseException as nested:
                append(root / 'closure-errors.jsonl', {'error': str(nested)[:2000], 'closure_error': closure_error})
        finally:
            planner.close(); outbox.close()
    if closure_error is not None:
        raise RuntimeError('terminal closure failed (no clean result; review required): ' + closure_error)
    return {'state': state, 'new_optimizer_steps': steps, 'absolute_optimizer_steps': origin_step + steps}
