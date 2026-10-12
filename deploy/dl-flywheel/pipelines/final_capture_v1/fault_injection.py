"""Fault injection for final_capture_v1: each key refusal is disabled in a scratch copy and its fixtures must fail.

Parent: checks SOURCE_PINS.json first, then for every mutant copies the package into a new scratch tree, applies one
exact single-occurrence replacement, runs the named fixtures in a child process against the mutant and requires
each to fail. A control run of an unmutated scratch copy must pass the union of those fixtures. Scratch trees are
removed; the original package bytes are re-proven against SOURCE_PINS.json afterwards. Writes a create-only
receipt plus one create-only log per run next to it. CPU only; private temporary worlds only.
r2: every failing (sub)test is also classified from its assertion text: 'accepted' (the counterexample was captured:
"ValueError not raised"), 'refused_otherwise' (a different, later refusal still caught it: "does not match"),
'error' (an exception other than the expected ValueError) or 'other'.
Child (internal): ``--child SCRATCH_DF TEST_ID...`` prints one JSON line {outcomes, details, module, tests_run}.
"""
import argparse, json, os, shutil, subprocess, sys, tempfile, time, unittest
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE))
import run_offline as R  # noqa: E402

T = R.T
OWNER, FAILED, PENDING, PARAM, CKPT, PROOF = (T + 'test_b_owner_active_or_cgroup_nonempty_refused', T + 'test_b_failed_or_uncertain_terminal_state_refused',
                                              T + 'test_b_pending_reservation_or_orphan_state_is_a_hold', T + 'test_b_parameter_sha_disagreement_refused',
                                              T + 'test_b_checkpoint_bytes_tamper_refused', T + 'test_b_proof_or_ack_evidence_mismatch_refused')
RESULT, MISSING, PARITY = T + 'test_a_present_result_must_join_exactly', T + 'test_a_missing_result_is_captured_and_fresh_journal_inherits', T + 'test_d_next_update_parity_on_isolated_scratch'
PRIVATE, PARTS = T + 'test_e_private_copy_only_pinned_runtime_and_create_only_output', T + 'test_g_chunked_parts_boundaries_tamper_missing_and_reorder'
Q = R.R  # r2 runtime_limit_stopped_idle fixtures
R_POS, R_A, R_B, R_WINDOW, R_PASSED = (Q + 'test_r_live_shape_runtime_limit_stopped_idle_is_captured', Q + 'test_r_a_timeout_with_unresolved_reservation_is_a_hold',
                                       Q + 'test_r_b_write_after_in_loop_deadline_refused', Q + 'test_r_b_in_loop_deadline_comes_from_window_and_config',
                                       Q + 'test_r_b_in_loop_deadline_must_have_passed')
R_C, R_C2, R_D, R_E, R_F = (Q + 'test_r_c_owner_cgroup_or_pid_refused', Q + 'test_r_c_restarted_unit_refused', Q + 'test_r_d_systemd_result_other_than_timeout_refused',
                            Q + 'test_r_e_running_failed_or_unknown_state_refused', Q + 'test_r_f_last_commit_progress_and_rolling_must_agree')
R_G, R_G2, R_H, R_BOUNDS = (Q + 'test_r_g_missing_or_forged_r17_receipt_refused', Q + 'test_r_g_clean_class_refuses_r17_inputs',
                            Q + 'test_r_h_symlink_or_special_file_in_root_refused', Q + 'test_r_h_walk_entry_and_byte_bounds_refused')
R_I, R_I2, R_J, R_K = (Q + 'test_r_i_live_walk_must_equal_private_copy_walk', Q + 'test_r_i_live_walk_changed_during_capture_refused',
                       Q + 'test_r_j_status_or_result_json_present_refused', Q + 'test_r_k_walk_parts_reader_refuses_tamper')
DISABLE = lambda s: s.replace('need(', 'need(True or ', 1)
MUTANTS = [
    ('owner-stopped-check', 'capture.py', "need(r['unit'] == unit and", [OWNER, R_C]),
    ('owner-stable-across-capture', 'capture.py', 'need(owner(owner_readback, unit) == before,', [OWNER]),
    ('failure-markers', 'capture.py', 'need(not markers,', [FAILED, R_E]),
    ('terminal-progress-state', 'capture.py', 'need(state in TERMINAL or state in IDLE_WAITING,', [FAILED, R_E]),
    ('pending-reservation-hold', 'capture.py', "need(head['unresolved_batches'] == 0 and pending == 0,", [PENDING, R_A]),
    ('orphan-proof-hold', 'capture.py', "need(not (training / f'batch-{absolute + 1:09d}.json').exists(),", [PENDING]),
    ('proof-commit-evidence-join', 'capture.py', "need(ev['origin'] == 'native_optimizer' and sha(raw) == ev['event_sha256'] and", [PROOF]),
    ('outbox-ack-join', 'capture.py', "need(v1q['schema'] == pq['schema']", [PROOF]),
    ('checkpoint-closure', 'capture.py', "need(m == proof['checkpoint'] and", [CKPT]),
    ('parameter-sha-join', 'capture.py', "need(planner_param == param == progress['checkpoint_sha256']", [PARAM]),
    ('result-json-join', 'capture.py', "need(r['schema'] == 'jope.continuous-neural-result/v1'", [RESULT]),
    ('private-copy-only', 'capture.py', "need(Path(c['owned_root']) != root", [PRIVATE]),
    ('part-hash-chain', 'capture.py', "need(sha(raw) == part['sha256'] and", [PARTS]),
    ('join-record-order', 'capture.py', 'need(joins and all(', [PARTS]),
    # r2: one mutant per new runtime_limit_stopped_idle guard
    ('r17-timeout-result', 'capture.py', "need(readback['Result'] == 'timeout',", [R_D]),
    ('r17-no-restarts', 'capture.py', "need(readback['NRestarts'] == 0,", [R_C2]),
    ('r17-sealed-receipt', 'capture.py', "need(type(owner_ruling) is bytes and sha(owner_ruling) == R17['receipt_sha256'],", [R_G]),
    ('clean-class-takes-no-r17-inputs', 'capture.py', 'need(owner_ruling is None and live is None,', [R_G2]),
    ('window-config-deadline-join', 'capture.py', "need(type(w) is dict and w.get('config_sha256') == campaign", [R_WINDOW]),
    ('in-loop-deadline-passed', 'capture.py', 'need(time.time_ns() > cutoff,', [R_PASSED]),
    ('walk-regular-files-and-dirs-only', 'capture.py', "need(kind in ('file', 'dir'),", [R_H]),
    ('walk-entry-byte-bounds', 'capture.py', 'need(count[0] <= WALK_MAX_ENTRIES', [R_BOUNDS]),
    ('no-write-after-in-loop-deadline', 'capture.py', 'need(not late,', [R_B]),
    ('live-walk-equals-copy-walk', 'capture.py', 'need(type(live) is dict and live == summary,', [R_I]),
    ('live-walk-stable-across-capture', 'capture.py', 'need(live_walk is None or live_walk() == live,', [R_I2]),
    ('r17-closure-commit-progress-rolling', 'capture.py', "need(index[-1]['absolute_step'] == absolute", [R_F]),
    ('status-result-absent', 'capture.py', 'need(not any(p.exists() or p.is_symlink() for p in (status_path, result_path)),', [R_J]),
    ('walk-records-reader', 'capture.py', "need(body['owner_ruling'] == R17 and walked", [R_K]),
]
EDITS = [  # mutants that change a value rather than disable a refusal
    ('ancestry-lineage-epochs', 'capture.py', "'checkpoint_sha256': param, 'optimizer_steps': absolute, 'completed_epochs': lineage_epochs, 'fingerprints': fps,",
     "'checkpoint_sha256': param, 'optimizer_steps': absolute, 'completed_epochs': local_completed, 'fingerprints': fps,", [MISSING, PARITY]),
    ('probe-ignores-carried-lr-groups', 'probe.py', "for g, r in zip(opt.param_groups, inputs['learning_rates']): g['lr'] = r", 'pass', [PARITY]),
    # r2: the positive live-shape fixture is admitted by the idle allow-list and nothing else
    ('idle-allow-list-emptied', 'capture.py', "IDLE_WAITING = ('waiting_for_mandatory_gp_capacity', 'waiting_for_output_capacity', 'waiting_for_unique_batch')",
     'IDLE_WAITING = ()', [R_POS]),
]


def detail(test, text, kind):
    """Classify one failing (sub)test of a mutant run from its assertion text."""
    if kind == 'error': how = 'error'
    elif 'not raised' in text: how = 'accepted'
    elif 'does not match' in text: how = 'refused_otherwise'
    else: how = 'other'
    last = [l for l in text.strip().splitlines() if l.strip()][-1:]
    return {'subtest': str(test)[:300], 'kind': how, 'last_line': (last[0] if last else '')[:400]}


def child(df, names):
    base = R.base_root(); sys.dont_write_bytecode = True; sys.path[:0] = R.search_path(base, Path(df)); os.environ['CUDA_VISIBLE_DEVICES'] = ''
    suite = unittest.defaultTestLoader.loadTestsFromNames(names); result = unittest.TextTestRunner(stream=sys.stderr, verbosity=2).run(suite)
    out = {i: 'ok' for i in names}; details = {i: [] for i in names}
    for kind, rows in (('failure', result.failures), ('error', result.errors), ('skip', result.skipped)):
        for test, text in rows:
            tid = getattr(test, 'test_case', test).id()  # a failed subTest counts against its parent fixture
            if out.get(tid) == 'ok': out[tid] = kind
            if tid in details and kind != 'skip': details[tid].append(detail(test, text, kind))
    import pipelines.final_capture_v1.capture as loaded
    print(json.dumps({'outcomes': out, 'details': details, 'module': loaded.__file__, 'tests_run': result.testsRun}))


def run(scratch_root, name, edit, tests, logdir, stem):
    tree = Path(tempfile.mkdtemp(prefix=f'w10-fi-{name}-', dir=scratch_root)); df = tree / 'deploy/dl-flywheel'
    shutil.copytree(PACKAGE, df / 'pipelines/final_capture_v1', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    record = {'id': name, 'tests': tests}
    try:
        if edit is not None:
            file, old, new = edit; path = df / 'pipelines/final_capture_v1' / file; text = path.read_text()
            R.require(text.count(old) == 1, 'mutation target occurs exactly once: ' + name)
            os.chmod(path, 0o644); path.write_text(text.replace(old, new)); record.update(file=file, old=old, new=new, mutated_sha256=R.file_sha(path))
        proc = subprocess.run([sys.executable, '-B', __file__, '--child', str(df), *tests], capture_output=True, text=True, timeout=1800)
        R.create_only(logdir / f'{stem}-{name}.log', proc.stdout + '\n--- stderr ---\n' + proc.stderr)
        lines = [l for l in proc.stdout.splitlines() if l.startswith('{"outcomes"')]
        R.require(len(lines) == 1, 'child produced one outcome line: ' + name); value = json.loads(lines[0])
        R.require(value['module'].startswith(str(df)), 'child imported the scratch copy: ' + name)
        record.update(outcomes=value['outcomes'], details=value['details'], tests_run=value['tests_run'],
                      log_sha256=R.file_sha(logdir / f'{stem}-{name}.log'))
        kinds = [d['kind'] for v in value['details'].values() for d in v]
        record['failing_cases'] = {k: kinds.count(k) for k in ('accepted', 'refused_otherwise', 'error', 'other')}
    finally: shutil.rmtree(tree)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument('--receipt', type=Path); parser.add_argument('--child', nargs='+')
    args = parser.parse_args(argv)
    if args.child: return child(args.child[0], args.child[1:])
    pins, pins_raw, base = R.checked_pins(); started = time.time()
    logdir, stem = args.receipt.parent, args.receipt.stem; scratch = Path(os.environ.get('TMPDIR', tempfile.gettempdir())).resolve()
    mutants = [(n, f, o, DISABLE(o), t) for n, f, o, t in MUTANTS] + EDITS
    union = sorted({t for *_, tests in mutants for t in tests})
    control = run(scratch, 'control-unmutated', None, union, logdir, stem)
    control['passed'] = all(control['outcomes'][t] == 'ok' for t in union) and control['tests_run'] == len(union)
    rows = []
    for name, file, old, new, tests in mutants:
        row = run(scratch, name, (file, old, new), tests, logdir, stem)
        row['caught'] = row['tests_run'] == len(tests) and all(row['outcomes'][t] in ('failure', 'error') for t in tests); rows.append(row)
    restored = R.source_pins(PACKAGE) == pins['source_pins']
    receipt = {'schema': 'jope.w10-fault-injection/v1', 'package': 'final_capture_v1', 'revision': 'r2', 'source_pins_sha256': R.hashlib.sha256(pins_raw).hexdigest(),
               'control': control, 'mutants': rows, 'all_caught': all(r['caught'] for r in rows), 'original_restored_and_pinned': restored,
               'passed': control['passed'] and all(r['caught'] for r in rows) and restored, 'base_root': str(base),
               'started_unix': started, 'finished_unix': time.time(),
               'scope': 'Scratch copies only; each mutant disables or alters exactly one check; CPU fixtures on private worlds.'}
    R.create_only(args.receipt, json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    print(json.dumps({'passed': receipt['passed'], 'control': control['passed'], 'caught': sum(r['caught'] for r in rows), 'mutants': len(rows),
                      'restored': restored}))
    if not receipt['passed']: raise SystemExit(1)


if __name__ == '__main__':
    main()
