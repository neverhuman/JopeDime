#!/usr/bin/env python3
"""H76 fixed-final adapter updates; CPU helpers never open corpus payloads."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ARMS = ("CONTROL", "K8", "SAM", "ASAM")
SEEDS = (12, 13, 14)
ROUNDS = 20
BATCH = 16
MARGIN = 0.001546
SELECTED_SHA = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
FUSED_SHA = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


h59 = load("h76_h59", HERE.parent / "h59/runner.py")
h55 = load("h76_h55", h59.H55_PATH)


def safe_path(path):
    """Reject protected/live aliases before opening even a metadata file."""
    path = Path(path)
    for candidate in (path.absolute(), path.resolve()):
        if any(any(word in part.lower() for word in
                   ("sealed", "holdout", "meta-test", "meta_test", "flywheel-full-20261030"))
               for part in candidate.parts):
            raise RuntimeError(f"protected/live path rejected: {path}")
    return path


def validate_paths(input_dir, rows):
    for row in rows:
        cid = row["context_sha256"]
        if not re.fullmatch(r"[0-9a-f]{64}", cid):
            raise RuntimeError("invalid context identity")
        if row.get("task") != "regression" or row.get("dl_split") not in ("DL-Train", "DL-Val"):
            raise RuntimeError("unapproved roster row")
        for directory, suffix in (("pools", ".npz"), ("tables", ".jtt1")):
            safe_path(input_dir / directory / (cid + suffix))


def batch_plan(pools, sampler, seed):
    ids = tuple(sorted(pools))
    orders = {cid: np.argsort(pool["fitness"]) for cid, pool in pools.items()}
    state = np.random.get_state()
    plan = []
    try:
        for rnd in range(1, ROUNDS + 1):
            key = seed * 1_000_003 + 10_007
            np.random.seed((key + rnd * 104729) % 2**32)
            rotated = ids[rnd % len(ids):] + ids[:rnd % len(ids)]
            for step in range(8):
                stream = int.from_bytes(hashlib.sha256(
                    f"flywheel-batch/v1:{key}:{rnd}:{step}".encode()).digest()[:8], "little")
                cids = tuple(random.Random(stream).choices(rotated, k=BATCH))
                ix = tuple(np.asarray(sampler(pools[c], orders[c], 192), dtype=np.int64).copy() for c in cids)
                if any(x.shape != (192,) for x in ix):
                    raise RuntimeError("incorrect candidate slate")
                plan.append(h59.TrainingBatch(rnd, step, stream, cids, ix))
    finally:
        np.random.set_state(state)
    return plan


def optimizer_step(named, optimizer, closure, arm):
    """One AdamW step, with exact rollback and matched RNG on SAM's two passes."""
    if arm not in ARMS:
        raise ValueError(arm)
    named = [(n, p) for n, p in named if p.requires_grad]
    params = [p for _, p in named]
    optimizer.zero_grad(set_to_none=True)
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if any(p.is_cuda for p in params) else None
    loss = closure()
    if not torch.isfinite(loss):
        raise RuntimeError("nonfinite loss")
    loss.backward()
    originals = []
    if arm in ("SAM", "ASAM"):
        active = [(n, p) for n, p in named if p.grad is not None]
        scales = [p.detach().abs() + .01 if arm == "ASAM" and "weight" in n
                  else torch.ones_like(p) for n, p in active]
        norm = torch.linalg.vector_norm(torch.stack([
            torch.linalg.vector_norm(t * p.grad) for (_, p), t in zip(active, scales, strict=True)]))
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite perturbation norm")
        rho = .5 if arm == "ASAM" else .05
        try:
            with torch.no_grad():
                for (_, p), t in zip(active, scales, strict=True):
                    originals.append((p, p.detach().clone()))
                    p.add_(rho * t.square() * p.grad / (norm + 1e-16))
            optimizer.zero_grad(set_to_none=True)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            loss = closure()
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite perturbed loss")
            loss.backward()
        finally:
            with torch.no_grad():
                for p, original in originals:
                    p.copy_(original)
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm)


def elo_delta(p):
    if p == 0:
        return "-Infinity"
    if p == 1:
        return "Infinity"
    return 400 * math.log10(p / (1 - p))


def summarize(cells, draws=10_000):
    expected = {(a, s) for a in ARMS for s in SEEDS}
    indexed = {(c["arm"], c["seed"]): c for c in cells}
    if len(cells) != 12 or set(indexed) != expected:
        raise RuntimeError("all twelve unique paired cells required")
    identities = set()
    for cell in cells:
        expected_updates = 160 if cell['arm'] == 'K8' else 40
        if (cell['timing_only'] or cell['updates'] != expected_updates or
            cell['trainable_count'] != 29211 or cell['dope_before'] != h59.DOPE_SHA or
            cell['dope_after'] != h59.DOPE_SHA or cell['checkpoint_sha256'] != h59.R882_SHA or
            cell['manifest_sha256'] != h59.MANIFEST_SHA or cell['selected_sha256'] != SELECTED_SHA or
            cell['initial_adapter_sha256'] == cell['final_adapter_sha256']):
            raise RuntimeError('cell integrity proof mismatch')
        identities.add((cell['prereg_sha256'], cell['source_commit'], str(cell['folds']), cell['initial_adapter_sha256']))
    if len(identities) != 1:
        raise RuntimeError('cell source/prereg/fold identity mismatch')
    families = sorted(indexed["CONTROL", 12]["dev"]["per_family"])
    if len(families) != 19:
        raise RuntimeError("nineteen dev families required")
    for cell in cells:
        if sorted(cell["dev"]["per_family"]) != families:
            raise RuntimeError("family roster mismatch")
    samples = np.random.default_rng(76000).integers(0, 19, size=(draws, 19))
    result = {"schema": "jope.h76.metrics/v1", "complete": True,
              "reference_elo": 1500, "nominal_comparisons": 3, "arms": {}}
    for arm in ARMS:
        values = np.asarray([[indexed[arm, s]["dev"]["per_family"][f] -
                              indexed["CONTROL", s]["dev"]["per_family"][f]
                              for f in families] for s in SEEDS], dtype=float)
        if not np.isfinite(values).all():
            raise RuntimeError("nonfinite family outcome")
        per_seed = values.mean(axis=1)
        primary = float(np.median(per_seed))
        boot = np.median(values[:, samples].mean(axis=2), axis=0)
        wins = (values > 0).astype(float) + .5 * (values == 0)
        p = float(wins.mean())
        p_ci = np.quantile(wins[:, samples].mean(axis=(0, 2)), [.025, .975])
        lcb = float(np.quantile(boot, .05))
        delta = elo_delta(p)
        result["arms"][arm] = {
            "per_seed_val_reg_gain": per_seed.tolist(), "median_gain": primary,
            "ci95": np.quantile(boot, [.025, .975]).tolist(), "lcb95": lcb,
            "bonferroni_lcb": float(np.quantile(boot, .05 / 3)),
            "admitted": arm != "CONTROL" and lcb > MARGIN and primary >= 0,
            "family_win_rate": p, "elo_delta": delta,
            "elo": delta if isinstance(delta, str) else 1500 + delta,
            "elo_delta_ci95": [elo_delta(float(x)) for x in p_ci],
            "seeds": [{"seed": s, "train_reg": indexed[arm, s]["train"]["mean"],
                       "val_reg": indexed[arm, s]["dev"]["mean"],
                       "gap": indexed[arm, s]["train"]["mean"] - indexed[arm, s]["dev"]["mean"]}
                      for s in SEEDS],
        }
    return result


def assert_lease(fd, path=None):
    """Validate the inherited descriptor and parent-owned kernel flock."""
    path = path or Path.home() / "jope_plan/qualification.lock"
    actual, expected = os.fstat(fd), path.stat()
    if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError("lease descriptor is not qualification.lock")
    key = f"{os.major(actual.st_dev):x}:{os.minor(actual.st_dev):02x}:{actual.st_ino}"
    entries = [line.split() for line in Path('/proc/locks').read_text().splitlines()]
    if not any(len(f) >= 6 and f[1:4] == ['FLOCK', 'ADVISORY', 'WRITE'] and
               f[4] == str(os.getppid()) and f[5] == key for f in entries):
        raise RuntimeError("parent does not own exclusive kernel lease")


def run_cell(args):
    assert_lease(args.lock_fd)
    torch.set_num_threads(4)
    safe_path(args.input)
    safe_path(args.checkpoint)
    for name in ('source_manifest.json', 'selected.json', 'cpu_exact_fused.py'):
        safe_path(args.input / name)
    if h59.sha256(args.input / 'selected.json') != SELECTED_SHA:
        raise RuntimeError('selected roster SHA mismatch')
    if h59.sha256(args.input / 'cpu_exact_fused.py') != FUSED_SHA:
        raise RuntimeError('fused model source SHA mismatch')
    h48 = load('h76_h48', h59.H48_PATH)
    selected, train_rows, dev_rows = h55.input_metadata(args.input, h48)
    validate_paths(args.input, train_rows + dev_rows)
    folds = h55.family_folds(train_rows, h48)
    staged = h59.stage_r882_copy(args.checkpoint, args.out.parent / 'scratch')
    pools, tables = h55.load_rows(args.input, train_rows, h48)
    net, params, guard, checkpoint = h55.build_model(args.input, staged, torch.device('cuda:0'), h48)
    tokens, masks = h48.h46.make_tokens(net, tables, torch.device('cuda:0'))
    predict = h48.make_predictor(net, pools, tokens, masks, torch.device('cuda:0'))
    ctx = {'net': net, 'h48': h48, 'guard': guard}
    names = h59.verify_cell_integrity(ctx)
    initial = h55.adapter_digest(h55.adapter_state(net))
    sampler = load('h76_h53b', h55.H53B_PATH)
    training = load('h76_sampling_reference', h55.TRAINING_PATH)
    orders = {cid: np.argsort(pool['fitness']) for cid, pool in pools.items()}
    state = np.random.get_state()
    np.random.seed(61353)
    probe = sorted(pools)[:16]
    reference = [training.training_indices(pools[c], 192) for c in probe]
    np.random.seed(61353)
    cached = [sampler.cached_training_indices(pools[c], orders[c], 192) for c in probe]
    np.random.set_state(state)
    if any(not np.array_equal(a, b) for a, b in zip(reference, cached, strict=True)):
        raise RuntimeError('cached sampler differs from reference')
    plan = [b for b in batch_plan(pools, sampler.cached_training_indices, args.seed)
            if args.arm == 'K8' or b.step_index < 2]
    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    h48.base.guard.assert_optimizer_excludes_dope(optimizer, net)
    losses = []
    for batch in plan:
        # Pair stochastic model operations independently of arm compute count.
        torch.manual_seed(batch.seed)
        def closure():
            prediction = predict(list(batch.context_ids), list(batch.candidate_indices))
            return h59.loss_by_arm('CONTROL', prediction,
                                   h59._batch_targets(pools, batch, prediction.device))
        losses.append(optimizer_step(net.named_parameters(), optimizer, closure, args.arm)[0])
    h59.verify_cell_integrity(ctx)
    final = h55.adapter_digest(h55.adapter_state(net))
    if initial == final:
        raise RuntimeError('no adapter parameter changed')
    if not args.timing:
        h48.save_adapter_snapshot(args.out.with_suffix('.adapter.pt'), h55.adapter_state(net), h59.R882_SHA)
    for p in params:
        p.requires_grad_(False)
    net.eval()
    h61 = load('h76_h61', HERE.parent / 'h61/runner.py')
    def evaluate(packet, predictor):
        scores = h61.predict_pool_scores(packet, packet, predictor)
        if any(not np.isfinite(v).all() for v in scores.values()):
            raise RuntimeError('nonfinite terminal predictions')
        return h61.evaluate_pool_predictions(packet, packet, scores)
    train = evaluate(pools, predict)
    dev = None
    if not args.timing:
        if h59.sha256(args.input / 'selected.json') != SELECTED_SHA:
            raise RuntimeError('selected roster changed before dev read')
        h49 = load('h76_h49', h59.H49_SCORE_PATH)
        rows, dev_pools, dev_tables = h49.load_validation_packet(args.input)
        if [r['context_sha256'] for r in rows] != [r['context_sha256'] for r in dev_rows]:
            raise RuntimeError('dev roster changed')
        dt, dm = h48.h46.make_tokens(net, dev_tables, torch.device('cuda:0'))
        dp = h48.make_predictor(net, dev_pools, dt, dm, torch.device('cuda:0'))
        dev = evaluate(dev_pools, dp)
    for p in params:
        p.requires_grad_(True)
    h59.verify_cell_integrity(ctx)
    assert_lease(args.lock_fd)
    result = {'arm': args.arm, 'seed': args.seed, 'timing_only': args.timing,
              'updates': len(plan), 'backward_passes': len(plan) * (2 if args.arm in ('SAM', 'ASAM') else 1),
              'trainable_count': sum(p.numel() for p in params), 'trainable_names': list(names),
              'dope_before': h59.DOPE_SHA, 'dope_after': h59.DOPE_SHA,
              'checkpoint_sha256': h59.R882_SHA, 'manifest_sha256': h59.MANIFEST_SHA,
              'selected_sha256': h59.sha256(args.input / 'selected.json'),
              'prereg_sha256': h59.sha256(HERE / 'PREREG.md'),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'source_hashes': h55.source_hashes(args.input),
              'folds': [sorted(f) for f in folds], 'initial_adapter_sha256': initial,
              'final_adapter_sha256': final, 'final_loss': losses[-1],
              'holdout_payloads_opened': 0, 'outer_rows_delivered': 0}
    if not args.timing:
        result.update(train=train, dev=dev)
    h59.atomic_json(args.out, result)
    staged.unlink()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', choices=ARMS, required=True)
    parser.add_argument('--seed', type=int, choices=SEEDS, required=True)
    parser.add_argument('--input', type=Path, default=h59.DEFAULT_INPUT)
    parser.add_argument('--checkpoint', type=Path, default=h59.DEFAULT_CHECKPOINT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    parser.add_argument('--timing', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise RuntimeError('refusing repeated cell')
    run_cell(args)


if __name__ == '__main__':
    main()
