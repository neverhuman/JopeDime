#!/usr/bin/env python3
"""One guarded H212 CPU attempt: native r882 selection over width-specific GP candidate pools."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

import numpy as np

import analysis as stat

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MANIFEST = HERE / "manifest.json"
FOLD_MANIFEST = HERE / "fold_manifest.json"
INPUT = Path.home() / "jope-data/h50/input"
CACHE = Path.home() / "jope-data/h58/cache"
REFERENCE_DIR = Path.home() / "jope-data/h195/inputs"
R882 = Path.home() / "jope-data/h51/ckpt/round-882.pt"
MICRO = Path.home() / "JopeDimeMicro"
H124_PATH = REPO / "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py"
H139_PATH = REPO / "research/flywheel_selfplay/h139_selection_audit/runner.py"
GENERATOR_PATH = MICRO / "python/rust_evaluator.py"
H124_SHA = "9ebce9652a7caadc7e50896a9c1b667de5aea53f1e3cf4f8797f815d8a16285a"
H139_SHA = "7a37821db82c89ddf8e48266383c681b20535ff31b95504e259c74dbe5a9c686"
GENERATOR_SHA = "53422343699172b61ed69ed707015558f7cf1a344e5449f30e1eb9d708aef900"
R882_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
DOPE_SHA = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
REFERENCE_MEAN = 1.1377941898164525
MAX_USED = 0.938
MIN_MEM = 16 * 2**30
MAX_RSS = 6_000_000_000
MAX_OUTPUT = 500_000_000
MAX_WALL = 10_800


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".new")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def require_host(host: str | None = None) -> None:
    actual = (host or socket.gethostname()).split(".")[0]
    if actual != "xbabe2":
        raise RuntimeError("H212 is CPU-only on xbabe2")


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable unavailable")


def cpu_ticks() -> tuple[int, int]:
    parts = Path("/proc/stat").read_text().splitlines()[0].split()[1:]
    values = [int(value) for value in parts]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    return sum(values), idle


def host_cpu_pct(sample_seconds: float = 0.2) -> float:
    total0, idle0 = cpu_ticks()
    time.sleep(sample_seconds)
    total1, idle1 = cpu_ticks()
    total, idle = total1 - total0, idle1 - idle0
    return 100.0 * (1.0 - idle / total) if total > 0 else 0.0


def descendants_rss_bytes(root_pid: int | None = None) -> int:
    root_pid = os.getpid() if root_pid is None else root_pid
    table: dict[int, tuple[int, int]] = {}
    for line in Path("/proc").glob("[0-9]*/stat"):
        try:
            text = line.read_text()
            right = text.rfind(")")
            parts = text[right + 2 :].split()
            pid = int(line.parent.name)
            ppid = int(parts[1])
            rss_pages = int(parts[21])
            table[pid] = (ppid, rss_pages)
        except (OSError, ValueError, IndexError):
            continue
    owned = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (ppid, _) in table.items():
            if ppid in owned and pid not in owned:
                owned.add(pid)
                changed = True
    page = os.sysconf("SC_PAGE_SIZE")
    return sum(table[pid][1] * page for pid in owned if pid in table)


def guard(root: Path | None = None) -> dict:
    require_host()
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("H212 requires nice -n 19")
    usage = shutil.disk_usage("/")
    used = usage.used / (usage.used + usage.free)
    available = mem_available()
    rss = descendants_rss_bytes()
    if used > MAX_USED:
        raise RuntimeError(f"disk guard: / used {used:.5%} > 93.8%")
    if available < MIN_MEM:
        raise RuntimeError(f"memory guard: MemAvailable {available} < 16 GiB")
    if rss >= MAX_RSS:
        raise RuntimeError(f"aggregate RSS guard: {rss} >= 6 GB")
    scratch = 0
    if root is not None and root.exists():
        scratch = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        if scratch >= MAX_OUTPUT:
            raise RuntimeError(f"attempt output {scratch} >= 500 MB")
    return {
        "host": "xbabe2",
        "disk_free_bytes": usage.free,
        "disk_used_fraction": used,
        "mem_available_bytes": available,
        "aggregate_rss_bytes": rss,
        "attempt_output_bytes": scratch,
        "load1": os.getloadavg()[0],
        "online_cores": os.cpu_count(),
    }


def wait_for_headroom(root: Path, deadline: float) -> float:
    paused = 0.0
    max_load = 0.85 * (os.cpu_count() or 1)
    while True:
        guard(root)
        if time.monotonic() >= deadline:
            raise TimeoutError("three-hour wall cap reached while waiting for host load")
        cpu = host_cpu_pct()
        load = os.getloadavg()[0]
        if load <= max_load and cpu <= 85.0:
            return paused
        print(f"PAUSE load1={load:.2f}/{max_load:.2f} cpu={cpu:.1f}%/85%", flush=True)
        t0 = time.monotonic()
        time.sleep(30 if load > max_load else 1)
        paused += time.monotonic() - t0


def load_source(path: Path, name: str, expected_sha: str):
    if sha(path) != expected_sha:
        raise RuntimeError(f"pinned source SHA mismatch: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify_launch(launch_path: Path) -> dict:
    launch = json.loads(Path(launch_path).read_text())
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    if head != launch.get("source_commit"):
        raise RuntimeError("run checkout is not the exact merged H212 source commit")
    prereg_sha = sha(HERE / "PREREG.md")
    if prereg_sha != launch.get("prereg_sha256"):
        raise RuntimeError("merged H212 preregistration hash mismatch")
    for rel, expected in launch.get("bound_source_sha256", {}).items():
        local = REPO / rel
        if sha(local) != expected:
            raise RuntimeError(f"executed source pin mismatch: {rel}")
        committed = subprocess.check_output(["git", "show", f"{head}:{rel}"], cwd=REPO)
        if hashlib.sha256(committed).hexdigest() != expected:
            raise RuntimeError(f"merged source commit differs from bound bytes: {rel}")
    return launch


def verify_inputs() -> dict:
    doc = json.loads(MANIFEST.read_text())
    for key in (
        "markdown_sha256",
        "machine_sha256",
    ):
        row = doc["catalog"]
        path = Path(row["markdown_path" if key == "markdown_sha256" else "machine_path"])
        if sha(path) != row[key]:
            raise RuntimeError(f"catalog pin changed: {path}")
    for key, value in doc["inputs"].items():
        if key.endswith("_path"):
            path = Path(value)
            expected = doc["inputs"].get(key.removesuffix("_path") + "_sha256")
            if expected is not None and sha(path) != expected:
                raise RuntimeError(f"input pin changed: {path}")
    for rel, expected in doc["source_pins"].items():
        path = {
            "h124_runner": H124_PATH,
            "h139_runner": H139_PATH,
            "repo_rust_evaluator": REPO / "research/flywheel_selfplay/rust_evaluator.py",
            "micro_rust_evaluator": GENERATOR_PATH,
            "micro_model": MICRO / "python/model.py",
            "micro_phase12_model": MICRO / "python/phase12_multimodal_model.py",
            "dope_model": MICRO / "assets/models/dope_mlc_v3_multitask_best.pt",
        }[rel]
        if sha(path) != expected:
            raise RuntimeError(f"source dependency changed: {path}")
    if sha(FOLD_MANIFEST) != doc["fold_manifest_sha256"]:
        raise RuntimeError("family fold manifest hash mismatch")
    return doc


def predict_dynamic(net, h124, context_row, pool, projection, proposals, generator):
    import torch

    cid = context_row["context_sha256"]
    if not proposals:
        return np.empty(0, dtype=np.float64)
    pairs = [generator.featurize_dsl_412(item["dsl"]) for item in proposals]
    feat = np.stack([pair[0] for pair in pairs]).astype(np.float32)
    proj = np.stack([pair[1] for pair in pairs]).astype(np.float32)
    original = h124.program({cid: pool}, cid)[:4]
    lo, hi = projection[1][cid]
    anchors_proj = projection[0].rows(lo, lo + 4)
    programs = np.concatenate((original, feat), axis=0)
    projections = np.concatenate((anchors_proj, proj), axis=0)
    ctx = context_row
    with torch.no_grad():
        out = net.adapter_10tok.predict_costs(
            ctx_feat=torch.zeros((1, 584)),
            prog_412=torch.from_numpy(programs)[None],
            proj_16=torch.from_numpy(projections)[None],
            base_prior_cost=torch.zeros((1, len(programs))),
            scale=torch.tensor([0.4208]),
            sos_prior=torch.zeros((1, len(programs), 2)),
            mm_tokens_10x32=torch.from_numpy(ctx["tokens_10x32"])[None],
            mm_mask_10=torch.from_numpy(ctx["mask_10"])[None].bool(),
            revealed_indices=torch.arange(4)[None],
            revealed_raw_f=torch.as_tensor(pool["fitness"][:4], dtype=torch.float32)[None],
            revealed_gate=torch.as_tensor(pool["gate"][:4], dtype=torch.float32)[None],
        )
    values = out["total_cost"][0].detach().cpu().numpy().astype(np.float64)
    return values


def eval_family_elo(h124, train, utilities_by_width, seed: int):
    helper = h124.source_module("python/src/jope/north_star/elo_tracker.py", "h212_elo")
    names = ("N16", "N32", "N64")
    rows_by_family: dict[str, list[int]] = {}
    for i, row in enumerate(train):
        rows_by_family.setdefault(row["family_id"], []).append(i)
    families = sorted(rows_by_family)

    def ratings(sampled_families):
        records = []
        for occurrence, family in enumerate(sampled_families):
            for i in rows_by_family[family]:
                records.append(
                    helper.DatasetEvalRecord(
                        f"{occurrence}:{train[i]['context_sha256']}",
                        f"{occurrence}:{family}",
                        "regression",
                        {name: float(utilities_by_width[name][i]) for name in names},
                    )
                )
        return helper.compute_family_grouped_bradley_terry_elo(
            records, "N16", anchor_elo=1500.0, tie_margin=1e-4, prior=0.01, n_iters=400
        )["ratings"]

    point = ratings(families)
    rng = np.random.default_rng(stat.ELO_BOOT_SEED + seed - stat.SEEDS[0])
    draws = {name: [] for name in names}
    for _ in range(stat.ELO_DRAWS):
        sampled = [families[k] for k in rng.integers(0, len(families), len(families))]
        score = ratings(sampled)
        for name in names:
            draws[name].append(float(score[name]))
    return {
        "ratings": {k: float(v) for k, v in point.items()},
        "ci95": {k: [float(v) for v in np.quantile(x, [0.025, 0.975])] for k, x in draws.items()},
        "anchor": "N16",
        "anchor_elo": 1500.0,
    }


def run(root: Path, launch_path: Path) -> dict:
    # Host gate is first: no external file is opened on an unauthorized host.
    require_host()
    if os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise RuntimeError("launch with nice -n 19")
    launch_guard = guard(None)
    launch = verify_launch(launch_path)
    verify_inputs()
    if sha(R882) != R882_SHA:
        raise RuntimeError("owned r882 checkpoint copy does not match the frozen SHA")
    root = Path(root)
    expected_parent = Path.home() / "jope-data/h212"
    if root.parent != expected_parent or root.name != "attempt-001" or root.exists():
        raise RuntimeError("H212 permits one fresh attempt-001 under ~/jope-data/h212")
    launch_guard = guard(None)
    root.mkdir(parents=True)
    started = time.monotonic()
    deadline = started + MAX_WALL
    paused = 0.0
    progress_path = root / "progress.jsonl"
    metrics = {"schema": "jope.h212.pool-width-metrics/v1", "status": "INCOMPLETE"}
    try:
        h139 = load_source(H139_PATH, "h212_h139", H139_SHA)
        h124 = h139.load_h124()
        h124.INPUT, h124.CACHE, h124.R882 = INPUT, CACHE, R882
        h124.FILES[R882] = R882_SHA
        h139.INPUT, h139.ARM_DIR = INPUT, REFERENCE_DIR
        train, val, reference = h139.roster(h124)
        if len(train) != 1219 or len(val) != 91:
            raise RuntimeError("approved H139/H195 roster size mismatch")
        if len({r["family_id"] for r in train}) != 205:
            raise RuntimeError("approved TRAIN family count mismatch")
        source_manifest = json.loads((INPUT / "source_manifest.json").read_text())
        if source_manifest.get("excluded_payloads_opened") != 0 or source_manifest.get("outer_rows_delivered") != 0:
            raise RuntimeError("pinned source manifest violates the read boundary")
        if {r["family_id"] for r in train} & {r["family_id"] for r in val}:
            raise RuntimeError("TRAIN and DL-Val families overlap")
        for records, split in ((train, "DL-Train"), (val, "DL-Val")):
            if any(
                row.get("dl_split") != split
                or row.get("task") != "regression"
                or row.get("has_meta_test") is True
                or row.get("origin") not in (None, "real")
                for row in records
            ):
                raise RuntimeError("unsupported or protected cohort row")
        rows = val + train
        context, reads = h124.train_context(rows, [])
        projection = h124.train_projection(rows, [])
        val_pools = h139.pools_for(val)
        train_pools = h124.load_pools(train)
        pools = {**val_pools, **train_pools}
        static_costs, net = h124.baseline(root, rows, context, pools, projection)
        if net.training or any(p.requires_grad for p in net.parameters()):
            raise RuntimeError("r882 must remain in eval mode with all gradients disabled")
        val_values = np.asarray(
            [h139.native(static_costs[r["context_sha256"]], pools[r["context_sha256"]]["fitness"],
                         pools[r["context_sha256"]]["gate"]) for r in val], dtype=np.float64
        )
        parity = h139.regression_check(val_values, reference)
        if float(val_values.mean()) != REFERENCE_MEAN:
            raise RuntimeError(f"native r882 mean not exact: {val_values.mean()!r}")
        write_json(root / "baseline-parity.json", {**parity, "exact_mean": float(val_values.mean())})

        # Treatment data is accessed only after baseline parity passes.
        generator = load_source(GENERATOR_PATH, "h212_gp_generator", GENERATOR_SHA)
        generator.EVAL_SCRATCH_ROOT = root / "rust_eval"
        cache_manifest = json.loads((CACHE / "cache_manifest.json").read_text())
        if cache_manifest.get("excluded_payloads_opened") != 0 or cache_manifest.get("dope_sha256_after") != DOPE_SHA:
            raise RuntimeError("H58 cache provenance or DOPE freeze check failed")
        zipped = json.loads((CACHE / "jtt1_gz_manifest.json").read_text())
        champ_file = json.loads((CACHE / "champions_featurized.json").read_text())
        champion_by_cid = {item["context_sha256"]: item for item in champ_file}
        fold_doc = json.loads(FOLD_MANIFEST.read_text())
        fold_by_family = {
            family: fold["fold"] for fold in fold_doc["folds"] for family in fold["families"]
        }
        family_ids = sorted({row["family_id"] for row in train})
        if set(fold_by_family) != set(family_ids) or len(champion_by_cid) != 1219:
            raise RuntimeError("family fold or champion bank is incomplete")
        bank_by_fold = {
            fold: [item for item in champ_file if fold_by_family[item["family_id"]] != fold]
            for fold in range(5)
        }
        if any(len(bank) != 1219 - sum(1 for row in train if fold_by_family[row["family_id"]] == fold)
               for fold, bank in bank_by_fold.items()):
            raise RuntimeError("family-excluded champion bank count mismatch")

        utility_panels = {seed: {width: np.empty(len(train), dtype=np.float64) for width in stat.WIDTHS}
                          for seed in stat.SEEDS}
        oracle_panels = {seed: {width: np.empty(len(train), dtype=np.float64) for width in stat.WIDTHS}
                         for seed in stat.SEEDS}
        seed_index = {seed: i for i, seed in enumerate(stat.SEEDS)}
        candidate_sha = np.full((len(stat.SEEDS), len(train), 112), b"", dtype="S64")
        candidate_costs = np.full((len(stat.SEEDS), len(train), 112), np.nan, dtype=np.float32)
        candidate_fitness = np.full((len(stat.SEEDS), len(train), 112), np.nan, dtype=np.float32)
        candidate_gate = np.zeros((len(stat.SEEDS), len(train), 112), dtype=bool)
        proposal_sha_by_width = np.full((len(stat.SEEDS), len(train), len(stat.WIDTHS), 64), b"", dtype="S64")
        families = [row["family_id"] for row in train]
        generator.EVAL_SCRATCH_ROOT.mkdir(parents=True)
        with torch_context():
            for seed in stat.SEEDS:
                for index, row in enumerate(train):
                    paused += wait_for_headroom(root, deadline)
                    if time.monotonic() >= deadline:
                        raise TimeoutError("three-hour wall cap reached")
                    cid, family = row["context_sha256"], row["family_id"]
                    fold = fold_by_family[family]
                    parents = [champion_by_cid[cid]]
                    stream = stat.stream_seed(seed, cid)
                    pool = train_pools[cid]
                    with np.load(pool["path"], allow_pickle=False) as archive:
                        known = {x.decode("ascii") if isinstance(x, bytes) else str(x)
                                 for x in archive["program_sha256"].tolist()}
                    raw_by_width = {}
                    novel_by_width = {}
                    union_by_sha = {}
                    for width in stat.WIDTHS:
                        raw = generator.propose_flywheel_candidates(
                            parents, context[cid]["lm48_z"], bank_by_fold[fold], family,
                            n_proposals=width, seed=stream,
                        )
                        stat.validate_width_proposals(raw, width)
                        if any(hashlib.sha256(p["dsl"].encode()).hexdigest() != p["program_sha256"] for p in raw):
                            raise RuntimeError(f"proposal DSL hash mismatch for {cid}")
                        raw_by_width[width] = raw
                        novel_by_width[width] = [p for p in raw if p["program_sha256"] not in known]
                        for proposal in novel_by_width[width]:
                            union_by_sha.setdefault(proposal["program_sha256"], proposal)
                        proposal_sha_by_width[seed_index[seed], index, stat.WIDTHS.index(width), :width] = np.asarray(
                            [p["program_sha256"].encode("ascii") for p in raw], dtype="S64"
                        )
                    novel = list(union_by_sha.values())
                    jtt = CACHE / "jtt1_gz" / f"{cid}.jtt1.gz"
                    if cid not in zipped or sha(jtt) != zipped[cid]:
                        raise RuntimeError(f"H58 train-only JTT1 source mismatch: {cid}")
                    if hashlib.sha256(gzip.decompress(jtt.read_bytes())).hexdigest() != row["train_only_jtt1_sha256"]:
                        raise RuntimeError(f"decompressed TRAIN JTT1 identity mismatch: {cid}")
                    results = generator.evaluate_gp_batch_rust(
                        str(jtt), novel, run_outer_test=False, max_workers=4, timeout_s=25.0
                    )
                    if len(results) != len(novel) or any(r.get("status") != "completed" for r in results):
                        raise RuntimeError(f"candidate evaluation incomplete for {cid}")
                    if [r["program_sha256"] for r in results] != [p["program_sha256"] for p in novel]:
                        raise RuntimeError(f"candidate result order/hash mismatch for {cid}")
                    costs = predict_dynamic(net, h124, context[cid], pool, projection, novel, generator)
                    if len(costs) != 4 + len(novel) or not np.isfinite(costs).all():
                        raise RuntimeError(f"r882 prediction panel malformed for {cid}")
                    if not np.allclose(costs[:4], static_costs[cid][:4], rtol=0, atol=1e-6):
                        raise RuntimeError(f"r882 revealed-anchor prediction drift for {cid}")
                    anchor_util = np.where(pool["gate"][:4], -pool["fitness"][:4], -2.0)
                    cand_fit = np.asarray([r["fitness"] for r in results], dtype=np.float64)
                    cand_gate = np.asarray([r["cv_gate_passed"] for r in results], dtype=bool)
                    if not np.isfinite(cand_fit).all():
                        raise RuntimeError(f"candidate native fitness is non-finite for {cid}")
                    cand_util = np.where(cand_gate, -cand_fit, -2.0)
                    result_index = {p["program_sha256"]: i for i, p in enumerate(novel)}
                    for width in stat.WIDTHS:
                        arm_ids = [p["program_sha256"] for p in novel_by_width[width]]
                        keep = [result_index[program_id] for program_id in arm_ids]
                        width_cost = np.concatenate((costs[:4], costs[4:][keep]))
                        width_fit = np.concatenate((pool["fitness"][:4], cand_fit[keep]))
                        width_gate = np.concatenate((pool["gate"][:4], cand_gate[keep]))
                        selected_utility, _ = stat.first_argmin_utility(width_cost, width_fit, width_gate)
                        utility_panels[seed][width][index] = selected_utility
                        oracle_panels[seed][width][index] = float(max(np.max(anchor_util), np.max(cand_util[keep]) if keep else -np.inf))
                    candidate_sha[seed_index[seed], index, :len(novel)] = np.asarray(
                        [p["program_sha256"].encode("ascii") for p in novel], dtype="S64"
                    )
                    candidate_costs[seed_index[seed], index, :len(novel)] = costs[4:]
                    candidate_fitness[seed_index[seed], index, :len(novel)] = cand_fit
                    candidate_gate[seed_index[seed], index, :len(novel)] = cand_gate
                    ctx_tag = hashlib.sha256(str(jtt).encode()).hexdigest()[:16]
                    shutil.rmtree(generator.EVAL_SCRATCH_ROOT / f"ctx-{ctx_tag}", ignore_errors=True)
                    if index % 10 == 0:
                        row_log = {
                            "seed": seed, "context_sha256": cid, "family_id": family,
                            "fold": fold, "generated_by_width": {str(w): len(raw_by_width[w]) for w in stat.WIDTHS},
                            "novel_union": len(novel),
                            "elapsed_seconds": time.monotonic() - started,
                            "guard": guard(root),
                        }
                        with progress_path.open("a") as stream_out:
                            stream_out.write(json.dumps(row_log, sort_keys=True) + "\n")
                    guard(root)
                    if time.monotonic() - started >= MAX_WALL:
                        raise TimeoutError("three-hour wall cap reached")
        # Save all primary panels and a deterministic family-level analysis.
        arrays = {
            "candidate_sha256": candidate_sha,
            "candidate_predicted_cost": candidate_costs,
            "candidate_fitness": candidate_fitness,
            "candidate_gate": candidate_gate,
            "proposal_sha256_by_width": proposal_sha_by_width,
        }
        summary = {}
        bootstrap_ix = np.random.default_rng(stat.BOOT_SEED).integers(0, len(set(families)), size=(stat.BOOT_DRAWS, len(set(families))))
        for seed in stat.SEEDS:
            # Construct context-to-family aggregation explicitly so every family receives equal weight.
            family_order = sorted(set(families))
            fam_idx = {family: i for i, family in enumerate(family_order)}
            means = {}
            oracle_means = {}
            for width in stat.WIDTHS:
                sums = np.zeros(205, dtype=np.float64)
                counts = np.zeros(205, dtype=np.int64)
                osums = np.zeros(205, dtype=np.float64)
                for i, family in enumerate(families):
                    j = fam_idx[family]
                    sums[j] += utility_panels[seed][width][i]
                    osums[j] += oracle_panels[seed][width][i]
                    counts[j] += 1
                means[width] = sums / counts
                oracle_means[width] = osums / counts
                arrays[f"utility_s{seed}_n{width}"] = utility_panels[seed][width]
                arrays[f"oracle_s{seed}_n{width}"] = oracle_panels[seed][width]
            summary[str(seed)] = {}
            summary[str(seed)]["N16_baseline"] = {
                "family_mean_utility": float(means[16].mean()),
                "family_median_utility": float(np.median(means[16])),
            }
            for width in (32, 64):
                gain = means[width] - means[16]
                stats = stat.family_gain_stats(gain, bootstrap_ix)
                summary[str(seed)][f"N{width}_vs_N16"] = stats
                summary[str(seed)][f"N{width}_oracle_headroom"] = float(np.median(oracle_means[width] - means[width]))
        elo_by_seed = {}
        for seed in stat.SEEDS:
            elo_by_seed[str(seed)] = eval_family_elo(
                h124, train,
                {f"N{width}": utility_panels[seed][width] for width in stat.WIDTHS},
                seed,
            )
        primary = {seed: summary[str(seed)]["N64_vs_N16"] for seed in stat.SEEDS}
        elo_lowers = {seed: elo_by_seed[str(seed)]["ci95"]["N64"][0] for seed in stat.SEEDS}
        decision = stat.decide(primary, elo_lowers)
        pooled = {}
        for width in (32, 64):
            gain = np.mean(
                [
                    np.array(
                        [
                            np.mean([utility_panels[s][width][i] - utility_panels[s][16][i]
                                     for i, fam in enumerate(families) if fam == family])
                            for family in sorted(set(families))
                        ]
                    )
                    for s in stat.SEEDS
                ],
                axis=0,
            )
            pooled[f"N{width}_vs_N16"] = stat.family_gain_stats(gain, bootstrap_ix)
        save_panel = root / "candidate-panels.npz"
        np.savez_compressed(save_panel, **arrays)
        metrics = {
            "schema": "jope.h212.pool-width-metrics/v1",
            "status": "COMPLETE",
            "verdict": decision["verdict"],
            "failed_gates": decision["failed_gates"],
            "attempt": "attempt-001",
            "source_commit": launch["source_commit"],
            "prereg_sha256": sha(HERE / "PREREG.md"),
            "baseline_parity": parity,
            "contexts": len(train),
            "families": len(set(families)),
            "fits_or_updates": 0,
            "widths": list(stat.WIDTHS),
            "primary_width": stat.PRIMARY_WIDTH,
            "family_statistics": summary,
            "pooled_family_statistics": pooled,
            "elo": elo_by_seed,
            "optimizer_updates": 0,
            "dope_sha256_before": DOPE_SHA,
            "dope_sha256_after": DOPE_SHA,
            "CUDA_initialized": False,
            "oracle": "diagnostic only",
            "runtime_seconds": time.monotonic() - started,
            "paused_load_seconds": paused,
            "launch_guard": launch_guard,
            "read_boundary": {"train": 1219, "validation_baseline_only": 91,
                              "holdout": 0, "sealed": 0, "meta_test": 0,
                              "outer_rows": 0, "excluded_payloads": 0},
            "panels_sha256": sha(save_panel),
            "panels_bytes": save_panel.stat().st_size,
        }
        write_json(root / "metrics.json", metrics)
        print(json.dumps({"status": metrics["status"], "verdict": metrics["verdict"]}), flush=True)
        return metrics
    except Exception as exc:
        metrics.update({"error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started})
        write_json(root / "metrics.json", metrics)
        raise


class torch_context:
    """Small context manager keeping Torch in deterministic, inference-only CPU mode."""

    def __enter__(self):
        import torch
        torch.set_num_threads(4)
        torch.set_grad_enabled(False)
        return torch

    def __exit__(self, exc_type, exc, tb):
        import torch
        torch.set_grad_enabled(True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True)
    args = parser.parse_args()
    require_host()
    result = run(args.root, args.launch)
    print(json.dumps({"status": result["status"], "verdict": result["verdict"]}))


if __name__ == "__main__":
    main()
