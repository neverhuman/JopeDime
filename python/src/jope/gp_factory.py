"""jain-GP champion factory: run the REAL feat-math GP per dataset on xbabe1's spare
cores, seeded with our current champions, and harvest winners into the training corpus.

Why: the corpus incumbents ARE historical jain_regressor scores — found once with weak
configs (pop=64, 42-1000s). The engine itself (feat-math ``evolve``: populations,
crossover, ridge-surrogate proxy fitness -> honest-CV promote, gen-0 seed injection) can
push far past them. Each slice:

  1. write slice hashes + seed genomes (per-dataset champions ∪ policy picks), scp to xbabe1
  2. ssh: ``jain jain-regressor --hashes-file ... --seed-genomes-file ... --model-name
     jope_gp_v1 --overwrite`` — reports land in each dataset's ``reports/jope_gp_v1.json``
     on xbabe1 (standard schema: genome + cv.oof_r2 + finalists[10])
  3. rsync the reports back, harvest: winner + finalists -> TrainingRow JSONL
     ``champion_rows_gp<batch>.jsonl`` in the art dir — the self-play trainer's
     ``champion_rows_g*`` glob picks them up automatically
  4. update the factory's own champion bars + telemetry (factory_state.json, TUI-friendly)

The factory loops slices continuously and independently of the self-play orchestrator.

Game mode (``--game``): each slice becomes an iterate-to-beat match. Per-dataset target =
max(champion bar, the model's best MEASURED candidate-pool score — the model's own claimed
frontier); jain receives a per-shard ``--targets-file`` (+ ``--fitness``), and unbeaten
datasets re-enter the next round (up to ``--game-rounds``) with a reseeded gen-0
(champions + candidate-pool model seeds + optional fresh CPU proposals + the previous
round's top finalists). Per-dataset/round outcomes append to ``game_ledger.jsonl``.
"""

from __future__ import annotations

import glob as globmod
import hashlib
import json
import math
import os
import re
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .champion_miner import build_champions
from .data.active_v3 import robust_score
from .prime_config import load_config
from .prime_staging import stage_read_only_dataset
from .propose import embedding_from_vs, target_summary_from_vs
from .schema import TrainingRow

# Fresh policy-model proposals to seed the GP with (idea: warm-start gen-0 with what the
# current model thinks is good, not just historical champions). These live in the self-play
# run's per-generation files; we scan the most recent few for the slice's datasets.
SELF_ROWS_GLOB = "self_rows_*_g*.jsonl"
SELF_ROWS_GLOBS = ("self_rows_g*.jsonl", SELF_ROWS_GLOB)
MODEL_SEED_RECENT = 3

GP_HOST = "xbabe2"
# Secondary worker host: xbabe3's spare cores (3 workers x 24 threads = 72, nice-5;
# it also serves DDP rank1 + eval bursts, which win contention at normal priority).
GP_HOST2 = "xbabe3"
GP_HOST2_WORKERS = 5
_PRIME_CONFIG = load_config()


def _gp_host_map(default_workers: int) -> dict[str, int]:
    """host→worker-count map; override via JOPE_GP_HOSTS="xbabe1:4,xbabe3:4,xbabe2:2"."""
    spec = os.environ.get("JOPE_GP_HOSTS", "").strip()
    if not spec:
        return {GP_HOST: default_workers, GP_HOST2: GP_HOST2_WORKERS}
    out: dict[str, int] = {}
    for part in spec.split(","):
        host, _, n = part.strip().partition(":")
        if host:
            out[host] = max(0, int(n or default_workers))
    return out or {GP_HOST: default_workers}
JAIN_BIN = str(_PRIME_CONFIG.jain_binary)
GPU_CACHE = str(_PRIME_CONFIG.gpu_cache)
REG_ROOT = str(_PRIME_CONFIG.regression_dataset_root)
GP_WORK_ROOT = _PRIME_CONFIG.cache / "gp_datasets"
MODEL_NAME = "jope_gp_v1"

# Game mode: per-dataset model bars + model seeds come from the NEWEST self-play candidate
# pool file (every proposal the model made last generation, measured or not) — not the
# self_rows scan, which only carries the evaluated top picks.
CANDIDATE_POOLS_GLOB = str(
    _PRIME_CONFIG.legacy_root / "runs/self_play_v33/candidate_pools/candidate_pools_g*.jsonl"
)
POOL_SEEDS_PER_DATASET = 8


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


class GpFactory:
    def __init__(
        self,
        art_dir: str | Path,
        run_dir: str | Path = _PRIME_CONFIG.runs / "gp_factory",
        *,
        slice_size: int = 48,
        workers: int = 4,  # concurrent jain processes on the GP host
        threads_per_worker: int = 16,  # 4*24=96 <= 100-core cap; host is shared
        # Probe-calibrated profile (dataset 0014b9…, 42s search → oof 0.8622 vs incumbent
        # 0.8742): search time is where wins come from; the honest-CV finalist gate is the
        # fixed cost (~37s per finalist). 240s search + top-4 gate ≈ 6.5 min/dataset,
        # ~900 datasets/day at 4 workers.
        gens: int = 5000,  # effectively uncapped; the TIME budget is the limiter
        pop: int = 128,
        folds: int = 3,
        search_rows: int = 768,
        time_budget_secs: int = 240,
        top_k: int = 4,
        gpu_kernel_configs: int = 0,  # CPU-only GP tier (jain GPU runtime unavailable on these hosts)
        seed: int = 20260707,
        game_mode: bool = False,  # iterate-to-beat: targets = max(champion bar, model bar)
        game_rounds: int = 3,
        fitness: str = "v1",  # preserve the programmatic API; the CLI explicitly opts into v2
        fresh_proposals_ckpt: str | Path | None = None,  # policy ckpt for CPU propose_v3 seeds (game mode)
        candidate_pools_glob: str | None = None,  # None -> CANDIDATE_POOLS_GLOB
        log=print,
    ) -> None:
        self.art_dir = Path(art_dir).resolve()
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.slice_size = slice_size
        self.workers = workers
        self.threads = threads_per_worker
        self.gens = gens
        self.pop = pop
        self.folds = folds
        self.search_rows = search_rows
        self.time_budget = time_budget_secs
        self.top_k = top_k
        self.gpu_kernel_configs = gpu_kernel_configs
        self.seed = seed
        self.game_mode = game_mode
        self.game_rounds = max(1, int(game_rounds))
        self.fitness = fitness
        self.fresh_proposals_ckpt = str(fresh_proposals_ckpt) if fresh_proposals_ckpt else None
        self.candidate_pools_glob = candidate_pools_glob or CANDIDATE_POOLS_GLOB
        self.log = log

        manifest = json.loads((self.art_dir / "split_manifest.json").read_text())["datasets"]
        self.datasets = {h: e for h, e in sorted(manifest.items()) if e.get("split") == "train"}
        self.standardization = json.loads((self.art_dir / "standardization.json").read_text())
        self.state_path = self.run_dir / "factory_state.json"
        self.state: dict[str, Any] = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else
            {"batch": 0, "cursor": 0, "harvested_rows": 0, "improved_datasets": 0}
        )

    # -- seeds -------------------------------------------------------------------
    def _model_seed_index(self) -> dict[str, list[str]]:
        """Cached {dataset_hash: [featmath canonical genomes]} from the last
        MODEL_SEED_RECENT self-play generations. The self_rows files are 275-418 MB each
        (every row carries a 1536-d embedding), so we scan them ONCE per factory run —
        with a cheap ``"featmath"`` substring pre-filter that skips the ~64% newmath rows
        without a full JSON parse — and persist the tiny index to disk so restarts and
        every subsequent slice are O(1). Rebuilt only when a newer gen file appears."""
        all_files: list[Path] = []
        seen: set[Path] = set()
        for glob in SELF_ROWS_GLOBS:
            for path in sorted(self.art_dir.glob(glob)):
                if path not in seen:
                    seen.add(path)
                    all_files.append(path)
        files = sorted(
            all_files,
            key=lambda p: int(m.group(1)) if (m := re.search(r"_g(\d+)", p.name)) else 0,
        )[-MODEL_SEED_RECENT:]
        sig = max((f.stat().st_mtime for f in files), default=0.0)
        if getattr(self, "_msi_sig", None) == sig and getattr(self, "_msi", None) is not None:
            return self._msi
        cache_path = self.run_dir / "model_seed_index.json"
        if cache_path.exists() and cache_path.stat().st_mtime >= sig and sig > 0:
            idx = json.loads(cache_path.read_text())
        else:
            idx = {}
            for f in files:
                with f.open("r", encoding="utf-8") as fh:
                    for line in fh:
                        if '"featmath"' not in line:  # skip newmath rows without parsing
                            continue
                        try:
                            r = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if r.get("engine") != "featmath":
                            continue
                        did = (r.get("metadata") or {}).get("dataset_hash") or r.get("dataset_id")
                        g = r.get("canonical_genome") or r.get("genome")
                        if did and g:
                            lst = idx.setdefault(did, [])
                            if g not in lst:
                                lst.append(g)
            cache_path.write_text(json.dumps(idx))
        self._msi, self._msi_sig = idx, sig
        return idx

    def _load_model_seeds(self, hashes: list[str]) -> dict[str, list[str]]:
        """Fresh featmath model proposals for a slice's datasets (O(1) cache lookups)."""
        idx = self._model_seed_index()
        return {h: idx[h] for h in hashes if h in idx}

    # -- game mode: model bars + seeds from candidate pools -------------------------
    def _model_pool_index(self) -> dict[str, dict[str, Any]]:
        """Cached ``{dataset_hash: {seeds, model_bar, pool_file, generation}}`` from the
        LATEST candidate_pools_g<N>.jsonl (by trailing generation number). ``seeds`` =
        featmath canonical genomes ordered by reward_percentile_in_engine desc (cap
        POOL_SEEDS_PER_DATASET); ``model_bar`` = max measured score_robust over ALL
        candidates any-engine (fallback measured_oof) — the model's best measured outcome,
        i.e. the bar the GP must beat to add information. Streamed once (the file is
        ~135 MB) and persisted; rebuilt only when a newer pools file appears."""
        pattern = getattr(self, "candidate_pools_glob", None) or CANDIDATE_POOLS_GLOB
        files: list[tuple[int, Path]] = []
        for name in globmod.glob(pattern):
            if m := re.search(r"_g(\d+)\.jsonl$", name):
                files.append((int(m.group(1)), Path(name)))
        if not files:
            return {}
        generation, latest = max(files)
        if getattr(self, "_mpi_src", None) == latest.name and getattr(self, "_mpi", None) is not None:
            return self._mpi
        cache_path = self.run_dir / "model_pool_index.json"
        idx: dict[str, dict[str, Any]] | None = None
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text())
            except (OSError, json.JSONDecodeError):
                cached = {}
            if cached and next(iter(cached.values())).get("pool_file") == latest.name:
                idx = cached
        if idx is None:
            ranked: dict[str, list[tuple[float, str]]] = defaultdict(list)
            bars: dict[str, float] = {}
            with latest.open("r", encoding="utf-8") as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    h = r.get("dataset_hash")
                    if not h:
                        continue
                    measured = r.get("score_robust")
                    if not isinstance(measured, (int, float)) or not math.isfinite(measured):
                        measured = r.get("measured_oof")
                    if isinstance(measured, (int, float)) and math.isfinite(measured):
                        measured = float(measured)
                        if h not in bars or measured > bars[h]:
                            bars[h] = measured
                    if r.get("engine") != "featmath":
                        continue
                    g = r.get("canonical_genome")
                    if not g:
                        continue
                    pct = r.get("reward_percentile_in_engine")
                    pct = float(pct) if isinstance(pct, (int, float)) and math.isfinite(pct) else float("-inf")
                    ranked[h].append((pct, g))
            idx = {}
            for h in set(ranked) | set(bars):
                seeds: list[str] = []
                for _, g in sorted(ranked.get(h, []), key=lambda t: t[0], reverse=True):
                    if g not in seeds:
                        seeds.append(g)
                    if len(seeds) >= POOL_SEEDS_PER_DATASET:
                        break
                idx[h] = {"seeds": seeds, "model_bar": bars.get(h),
                          "pool_file": latest.name, "generation": generation}
            cache_path.write_text(json.dumps(idx))
        self._mpi, self._mpi_src = idx, latest.name
        return idx

    def _fresh_policy_model(self):
        """Lazy one-time CPU policy load for --fresh-proposals (mirrors self_play)."""
        if getattr(self, "_fresh_policy", None) is None:
            import torch  # noqa: PLC0415

            from .policy.model_v3 import load_policy_v3  # noqa: PLC0415

            state = torch.load(self.fresh_proposals_ckpt, map_location="cpu", weights_only=False)
            self._fresh_policy = load_policy_v3(state)
        return self._fresh_policy

    def _fresh_proposals(self, hashes: list[str], batch: int) -> dict[str, list[str]]:
        """Fresh CPU propose_v3 featmath genomes per game dataset (n=4, seed=batch).
        SLOW (~seconds/dataset on CPU) — game mode only, cached per (batch, dataset).
        Fail-open with loud warnings: seeding must never take the factory down."""
        ckpt = getattr(self, "fresh_proposals_ckpt", None)
        if not ckpt:
            return {}
        cache: dict[tuple[int, str], list[str]] = getattr(self, "_fresh_cache", {})
        self._fresh_cache = cache
        missing = [h for h in hashes if (batch, h) not in cache and h in self.datasets]
        if missing:
            try:
                model = self._fresh_policy_model()
            except Exception as exc:  # noqa: BLE001
                self.log(f"[factory] WARNING: --fresh-proposals disabled — cannot load {ckpt}: {exc!r}")
                self.fresh_proposals_ckpt = None
                return {}
            from .engines import EngineRouter  # noqa: PLC0415
            from .policy.decode import propose_v3  # noqa: PLC0415

            router = EngineRouter()
            for h in missing:
                entry = self.datasets[h]
                try:
                    emb = embedding_from_vs(entry["vs_path"], self.standardization)
                    tgt = target_summary_from_vs(entry.get("train_target_vs"))
                    props = propose_v3(
                        emb, target_summary=tgt, model=model, router=router, n=4,
                        engines=("featmath",), device="cpu", seed=batch,
                    )
                    genomes: list[str] = []
                    for p in props:
                        g = p.get("canonical_genome") or p.get("genome")
                        if g and g not in genomes:
                            genomes.append(g)
                    cache[(batch, h)] = genomes
                except Exception as exc:  # noqa: BLE001
                    self.log(f"[factory] WARNING: fresh proposal failed for {h}: {exc!r}")
                    cache[(batch, h)] = []
        return {h: cache[(batch, h)] for h in hashes if cache.get((batch, h))}

    def _seed_genomes(
        self,
        hashes: list[str],
        champions: dict[str, dict],
        *,
        use_model: bool = True,
        batch: int = 0,
        extra: dict[str, list[str]] | None = None,
    ) -> list[str]:
        """Gen-0 seeds for a slice (featmath only): each dataset's champion PLUS (in the
        model arm) the current model's fresh featmath proposals, interleaved per dataset so
        the cap keeps a champion/model mix. Deduped; capped at pop (jain validates seeds).

        Game mode: model seeds come from the candidate-pool index instead (first-class,
        no A/B), with per-dataset buckets ordered previous-round finalists (``extra``) →
        fresh CPU proposals → pool seeds."""
        game_labels: dict[tuple[str, str], str] = {}
        if getattr(self, "game_mode", False):
            pool_idx = self._model_pool_index()
            fresh = self._fresh_proposals(hashes, batch)
            model_seeds = {}
            for h in hashes:
                merged: list[str] = []
                for label, bucket in (
                    ("gp_finalist", (extra or {}).get(h) or []),
                    ("model_fresh", fresh.get(h) or []),
                    ("model_pool", (pool_idx.get(h) or {}).get("seeds") or []),
                ):
                    for g in bucket:
                        if g and g not in merged:
                            merged.append(g)
                            game_labels[(h, g)] = label
                if merged:
                    model_seeds[h] = merged
        else:
            model_seeds = self._load_model_seeds(hashes) if use_model else {}
        seeds: list[str] = []
        seen: set[str] = set()
        seed_sources: dict[str, list[dict[str, Any]]] = defaultdict(list)
        champion_hits: set[str] = set()
        model_hits: set[str] = set()

        def add(g: str | None, source: dict[str, Any]) -> bool:
            if not g:
                return False
            seed_sources[g].append(source)
            if g not in seen and len(seeds) < self.pop:
                seen.add(g)
                seeds.append(g)
                return True
            return False

        for h in hashes:
            champ = champions.get(h)
            if champ and champ.get("engine") == "featmath":
                if add(
                    champ.get("genome"),
                    {
                        "source": "champion",
                        "dataset_hash": h,
                        "seed_rank": 0,
                        "seed_score": champ.get("score"),
                    },
                ):
                    champion_hits.add(h)
        max_model_depth = max((len(model_seeds.get(h, [])) for h in hashes), default=0)
        for rank in range(max_model_depth):
            if len(seeds) >= self.pop:
                break
            for h in hashes:
                bucket = model_seeds.get(h, [])
                if rank >= len(bucket):
                    continue
                if add(
                    bucket[rank],
                    {
                        "source": game_labels.get((h, bucket[rank]), "model"),
                        "dataset_hash": h,
                        "seed_rank": rank,
                        "seed_score": None,
                    },
                ):
                    model_hits.add(h)
                if len(seeds) >= self.pop:
                    break
        self._last_seed_sources = dict(seed_sources)
        self._last_seed_coverage = {
            "datasets": len(hashes),
            "own_champion": len(champion_hits),
            "own_model_seed": len(model_hits),
            "own_champion_rate": len(champion_hits) / max(1, len(hashes)),
            "own_model_seed_rate": len(model_hits) / max(1, len(hashes)),
        }
        return seeds

    def _ab_arm(self, batch: int) -> bool:
        """A/B: which batches get model seeds. Pseudo-random on the batch number so the arm
        decorrelates from the weakest-champion-first slice ordering (a plain even/odd split
        would confound arm with dataset headroom)."""
        return int(hashlib.sha256(str(batch).encode()).hexdigest(), 16) % 2 == 0

    # -- one slice ----------------------------------------------------------------
    def _write_targets_file(
        self, tag: str, w: int, shard: list[str], targets: dict[str, float]
    ) -> Path | None:
        """Per-shard ``hash target`` lines for jain's --targets-file (game mode)."""
        lines = [f"{h} {targets[h]}" for h in shard if h in targets]
        if not lines:
            return None
        path = self.run_dir / f"targets_{tag}_{w}.txt"
        path.write_text("\n".join(lines) + "\n")
        return path

    def _jain_cmd(
        self, remote_dir: str, shard_file: str, w: int, seeds_name: str,
        jain_seed: int, remote_targets: str | None,
    ) -> str:
        cmd = (
            f"nice -n 5 {JAIN_BIN} jain-regressor --root {GP_WORK_ROOT} "
            f"--hashes-file {shard_file} "
            f"--gens {self.gens} --pop {self.pop} --folds {self.folds} "
            f"--search-rows {self.search_rows} --time-budget-secs {self.time_budget} "
            f"--top-k {self.top_k} --gpu-kernel-configs {self.gpu_kernel_configs} "
            f"--gpu-cache-dir {GPU_CACHE} "
            f"--threads {self.threads} --seed {jain_seed} "
            f"--seed-genomes-file {remote_dir}/{seeds_name} "
            f"--model-name {MODEL_NAME} --overwrite --no-predictions "
            f"--status-jsonl {remote_dir}/status_{w}.jsonl"
        )
        # New jain flags (landing concurrently on the Rust side): appended ONLY when
        # configured so the factory keeps working against the current binary.
        if self.fitness != "v1" or self.game_mode:
            cmd += f" --fitness {self.fitness}"
        if self.game_mode and remote_targets:
            cmd += f" --targets-file {remote_targets}"
        return cmd

    def run_slice(
        self,
        batch: int,
        hashes: list[str],
        champions: dict[str, dict],
        *,
        tag: str | None = None,
        extra: dict[str, list[str]] | None = None,
        targets: dict[str, float] | None = None,
        jain_seed: int | None = None,
    ) -> dict[str, Any]:
        tag = tag or f"b{batch}"
        for dataset_hash in hashes:
            stage_read_only_dataset(
                Path(REG_ROOT) / dataset_hash,
                config=_PRIME_CONFIG,
                destination=GP_WORK_ROOT / dataset_hash,
            )
        remote_dir = f"{_PRIME_CONFIG.remote_scratch_root}/gp_factory/{tag}"
        slice_file = self.run_dir / f"slice_{tag}.txt"
        seeds_file = self.run_dir / f"seeds_{tag}.txt"
        slice_file.write_text("\n".join(hashes) + "\n")
        # A/B alternates model-seeded vs champion-only batches; in game mode model seeding
        # is first-class (the target already prices the model's contribution in).
        use_model = True if self.game_mode else self._ab_arm(batch)
        seeds = self._seed_genomes(hashes, champions, use_model=use_model, batch=batch, extra=extra)
        seeds_file.write_text("\n".join(seeds) + "\n")
        jain_seed = self.seed + batch if jain_seed is None else jain_seed
        # worker pool spans hosts (user ask 2026-07-08: use xbabe3 CPUs too). xbabe3
        # shares cores with DDP rank1 + eval bursts, so it gets fewer workers; jain
        # runs nice-5 so ladder evals win contention.
        # compute-saturation guard: the factory is opportunistic (nice -5), so shed workers
        # on any host that is already loaded (by our eval/miner or the user's own jobs) and
        # skip it entirely when badly oversaturated. Fail-open (None load → full workers).
        from .cluster_monitor import factory_workers
        want = _gp_host_map(self.workers)
        assignments = []
        fallback_host = None
        base_w = 0  # disjoint per-host worker-index blocks: shard/status/log names never collide
        for host, req in want.items():
            got = factory_workers(host, req)
            if got != req:
                self.log(f"[factory {tag}] backoff host={host} → workers {req}→{got}")
            if got > 0 and fallback_host is None:
                fallback_host = host
            assignments += [(host, base_w + w) for w in range(got)]
            base_w += req
        if not assignments:  # all hosts saturated — run a minimal shard on GP_HOST anyway
            self.log(f"[factory {tag}] all hosts saturated; running 1 worker on {GP_HOST}")
            assignments = [(GP_HOST, 0)]
            fallback_host = GP_HOST
        # Split before staging remote workspaces so each host receives only its
        # assigned read-only input symlinks and Prime-owned output directories.
        n_workers = len(assignments)
        shards = [hashes[i::n_workers] for i in range(n_workers)]
        host_hashes: dict[str, set[str]] = {}
        for (host, _), shard in zip(assignments, shards):
            host_hashes.setdefault(host, set()).update(shard)
        for host in {h for h, _ in assignments}:
            subprocess.run(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    host,
                    f"mkdir -p {remote_dir} {GP_WORK_ROOT}",
                ],
                check=True,
            )
            subprocess.run(["rsync", "-a", str(slice_file), str(seeds_file), f"{host}:{remote_dir}/"], check=True)
            for dataset_hash in sorted(host_hashes.get(host, ())):
                remote_dataset = f"{GP_WORK_ROOT}/{dataset_hash}"
                subprocess.run(
                    [
                        "ssh",
                        "-o",
                        "BatchMode=yes",
                        host,
                        f"mkdir -p {remote_dataset}/reports {remote_dataset}/outputs",
                    ],
                    check=True,
                )
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        "--exclude",
                        "reports/",
                        "--exclude",
                        "outputs/",
                        f"{GP_WORK_ROOT / dataset_hash}/",
                        f"{host}:{remote_dataset}/",
                    ],
                    check=True,
                )

        # split the slice across all concurrent jain processes (each rayon-parallel inside)
        shard_host: dict[str, str] = {}
        procs: list[subprocess.Popen] = []
        for (host, w), shard in zip(assignments, shards):
            if not shard:
                continue
            for hh in shard:
                shard_host[hh] = host
            shard_file = f"{remote_dir}/shard_{w}.txt"
            subprocess.run(
                ["ssh", "-o", "BatchMode=yes", host, f"printf '%s\\n' {' '.join(shard)} > {shard_file}"],
                check=True,
            )
            remote_targets = None
            if self.game_mode and targets:
                tf = self._write_targets_file(tag, w, shard, targets)
                if tf is not None:
                    subprocess.run(["rsync", "-a", str(tf), f"{host}:{remote_dir}/"], check=True)
                    remote_targets = f"{remote_dir}/{tf.name}"
            reports = " ".join(
                f"{GP_WORK_ROOT}/{dataset_hash}/reports/{MODEL_NAME}.json"
                for dataset_hash in shard
            )
            cmd = (
                f"rm -f {reports}; "
                + self._jain_cmd(
                    remote_dir,
                    shard_file,
                    w,
                    seeds_file.name,
                    jain_seed,
                    remote_targets,
                )
            )
            log_path = self.run_dir / f"jain_{tag}_w{w}.log"
            procs.append(subprocess.Popen(["ssh", "-o", "BatchMode=yes", host, cmd],
                                          stdout=log_path.open("w"), stderr=subprocess.STDOUT))
        rc = [p.wait() for p in procs]
        seed_source = "game" if self.game_mode else ("model+champion" if use_model else "champion")
        self.log(f"[factory {tag}] jain workers done rc={rc} "
                 f"seed_source={seed_source} n_seeds={len(seeds)} "
                 f"seed_coverage={getattr(self, '_last_seed_coverage', {})}")

        # pull reports back from the host that ran each dataset's shard
        pulled = 0
        for h in hashes:
            # a dropped host may not have run this shard; fall back to whichever host did run
            src = f"{shard_host.get(h, fallback_host or GP_HOST)}:{GP_WORK_ROOT}/{h}/reports/{MODEL_NAME}.json"
            dst = GP_WORK_ROOT / h / "reports"
            subprocess.run(["rsync", "-a", src, f"{dst}/"], capture_output=True, check=True)
            pulled += 1
        return {"rc": rc, "reports_pulled": pulled,
                "seed_source": seed_source, "n_seeds": len(seeds),
                "seed_coverage": getattr(self, "_last_seed_coverage", {})}

    # -- harvest ------------------------------------------------------------------
    def harvest_slice(self, batch: int, hashes: list[str], champions: dict[str, dict]) -> dict[str, Any]:
        rows: list[dict] = []
        ledger_rows: list[dict[str, Any]] = []
        improved = 0
        deltas: list[float] = []
        seed_sources: dict[str, list[dict[str, Any]]] = getattr(self, "_last_seed_sources", {})
        for h in hashes:
            rp = GP_WORK_ROOT / h / "reports" / f"{MODEL_NAME}.json"
            if not rp.exists():
                continue
            try:
                rep = json.loads(rp.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            entry = self.datasets.get(h)
            if entry is None:
                continue
            bar = (champions.get(h) or {}).get("score")
            emb = embedding_from_vs(entry["vs_path"], self.standardization)
            tgt = target_summary_from_vs(entry.get("train_target_vs"))
            candidates = [{
                "genome": rep.get("genome"), "cv": rep.get("cv") or {}, "rank": 0,
            }]
            for i, f in enumerate(rep.get("finalists") or []):
                candidates.append({"genome": f.get("genome"), "cv": f.get("cv") or {}, "rank": i + 1})
            best_new = None
            best_new_genome = None
            for c in candidates:
                oof = (c["cv"] or {}).get("oof_r2")
                genome = c.get("genome")
                if not genome or not isinstance(oof, (int, float)) or not math.isfinite(oof):
                    continue
                score = robust_score(float(oof), (c["cv"] or {}).get("fold_r2"))
                if score is None:
                    continue
                beats = bar is None or score > bar
                if beats and (best_new is None or score > best_new):
                    best_new = score
                    best_new_genome = genome
                rows.append(TrainingRow(
                    dataset_id=h,
                    source="gp_factory",
                    embedding=emb,
                    target_embedding=tgt,
                    genome=genome,
                    canonical_genome=genome,
                    engine="featmath",
                    score=max(float(oof), -1.0),
                    metadata={
                        "provenance": f"gp_factory_b{batch}",
                        "task_type": "regression",
                        "split": "train",
                        "oof_r2_raw": float(oof),
                        "score_robust": score,
                        "rank_pct": 1.0 if beats and c["rank"] == 0 else (0.9 if beats else 0.6),
                        "finalist_rank": c["rank"],
                        "beat_bar": beats,
                        "algorithm": MODEL_NAME,
                    },
                ).to_json())
                for seed in seed_sources.get(genome, []):
                    ledger_rows.append({
                        "gp_batch": batch,
                        "dataset_hash": h,
                        "seed_source": seed.get("source"),
                        "seed_dataset_hash": seed.get("dataset_hash"),
                        "seed_rank": seed.get("seed_rank"),
                        "seed_score": seed.get("seed_score"),
                        "seed_genome": genome,
                        "descendant_type": "finalist",
                        "finalist_rank": c["rank"],
                        "descendant_genome": genome,
                        "descendant_robust_score": score,
                        "delta": score - bar if bar is not None else None,
                        "cost_ms": (rep.get("timings") or {}).get("total_ms"),
                        "exact_seed_match": True,
                    })
            for row in ledger_rows:
                if row.get("dataset_hash") == h and best_new_genome is not None and row.get("descendant_genome") == best_new_genome:
                    row["descendant_type"] = "champion"
            if best_new is not None:
                improved += 1
                if bar is not None:
                    deltas.append(best_new - bar)
                champions[h] = {"genome": best_new_genome, "engine": "featmath", "score": best_new}
        out = self.art_dir / f"champion_rows_gp{batch}.jsonl"
        with out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        if ledger_rows:
            ledger = self.run_dir / "seed_attribution_ledger.jsonl"
            with ledger.open("a", encoding="utf-8") as fh:
                for row in ledger_rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
        return {
            "rows": len(rows), "improved_datasets": improved,
            "mean_delta": round(sum(deltas) / len(deltas), 5) if deltas else None,
            "seed_attribution_rows": len(ledger_rows),
            "out": str(out),
        }

    # -- game mode ------------------------------------------------------------------
    def _report_outcome(self, h: str) -> dict[str, Any] | None:
        """Best robust score + top finalist genomes from a dataset's pulled-back report.
        Game-mode outcome check only — harvest logic stays untouched."""
        rp = GP_WORK_ROOT / h / "reports" / f"{MODEL_NAME}.json"
        if not rp.exists():
            return None
        try:
            rep = json.loads(rp.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        cands = [{"genome": rep.get("genome"), "cv": rep.get("cv") or {}}]
        cands += [{"genome": f.get("genome"), "cv": f.get("cv") or {}} for f in rep.get("finalists") or []]
        scored: list[tuple[float, str]] = []
        for c in cands:
            oof = (c["cv"] or {}).get("oof_r2")
            g = c.get("genome")
            if not g or not isinstance(oof, (int, float)) or not math.isfinite(oof):
                continue
            score = robust_score(float(oof), (c["cv"] or {}).get("fold_r2"))
            if score is not None:
                scored.append((score, g))
        scored.sort(key=lambda t: t[0], reverse=True)
        top: list[str] = []
        for _, g in scored:
            if g not in top:
                top.append(g)
        return {
            "best": scored[0][0] if scored else None,
            "target_beaten": rep.get("target_beaten"),  # jain --fitness/--targets contract
            "target": rep.get("target"),
            "top_genomes": top[:3],
        }

    def _run_game_batch(self, batch: int, hashes: list[str], champions: dict[str, dict]) -> dict[str, Any]:
        """Iterate-to-beat: per-dataset target = max(champion bar, model bar); unbeaten
        datasets re-enter the next round (same batch, reseeded gen-0 + the round's top
        finalists) until beaten or ``game_rounds`` exhausted. Harvest still runs every
        round — the training payoff flows regardless of who wins the game."""
        pool_idx = self._model_pool_index()
        per_ds: dict[str, dict[str, Any]] = {}
        targets: dict[str, float] = {}
        for h in hashes:
            cb = (champions.get(h) or {}).get("score")
            mb = (pool_idx.get(h) or {}).get("model_bar")
            finite = [float(v) for v in (cb, mb) if isinstance(v, (int, float)) and math.isfinite(v)]
            per_ds[h] = {"champion_bar": cb, "model_bar": mb, "target": max(finite) if finite else None}
            if finite:
                targets[h] = max(finite)
        remaining = list(hashes)
        extra: dict[str, list[str]] = {}
        beaten_round: dict[str, int] = {}
        rows_lines: list[str] = []
        seen_rows: set[tuple[Any, Any]] = set()
        improved_total = 0
        out_path: Path | None = None
        ledger_path = self.run_dir / "game_ledger.jsonl"
        rounds_used = 0
        for r in range(1, self.game_rounds + 1):
            rounds_used = r
            t0 = time.time()
            self.state.update({"phase": f"GAME r{r}", "slice": len(remaining), "ts": time.time()})
            _atomic_json(self.state_path, self.state)
            run_info = self.run_slice(
                batch, remaining, champions, tag=f"b{batch}r{r}",
                extra=extra or None, targets=targets,
                jain_seed=self.seed + batch * 10 + r,
            )
            h_info = self.harvest_slice(batch, remaining, champions)
            improved_total += h_info["improved_datasets"]
            # rounds share champion_rows_gp{batch}.jsonl (harvest overwrites it) — fold
            # each round's rows into one deduped payload, rewritten after the game.
            out_path = Path(h_info["out"])
            if out_path.exists():
                for line in out_path.read_text(encoding="utf-8").splitlines():
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    key = (row.get("dataset_id"), row.get("genome"))
                    if key not in seen_rows:
                        seen_rows.add(key)
                        rows_lines.append(line)
            wall = round(time.time() - t0, 1)
            still: list[str] = []
            extra = {}
            ts = time.time()
            with ledger_path.open("a", encoding="utf-8") as fh:
                for h in remaining:
                    outcome = self._report_outcome(h) or {}
                    best = outcome.get("best")
                    target = per_ds[h]["target"]
                    if outcome.get("target_beaten"):
                        beaten = True
                    elif target is None:
                        beaten = best is not None  # no bar to beat: any measured result wins
                    else:
                        beaten = best is not None and best >= target
                    fh.write(json.dumps({
                        "batch": batch, "round": r, "dataset": h,
                        "model_bar": per_ds[h]["model_bar"],
                        "champion_bar": per_ds[h]["champion_bar"],
                        "target": target, "gp_best_score": best,
                        "beaten": beaten, "wall_s": wall, "ts": ts,
                    }, sort_keys=True) + "\n")
                    if beaten:
                        beaten_round[h] = r
                    else:
                        still.append(h)
                        if tops := outcome.get("top_genomes"):
                            extra[h] = tops[:3]  # reseed next round with this round's best
            self.log(f"[factory b{batch} r{r}/{self.game_rounds}] "
                     f"beat {len(remaining) - len(still)}/{len(remaining)} targets, "
                     f"{len(still)} carry over rc={run_info.get('rc')} secs={wall}")
            remaining = still
            if not remaining:
                break
        if out_path is not None:
            with out_path.open("w", encoding="utf-8") as fh:
                for line in rows_lines:
                    fh.write(line + "\n")
        hist: dict[str, int] = {}
        for h in hashes:
            key = str(beaten_round[h]) if h in beaten_round else "lost"
            hist[key] = hist.get(key, 0) + 1
        return {
            "rows": len(rows_lines),
            "improved_datasets": improved_total,
            "games_played": len(hashes),
            "games_won": len(beaten_round),
            "rounds_used": rounds_used,
            "rounds_histogram": hist,
            "unbeaten": remaining,
            "out": str(out_path) if out_path else None,
        }

    # -- main loop ------------------------------------------------------------------
    def run(self, passes: int = 1, max_batches: int | None = None) -> None:
        # Champions come from the fast corpus + champion_rows pool (globbing all 17 self_rows
        # files here would parse ~7GB of embeddings at startup). Fresh model proposals are
        # injected separately via the cached _model_seed_index (last few gens only).
        champions = build_champions(self.art_dir, log=self.log)
        # Mine weakest-champion datasets FIRST: lowest current best score = most headroom.
        # (The GP's marginal value is highest where the known bar is lowest.)
        hashes_all = sorted(
            self.datasets,
            key=lambda h: (champions.get(h) or {}).get("score", float("-inf")),
        )
        batches_done = 0
        for _ in range(passes):
            cursor = self.state.get("cursor", 0)
            while cursor < len(hashes_all):
                if max_batches is not None and batches_done >= max_batches:
                    return
                batch = self.state["batch"] + 1
                hashes = hashes_all[cursor : cursor + self.slice_size]
                t0 = time.time()
                self.state.update({"batch": batch, "cursor": cursor, "phase": "GP",
                                   "slice": len(hashes), "ts": time.time()})
                _atomic_json(self.state_path, self.state)
                if self.game_mode:
                    info = self._run_game_batch(batch, hashes, champions)
                    cursor += len(hashes)
                    hist = self.state.setdefault("game_rounds_histogram", {})
                    for k, v in info["rounds_histogram"].items():
                        hist[k] = hist.get(k, 0) + v
                    self.state.update({
                        "cursor": cursor, "phase": "IDLE",
                        "harvested_rows": self.state.get("harvested_rows", 0) + info["rows"],
                        "improved_datasets": self.state.get("improved_datasets", 0) + info["improved_datasets"],
                        "game_batches": self.state.get("game_batches", 0) + 1,
                        "games_played": self.state.get("games_played", 0) + info["games_played"],
                        "games_won": self.state.get("games_won", 0) + info["games_won"],
                        "last_batch": {**info, "secs": round(time.time() - t0, 1)},
                        "ts": time.time(),
                    })
                    _atomic_json(self.state_path, self.state)
                    self.log(f"[factory b{batch} game] won={info['games_won']}/{info['games_played']} "
                             f"rounds={json.dumps(info['rounds_histogram'], sort_keys=True)} "
                             f"rows={info['rows']} improved={info['improved_datasets']} "
                             f"secs={round(time.time() - t0, 1)}")
                    batches_done += 1
                    continue
                run_info = self.run_slice(batch, hashes, champions)
                self.state.update({"phase": "HARVEST", "ts": time.time()})
                _atomic_json(self.state_path, self.state)
                h_info = self.harvest_slice(batch, hashes, champions)
                cursor += len(hashes)
                # A/B tally: does seeding gen-0 with the model's fresh proposals raise the
                # per-batch improved-datasets rate vs champion-only seeding? (the merit gate)
                arm = "ab_model" if run_info.get("seed_source", "").startswith("model") else "ab_control"
                tally = self.state.setdefault(arm, {"batches": 0, "improved": 0, "delta_sum": 0.0, "datasets": 0})
                tally["batches"] += 1
                tally["improved"] += h_info["improved_datasets"]
                tally["delta_sum"] += float(h_info.get("mean_delta") or 0.0)
                tally["datasets"] += len(hashes)
                self.state.update({
                    "cursor": cursor, "phase": "IDLE",
                    "harvested_rows": self.state.get("harvested_rows", 0) + h_info["rows"],
                    "improved_datasets": self.state.get("improved_datasets", 0) + h_info["improved_datasets"],
                    "last_batch": {**run_info, **h_info, "secs": round(time.time() - t0, 1)},
                    "ts": time.time(),
                })
                _atomic_json(self.state_path, self.state)
                self.log(f"[factory b{batch}] {json.dumps({**run_info, **h_info, 'secs': round(time.time()-t0,1)})}")
                batches_done += 1
            self.state["cursor"] = 0  # next pass restarts the rotation
            _atomic_json(self.state_path, self.state)
