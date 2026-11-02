"""Training and proposal phases for self-play generations."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from collections import defaultdict
import json
import math
import subprocess
import time
import torch

from .self_play import (
    EngineRouter, MASTER_ADDR, NCCL_ENV, PROJ, TrainingRow, _atomic_json,
    _candidate_pool_id, _candidate_pool_record, _proposal_failure_exceeded,
    deployment_elo, embedding_from_vs, read_jsonl, target_summary_from_vs,
)

class SelfPlayTrainingMixin:
    def __init__(
        self,
        art_dir: str | Path,
        run_dir: str | Path | None = None,
        *,
        train_hosts: tuple[str, ...] = ("local", "xbabe3"),  # rank0 local (xbabe2), rank1 ssh
        eval_workers: int = 90,
        explore: int = 3,
        folds: int = 3,
        epochs: int = 14,
        batch_size: int = 512,
        d_model: int = 256,
        n_layers: int = 2,
        ffn_dim: int = 1024,
        dataset_limit: int | None = None,
        elo_every: int = 1,
        gate_beat_rate: float = 0.55,
        miner_slice: int = 400,  # datasets mined per generation (0 disables)
        lr: float = 5e-4,
        warm_lr: float | None = 1.5e-4,  # refinement LR for warm-started generations:
        # full-LR retrains churned the champion's reward head every generation
        # (conditional_advantage decayed 0.068->0.011 across ladder #1)
        selector_calibration: str | Path | None = None,
        selector_reranker: str | Path | None = None,
        samples_per_engine: int | None = None,
        candidate_pool_dir: str | Path | None = None,
        proposal_failure_threshold: float | None = 0.01,
        seed: int = 0,
        log=print,
    ) -> None:
        self.miner_slice = miner_slice
        self.lr = lr
        self.warm_lr = warm_lr
        # Absolute paths: art_dir is rsync'd to the same location on remote hosts, and a
        # relative path would resolve against the remote $HOME (rsync mkdir failure).
        self.art_dir = Path(art_dir).resolve()
        if run_dir is None:
            from .prime_config import load_config

            run_dir = load_config().runs / "self_play"
        self.run_dir = Path(run_dir).resolve()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.train_hosts = train_hosts
        self.eval_workers = eval_workers
        self.explore = explore
        self.folds = folds
        self.epochs = epochs
        self.batch_size = batch_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.elo_every = elo_every
        self.gate_beat_rate = gate_beat_rate
        self.selector_calibration = selector_calibration
        self.selector_reranker = selector_reranker
        self.samples_per_engine = samples_per_engine
        self.candidate_pool_dir = Path(candidate_pool_dir).resolve() if candidate_pool_dir else None
        if self.candidate_pool_dir is not None:
            self.candidate_pool_dir.mkdir(parents=True, exist_ok=True)
        self.proposal_failure_threshold = proposal_failure_threshold
        self.seed = seed
        self.log = log

        manifest = json.loads((self.art_dir / "split_manifest.json").read_text())["datasets"]
        self.datasets = {h: e for h, e in sorted(manifest.items()) if e.get("split") == "train"}
        if dataset_limit:
            self.datasets = dict(list(self.datasets.items())[:dataset_limit])
        self.standardization = json.loads((self.art_dir / "standardization.json").read_text())

        self.state_path = self.run_dir / "self_play_state.json"
        self.metrics_path = self.run_dir / "self_play_metrics.jsonl"
        # live gene stream for the TUI: proposals/evals/champions scroll in real time
        from .gene_feed import GeneFeed

        self.gene_feed = GeneFeed(self.run_dir / "gene_events.jsonl", run_id="self_play")
        self.state: dict[str, Any] = (
            json.loads(self.state_path.read_text())
            if self.state_path.exists()
            else {"generation": 0, "phase": "idle", "self_best": {}, "gen_top1": {}, "elo_history": []}
        )
        # per-dataset historical score pools (for rank_pct of new rows) — built lazily
        self._score_pool: dict[str, list[float]] | None = None

    # -- state/telemetry -------------------------------------------------------
    def _set_phase(self, phase: str, **extra) -> None:
        self.state["phase"] = phase
        self.state["phase_started"] = time.time()
        # reset per-phase counters so the TUI never shows the previous phase's progress
        for k in ("phase_done", "phase_total", "evals_per_sec"):
            self.state.pop(k, None)
        self.state.update(extra)
        _atomic_json(self.state_path, self.state)
        self.log(f"[gen {self.state['generation']}] {phase} {extra if extra else ''}", flush=True)

    def _progress(self, done: int, total: int) -> None:
        self.state["phase_done"] = done
        self.state["phase_total"] = total
        _atomic_json(self.state_path, self.state)

    # -- score pools -----------------------------------------------------------
    def _build_score_pool(self) -> dict[str, list[float]]:
        if self._score_pool is not None:
            return self._score_pool
        pool: dict[str, list[float]] = defaultdict(list)
        paths = [
            self.art_dir / "active_rows.jsonl",
            *sorted(self.art_dir.glob("self_rows_g*.jsonl")),
            *sorted(self.art_dir.glob(f"self_rows_{self.run_dir.name}_g*.jsonl")),
            *sorted(self.art_dir.glob("self_rows_*_g*.jsonl")),
            *sorted(self.art_dir.glob("champion_rows_g*.jsonl")),  # miner + GP factory
        ]
        seen_paths: set[Path] = set()
        for path in paths:
            if path in seen_paths:
                continue
            seen_paths.add(path)
            if not path.exists():
                continue
            for r in read_jsonl(path):
                md = r.get("metadata") or {}
                s = md.get("score_robust", r.get("score"))
                try:
                    s = float(s)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(s):
                    pool[str(r.get("dataset_id"))].append(s)
        for v in pool.values():
            v.sort()
        self._score_pool = dict(pool)
        return self._score_pool

    def _rank_pct(self, dataset: str, score: float) -> float:
        vals = self._build_score_pool().get(dataset) or []
        if not vals:
            return 0.5
        import bisect

        lo = bisect.bisect_left(vals, score)
        hi = bisect.bisect_right(vals, score)
        n = len(vals)
        return min(max((lo + 0.5 * (hi - lo)) / n, 0.0), 1.0)

    # -- phases ------------------------------------------------------------------
    def _launch_miner(self, g: int) -> subprocess.Popen | None:
        """Champion miner runs on the CPUs WHILE the GPUs train — zero idle resources.

        Each generation mines a rotating slice of train datasets: grammar-aware mutants
        of every champion race on the cluster; improvements become champion_rows_g<g>
        (imitation targets for the NEXT generation).
        """
        if self.miner_slice <= 0:
            return None
        hashes = list(self.datasets)
        size = min(self.miner_slice, len(hashes))
        start = (g * size) % max(1, len(hashes))
        chunk = hashes[start : start + size]
        if len(chunk) < size:  # wrap around
            chunk += hashes[: size - len(chunk)]
        slice_file = self.run_dir / f"miner_slice_g{g}.json"
        slice_file.write_text(json.dumps(chunk))
        cmd = (
            # spread miner evals across BOTH hosts' cores (local-only leaves xbabe3 idle)
            f"cd {PROJ} && JOPE_EVAL_HOSTS=local,xbabe3 PYTHONPATH=src "
            f"python3 -m jope.cli mine-champions "
            f"--art-dir {self.art_dir} --generation {g} --slice-file {slice_file} "
            f"--out {self.art_dir}/champion_rows_g{g}.jsonl "
            f"--workers {self.eval_workers} --ledger {self.run_dir}/miner_ledger.jsonl "
            f"--progress-file {self.run_dir}/miner_progress.json "
            f"--gene-feed {self.run_dir}/gene_events.jsonl"
        )
        log_path = self.run_dir / f"miner_g{g}.log"
        return subprocess.Popen(["bash", "-lc", cmd], stdout=log_path.open("w"), stderr=subprocess.STDOUT)

    def _best_checkpoint(self) -> Path | None:
        """Checkpoint of the highest-Elo former generation — the warm-start ancestor.

        Fresh-init-per-generation made quality a lottery (gen 3 drew −97 Elo); warm
        starting from the reigning champion turns generations into monotone refinement
        while the gate still enforces 'must beat the former self' on real evals.
        """
        history = [h for h in self.state.get("elo_history", []) if deployment_elo(h) is not None]
        # governance (V7): the gate is law — only SHIPPED generations may become the
        # warm-start ancestor (missing key = legacy/seed entry, treated as shipped).
        shipped = [h for h in history if h.get("shipped", True)]
        history = shipped or history
        if not history:
            return None
        best = max(history, key=lambda h: deployment_elo(h) or float("-inf"))
        ckpt = self.run_dir / f"gen{best['gen']}.pt"
        return ckpt if ckpt.exists() else None

    def train_generation(self, g: int) -> Path:
        ckpt = self.run_dir / f"gen{g}.pt"
        # RESUME: if this generation's training already completed (checkpoint + full
        # metrics), skip it — a mid-PROPOSE/EVAL restart must not redo a 45-min train.
        metrics = self.run_dir / f"gen{g}_train_metrics.jsonl"
        if ckpt.exists() and metrics.exists():
            rows = [l for l in metrics.read_text().splitlines() if l.strip()]
            if len(rows) >= self.epochs:
                self.log(f"[gen {g}] training already complete ({len(rows)} epochs) — skipping TRAIN")
                return ckpt
        # Corpus growth cap: ALL champion rows (monotone bars) but only the LAST 4
        # generations' self rows — older picks are dominated, and unbounded growth was
        # pushing per-generation startup (JSONL parse + artifact precompute) toward
        # unusable by gen 30 (~8k rows/gen).
        import re as _re

        self_files = sorted(
            self.art_dir.glob(f"self_rows_{self.run_dir.name}_g*.jsonl"),
            key=lambda p: int(_re.search(r"_g(\d+)", p.name).group(1)),
        )
        extra = [str(p) for p in self_files[-4:]]
        extra += sorted(str(p) for p in self.art_dir.glob("champion_rows_g*.jsonl"))
        warm = self._best_checkpoint()
        metrics = self.run_dir / f"gen{g}_train_metrics.jsonl"
        progress = self.run_dir / "train_progress.json"
        for p in (metrics,):
            p.unlink(missing_ok=True)
        args = (
            f"-m jope.cli train-policy --corpus {self.art_dir}/active_rows.jsonl "
            f"--checkpoint {ckpt} --epochs {self.epochs} --batch-size {self.batch_size} "
            f"--lr {(self.warm_lr if (warm is not None and self.warm_lr) else self.lr)} "
            f"--device cuda --d-model {self.d_model} --n-layers {self.n_layers} "
            f"--ffn-dim {self.ffn_dim} --amp --seed {self.seed + g} "
            f"--metrics-jsonl {metrics} --progress-jsonl {progress} "
            f"--greedy-eval-datasets 0 "
            + (f"--extra-rows {' '.join(extra)} " if extra else "")
            + (f"--init-from {warm}" if warm is not None else "")
        )
        nnodes = len(self.train_hosts)
        # Single-node training must rendezvous on localhost: the multi-node MASTER_ADDR
        # is xbabe2's IP, which cannot be bound on another host (torchrun hangs forever
        # at 0% CPU waiting for it — observed on xbabe3 after the migration).
        master = "127.0.0.1" if nnodes == 1 else MASTER_ADDR
        port = 29650 + (g % 40)
        procs = []
        for rank, host in enumerate(self.train_hosts):
            tr = (
                f"torchrun --no-python --nnodes={nnodes} --nproc_per_node=1 --node_rank={rank} "
                f"--master_addr={master} --master_port={port} python3 {args}"
            )
            log_path = self.run_dir / f"gen{g}_rank{rank}.log"
            if host == "local":
                cmd = f"cd {PROJ} && env {NCCL_ENV} {tr}"
                procs.append(
                    subprocess.Popen(
                        ["bash", "-lc", cmd], stdout=log_path.open("w"), stderr=subprocess.STDOUT
                    )
                )
            else:
                # ship fresh code + corpus labels + self rows before remote training
                subprocess.run(
                    [
                        "rsync",
                        "-a",
                        "--exclude",
                        "__pycache__",
                        f"{PROJ}/src/jope/",
                        f"{host}:{PROJ}/src/jope/",
                    ],
                    check=True,
                )
                subprocess.run(["rsync", "-a", f"{self.art_dir}/", f"{host}:{self.art_dir}/"], check=True)
                # warm-start checkpoint lives in run_dir on THIS host only — every rank
                # loads it locally, so it must exist at the same path on the remote
                # (rank1 FileNotFoundError -> dead rank -> NCCL hang, observed gen 4).
                if warm is not None:
                    subprocess.run(
                        ["ssh", "-o", "BatchMode=yes", host, f"mkdir -p {self.run_dir}"], check=True
                    )
                    subprocess.run(["rsync", "-a", str(warm), f"{host}:{warm}"], check=True)
                cmd = f"cd {PROJ} && env {NCCL_ENV} {tr}"
                procs.append(
                    subprocess.Popen(
                        ["ssh", "-o", "BatchMode=yes", host, cmd],
                        stdout=log_path.open("w"),
                        stderr=subprocess.STDOUT,
                    )
                )
        rc = [p.wait() for p in procs]
        if any(rc) or not ckpt.exists():
            raise RuntimeError(f"gen {g} training failed rc={rc}; see {self.run_dir}/gen{g}_rank*.log")
        return ckpt

    def propose_generation(
        self, g: int, ckpt: Path
    ) -> tuple[dict[str, list[dict]], list[dict], list[dict], dict[str, Any]]:
        from .policy.decode import propose_v3
        from .policy.model_v3 import load_policy_v3

        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = load_policy_v3(state)
        # Explicit CUDA attempt with loud telemetry: a long-lived orchestrator process was
        # observed with a wedged CUDA context (is_available()==False while fresh processes
        # saw the GPU) → proposals silently crawled on CPU at ~150x cost.
        device = "cpu"
        try:
            model = model.to("cuda")
            device = "cuda"
        except Exception as exc:  # noqa: BLE001
            self.log(f"[gen {g}] WARNING: propose on CPU (cuda unavailable: {exc})")
        self.state["propose_device"] = device
        _atomic_json(self.state_path, self.state)
        router = EngineRouter()
        picks: dict[str, list[dict]] = {}
        eval_inputs: list[dict] = []
        candidate_pool_rows: list[dict] = []
        candidate_pool_lookup: dict[tuple[str, str, str], dict] = {}
        proposal_errors: list[dict[str, Any]] = []
        hashes = list(self.datasets)
        n = 1 + self.explore
        for i, h in enumerate(hashes):
            entry = self.datasets[h]
            emb = embedding_from_vs(entry["vs_path"], self.standardization)
            tgt = target_summary_from_vs(entry.get("train_target_vs"))
            try:
                result = propose_v3(
                    emb,
                    target_summary=tgt,
                    model=model,
                    router=router,
                    n=n,
                    seed=self.seed + g * 7919 + i,
                    temperature=0.7,
                    engines=("featmath", "newmath"),
                    device=device,
                    samples_per_engine=self.samples_per_engine,
                    selector_calibration=self.selector_calibration,
                    selector_reranker=self.selector_reranker,
                    return_all_candidates=self.candidate_pool_dir is not None,
                    gene_feed=self.gene_feed,
                    gene_feed_dataset=h,
                )
                if self.candidate_pool_dir is not None:
                    props, all_candidates = result
                    for cand in all_candidates:
                        canonical = cand.get("canonical_genome") or cand.get("genome")
                        engine = str(cand.get("engine") or "")
                        if not canonical or not engine:
                            continue
                        cpid = _candidate_pool_id(g, h, engine, str(canonical))
                        row = _candidate_pool_record(
                            generation=g,
                            candidate_pool_id=cpid,
                            policy_checkpoint=str(ckpt),
                            dataset_hash=h,
                            row=cand,
                        )
                        candidate_pool_rows.append(row)
                        candidate_pool_lookup[(h, engine, str(canonical))] = row
                else:
                    props = result
            except Exception as exc:  # noqa: BLE001
                proposal_errors.append({"generation": g, "dataset_hash": h, "error": repr(exc)})
                continue
            if not props:
                proposal_errors.append({"generation": g, "dataset_hash": h, "error": "empty_proposals"})
                continue
            picks[h] = []
            for rank, p in enumerate(props[:n]):
                genome = p.get("canonical_genome") or p.get("genome")
                engine = str(p.get("engine") or "")
                pool_row = candidate_pool_lookup.get((h, engine, str(genome)))
                candidate_pool_id = pool_row.get("candidate_pool_id") if pool_row else None
                if pool_row is not None:
                    pool_row["selected_for_eval_rank"] = rank
                rec = {
                    "rank": rank,
                    "engine": p.get("engine"),
                    "genome": genome,
                    "deployed": rank == 0,
                    "candidate_pool_id": candidate_pool_id,
                    "reward": p.get("reward"),
                    "reward_std": p.get("reward_std"),
                    "reward_percentile_in_engine": p.get("reward_percentile_in_engine"),
                    "p_bad": p.get("p_bad"),
                    "selection_key": p.get("selection_key"),
                    "selection_score": p.get("selection_score"),
                    "selector_reranker_score": p.get("selector_reranker_score"),
                }
                picks[h].append(rec)
                eval_inputs.append(
                    {
                        "engine": p.get("engine"),
                        "dataset_hash": h,
                        "train_csv": entry["train_csv"],
                        "test_csv": entry.get("test_csv"),
                        "genome": genome,
                        "folds": self.folds,
                        "_h": h,
                        "_rank": rank,
                        "_candidate_pool_id": candidate_pool_id,
                    }
                )
            if (i + 1) % 10 == 0 or i + 1 == len(hashes):
                self._progress(i + 1, len(hashes))
        if proposal_errors:
            err_path = self.run_dir / f"proposal_errors_g{g}.jsonl"
            with err_path.open("w", encoding="utf-8") as fh:
                for row in proposal_errors:
                    fh.write(json.dumps(row) + "\n")
        failure_rate = len(proposal_errors) / max(1, len(hashes))
        stats = {
            "proposal_failures": len(proposal_errors),
            "proposal_failure_rate": failure_rate,
            "proposal_error_path": str(self.run_dir / f"proposal_errors_g{g}.jsonl")
            if proposal_errors
            else None,
        }
        if _proposal_failure_exceeded(len(proposal_errors), len(hashes), self.proposal_failure_threshold):
            raise RuntimeError(
                f"gen {g} proposal coverage below threshold: failure_rate={failure_rate:.4f} "
                f"threshold={self.proposal_failure_threshold}"
            )
        return picks, eval_inputs, candidate_pool_rows, stats

