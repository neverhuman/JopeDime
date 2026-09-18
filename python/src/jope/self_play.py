"""Self-play generation loop: each new model must beat ALL former model attempts.

Per generation g:
  TRAIN    2-node DDP (xbabe2 rank0 local + xbabe3 rank1 ssh) on corpus ∪ self_rows_g*
  PROPOSE  new checkpoint picks per train dataset: 1 deployed top-1 + E exploration samples
  EVAL     cluster CPU eval of all picks (ledger-cached)
  APPEND   rows provenance="self_g<g>"; per-dataset self-best bar rises monotonically
  GATE     beat_rate vs generation g-1 (ship iff ≥ threshold), new-bests, mean rank-pct
  BENCH    elo-bench top-1 on the held-out validation ladder

State/telemetry (TUI-tailable, atomic):
  <run_dir>/self_play_state.json     current gen + phase + progress
  <run_dir>/self_play_metrics.jsonl  one row per generation (the learning curve)
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .data.active_v3 import robust_score
from .engines import EngineRouter
from .eval_ledger import iter_eval_regression_with_ledger
from .io import read_jsonl
from .prime_config import load_config
from .propose import embedding_from_vs, target_summary_from_vs
from .schema import TrainingRow

PROJ = load_config().root
NCCL_ENV = (
    "CUDA_VISIBLE_DEVICES=0 NCCL_SOCKET_IFNAME=enp37s0f1 GLOO_SOCKET_IFNAME=enp37s0f1 "
    "NCCL_IB_DISABLE=1 NCCL_NET=Socket TORCH_NCCL_ASYNC_ERROR_HANDLING=1 "
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
    "OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 PYTHONPATH=src"
)
MASTER_ADDR = "192.168.68.87"  # xbabe2 (this host)


def _atomic_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def deployment_elo(row: dict[str, Any]) -> float | None:
    """Canonical deployed Elo: cheap-probe first, then CV-select, then legacy top1."""
    for key in ("elo_probe", "elo_cv", "elo_top1"):
        value = _finite_float(row.get(key))
        if value is not None:
            return value
    return None


def _resolve_ship_decision(gate_shipped: bool, elo: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, Any]:
    prior = [
        deployment_elo(h) for h in history
        if h.get("shipped") and deployment_elo(h) is not None
    ]
    cur = deployment_elo(elo)
    baseline = max(prior) if prior else None
    elo_record = bool(cur is not None and (baseline is None or cur >= baseline))
    elo_regressed = bool(cur is not None and baseline is not None and cur < baseline)
    return {
        "shipped": (bool(gate_shipped) and not elo_regressed) or elo_record,
        "elo_record": elo_record,
        "deployed_elo_baseline": baseline,
        "deployed_elo_delta": (cur - baseline) if cur is not None and baseline is not None else None,
        "deployed_elo_regressed": elo_regressed,
    }


def _choose_deployed_candidate(recs: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [r for r in recs if _finite_float(r.get("score_robust")) is not None]
    if not scored:
        return None
    return max(scored, key=lambda r: (_finite_float(r.get("score_robust")) or float("-inf"), _finite_float(r.get("_oof")) or float("-inf")))


def _proposal_failure_exceeded(failed: int, total: int, threshold: float | None) -> bool:
    if total <= 0 or threshold is None:
        return False
    return failed / max(1, total) > float(threshold)


def _candidate_pool_id(generation: int, dataset_hash: str, engine: str, canonical: str) -> str:
    raw = f"{generation}|{dataset_hash}|{engine}|{canonical}".encode()
    return hashlib.sha1(raw).hexdigest()[:16]


def _candidate_pool_record(
    *,
    generation: int,
    candidate_pool_id: str,
    policy_checkpoint: str,
    dataset_hash: str,
    row: dict[str, Any],
) -> dict[str, Any]:
    reward_mean = _finite_float(row.get("reward_mean", row.get("reward")))
    reward_std = _finite_float(row.get("reward_std"))
    reward_lcb = _finite_float(row.get("reward_lcb"))
    if reward_lcb is None and reward_mean is not None and reward_std is not None:
        reward_lcb = reward_mean - reward_std
    final_key = row.get("selector_reranker_score")
    if final_key is None:
        final_key = row.get("selection_key")
    return {
        "generation": generation,
        "candidate_pool_id": candidate_pool_id,
        "policy_checkpoint": policy_checkpoint,
        "dataset_hash": dataset_hash,
        "engine": row.get("engine"),
        "canonical_genome": row.get("canonical_genome") or row.get("genome"),
        "decoder": row.get("decoder"),
        "reward_mean": reward_mean,
        "reward_std": reward_std,
        "reward_lcb": reward_lcb,
        "reward_percentile_in_engine": row.get("reward_percentile_in_engine"),
        "p_bad": row.get("p_bad"),
        "candidate_risk_reason": row.get("candidate_risk_reason"),
        "family_bitmap": row.get("family_bitmap"),
        "head_type": row.get("head_type"),
        "candidate_diversity_key": row.get("candidate_diversity_key"),
        "selection_key": row.get("selection_key"),
        "selection_score": row.get("selection_score"),
        "selector_reranker_score": row.get("selector_reranker_score"),
        "final_selection_key": final_key,
        "selected_for_eval_rank": None,
        "measured_oof": None,
        "score_robust": None,
        "cost_ms": None,
        "deployed": False,
    }


class SelfPlay:
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
        self.datasets = {
            h: e for h, e in sorted(manifest.items()) if e.get("split") == "train"
        }
        if dataset_limit:
            self.datasets = dict(list(self.datasets.items())[:dataset_limit])
        self.standardization = json.loads((self.art_dir / "standardization.json").read_text())

        self.state_path = self.run_dir / "self_play_state.json"
        self.metrics_path = self.run_dir / "self_play_metrics.jsonl"
        # live gene stream for the TUI: proposals/evals/champions scroll in real time
        from .gene_feed import GeneFeed

        self.gene_feed = GeneFeed(self.run_dir / "gene_events.jsonl", run_id="self_play")
        self.state: dict[str, Any] = (
            json.loads(self.state_path.read_text()) if self.state_path.exists() else
            {"generation": 0, "phase": "idle", "self_best": {}, "gen_top1": {}, "elo_history": []}
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
        chunk = hashes[start:start + size]
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
        return subprocess.Popen(["bash", "-lc", cmd], stdout=log_path.open("w"),
                                stderr=subprocess.STDOUT)

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
                procs.append(subprocess.Popen(["bash", "-lc", cmd], stdout=log_path.open("w"),
                                              stderr=subprocess.STDOUT))
            else:
                # ship fresh code + corpus labels + self rows before remote training
                subprocess.run(["rsync", "-a", "--exclude", "__pycache__",
                                f"{PROJ}/src/jope/", f"{host}:{PROJ}/src/jope/"], check=True)
                subprocess.run(["rsync", "-a", f"{self.art_dir}/",
                                f"{host}:{self.art_dir}/"], check=True)
                # warm-start checkpoint lives in run_dir on THIS host only — every rank
                # loads it locally, so it must exist at the same path on the remote
                # (rank1 FileNotFoundError -> dead rank -> NCCL hang, observed gen 4).
                if warm is not None:
                    subprocess.run(["ssh", "-o", "BatchMode=yes", host,
                                    f"mkdir -p {self.run_dir}"], check=True)
                    subprocess.run(["rsync", "-a", str(warm), f"{host}:{warm}"], check=True)
                cmd = f"cd {PROJ} && env {NCCL_ENV} {tr}"
                procs.append(subprocess.Popen(["ssh", "-o", "BatchMode=yes", host, cmd],
                                              stdout=log_path.open("w"), stderr=subprocess.STDOUT))
        rc = [p.wait() for p in procs]
        if any(rc) or not ckpt.exists():
            raise RuntimeError(f"gen {g} training failed rc={rc}; see {self.run_dir}/gen{g}_rank*.log")
        return ckpt

    def propose_generation(self, g: int, ckpt: Path) -> tuple[dict[str, list[dict]], list[dict], list[dict], dict[str, Any]]:
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
                    emb, target_summary=tgt, model=model, router=router, n=n,
                    seed=self.seed + g * 7919 + i, temperature=0.7,
                    engines=("featmath", "newmath"), device=device,
                    samples_per_engine=self.samples_per_engine,
                    selector_calibration=self.selector_calibration,
                    selector_reranker=self.selector_reranker,
                    return_all_candidates=self.candidate_pool_dir is not None,
                    gene_feed=self.gene_feed, gene_feed_dataset=h,
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
                rec = {"rank": rank, "engine": p.get("engine"), "genome": genome,
                       "deployed": rank == 0, "candidate_pool_id": candidate_pool_id,
                       "reward": p.get("reward"), "reward_std": p.get("reward_std"),
                       "reward_percentile_in_engine": p.get("reward_percentile_in_engine"),
                       "p_bad": p.get("p_bad"), "selection_key": p.get("selection_key"),
                       "selection_score": p.get("selection_score"),
                       "selector_reranker_score": p.get("selector_reranker_score")}
                picks[h].append(rec)
                eval_inputs.append({
                    "engine": p.get("engine"), "dataset_hash": h,
                    "train_csv": entry["train_csv"], "test_csv": entry.get("test_csv"),
                    "genome": genome, "folds": self.folds, "_h": h, "_rank": rank,
                    "_candidate_pool_id": candidate_pool_id,
                })
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
            "proposal_error_path": str(self.run_dir / f"proposal_errors_g{g}.jsonl") if proposal_errors else None,
        }
        if _proposal_failure_exceeded(len(proposal_errors), len(hashes), self.proposal_failure_threshold):
            raise RuntimeError(
                f"gen {g} proposal coverage below threshold: failure_rate={failure_rate:.4f} "
                f"threshold={self.proposal_failure_threshold}"
            )
        return picks, eval_inputs, candidate_pool_rows, stats

    def evaluate_and_append(
        self,
        g: int,
        picks: dict,
        eval_inputs: list[dict],
        candidate_pool_rows: list[dict] | None = None,
    ) -> dict[str, Any]:
        router = EngineRouter()
        candidate_pool_rows = candidate_pool_rows or []
        pool_by_id = {
            str(row.get("candidate_pool_id")): row
            for row in candidate_pool_rows
            if row.get("candidate_pool_id")
        }
        # consume results INCREMENTALLY: live phase progress + evals/s + gene-feed
        # 'scored' events, so the TUI shows evaluation streaming in real time.
        by_idx: dict[int, dict] = {}
        t0 = time.time()
        done = 0
        for idx, res in iter_eval_regression_with_ledger(
            eval_inputs, ledger=str(self.run_dir / "ledger.jsonl"), router=router,
            workers=self.eval_workers,
        ):
            by_idx[idx] = res
            done += 1
            inp = eval_inputs[idx]
            cv0 = res.get("cv") if isinstance(res.get("cv"), dict) else {}
            try:
                from .gene_feed import gene_id as _gid

                self.gene_feed.scored(
                    _gid(inp.get("engine", ""), inp.get("_h", ""), inp.get("genome", "")),
                    engine=inp.get("engine", ""), dataset_hash=inp.get("_h", ""),
                    canonical=inp.get("genome", ""), oof_r2=cv0.get("oof_r2"),
                    cache_hit=bool(res.get("cache_hit")),
                )
            except Exception:
                pass
            if done % 25 == 0 or done == len(eval_inputs):
                self.state["evals_per_sec"] = round(done / max(time.time() - t0, 1e-6), 1)
                self._progress(done, len(eval_inputs))
        rows_out: list[dict] = []
        pool = self._build_score_pool()
        deployed_scores: dict[str, float] = {}
        # pass 1: score every evaluated candidate (stash oof/fold/res on its rec)
        for idx, inp in enumerate(eval_inputs):
            res = by_idx.get(idx) or {}
            cv = res.get("cv") if isinstance(res.get("cv"), dict) else {}
            oof = cv.get("oof_r2")
            fold_r2 = cv.get("fold_r2") if isinstance(cv.get("fold_r2"), list) else None
            rec = picks[inp["_h"]][inp["_rank"]]
            if not (isinstance(oof, (int, float)) and math.isfinite(float(oof))):
                rec["score_robust"] = None
                continue
            rec["score_robust"] = robust_score(float(oof), fold_r2)
            rec["_oof"] = float(oof)
            rec["_cost_ms"] = (res.get("timings") or {}).get("total_ms")
            pool_row = pool_by_id.get(str(inp.get("_candidate_pool_id") or ""))
            if pool_row is not None:
                pool_row["measured_oof"] = rec["_oof"]
                pool_row["score_robust"] = rec["score_robust"]
                pool_row["cost_ms"] = rec.get("_cost_ms")
        # pass 2: CHEAP-PROBE selection — the deployed pick is the best robust candidate among
        # the top-n reward proposals we just probed (not the reward-rank-0 pick). This closes
        # the top1->cv-select gap (+~94 Elo offline): the reward head prefilters, the cheap
        # CV probe picks. Measurement beats prediction (see [[jope-selection-gap]]).
        regrets: list[float] = []
        captures_at_6: list[int] = []
        for h, recs in picks.items():
            for r in recs:
                r["deployed"] = False
            best = _choose_deployed_candidate(recs)
            if best is not None:
                best["deployed"] = True
                deployed_scores[h] = best["score_robust"]
                measured = [r for r in recs if _finite_float(r.get("score_robust")) is not None]
                if measured:
                    best_measured = max(measured, key=lambda r: _finite_float(r.get("score_robust")) or float("-inf"))
                    regrets.append((best_measured["score_robust"] or 0.0) - (best["score_robust"] or 0.0))
                    captures_at_6.append(1 if int(best_measured.get("rank") or 0) < 6 else 0)
                pool_row = pool_by_id.get(str(best.get("candidate_pool_id") or ""))
                if pool_row is not None:
                    pool_row["deployed"] = True
                try:
                    from .gene_feed import gene_id as _gid

                    self.gene_feed.selected(
                        _gid(str(best.get("engine") or ""), h, str(best.get("genome") or "")),
                        engine=best.get("engine"), dataset_hash=h, canonical=best.get("genome"),
                        selection_slot=f"self_g{g}_deployed", reward_rank=best.get("rank"),
                        generation=g, candidate_pool_id=best.get("candidate_pool_id"),
                        selection_key=best.get("selection_key"), selection_score=best.get("selection_score"),
                        selected_for_eval=True,
                    )
                except Exception:
                    pass
        # pass 3: emit training rows
        for h, recs in picks.items():
            entry = self.datasets[h]
            former_best = self.state["self_best"].get(h)
            for rec in recs:
                if rec.get("score_robust") is None:
                    continue
                score = rec["score_robust"]
                row = TrainingRow(
                    dataset_id=h, source="self_play",
                    embedding=embedding_from_vs(entry["vs_path"], self.standardization),
                    target_embedding=target_summary_from_vs(entry.get("train_target_vs")),
                    genome=rec["genome"], canonical_genome=rec["genome"], engine=rec["engine"],
                    score=max(rec["_oof"], -1.0), cost_ms=rec.get("_cost_ms"),
                    metadata={
                        "provenance": f"self_g{g}", "generation": g,
                        "deployed": rec["deployed"], "task_type": "regression", "split": "train",
                        "oof_r2_raw": rec["_oof"], "score_robust": score,
                        "rank_pct": round(self._rank_pct(h, score), 6),
                        "former_best_score": former_best,
                        "candidate_pool_id": rec.get("candidate_pool_id"),
                        "selected_for_eval": True,
                        "selected_for_eval_rank": rec.get("rank"),
                        "reward": rec.get("reward"),
                        "reward_std": rec.get("reward_std"),
                        "reward_percentile_in_engine": rec.get("reward_percentile_in_engine"),
                        "p_bad": rec.get("p_bad"),
                        "selection_key": rec.get("selection_key"),
                        "selection_score": rec.get("selection_score"),
                        "selector_reranker_score": rec.get("selector_reranker_score"),
                    },
                )
                rows_out.append(row.to_json())
                pool.setdefault(h, []).append(score)
                pool[h].sort()
        out_path = self.art_dir / f"self_rows_{self.run_dir.name}_g{g}.jsonl"
        with out_path.open("w", encoding="utf-8") as fh:
            for r in rows_out:
                fh.write(json.dumps(r) + "\n")
        candidate_pool_path = None
        if self.candidate_pool_dir is not None:
            candidate_pool_path = self.candidate_pool_dir / f"candidate_pools_g{g}.jsonl"
            with candidate_pool_path.open("w", encoding="utf-8") as fh:
                for row in candidate_pool_rows:
                    fh.write(json.dumps(row, sort_keys=True) + "\n")
        selector_regret = sum(regrets) / len(regrets) if regrets else None
        capture_at_6 = sum(captures_at_6) / len(captures_at_6) if captures_at_6 else None
        return {
            "rows": len(rows_out),
            "deployed_scores": deployed_scores,
            "self_rows_path": str(out_path),
            "candidate_pool_path": str(candidate_pool_path) if candidate_pool_path is not None else None,
            "selector_regret_at_1": selector_regret,
            "capture_at_6": capture_at_6,
        }

    def gate(self, g: int, deployed: dict[str, float]) -> dict[str, Any]:
        prev, baseline_generation = self._last_shipped_deployed_scores(g)
        beats = ties = comps = 0
        for h, s in deployed.items():
            p = prev.get(h)
            if p is None:
                continue
            comps += 1
            if s > p:
                beats += 1
            elif s == p:
                ties += 1
        beat_rate = (beats + 0.5 * ties) / comps if comps else None
        new_bests = 0
        for h, s in deployed.items():
            best = self.state["self_best"].get(h)
            if best is None or s > best:
                self.state["self_best"][h] = s
                new_bests += 1
        mean_pct = (
            sum(self._rank_pct(h, s) for h, s in deployed.items()) / len(deployed)
            if deployed else None
        )
        self.state["gen_top1"][str(g)] = deployed
        shipped = beat_rate is None or beat_rate >= self.gate_beat_rate
        return {
            "beat_rate_vs_prev": beat_rate,
            "gate_baseline_generation": baseline_generation,
            "compared": comps,
            "new_bests": new_bests,
            "cum_best_datasets": len(self.state["self_best"]),
            "mean_rank_pct": mean_pct,
            "shipped": shipped,
        }

    def _last_shipped_deployed_scores(self, g: int) -> tuple[dict[str, float], int | None]:
        explicit = self.state.get("last_shipped_deployed")
        if isinstance(explicit, dict):
            return {str(k): float(v) for k, v in explicit.items() if _finite_float(v) is not None}, self.state.get("last_shipped_generation")
        gen_top1 = self.state.get("gen_top1") if isinstance(self.state.get("gen_top1"), dict) else {}
        shipped_gens = [
            int(row.get("gen"))
            for row in self.state.get("elo_history", [])
            if row.get("gen") is not None and int(row.get("gen")) < g and row.get("shipped", True)
        ]
        for gen in sorted(shipped_gens, reverse=True):
            scores = gen_top1.get(str(gen))
            if isinstance(scores, dict):
                return {str(k): float(v) for k, v in scores.items() if _finite_float(v) is not None}, gen
        legacy = gen_top1.get(str(g - 1))
        if isinstance(legacy, dict):
            return {str(k): float(v) for k, v in legacy.items() if _finite_float(v) is not None}, g - 1
        return {}, None

    def bench(self, g: int, ckpt: Path) -> dict[str, Any]:
        import re

        from .elo_bench import run_elo_bench

        def _bench_log(*a, **k) -> None:
            # surface elo-bench's own progress lines ("proposed i/N datasets") as phase ticks
            msg = " ".join(str(x) for x in a)
            m = re.search(r"proposed (\d+)/(\d+)", msg)
            if m:
                self._progress(int(m.group(1)), int(m.group(2)))

        report = run_elo_bench(
            ckpt, self.art_dir, k=16, out_dir=self.run_dir / f"elo_g{g}",
            ledger=self.run_dir / "elo_ledger.jsonl", workers=self.eval_workers,
            selector_calibration=self.selector_calibration,
            selector_reranker=self.selector_reranker,
            samples_per_engine=self.samples_per_engine,
            candidate_pool_dir=self.candidate_pool_dir,
            proposal_failure_threshold=self.proposal_failure_threshold,
            log=_bench_log,
        )
        top1 = report["modes"]["top1"]
        probe = report["modes"].get("cv_top6", report["modes"]["cv_select"])
        out = {"elo_top1": top1["elo"], "elo_rank": top1["field_rank"],
                "elo_top1_dh": top1.get("double_holdout_elo"),
                # cheap-probe deployed selector (probe top-6 reward candidates, pick best robust score)
                "elo_probe": probe["elo"], "elo_probe_rank": probe["field_rank"],
                "elo_probe_dh": probe.get("double_holdout_elo"),
                "battles_won": probe.get("battles_won"),
                "battles_top3": probe.get("battles_top3"),
                "elo_cv": report["modes"]["cv_select"]["elo"],
                "elo_best_k": report["modes"]["best_k"]["elo"]}
        out["elo_deployed"] = deployment_elo(out)
        if _finite_float(out.get("elo_best_k")) is not None and _finite_float(out.get("elo_top1")) is not None:
            out["top1_regret_vs_best_k"] = float(out["elo_best_k"]) - float(out["elo_top1"])
        if _finite_float(out.get("elo_best_k")) is not None and _finite_float(out.get("elo_probe")) is not None:
            out["probe_regret_vs_best_k"] = float(out["elo_best_k"]) - float(out["elo_probe"])
        return out

    # -- main loop ---------------------------------------------------------------
    def run(self, generations: int) -> None:
        start_gen = self.state["generation"] + 1
        for g in range(start_gen, start_gen + generations):
            t0 = time.time()
            self.state["generation"] = g
            # miner races champions on the CPUs while the GPUs train (concurrent)
            miner_proc = self._launch_miner(g)
            self._set_phase("TRAIN", miner_active=miner_proc is not None)
            ckpt = self.train_generation(g)
            train_sec = time.time() - t0
            miner_report: dict[str, Any] = {}
            if miner_proc is not None:
                miner_proc.wait()
                mlog = self.run_dir / f"miner_g{g}.log"
                for line in reversed(mlog.read_text().splitlines() if mlog.exists() else []):
                    if '"new_champions"' in line:
                        try:
                            miner_report = json.loads(line[line.index("{"):])
                        except Exception:
                            pass
                        break
                self._score_pool = None  # champion rows extend the pools

            self._set_phase("PROPOSE")
            t1 = time.time()
            picks, eval_inputs, candidate_pool_rows, proposal_stats = self.propose_generation(g, ckpt)
            propose_sec = time.time() - t1

            self._set_phase("EVAL", phase_total=len(eval_inputs))
            t2 = time.time()
            appended = self.evaluate_and_append(g, picks, eval_inputs, candidate_pool_rows)
            eval_sec = time.time() - t2

            self._set_phase("GATE")
            gate = self.gate(g, appended["deployed_scores"])

            elo: dict[str, Any] = {}
            if g % self.elo_every == 0:
                self._set_phase("BENCH")
                elo = self.bench(g, ckpt)
                # Ship on the deployment metric too: the beat-rate gate (per-dataset
                # train wins) can reject a generation that sets a big held-out Elo
                # RECORD by improving unevenly. A genuine Elo record must become the
                # warm-start ancestor, else future gens build on a weaker checkpoint.
                # deployed capability is now the cheap-probe selector (elo_probe), so the
                # warm-start ancestor is chosen by THAT, not the reward-rank elo_top1.
                gate.update(_resolve_ship_decision(
                    bool(gate.get("shipped", True)),
                    elo,
                    self.state.get("elo_history", []),
                ))
                self.state["elo_history"].append({"gen": g, "shipped": bool(gate["shipped"]), **elo})
            if gate.get("shipped"):
                self.state["last_shipped_deployed"] = appended["deployed_scores"]
                self.state["last_shipped_generation"] = g

            # read last train metrics row for pairwise/conditional advantage
            last_train: dict[str, Any] = {}
            m = self.run_dir / f"gen{g}_train_metrics.jsonl"
            if m.exists():
                lines = m.read_text().splitlines()
                if lines:
                    last = json.loads(lines[-1])
                    last_train = {
                        "val_pairwise_win": last.get("val_pairwise_win_accuracy"),
                        "conditional_advantage": last.get("val_conditional_advantage"),
                        "train_loss": last.get("train_loss"),
                        "val_loss": last.get("val_loss"),
                    }

            row = {
                "generation": g,
                "checkpoint": str(ckpt),
                "rows_appended": appended["rows"],
                "miner_new_champions": miner_report.get("new_champions"),
                "miner_mutants_raced": miner_report.get("mutants_raced"),
                "candidate_pool_path": appended.get("candidate_pool_path"),
                **gate,
                **elo,
                **last_train,
                "selector_regret_at_1": appended.get("selector_regret_at_1"),
                "capture_at_6": appended.get("capture_at_6"),
                "proposal_failure_rate": proposal_stats.get("proposal_failure_rate"),
                "seed_coverage": miner_report.get("seed_coverage"),
                "mean_seed_uplift": miner_report.get("mean_seed_uplift"),
                "train_sec": round(train_sec, 1),
                "propose_sec": round(propose_sec, 1),
                "eval_sec": round(eval_sec, 1),
                "total_sec": round(time.time() - t0, 1),
                "ts": time.time(),
            }
            with self.metrics_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
            self._set_phase("DONE", last_gen_summary=row)
            self.log(json.dumps(row, indent=2), flush=True)


def run_self_play(art_dir: str, run_dir: str, generations: int, **kw) -> None:
    SelfPlay(art_dir, run_dir, **kw).run(generations)
