"""Evaluation, deployment gate, benchmark, and run loop for self-play."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from .self_play import (
    EngineRouter,
    TrainingRow,
    _choose_deployed_candidate,
    _finite_float,
    _resolve_ship_decision,
    deployment_elo,
    embedding_from_vs,
    iter_eval_regression_with_ledger,
    robust_score,
    target_summary_from_vs,
)
from .self_play import (
    _atomic_json as _atomic_json,
)


class SelfPlayEvaluationMixin:
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
            eval_inputs,
            ledger=str(self.run_dir / "ledger.jsonl"),
            router=router,
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
                    engine=inp.get("engine", ""),
                    dataset_hash=inp.get("_h", ""),
                    canonical=inp.get("genome", ""),
                    oof_r2=cv0.get("oof_r2"),
                    cache_hit=bool(res.get("cache_hit")),
                )
            except (Exception, ArithmeticError):
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
                    best_measured = max(
                        measured, key=lambda r: _finite_float(r.get("score_robust")) or float("-inf")
                    )
                    regrets.append((best_measured["score_robust"] or 0.0) - (best["score_robust"] or 0.0))
                    captures_at_6.append(1 if int(best_measured.get("rank") or 0) < 6 else 0)
                pool_row = pool_by_id.get(str(best.get("candidate_pool_id") or ""))
                if pool_row is not None:
                    pool_row["deployed"] = True
                try:
                    from .gene_feed import gene_id as _gid

                    self.gene_feed.selected(
                        _gid(str(best.get("engine") or ""), h, str(best.get("genome") or "")),
                        engine=best.get("engine"),
                        dataset_hash=h,
                        canonical=best.get("genome"),
                        selection_slot=f"self_g{g}_deployed",
                        reward_rank=best.get("rank"),
                        generation=g,
                        candidate_pool_id=best.get("candidate_pool_id"),
                        selection_key=best.get("selection_key"),
                        selection_score=best.get("selection_score"),
                        selected_for_eval=True,
                    )
                except (Exception, ArithmeticError):
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
                    dataset_id=h,
                    source="self_play",
                    embedding=embedding_from_vs(entry["vs_path"], self.standardization),
                    target_embedding=target_summary_from_vs(entry.get("train_target_vs")),
                    genome=rec["genome"],
                    canonical_genome=rec["genome"],
                    engine=rec["engine"],
                    score=max(rec["_oof"], -1.0),
                    cost_ms=rec.get("_cost_ms"),
                    metadata={
                        "provenance": f"self_g{g}",
                        "generation": g,
                        "deployed": rec["deployed"],
                        "task_type": "regression",
                        "split": "train",
                        "oof_r2_raw": rec["_oof"],
                        "score_robust": score,
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
            sum(self._rank_pct(h, s) for h, s in deployed.items()) / len(deployed) if deployed else None
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
            return {
                str(k): float(v) for k, v in explicit.items() if _finite_float(v) is not None
            }, self.state.get("last_shipped_generation")
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
            ckpt,
            self.art_dir,
            k=16,
            out_dir=self.run_dir / f"elo_g{g}",
            ledger=self.run_dir / "elo_ledger.jsonl",
            workers=self.eval_workers,
            selector_calibration=self.selector_calibration,
            selector_reranker=self.selector_reranker,
            samples_per_engine=self.samples_per_engine,
            candidate_pool_dir=self.candidate_pool_dir,
            proposal_failure_threshold=self.proposal_failure_threshold,
            log=_bench_log,
        )
        top1 = report["modes"]["top1"]
        probe = report["modes"].get("cv_top6", report["modes"]["cv_select"])
        out = {
            "elo_top1": top1["elo"],
            "elo_rank": top1["field_rank"],
            "elo_top1_dh": top1.get("double_holdout_elo"),
            # cheap-probe deployed selector (probe top-6 reward candidates, pick best robust score)
            "elo_probe": probe["elo"],
            "elo_probe_rank": probe["field_rank"],
            "elo_probe_dh": probe.get("double_holdout_elo"),
            "battles_won": probe.get("battles_won"),
            "battles_top3": probe.get("battles_top3"),
            "elo_cv": report["modes"]["cv_select"]["elo"],
            "elo_best_k": report["modes"]["best_k"]["elo"],
        }
        out["elo_deployed"] = deployment_elo(out)
        if (
            _finite_float(out.get("elo_best_k")) is not None
            and _finite_float(out.get("elo_top1")) is not None
        ):
            out["top1_regret_vs_best_k"] = float(out["elo_best_k"]) - float(out["elo_top1"])
        if (
            _finite_float(out.get("elo_best_k")) is not None
            and _finite_float(out.get("elo_probe")) is not None
        ):
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
                            miner_report = json.loads(line[line.index("{") :])
                        except (Exception, ArithmeticError):
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
                gate.update(
                    _resolve_ship_decision(
                        bool(gate.get("shipped", True)),
                        elo,
                        self.state.get("elo_history", []),
                    )
                )
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
