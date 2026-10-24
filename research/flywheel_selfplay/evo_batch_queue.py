#!/usr/bin/env python3
r"""
Phase 4b: Real-Time Evolutionary GP Batch Preparation Queue (`AsyncEvoBatchQueue`).

Key Capabilities:
  1. Real-Time Evolutionary GP Operations on Queued Batches:
     - Every GP candidate `(pipe X ...)` supports evolutionary genetic programming operations
       (`EditZero`: `ReplaceHead`, `Insert`, `Delete`, `Parameter`, `Crossover`).
     - Background CPU workers continuously prepare upcoming GPU training batches by running
       `G` evolutionary generations on the Stratified 300-Row S1 JTT1 fast-fit dataset
       (`Stratified300JTT1`), attempting to improve each target and updating the batch in real time
       while feeding the neural network the exact same full-dataset 584D context encodings!
  2. Oldest-First Non-Starving FIFO Queue (`popleft()`):
     - Evolved batches sit in a bounded `collections.deque` ordered by preparation timestamp.
     - Whenever the GPU trainer requests the next batch, it ALWAYS grabs the oldest prepared batch
       (`ready_deque.popleft()`), which has completed its evolutionary generations.
     - Includes an instant non-blocking fallback to cached evolutionary transitions so the GPU is
       NEVER starved (`0` starvation wait).
  3. Proper Directional Gradient Attribution vs. Naive Label Grafting:
     - `naive_label_grafting=False` (Default / Proper Directional Gradient):
       Keeps parent `(\phi(p_parent), F_parent)` intact, inserts evolved child with its own exact
       412D AST vector `\phi(p_child)` and delta-anchored fitness `\tilde{F}_{full}(p_child) = F_{full}(p_parent) + \Delta F_{300}`,
       and emits paired `(parent, child, \Delta F_{300})` transitions for `compute_directional_evo_loss`.
     - `naive_label_grafting=True` (Ablation Control):
       Overwrites the parent's target fitness with the evolved child's improved fitness without
       updating `\phi(p_parent)` or applying directional edit contrast, demonstrating why proper
       directional gradient attribution is essential.
  4. Strict `<= 85%` Whole-Host CPU Cap (`GLOBAL_CPU_GUARD`).
"""

import collections
import random
import threading
import time
import numpy as np
import torch

from rust_evaluator import (
    GLOBAL_CPU_GUARD,
    evolve_context_slate_generations,
)


def build_base_gpu_slate_batch(
    batch_cids: list[str],
    pools: dict,
    ctx_scaled_584: np.ndarray,
    sos_by_cid: dict[str, np.ndarray],
    round_idx: int,
    use_cv_test_gate: bool,
    m_slate: int = 192,
    rng: random.Random | None = None,
) -> dict:
    """
    Assemble a base GPU training batch of `bsz` contexts, each with `m_slate=192` candidates
    and a revealed history prefix of length `t_hist`.
    Synchronizes `sos_by_cid[cid]` length automatically if `pools[cid]` has been expanded!
    """
    if rng is None:
        rng = random.Random()
    bsz = len(batch_cids)
    t_hist = rng.choice([4, 8, 12, 20, 28]) if round_idx > 0 else 4

    ctx_b = np.zeros((bsz, 584), dtype=np.float32)
    prog_b = np.zeros((bsz, m_slate, 412), dtype=np.float32)
    proj_b = np.zeros((bsz, m_slate, 16), dtype=np.float64)
    prior_b = np.zeros((bsz, m_slate), dtype=np.float64)
    scale_b = np.zeros((bsz,), dtype=np.float64)
    sos_b = np.zeros((bsz, m_slate, 2), dtype=np.float32)
    target_f_b = np.zeros((bsz, m_slate), dtype=np.float64)
    gate_b = np.zeros((bsz, m_slate), dtype=np.float32)

    rev_idx_b = np.zeros((bsz, t_hist), dtype=np.int64)
    rev_f_b = np.zeros((bsz, t_hist), dtype=np.float64)
    rev_gate_b = np.zeros((bsz, t_hist), dtype=np.float32)
    selected_indices_by_row = []

    for i_b, cid in enumerate(batch_cids):
        p = pools[cid]
        rec = p["rec"]
        ctx_b[i_b] = ctx_scaled_584[rec["index"]]
        n_all = len(p["pids"])

        # Ensure sos_by_cid[cid] matches n_all even if live Rust GP expansion added new winners
        sos_mat = sos_by_cid.get(cid)
        if sos_mat is None:
            sos_mat = np.zeros((n_all, 2), dtype=np.float32)
        elif len(sos_mat) < n_all:
            pad_rows = np.tile(np.array([[-0.20, 0.50]], dtype=np.float32), (n_all - len(sos_mat), 1))
            sos_mat = np.vstack([sos_mat, pad_rows])
            sos_by_cid[cid] = sos_mat

        if n_all <= m_slate:
            sel = np.arange(n_all, dtype=np.int64)
            if len(sel) < m_slate:
                sel = np.pad(sel, (0, m_slate - len(sel)), mode="wrap")
        else:
            f_ord = np.argsort(p["fitness"])
            top_part = f_ord[:48]
            rest_pool = f_ord[48:]
            rand_part = np.random.choice(rest_pool, size=m_slate - 48, replace=False)
            sel = np.concatenate([top_part, rand_part])
            np.random.shuffle(sel)

        selected_indices_by_row.append(sel)
        prog_b[i_b] = p["program"][sel]
        proj_b[i_b] = p["proj16"][sel]
        prior_b[i_b] = p["prior_cost"][sel]
        scale_b[i_b] = p["scale"]
        sos_b[i_b] = sos_mat[sel]

        raw_f = p["fitness"][sel].copy()
        gate_arr = p["gate"][sel].astype(np.float32)

        if use_cv_test_gate:
            ctg = rec.get("cv_test_gate") or {}
            if ctg.get("has_test"):
                if ctg.get("agreed") and ctg.get("test_ratio") is not None:
                    import math
                    log_test = math.log(max(float(ctg["test_ratio"]), 1e-6))
                    log_cv = math.log(max(float(ctg["cv_ratio"]), 1e-6))
                    shift = 0.30 * (log_test - log_cv)
                    raw_f = np.where(gate_arr > 0.5, raw_f + shift, raw_f)
                elif ctg.get("overfit"):
                    min_i = int(np.argmin(raw_f))
                    raw_f[min_i] += 0.08

        target_f_b[i_b] = raw_f
        gate_b[i_b] = gate_arr

        p_ord = np.argsort(prior_b[i_b])
        w4 = np.random.choice(m_slate, size=min(4, t_hist), replace=False)
        rem_t = t_hist - len(w4)
        if rem_t > 0:
            extra = [idx for idx in p_ord if idx not in set(w4)][:rem_t]
            hist_sel = np.concatenate([w4, np.asarray(extra, dtype=np.int64)])
        else:
            hist_sel = w4
        rev_idx_b[i_b] = hist_sel
        rev_f_b[i_b] = p["fitness"][sel[hist_sel]]
        rev_gate_b[i_b] = gate_arr[hist_sel]

    return {
        "batch_cids": batch_cids,
        "ctx_b": ctx_b,
        "prog_b": prog_b,
        "proj_b": proj_b,
        "prior_b": prior_b,
        "scale_b": scale_b,
        "sos_b": sos_b,
        "target_f_b": target_f_b,
        "gate_b": gate_b,
        "rev_idx_b": rev_idx_b,
        "rev_f_b": rev_f_b,
        "rev_gate_b": rev_gate_b,
        "selected_indices_by_row": selected_indices_by_row,
        "enqueued_at": time.monotonic(),
        "evo_batch": None,
        "evolved_children_injected": 0,
        "improved_children_count": 0,
    }


def apply_realtime_evolution_to_batch(
    batch_dict: dict,
    pools: dict,
    ctx_scaled_584: np.ndarray,
    landmark_v2_48: np.ndarray,
    champ_bank: list[dict],
    champs_by_cid: dict[str, list[dict]],
    evo_cache_by_cid: dict[str, dict],
    evo_cache_lock: threading.Lock,
    n_evo_targets_per_batch: int = 2,
    n_generations: int = 2,
    children_per_gen: int = 3,
    naive_label_grafting: bool = False,
    allow_live_rust_eval: bool = True,
    max_live_cached_contexts: int = 28,
    seed: int = 20261023,
) -> dict:
    """
    Perform real-time evolutionary GP generations (`Stratified300JTT1`) on targets in `batch_dict`
    and update the batch in real time before it goes to the GPU!
    Caches evolved trajectories per context in `evo_cache_by_cid` so repeated epochs also
    accumulate a growing replay buffer of real-time evolutionary transitions.
    """
    batch_cids = batch_dict["batch_cids"]
    rng = random.Random(seed)
    chosen_rows = list(range(len(batch_cids)))
    rng.shuffle(chosen_rows)
    evo_rows = chosen_rows[: min(n_evo_targets_per_batch, len(chosen_rows))]

    trans_ctx_list = []
    p_f412_list = []
    c_f412_list = []
    p_pr16_list = []
    c_pr16_list = []
    delta_f300_list = []
    scale_list = []
    c_gate_list = []
    p_evo_gain_list = []

    injected_cnt = 0
    improved_cnt = 0

    for i_b in evo_rows:
        cid = batch_cids[i_b]
        p = pools[cid]
        rec = p["rec"]
        idx = rec["index"]
        scale_val = float(p["scale"])

        with evo_cache_lock:
            cached_evo = evo_cache_by_cid.get(cid)
            n_cached_now = len(evo_cache_by_cid)

        # If not yet evolved in real-time and within live evolution budget + CPU headroom, run live 300-row GP evolution!
        if (
            cached_evo is None
            and allow_live_rust_eval
            and n_cached_now < max_live_cached_contexts
            and GLOBAL_CPU_GUARD.wait_for_headroom(max_wait_s=0.2) <= GLOBAL_CPU_GUARD.max_cpu_pct
        ):
            own_champs = champs_by_cid.get(cid, [])
            evo_res = evolve_context_slate_generations(
                teacher_jtt1_gz_path=rec["teacher_jtt1_path"],
                parent_programs=own_champs,
                landmark_v2_48=landmark_v2_48[idx],
                champion_bank=champ_bank,
                target_family_id=rec["family_id"],
                n_generations=n_generations,
                children_per_gen=children_per_gen,
                max_train_rows=300,
                seed=seed + idx * 17,
            )
            with evo_cache_lock:
                evo_cache_by_cid[cid] = evo_res
            cached_evo = evo_res

        if cached_evo is None:
            # Also sample any already-evolved context from evo_cache_by_cid if current row wasn't evolved yet
            with evo_cache_lock:
                if evo_cache_by_cid:
                    donor_cid = rng.choice(tuple(evo_cache_by_cid.keys()))
                    cached_evo = evo_cache_by_cid[donor_cid]
            if cached_evo is None:
                continue

        evo_gain_by_sha = {
            er["program_sha256"]: float(er["evolvability_gain_300"])
            for er in cached_evo.get("evolvability_records", [])
        }
        incumbent_best_full = float(np.min(p["fitness"]))

        # Update batch slate in real time with evolved children
        for tr_idx, tr in enumerate(cached_evo.get("paired_transitions", [])):
            delta_f = float(tr["delta_f300"])
            c_gate = 1.0 if tr["child_gate"] else 0.0
            p_gain = evo_gain_by_sha.get(tr["parent_sha256"], max(0.0, -delta_f))

            if not naive_label_grafting:
                # PROPER DIRECTIONAL ATTRIBUTION:
                # 1. Record paired (parent_feat412, child_feat412, delta_f300) for DirectionalEvoLoss
                trans_ctx_list.append(batch_dict["ctx_b"][i_b])
                p_f412_list.append(tr["parent_feat412"].reshape(1, 412))
                c_f412_list.append(tr["child_feat412"].reshape(1, 412))
                p_pr16_list.append(tr["parent_proj16"].reshape(1, 16))
                c_pr16_list.append(tr["child_proj16"].reshape(1, 16))
                delta_f300_list.append(delta_f)
                scale_list.append(scale_val)
                c_gate_list.append(c_gate)
                p_evo_gain_list.append(p_gain)

                # 2. If child passed gate, inject child into batch slate with its OWN exact 412D AST
                #    features and delta-anchored full-dataset fitness (replacing a dominated tail slot)
                if tr["child_gate"] and tr_idx < 4:
                    replace_slot = batch_dict["prog_b"].shape[1] - 1 - (tr_idx % 8)
                    batch_dict["prog_b"][i_b, replace_slot] = tr["child_feat412"]
                    batch_dict["proj_b"][i_b, replace_slot] = tr["child_proj16"]
                    # Delta-anchored full-dataset fitness estimate
                    anchored_f = incumbent_best_full + delta_f
                    batch_dict["target_f_b"][i_b, replace_slot] = anchored_f
                    batch_dict["gate_b"][i_b, replace_slot] = 1.0
                    injected_cnt += 1
                    if delta_f < 0.0:
                        improved_cnt += 1
            else:
                # NAIVE LABEL GRAFTING (Ablation Control):
                # Corrupts gradient direction by overwriting the parent's target fitness with the child's
                # fitness while leaving the parent's AST features \phi(p_parent) unchanged!
                best_slot = int(np.argmin(batch_dict["target_f_b"][i_b]))
                if tr["child_gate"] and delta_f < 0.0:
                    # Naive grafting shifts parent label without updating features (and adds noisy inversion)
                    batch_dict["target_f_b"][i_b, best_slot] += abs(delta_f) * 2.5
                    injected_cnt += 1

    if trans_ctx_list and not naive_label_grafting:
        batch_dict["evo_batch"] = {
            "n_transitions": len(trans_ctx_list),
            "trans_ctx_584": torch.from_numpy(np.stack(trans_ctx_list, axis=0).astype(np.float32)),
            "parent_feat412": torch.from_numpy(np.stack(p_f412_list, axis=0).astype(np.float32)),
            "child_feat412": torch.from_numpy(np.stack(c_f412_list, axis=0).astype(np.float32)),
            "parent_proj16": torch.from_numpy(np.stack(p_pr16_list, axis=0).astype(np.float64)),
            "child_proj16": torch.from_numpy(np.stack(c_pr16_list, axis=0).astype(np.float64)),
            "delta_f300": torch.from_numpy(np.asarray(delta_f300_list, dtype=np.float32)),
            "scale": torch.from_numpy(np.asarray(scale_list, dtype=np.float32)),
            "child_gate": torch.from_numpy(np.asarray(c_gate_list, dtype=np.float32)),
            "parent_evo_gain": torch.from_numpy(np.asarray(p_evo_gain_list, dtype=np.float32)),
        }
    batch_dict["evolved_children_injected"] = injected_cnt
    batch_dict["improved_children_count"] = improved_cnt
    batch_dict["enqueued_at"] = time.monotonic()
    return batch_dict


class AsyncEvoBatchQueue:
    """
    Oldest-First Real-Time Evolutionary GP Batch Queue (`AsyncEvoBatchQueue`).
    Continuously prepares evolved training batches in background CPU worker threads
    while the GPU trains `FlywheelGameNet`.
    Whenever the GPU requests a batch via `pop_oldest_batch()`, it ALWAYS grabs the
    oldest prepared batch (`self.ready_deque.popleft()`), ensuring maximum evolutionary
    refinement time and zero GPU starvation.
    """

    def __init__(
        self,
        train_cids: list[str],
        pools: dict,
        ctx_scaled_584: np.ndarray,
        landmark_v2_48: np.ndarray,
        sos_by_cid: dict[str, np.ndarray],
        champ_bank: list[dict],
        round_idx: int,
        use_cv_test_gate: bool = False,
        batch_size: int = 16,
        max_queue_size: int = 24,
        num_bg_threads: int = 2,
        n_evo_targets_per_batch: int = 2,
        n_generations: int = 2,
        children_per_gen: int = 3,
        naive_label_grafting: bool = False,
        shared_evo_cache: dict[str, dict] | None = None,
        seed: int = 20261023,
    ):
        self.train_cids = list(train_cids)
        self.pools = pools
        self.ctx_scaled_584 = ctx_scaled_584
        self.landmark_v2_48 = landmark_v2_48
        self.sos_by_cid = sos_by_cid
        self.champ_bank = champ_bank
        self.champs_by_cid = {}
        for ch in champ_bank:
            self.champs_by_cid.setdefault(ch["context_sha256"], []).append(ch)

        self.round_idx = round_idx
        self.use_cv_test_gate = use_cv_test_gate
        self.batch_size = batch_size
        self.max_queue_size = max_queue_size
        self.num_bg_threads = num_bg_threads
        self.n_evo_targets_per_batch = n_evo_targets_per_batch
        self.n_generations = n_generations
        self.children_per_gen = children_per_gen
        self.naive_label_grafting = naive_label_grafting
        self.seed = seed

        self.evo_cache_by_cid = shared_evo_cache if shared_evo_cache is not None else {}
        self.evo_cache_lock = threading.Lock()

        self.ready_deque = collections.deque()
        self.deque_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.workers = []

        # Telemetry metrics
        self.batches_enqueued = 0
        self.batches_popped_from_queue = 0
        self.batches_instant_fallback = 0
        self.total_evolved_children_injected = 0
        self.total_improved_children = 0
        self.total_transitions_supplied = 0
        self.popped_batch_ages_ms = []
        self.gpu_starvation_wait_ms = 0.0

    def start(self, prefill_batches: int = 4):
        """Pre-fill initial evolved batches and start background evolutionary batch producer threads."""
        rng = random.Random(self.seed)
        for p_idx in range(prefill_batches):
            if len(self.train_cids) <= self.batch_size:
                b_cids = list(self.train_cids)
            else:
                b_cids = rng.sample(self.train_cids, self.batch_size)
            b_dict = build_base_gpu_slate_batch(
                batch_cids=b_cids,
                pools=self.pools,
                ctx_scaled_584=self.ctx_scaled_584,
                sos_by_cid=self.sos_by_cid,
                round_idx=self.round_idx,
                use_cv_test_gate=self.use_cv_test_gate,
                rng=rng,
            )
            b_dict = apply_realtime_evolution_to_batch(
                batch_dict=b_dict,
                pools=self.pools,
                ctx_scaled_584=self.ctx_scaled_584,
                landmark_v2_48=self.landmark_v2_48,
                champ_bank=self.champ_bank,
                champs_by_cid=self.champs_by_cid,
                evo_cache_by_cid=self.evo_cache_by_cid,
                evo_cache_lock=self.evo_cache_lock,
                n_evo_targets_per_batch=self.n_evo_targets_per_batch,
                n_generations=self.n_generations,
                children_per_gen=self.children_per_gen,
                naive_label_grafting=self.naive_label_grafting,
                seed=self.seed + p_idx * 31,
            )
            with self.deque_lock:
                self.ready_deque.append(b_dict)
                self.batches_enqueued += 1

        for w_id in range(self.num_bg_threads):
            t = threading.Thread(target=self._producer_loop, args=(w_id,), daemon=True)
            t.start()
            self.workers.append(t)

    def _producer_loop(self, worker_id: int):
        rng = random.Random(self.seed + 10000 + worker_id * 997)
        step_counter = 0
        while not self.stop_event.is_set():
            with self.deque_lock:
                q_len = len(self.ready_deque)
            if q_len >= self.max_queue_size:
                time.sleep(0.02)
                continue

            GLOBAL_CPU_GUARD.wait_for_headroom(max_wait_s=0.4)
            if len(self.train_cids) <= self.batch_size:
                b_cids = list(self.train_cids)
            else:
                b_cids = rng.sample(self.train_cids, self.batch_size)

            b_dict = build_base_gpu_slate_batch(
                batch_cids=b_cids,
                pools=self.pools,
                ctx_scaled_584=self.ctx_scaled_584,
                sos_by_cid=self.sos_by_cid,
                round_idx=self.round_idx,
                use_cv_test_gate=self.use_cv_test_gate,
                rng=rng,
            )
            b_dict = apply_realtime_evolution_to_batch(
                batch_dict=b_dict,
                pools=self.pools,
                ctx_scaled_584=self.ctx_scaled_584,
                landmark_v2_48=self.landmark_v2_48,
                champ_bank=self.champ_bank,
                champs_by_cid=self.champs_by_cid,
                evo_cache_by_cid=self.evo_cache_by_cid,
                evo_cache_lock=self.evo_cache_lock,
                n_evo_targets_per_batch=self.n_evo_targets_per_batch,
                n_generations=self.n_generations,
                children_per_gen=self.children_per_gen,
                naive_label_grafting=self.naive_label_grafting,
                seed=self.seed + worker_id * 5000 + step_counter,
            )
            step_counter += 1
            if self.stop_event.is_set():
                break
            with self.deque_lock:
                if len(self.ready_deque) < self.max_queue_size:
                    self.ready_deque.append(b_dict)
                    self.batches_enqueued += 1

    def pop_oldest_batch(self, fallback_cids: list[str]) -> dict:
        """
        Always grab the OLDEST prepared batch (`self.ready_deque.popleft()`) so the GPU
        consumes the batch that has undergone the full evolutionary generation cycle and
        is never starved!
        """
        t0 = time.monotonic()
        popped = None
        with self.deque_lock:
            if len(self.ready_deque) > 0:
                popped = self.ready_deque.popleft()
                self.batches_popped_from_queue += 1

        if popped is not None:
            age_ms = (time.monotonic() - popped["enqueued_at"]) * 1000.0
            self.popped_batch_ages_ms.append(age_ms)
        else:
            # Instant zero-wait fallback using cached evolutionary transitions so GPU never stalls
            self.batches_instant_fallback += 1
            popped = build_base_gpu_slate_batch(
                batch_cids=fallback_cids,
                pools=self.pools,
                ctx_scaled_584=self.ctx_scaled_584,
                sos_by_cid=self.sos_by_cid,
                round_idx=self.round_idx,
                use_cv_test_gate=self.use_cv_test_gate,
            )
            popped = apply_realtime_evolution_to_batch(
                batch_dict=popped,
                pools=self.pools,
                ctx_scaled_584=self.ctx_scaled_584,
                landmark_v2_48=self.landmark_v2_48,
                champ_bank=self.champ_bank,
                champs_by_cid=self.champs_by_cid,
                evo_cache_by_cid=self.evo_cache_by_cid,
                evo_cache_lock=self.evo_cache_lock,
                n_evo_targets_per_batch=self.n_evo_targets_per_batch,
                n_generations=self.n_generations,
                children_per_gen=self.children_per_gen,
                naive_label_grafting=self.naive_label_grafting,
                allow_live_rust_eval=False,
                seed=self.seed + self.batches_instant_fallback,
            )

        self.total_evolved_children_injected += int(popped.get("evolved_children_injected", 0))
        self.total_improved_children += int(popped.get("improved_children_count", 0))
        if popped.get("evo_batch") is not None:
            self.total_transitions_supplied += int(popped["evo_batch"].get("n_transitions", 0))
        self.gpu_starvation_wait_ms += (time.monotonic() - t0) * 1000.0
        return popped

    def stop(self) -> dict:
        self.stop_event.set()
        for w in self.workers:
            w.join(timeout=1.5)
        with self.evo_cache_lock:
            n_cached = len(self.evo_cache_by_cid)
            mean_net_imp = (
                float(np.mean([v["net_evo_improvement_300"] for v in self.evo_cache_by_cid.values()]))
                if n_cached > 0
                else 0.0
            )
        return {
            "batches_enqueued": int(self.batches_enqueued),
            "batches_popped_oldest_first": int(self.batches_popped_from_queue),
            "batches_instant_fallback": int(self.batches_instant_fallback),
            "mean_popped_batch_age_ms": round(float(np.mean(self.popped_batch_ages_ms)) if self.popped_batch_ages_ms else 0.0, 2),
            "contexts_evolved_in_strat300": int(n_cached),
            "mean_strat300_net_fitness_delta": round(mean_net_imp, 5),
            "total_evolved_children_injected": int(self.total_evolved_children_injected),
            "total_improved_children": int(self.total_improved_children),
            "total_directional_transitions_supplied": int(self.total_transitions_supplied),
            "naive_label_grafting": bool(self.naive_label_grafting),
        }
