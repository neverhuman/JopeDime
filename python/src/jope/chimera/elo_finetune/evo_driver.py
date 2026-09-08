"""Evolution driver (runs on xbabe2): generations, racing, swarm updates,
gates, promotions, metrics emission.

Control plane: JSON specs rsync-pushed to each node's evo/inbox; results
polled from evo/outbox. Workers are stateless (champion = θ₀ + promo deltas).
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from .ckpt import CheckpointManager, save_atomic
from .evo_blocks import (BlockTable, CandidateSpec, DeltaEngine, MODULE_GROUPS,
                         delta_to_sparse, sample_mask, save_sparse_delta)
from .evo_episodes import episode_specs_guard, episode_specs_narrow
from .evo_fitness import paired_stats
from .evo_swarm import Swarm
from .metrics import JsonlWriter, new_run_id
from .model_io import CONFIG_NAME, default_weights_dir
from .pack import FablePack
from .paths import import_regressor_train

DEFAULT_NODES = ["xbabe1", "local", "xbabe3"]
DEFAULT_RUNGS = [(128, 6), (256, 3), (640, 1)]  # extra episodes per rung, survivors
PROMOTE_P, PROMOTE_DELTA, GUARD_EPS = 0.01, 0.001, 0.003


def sh(cmd: str) -> None:
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


class Driver:
    def __init__(self, root: Path, cfg: dict):
        self.root, self.cfg = root, cfg
        self.nodes = cfg.get("nodes", DEFAULT_NODES)
        self.rungs = cfg.get("rungs", DEFAULT_RUNGS)
        self.evo = root / "evo"
        for d in ("inbox", "outbox", "exchange/champion", "journal", "state"):
            (self.evo / d).mkdir(parents=True, exist_ok=True)

        tm = import_regressor_train()
        mcfg = json.loads((default_weights_dir() / CONFIG_NAME).read_text())
        self.model = tm.get_architecture(tm.TabPFNV3Config.from_dict(mcfg))
        payload = CheckpointManager.load(root / "ckpt" / "step_1500.ckpt",
                                         map_location="cpu")
        self.model.load_state_dict(payload["ema_model"], strict=True)
        self.model.eval()  # CPU replica: delta materialization only
        self.table = BlockTable(self.model.state_dict())
        self.engine = DeltaEngine(self.model, self.table)

        pack = FablePack(root / "pack" / "fable_real_v1.pack")
        self.train_tasks = [pack.dataset_ids[i]
                            for i in pack.split_indices("train")]
        self.val_tasks = [pack.dataset_ids[i] for i in pack.split_indices("val")]

        state_path = self.evo / "state" / "state.json"
        if state_path.exists():
            self.state = json.loads(state_path.read_text())
            self.swarm = Swarm.load(str(self.evo / "state" / "swarm.npz"),
                                    self.table.n_blocks)
            # RESUME: reconstruct the champion on the driver's engine (base +
            # all promo deltas) so delta scaling / refresh_stats match the
            # workers' reconstruction. Without this the driver diverges from
            # the fleet after any restart-with-promotions.
            for promo in self.state.get("promos", []):
                self.engine.apply_sparse_file(
                    str(self.evo / "exchange" / "champion" / promo))
            if self.state.get("promos"):
                self.engine.refresh_stats()
        else:
            self.state = {"gen": 0, "promos": [], "elite": None,
                          "top_last_gen": [], "thresholdout_used": 0,
                          "incumbent_val": None, "grad_promos_streak": 0,
                          "run_id": new_run_id(), "f_broad_theta0": None}
            self.swarm = Swarm(self.table.n_blocks)
        self.state["run_id"] = new_run_id()  # fresh per process (TUI dedupe)
        self.run_id = self.state["run_id"]
        self.metrics = JsonlWriter(root / "metrics" / "evo_metrics.jsonl",
                                   self.run_id)
        self._write_layout()

    # ---- transport ---------------------------------------------------------
    def push_spec(self, name: str, spec: dict, nodes=None) -> None:
        spec_json = json.dumps(spec)
        target_nodes = list(nodes or self.nodes)
        for shard_ix, node in enumerate(target_nodes):
            s = dict(spec)
            s["shard"] = [shard_ix, len(target_nodes)]
            s["episode_specs"] = spec["episode_specs"][shard_ix::len(target_nodes)]
            local = self.evo / "state" / f"{name}.{node}.json"
            local.write_text(json.dumps(s))
            if node == "local":
                (self.evo / "inbox" / f"{name}.json").write_text(json.dumps(s))
            else:
                sh(f"rsync -a {local} {node}:~/fable_god_run/evo/inbox/{name}.json")

    def worker_alive(self, node: str) -> bool:
        if node == "local":
            return subprocess.run("tmux has-session -t evo_worker",
                                  shell=True, capture_output=True).returncode == 0
        return subprocess.run(
            f"ssh {node} tmux has-session -t evo_worker",
            shell=True, capture_output=True).returncode == 0

    def respawn_worker(self, node: str) -> None:
        cmd = ("cd ~/fable_god_run/code && OMP_NUM_THREADS=2 "
               "FABLE_PACK_DIR=$HOME/fable_god_run/pack "
               "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
               "python3 -m python.elo_finetune.evo_worker "
               "--root $HOME/fable_god_run 2>&1 | tee -a "
               "$HOME/fable_god_run/logs/evo_worker.log")
        if node == "local":
            subprocess.run("tmux kill-session -t evo_worker 2>/dev/null; "
                           f"tmux new-session -d -s evo_worker '{cmd}'",
                           shell=True, capture_output=True)
        else:
            subprocess.run(
                f"ssh {node} \"tmux kill-session -t evo_worker 2>/dev/null; "
                f"tmux new-session -d -s evo_worker '{cmd}'\"",
                shell=True, capture_output=True)
        self.metrics.event(self.state.get("gen", 0),
                           f"respawned dead worker on {node}", level="warn")
        print(f"respawned worker on {node}", flush=True)

    def collect(self, name: str, nodes=None, timeout: float = 2400) -> dict:
        nodes = list(nodes or self.nodes)
        results, t0, last_watchdog = {}, time.time(), time.time()
        while len(results) < len(nodes):
            for node in nodes:
                if node in results:
                    continue
                if node == "local":
                    p = self.evo / "outbox" / f"{name}.result.json"
                    if p.exists():
                        results[node] = json.loads(p.read_text())
                        p.unlink()
                else:
                    dst = self.evo / "state" / f"{name}.{node}.result.json"
                    subprocess.run(
                        f"rsync -a {node}:~/fable_god_run/evo/outbox/"
                        f"{name}.result.json {dst}",
                        shell=True, capture_output=True)
                    if dst.exists():
                        results[node] = json.loads(dst.read_text())
                        sh(f"ssh {node} rm -f "
                           f"~/fable_god_run/evo/outbox/{name}.result.json")
                        dst.unlink()
            # watchdog: a dead worker never produces output — respawn it and
            # re-push its spec so this generation can still complete.
            if time.time() - last_watchdog > 120:
                last_watchdog = time.time()
                for node in nodes:
                    if node in results or self.worker_alive(node):
                        continue
                    self.respawn_worker(node)
                    time.sleep(20)
                    src = self.evo / "state" / f"{name}.{node}.json"
                    if src.exists() and node != "local":
                        sh(f"rsync -a {src} "
                           f"{node}:~/fable_god_run/evo/inbox/{name}.json")
                    elif src.exists():
                        (self.evo / "inbox" / f"{name}.json").write_text(
                            src.read_text())
            if time.time() - t0 > timeout:
                raise TimeoutError(f"{name}: got {list(results)} of {nodes}")
            if len(results) < len(nodes):
                time.sleep(2.0)
        for node, r in results.items():
            if r.get("kind") == "error":
                raise RuntimeError(f"{node} failed {name}: {r['error']}\n"
                                   f"{r.get('trace', '')}")
        return results

    def push_file(self, path: Path, nodes=None) -> None:
        for node in (nodes or self.nodes):
            if node != "local":
                sh(f"rsync -a {path} {node}:~/fable_god_run/evo/exchange/")

    # ---- candidates --------------------------------------------------------
    def make_candidates(self, gen: int) -> list[dict]:
        cands = []
        base = gen * 1_000
        for pair in range(self.cfg["n_pairs"]):
            seeds = {"mask_seed": base + pair * 3 + 1,
                     "sigma_seed": base + pair * 3 + 2,
                     "value_seed": base + pair * 3 + 3}
            mask = sample_mask(self.table, self.swarm.p, seeds["mask_seed"])
            for sign in (+1, -1):
                cands.append({"kind": "seed", "cand_id": len(cands),
                              "pair": pair, "sign": sign,
                              "mask_blocks": mask.tolist(), **seeds})
        for arm in range(self.cfg["n_grad"]):
            mask = sample_mask(self.table, self.swarm.p, base + 500 + arm)
            cands.append({"kind": "file", "cand_id": len(cands), "pair": None,
                          "arm_id": arm, "grad": True,
                          "mask_blocks": mask.tolist(),
                          "path": f"grad_g{gen}_a{arm}.npz"})
        if self.state["top_last_gen"]:
            path = self._materialize_recomb(gen)
            if path:
                cands.append({"kind": "file", "cand_id": len(cands),
                              "pair": None, "recomb": True, "path": path,
                              "mask_blocks": []})
        if self.state["elite"]:
            cands.append({"kind": "file", "cand_id": len(cands), "pair": None,
                          "elite": True, "path": self.state["elite"]["path"],
                          "mask_blocks": self.state["elite"]["mask_blocks"]})
        return cands

    def _materialize_seed_delta(self, gen: int, cand: dict) -> Path:
        cs = CandidateSpec(gen, cand["cand_id"], cand["mask_seed"],
                           cand["sigma_seed"], cand["value_seed"], cand["sign"])
        delta = self.engine.build_delta(
            cs, np.array(cand["mask_blocks"], dtype=np.int64),
            self.swarm.sigma_groups)
        path = self.evo / "exchange" / f"seed_g{gen}_c{cand['cand_id']}.npz"
        save_sparse_delta(str(path), delta_to_sparse(delta))
        return path

    def _materialize_recomb(self, gen: int) -> str | None:
        parts = self.state["top_last_gen"][:4]
        if not parts:
            return None
        acc: dict[str, dict[int, float]] = {}
        rank = 0
        for ref in parts:
            src = self.evo / "exchange" / ref["path"]
            if not src.exists():  # cleaned transient — skip, don't crash
                continue
            rank += 1
            w = 0.7 * (math.log(5) - math.log(rank))
            data = np.load(src)
            for name in data["tensors"]:
                d = acc.setdefault(str(name), {})
                for i, val in zip(data[f"{name}::idx"], data[f"{name}::val"]):
                    d[int(i)] = d.get(int(i), 0.0) + w * float(val)
        if not acc:
            return None
        sparse = {name: (np.array(sorted(vals)),
                         np.array([vals[i] for i in sorted(vals)], np.float32))
                  for name, vals in acc.items()}
        path = f"recomb_g{gen}.npz"
        save_sparse_delta(str(self.evo / "exchange" / path), sparse)
        self.push_file(self.evo / "exchange" / path)
        return path

    # ---- metrics emission (TUI contract) -----------------------------------
    def _write_layout(self) -> None:
        layout = self.table.layout_json()
        layout.update({"run_id": self.run_id, "t": time.time(),
                       "sigma0": 0.02,
                       "f_broad_theta0": self.state.get("f_broad_theta0"),
                       "f_broad_eps": GUARD_EPS,
                       "thresholdout_budget": self.cfg["thresholdout_budget"]})
        (self.root / "metrics" / "evo_layout.json").write_text(
            json.dumps(layout))

    @staticmethod
    def _q16(v: np.ndarray) -> dict:
        mx = float(max(v.max(), 1e-12))
        q = np.round(65535 * np.clip(v, 0, mx) / mx).astype(np.uint16)
        return {"n": len(v), "max": mx,
                "q": base64.b64encode(q.tobytes()).decode()}

    def _q8s(self, v: np.ndarray) -> dict:
        mx = float(max(np.abs(v).max(), 1e-12))
        q = np.round(127 * np.clip(v / mx, -1, 1)).astype(np.int8)
        return {"n": len(v), "max": mx,
                "q": base64.b64encode(q.tobytes()).decode()}

    def emit_grid(self, gen: int, dmove: np.ndarray) -> None:
        self.metrics.write("grid", gen, {
            "gen": gen, "p": self._q16(self.swarm.p),
            "d": self._q16(dmove), "c": self._q8s(self.swarm.credit)})

    # ---- generation --------------------------------------------------------
    def generation(self) -> None:
        gen = self.state["gen"] + 1
        t_start = time.time()
        n_total = sum(n for n, _ in self.rungs)
        ep_specs = episode_specs_narrow(self.train_tasks, gen, n_total, (0, 1))

        cands = self.make_candidates(gen)
        if self.cfg["n_grad"]:
            inner = episode_specs_narrow(self.train_tasks, gen * 31 + 7,
                                         self.cfg["n_grad"] * 8 * 16, (0, 1))
            self.push_spec(f"g{gen}_grad", {
                "kind": "grad", "gen": gen,
                "champion_promos": self.state["promos"],
                "arms": [{"arm_id": c["arm_id"],
                          "mask_blocks": c["mask_blocks"]}
                         for c in cands if c.get("grad")],
                "inner_episode_specs": inner, "episode_specs": [],
                "k_steps": 8, "lr": 3e-5}, nodes=[self.nodes[0]])
            self.collect(f"g{gen}_grad", nodes=[self.nodes[0]])
            for c in cands:
                if c.get("grad"):
                    if self.nodes[0] != "local":
                        sh(f"rsync -a {self.nodes[0]}:~/fable_god_run/evo/"
                           f"exchange/{c['path']} {self.evo / 'exchange'}/")
                    others = [n for n in self.nodes
                              if n not in ("local", self.nodes[0])]
                    if others:
                        self.push_file(self.evo / "exchange" / c["path"],
                                       nodes=others)

        alive = cands
        offset = 0
        margins: dict[str, list] = {}
        incumbent: list[float] = []
        ep_order: list[str] = []
        for rung, (n_eps, keep) in enumerate(self.rungs, start=1):
            spec = {"kind": "eval", "gen": gen, "rung": rung,
                    "champion_promos": self.state["promos"],
                    "sigma_groups": self.swarm.sigma_groups.tolist(),
                    "episode_specs": ep_specs[offset : offset + n_eps],
                    "candidates": [{k: v for k, v in c.items()
                                    if k != "mask_blocks"} | {
                                       "mask_blocks": c["mask_blocks"]}
                                   for c in alive]}
            self.push_spec(f"g{gen}_r{rung}", spec)
            res = self.collect(f"g{gen}_r{rung}")
            for node in self.nodes:
                r = res[node]
                ep_order += r["episode_ids"]
                incumbent += r["margins"]["__incumbent__"]
                for cid, vals in r["margins"].items():
                    if cid != "__incumbent__":
                        margins.setdefault(cid, []).extend(vals)
            offset += n_eps
            units = self._rank_units(alive, margins, incumbent)
            alive_ids = set()
            for u in units[:keep]:
                alive_ids.update(m["cand_id"] for m in u["members"])
            alive = [c for c in cands if c["cand_id"] in alive_ids]

        challenger_unit = units[0]
        challenger = max(challenger_unit["members"],
                         key=lambda m: m["stats"]["delta"])
        stats = challenger["stats"]

        # swarm + sigma updates from ALL rung-1 units
        all_units = self._rank_units(cands, margins, incumbent)
        self.swarm.update([{"mask_blocks": np.array(u["mask_blocks"],
                                                    dtype=np.int64),
                            "delta": u["delta"]} for u in all_units
                           if len(u["mask_blocks"])])
        # A "success" must be a REAL positive (clears the promotion Δ bar), not
        # just max(±δ)>0 which noise satisfies ~50% of the time for antithetic
        # pairs — that false success was pinning σ at its ceiling.
        successes = sum(1 for u in all_units if u["delta"] > PROMOTE_DELTA)
        self.swarm.fifth_rule(successes, len(all_units))

        if self.state["f_broad_theta0"] is None or gen % 5 == 1:
            probe = self._guard_check(gen, None)
            self.state["f_broad_last"] = probe["incumbent"]
        promoted, gate_info = self._maybe_promote(gen, challenger, stats, cands)
        self._journal(gen, cands, margins, incumbent, ep_order)
        self._emit_gen(gen, cands, margins, incumbent, challenger, stats,
                       promoted, gate_info, time.time() - t_start,
                       len(incumbent))
        self.state["gen"] = gen
        self._persist()
        self._cleanup_exchange(gen)

    def _cleanup_exchange(self, gen: int) -> None:
        """Delete transient deltas from gens < `gen` everywhere. After a
        generation, the only referenced transients (current elite + recomb
        sources) are from THIS gen; champion/ promos live in a subdir and are
        never matched. Numeric rule = robust across ssh (no name lists).
        Without this, exchange/ grows ~50MB/gen → disk-full."""
        import re

        exch = self.evo / "exchange"
        rx = re.compile(r"_g(\d+)")
        for f in exch.glob("*.npz"):
            if not f.name.startswith(("grad_g", "seed_g", "recomb_g")):
                continue
            m = rx.search(f.name)
            if m and int(m.group(1)) < gen:
                f.unlink(missing_ok=True)
        # pure shell param-expansion (no sed → no single quotes that would
        # break out of the ssh '...' wrapper; that bug left remotes uncleaned)
        remote = (
            "cd ~/fable_god_run/evo/exchange 2>/dev/null && "
            "for f in grad_g* seed_g* recomb_g*; do [ -e \"$f\" ] || continue; "
            "n=${f#*_g}; n=${n%%[!0-9]*}; "
            f"[ \"$n\" -lt {gen} ] 2>/dev/null && rm -f \"$f\"; done")
        for node in self.nodes:
            if node != "local":
                subprocess.run(f"ssh {node} '{remote}'", shell=True,
                               capture_output=True)
        # cap journal: keep last 200 + every 25th
        js = sorted(self.evo.joinpath("journal").glob("gen_*.pt"),
                    key=lambda p: int(p.stem.split("_")[1]))
        for p in js[:-200]:
            if int(p.stem.split("_")[1]) % 25 != 0:
                p.unlink(missing_ok=True)

    def _rank_units(self, cands: list[dict], margins: dict,
                    incumbent: list[float]) -> list[dict]:
        by_pair: dict = {}
        for c in cands:
            cid = str(c["cand_id"])
            if cid not in margins:
                continue
            st = paired_stats(margins[cid], incumbent[: len(margins[cid])])
            member = {"cand_id": c["cand_id"], "stats": st, "cand": c}
            key = ("pair", c["pair"]) if c.get("pair") is not None \
                else ("solo", c["cand_id"])
            by_pair.setdefault(key, {"members": [],
                                     "mask_blocks": c["mask_blocks"]})[
                "members"].append(member)
        units = []
        for key, u in by_pair.items():
            deltas = [m["stats"]["delta"] for m in u["members"]]
            if any(not math.isfinite(d) for d in deltas):
                continue  # NaN kills the whole unit
            u["delta"] = max(deltas)
            units.append(u)
        return sorted(units, key=lambda u: -u["delta"])

    def _maybe_promote(self, gen, challenger, stats, cands):
        info = {"wilcoxon_p": stats["wilcoxon_p"], "guard": None,
                "thresholdout": None}
        ok = (stats["wilcoxon_p"] is not None
              and stats["wilcoxon_p"] < PROMOTE_P
              and stats["delta"] >= PROMOTE_DELTA)
        cand = challenger["cand"]
        if cand.get("grad") and self.state["grad_promos_streak"] >= 2:
            ok = False
            info["blocked"] = "grad-arm streak cap"
        if not ok:
            self._set_elite(gen, cand)
            return False, info

        path = (cand["path"] if cand["kind"] == "file"
                else self._materialize_seed_delta(gen, cand).name)
        self.push_file(self.evo / "exchange" / path)

        guard = self._guard_check(gen, path)
        info["guard"] = guard
        if not guard["ok"]:
            self._set_elite(gen, cand, path)
            return False, info
        gate = self._val_gate(gen, path)
        info["thresholdout"] = gate
        if not gate["ok"]:
            self._set_elite(gen, cand, path)
            return False, info

        promo_name = f"promo_{len(self.state['promos']):04d}.npz"
        src = self.evo / "exchange" / path
        dst = self.evo / "exchange" / "champion" / promo_name
        dst.write_bytes(src.read_bytes())
        for node in self.nodes:
            if node != "local":
                sh(f"rsync -a {dst} {node}:~/fable_god_run/evo/exchange/champion/")
        self.engine.apply_sparse_file(str(dst))
        self.engine.refresh_stats()
        self.state["promos"].append(promo_name)
        self.state["grad_promos_streak"] = (
            self.state["grad_promos_streak"] + 1 if cand.get("grad") else 0)
        self.state["elite"] = None
        self.state["top_last_gen"] = []  # its files are cleaned; don't recomb
        self.metrics.event(gen, f"PROMOTED gen {gen} Δ={stats['delta']:+.4f} "
                                f"p={stats['wilcoxon_p']:.4f}")
        return True, info

    def _set_elite(self, gen, cand, path=None):
        if path is None and cand["kind"] == "seed":
            path = self._materialize_seed_delta(gen, cand).name
            self.push_file(self.evo / "exchange" / path)
        elif path is None:
            path = cand["path"]
        # Re-stamp the elite delta under THIS gen's name so the numeric
        # cleanup rule (delete gen < current) always protects it. Without this
        # a carried-forward grad/recomb elite keeps its old gen number and
        # gets deleted while still referenced → the g21 crash we just hit.
        src = self.evo / "exchange" / path
        dst_name = f"seed_g{gen}_elite.npz"
        dst = self.evo / "exchange" / dst_name
        if src.exists() and src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
            self.push_file(dst)
            path = dst_name
        self.state["elite"] = {"path": path,
                               "mask_blocks": cand["mask_blocks"]}
        self.state["top_last_gen"] = [{"path": path}]

    def _guard_check(self, gen, cand_path) -> dict:
        specs = episode_specs_guard(self.train_tasks, gen,
                                    self.cfg["guard_eps"], (0, 1))
        cand_list = ([] if cand_path is None
                     else [{"kind": "file", "cand_id": 0, "path": cand_path}])
        tag = "guard" if cand_path else "guardp"
        self.push_spec(f"g{gen}_{tag}", {
            "kind": "guard", "gen": gen,
            "champion_promos": self.state["promos"],
            "episode_specs": specs, "candidates": cand_list})
        res = self.collect(f"g{gen}_{tag}")
        inc, cha = [], []
        for node in self.nodes:
            inc += res[node]["margins"]["__incumbent__"]
            cha += res[node]["margins"].get("0", [])
        f_inc = float(np.nanmean(inc))
        f_cha = float(np.nanmean(cha)) if cha else f_inc
        if self.state["f_broad_theta0"] is None:
            self.state["f_broad_theta0"] = f_inc
            self._write_layout()
        floor = self.state["f_broad_theta0"] - GUARD_EPS
        return {"ok": f_cha >= floor, "incumbent": f_inc, "challenger": f_cha,
                "floor": floor}

    def _val_gate(self, gen, cand_path) -> dict:
        if self.state["thresholdout_used"] >= self.cfg["thresholdout_budget"]:
            return {"ok": False, "reason": "budget exhausted"}
        rng = np.random.default_rng([gen, 4242])
        tasks = rng.choice(self.val_tasks, size=min(256, len(self.val_tasks)),
                           replace=False)
        specs = [("val", str(t), gen * 600_011 + k)
                 for k, t in enumerate(tasks)]
        self.push_spec(f"g{gen}_val", {
            "kind": "valgate", "gen": gen,
            "champion_promos": self.state["promos"],
            "episode_specs": specs,
            "candidates": [{"kind": "file", "cand_id": 0, "path": cand_path}]})
        res = self.collect(f"g{gen}_val")
        inc, cha = [], []
        for node in self.nodes:
            inc += res[node]["margins"]["__incumbent__"]
            cha += res[node]["margins"]["0"]
        st = paired_stats(cha, inc)
        noise = float(np.random.default_rng([gen, 777]).laplace(0, st["se"]))
        self.state["thresholdout_used"] += 1
        ok = st["delta"] + noise > -st["se"]
        return {"ok": bool(ok), "delta": st["delta"], "se": st["se"],
                "used": self.state["thresholdout_used"]}

    def _journal(self, gen, cands, margins, incumbent, ep_order) -> None:
        save_atomic({"gen": gen, "cands": cands, "margins": margins,
                     "incumbent": incumbent, "episodes": ep_order},
                    self.evo / "journal" / f"gen_{gen:05d}.pt")

    @staticmethod
    def _fin(x, fallback=0.0):
        try:
            return float(x) if math.isfinite(float(x)) else fallback
        except (TypeError, ValueError):
            return fallback

    def _emit_gen(self, gen, cands, margins, incumbent, challenger, stats,
                  promoted, gate_info, wall_s, eps) -> None:
        cand_rows = []
        for c in cands:
            cid = str(c["cand_id"])
            if cid not in margins:
                continue
            st = paired_stats(margins[cid], incumbent[: len(margins[cid])])
            arm = ("grad" if c.get("grad") else
                   "recomb" if c.get("recomb") else
                   "elite" if c.get("elite") else
                   ("es+" if c.get("sign", 1) > 0 else "es-"))
            cand_rows.append({"id": c["cand_id"], "pair": c.get("pair"),
                              "arm": arm, "d": self._fin(st["delta"]),
                              "se": self._fin(st["se"]),
                              "n_ep": st["n"], "w_p": st["wilcoxon_p"]})
        f_inc = self._fin(np.nanmean(incumbent))
        guard_inc = (gate_info.get("guard") or {}).get("incumbent")
        if guard_inc is not None:
            self.state["f_broad_last"] = guard_inc
        f_broad = self._fin(self.state.get("f_broad_last"), f_inc)
        floor = self._fin(
            None if self.state["f_broad_theta0"] is None
            else self.state["f_broad_theta0"] - GUARD_EPS, f_broad - GUARD_EPS)
        self.metrics.write("gen", gen, {
            "gen": gen, "champion_gen": len(self.state["promos"]),
            "f_narrow": f_inc,
            "f_narrow_se": self._fin(np.nanstd(incumbent)
                                     / max(np.sqrt(len(incumbent)), 1)),
            "f_broad": f_broad, "f_broad_se": None,
            "f_broad_floor": floor,
            "sigma": float(self.swarm.sigma_groups.mean() * 0.02),
            "success_rate": self.swarm.success_rate(),
            "p_entropy_bits": self.swarm.entropy_bits(),
            "promoted": promoted, "wilcoxon_p": stats["wilcoxon_p"],
            "accept_id": challenger["cand_id"] if promoted else None,
            "candidates": cand_rows,
            "mask_blocks_winner": challenger["cand"]["mask_blocks"][:256],
            "gen_wall_s": round(wall_s, 1), "eps": eps,
            "thresholdout": {"queried": gate_info.get("thresholdout") is not None,
                             "used": self.state["thresholdout_used"],
                             "verdict": (None if gate_info.get("thresholdout")
                                         is None else
                                         ("PASS" if gate_info["thresholdout"].get("ok")
                                          else "FAIL"))},
        })
        dmove = self._cumulative_move()
        self.emit_grid(gen, dmove)

    def _cumulative_move(self) -> np.ndarray:
        move = np.zeros(self.table.n_blocks)
        for promo in self.state["promos"]:
            data = np.load(self.evo / "exchange" / "champion" / promo)
            for name in data["tensors"]:
                idx = data[f"{name}::idx"]
                val = data[f"{name}::val"]
                for b in self.table.blocks:
                    if b.tensor == str(name):
                        sel = (idx >= b.start) & (idx < b.end)
                        if sel.any():
                            move[b.id] += float((val[sel] ** 2).sum())
        return np.sqrt(move)

    def _persist(self) -> None:
        self.swarm.save(str(self.evo / "state" / "swarm.npz"))
        tmp = self.evo / "state" / "state.json.tmp"
        tmp.write_text(json.dumps(self.state))
        tmp.rename(self.evo / "state" / "state.json")

    def run(self, max_gens: int) -> None:
        fails = 0
        while self.state["gen"] < max_gens:
            try:
                self.generation()
                fails = 0
            except Exception as err:
                fails += 1
                self.metrics.event(self.state["gen"],
                                   f"generation failed ({fails}/3): {err!r}",
                                   level="error")
                print(f"generation failed ({fails}/3): {err!r}", flush=True)
                if fails >= 3:
                    raise
                time.sleep(30)  # workers recover (empty_cache), then retry
                continue
            print(f"gen {self.state['gen']} done "
                  f"(promos={len(self.state['promos'])})", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.home() / "fable_god_run")
    parser.add_argument("--max-gens", type=int, default=1000)
    parser.add_argument("--n-pairs", type=int, default=7)
    parser.add_argument("--n-grad", type=int, default=4)
    parser.add_argument("--guard-eps-episodes", type=int, default=512)
    parser.add_argument("--thresholdout-budget", type=int, default=40)
    parser.add_argument("--nodes", default="xbabe1,local,xbabe3")
    parser.add_argument("--rungs", default="128:6,256:3,640:1")
    parser.add_argument("--null", action="store_true",
                        help="sigma=0 trust test: all deltas exactly zero; "
                             "the promotion gate must never fire")
    args = parser.parse_args()
    rungs = [(int(a), int(b)) for a, b in
             (part.split(":") for part in args.rungs.split(","))]
    cfg = {"n_pairs": args.n_pairs, "n_grad": args.n_grad,
           "guard_eps": args.guard_eps_episodes,
           "thresholdout_budget": args.thresholdout_budget,
           "nodes": args.nodes.split(","), "rungs": rungs}
    driver = Driver(args.root, cfg)
    if args.null:
        driver.swarm.sigma_groups[:] = 0.0
    driver.run(args.max_gens)


if __name__ == "__main__":
    main()
