"""Per-node evolution worker: watches an inbox for JSON work specs, evaluates
candidates on its episode shard, writes per-episode margins to the outbox.

Spec kinds:
- "eval":  {gen, rung, episode_specs, candidates, champion_promos, p_snapshot?}
           candidates: [{"kind":"seed", cand_id, sign, mask_seed, sigma_seed,
                         value_seed, mask_blocks}, {"kind":"file", cand_id,
                         path}] — mask_blocks shipped explicitly (sampled by
           the driver from p) so workers never need p.
- "grad":  {gen, arm_id, mask_blocks, inner_episode_specs} → writes a sparse
           delta npz to the outbox (xbabe1 only).
- "guard"/"probe": episode specs evaluated on the incumbent only.

The incumbent champion = θ₀(step_1500 EMA) + all promotion deltas in
exchange/champion/, applied in order; verified by checksum in every spec.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .ckpt import CheckpointManager
from .eval_protocol import ProtocolRunner
from .evo_blocks import BlockTable, CandidateSpec, DeltaEngine
from .evo_episodes import EpisodeProducer, build_episode, _pool_init
from .evo_fitness import candidate_margins, grad_arm_delta
from .model_io import CONFIG_NAME, default_weights_dir
from .paths import import_regressor_train


class Worker:
    def __init__(self, root: Path, device: str, workers: int):
        self.root = root
        self.inbox = root / "evo" / "inbox"
        self.outbox = root / "evo" / "outbox"
        self.exchange = root / "evo" / "exchange"
        for d in (self.inbox, self.outbox, self.exchange):
            d.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(device)

        tm = import_regressor_train()
        cfg = json.loads((default_weights_dir() / CONFIG_NAME).read_text())
        self.model = tm.get_architecture(tm.TabPFNV3Config.from_dict(cfg))
        payload = CheckpointManager.load(root / "ckpt" / "step_1500.ckpt",
                                         map_location="cpu")
        self.model.load_state_dict(payload["ema_model"], strict=True)
        self.model = self.model.to(self.device).eval()
        self.table = BlockTable(self.model.state_dict())
        self.engine = DeltaEngine(self.model, self.table)
        self.runner = ProtocolRunner(self.model, self.device)
        self.applied_promos: list[str] = []
        self.producer = EpisodeProducer(root / "pack", workers=workers)
        self.incumbent_cache: dict[tuple, list[float]] = {}

    def sync_champion(self, promos: list[str]) -> None:
        for name in promos:
            if name in self.applied_promos:
                continue
            self.engine.apply_sparse_file(str(self.exchange / "champion" / name))
            self.applied_promos.append(name)
        self.engine.refresh_stats()
        self.incumbent_cache.clear()

    def episodes_for(self, specs: list) -> list:
        return self.producer.prepare([tuple(s) for s in specs])

    def handle_eval(self, spec: dict) -> dict:
        self.sync_champion(spec.get("champion_promos", []))
        episodes = self.episodes_for(spec["episode_specs"])
        ep_ids = [f"{e.task}:{e.seed}" for e in episodes]

        cache_key = (spec["gen"], spec["rung"])
        incumbent = candidate_margins(self.runner, episodes)
        results = {"__incumbent__": incumbent}
        sigma_groups = np.array(spec.get("sigma_groups", [1.0] * 5))

        for cand in spec["candidates"]:
            if cand["kind"] == "seed":
                cs = CandidateSpec(spec["gen"], cand["cand_id"],
                                   cand["mask_seed"], cand["sigma_seed"],
                                   cand["value_seed"], cand.get("sign", 1))
                mask = np.array(cand["mask_blocks"], dtype=np.int64)
                delta = self.engine.build_delta(cs, mask, sigma_groups)
                saved = self.engine.apply(delta)
                try:
                    results[str(cand["cand_id"])] = candidate_margins(
                        self.runner, episodes)
                finally:
                    self.engine.restore(delta, saved)
            else:  # file-based (grad/recomb/elite)
                path = self.exchange / cand["path"]
                saved = self.engine.apply_sparse_file_saved(str(path))
                try:
                    results[str(cand["cand_id"])] = candidate_margins(
                        self.runner, episodes)
                finally:
                    self.engine.restore_sparse_saved(saved)
        return {"kind": "eval", "gen": spec["gen"], "rung": spec["rung"],
                "episode_ids": ep_ids, "margins": results,
                "checksum": self.engine.checksum()}

    def handle_grad(self, spec: dict) -> dict:
        self.sync_champion(spec.get("champion_promos", []))
        inner = self.episodes_for(spec["inner_episode_specs"])
        for arm in spec["arms"]:
            sparse = grad_arm_delta(
                self.model, self.engine,
                np.array(arm["mask_blocks"], dtype=np.int64), inner,
                self.device, k_steps=spec.get("k_steps", 8),
                lr=spec.get("lr", 3e-5))
            from .evo_blocks import save_sparse_delta

            out = self.exchange / f"grad_g{spec['gen']}_a{arm['arm_id']}.npz"
            save_sparse_delta(str(out), sparse)
        return {"kind": "grad", "gen": spec["gen"], "done": True,
                "checksum": self.engine.checksum()}

    def handle_incumbent(self, spec: dict) -> dict:
        """guard / probe / val-gate: incumbent (and optionally challenger)."""
        self.sync_champion(spec.get("champion_promos", []))
        episodes = self.episodes_for(spec["episode_specs"])
        if spec.get("corrupt_context"):
            for ep in episodes:  # context-corruption probe: shuffle ctx labels
                rng = np.random.default_rng([ep.seed, 999])
                ep.ctx_y = ep.ctx_y[rng.permutation(len(ep.ctx_y))]
        out = {"__incumbent__": candidate_margins(self.runner, episodes)}
        for cand in spec.get("candidates", []):
            path = self.exchange / cand["path"]
            saved = self.engine.apply_sparse_file_saved(str(path))
            try:
                out[str(cand["cand_id"])] = candidate_margins(self.runner, episodes)
            finally:
                self.engine.restore_sparse_saved(saved)
        return {"kind": spec["kind"], "gen": spec["gen"], "margins": out,
                "episode_ids": [f"{e.task}:{e.seed}" for e in episodes]}

    def loop(self) -> None:
        print(f"worker up: {self.table.n_blocks} blocks, "
              f"checksum {self.engine.checksum()}", flush=True)
        while True:
            specs = sorted(self.inbox.glob("*.json"))
            if not specs:
                time.sleep(1.0)
                continue
            path = specs[0]
            try:
                spec = json.loads(path.read_text())
            except json.JSONDecodeError:
                time.sleep(0.5)
                continue
            t0 = time.time()
            handler = {"eval": self.handle_eval, "grad": self.handle_grad,
                       "guard": self.handle_incumbent,
                       "probe": self.handle_incumbent,
                       "valgate": self.handle_incumbent}[spec["kind"]]
            try:
                result = handler(spec)
                result["seconds"] = round(time.time() - t0, 1)
            except Exception as err:  # report, never die silently
                import traceback

                result = {"kind": "error", "gen": spec.get("gen"),
                          "error": repr(err),
                          "trace": traceback.format_exc()[-2000:]}
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # reclaim after OOM paths
            out = self.outbox / (path.stem + ".result.json")
            tmp = out.with_suffix(".tmp")
            tmp.write_text(json.dumps(result))
            tmp.rename(out)
            path.unlink(missing_ok=True)  # driver restart may clear inbox mid-run
            print(f"done {path.stem} in {result.get('seconds', '?')}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.home() / "fable_god_run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=48)
    args = parser.parse_args()
    Worker(args.root, args.device, args.workers).loop()


if __name__ == "__main__":
    main()
