"""Consume version-verified H278 frozen exports in read-only shadow mode."""

from pathlib import Path
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import time
import numpy as np
from h278_metrics import compare, advisory_gate, frozen_picks

DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
SEEDS = (278101, 278102, 278103)
IDENTITY = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    used = sum(p.stat().st_size for p in path.parent.rglob("*") if p.is_file())
    if used + len(raw) > 90_000_000:
        raise RuntimeError("shadow output ceiling; no deletion/writer action")
    temp = path.with_suffix(path.suffix + ".new")
    temp.write_bytes(raw)
    os.replace(temp, path)


def validate_packet_alignment(exported, local):
    """Bind candidate/order columns exactly; use exported features without alteration.

    Only the last context-norm column may differ by one float32 ULP in the local
    reference computation. It is not used to recover candidate identity.
    """
    assert exported.shape == local.shape == (91, 8, 568)
    assert exported.dtype == local.dtype == np.float32
    assert np.array_equal(exported[:, :, :567], local[:, :, :567])
    a = exported[:, :, 567].copy()
    b = local[:, :, 567].copy()
    assert (
        np.isfinite(a).all()
        and np.isfinite(b).all()
        and (a >= 0).all()
        and (b >= 0).all()
    )
    ulp = np.abs(a.view(np.int32).astype(np.int64) - b.view(np.int32).astype(np.int64))
    assert ulp.max() <= 1
    return {
        "candidate_and_order_columns_bit_exact": True,
        "exported_features_used_unmodified": True,
        "local_context_norm_max_float32_ULP_difference": int(ulp.max()),
        "local_context_norm_differing_contexts": int((ulp > 0).any(axis=1).sum()),
    }


class FrozenExport:
    def __init__(self, root, verification, old_packet, p2_packet, pool_input):
        import lightgbm as lgb

        self.root = Path(root)
        v = json.loads(Path(verification).read_text())
        assert v["all_objects_exact_VersionId_and_sha256_verified"]
        m = json.loads((self.root / "export_manifest.json").read_text())
        assert sha(self.root / "export_manifest.json") == v["manifest"]["sha256"]
        assert (
            m["schema"] == "jope.h278.frozen-rerank8-veto/v1"
            and m["source_commit"] == v["source_commit"]
        )
        assert (
            m["arm"] == "RERANK8_VETO"
            and m["fit_split"] == "DL-Train"
            and m["dl_val_reads_before_lock"] == 0
        )
        assert m["DOPE_sha256"] == DOPE and m["zero_updates"]
        for rec in v["objects"]:
            assert sha(self.root / rec["file"]) == rec["sha256"]
        self.manifest = m
        self.verification = v
        self.pool_input = Path(pool_input)
        self.meta = json.loads((Path(old_packet) / "packet.json").read_text())
        assert sorted(m["training_family_ids"]) == self.meta["training_family_ids"]
        assert not set(m["training_family_ids"]) & {
            r["family_id"] for r in self.meta["contexts"]
        }
        assert (
            len(self.meta["contexts"]) == 91
            and len({r["family_id"] for r in self.meta["contexts"]}) == 19
        )
        self.models = {}
        for cell in m["cells"]:
            assert cell["selected_tau"] == 0.1 and cell["seed"] in SEEDS
            rank = lgb.Booster(model_file=str(self.root / cell["ranker"]["file"]))
            veto = lgb.Booster(model_file=str(self.root / cell["veto"]["file"]))
            assert rank.num_feature() == veto.num_feature() == 568
            self.models[cell["seed"]] = (cell, rank, veto)
        assert set(self.models) == set(SEEDS)
        self.packets = {}
        self.packet_alignment = {}
        for name, folder, export_name in [
            ("P1_R882_TOP8", Path(old_packet), "dev-features-r882-top8.npz"),
            (
                "P2_CPU_R3175_TOP8_EXPLORATORY",
                Path(p2_packet),
                "dev-features-r3175-top8.npz",
            ),
        ]:
            meta = json.loads((folder / "packet.json").read_text())
            assert sha(folder / "dev-packet.npz") == meta["packet_sha256"]
            with np.load(folder / "dev-packet.npz", allow_pickle=False) as z:
                packet = {k: z[k].copy() for k in z.files}
            with np.load(self.root / export_name, allow_pickle=False) as z:
                assert set(z.files) == {"X", "cids"}
                X = z["X"].copy()
                cids = z["cids"].tolist()
            assert (
                X.shape == (91, 8, 568)
                and X.dtype == np.float32
                and np.isfinite(X).all()
            )
            assert cids == [r["cid"] for r in meta["contexts"]]
            # The shipped H278 features are authoritative; exact equality proves local pick/utility alignment.
            self.packet_alignment[name] = validate_packet_alignment(X, packet["X"])
            gates = []
            for i, r in enumerate(meta["contexts"]):
                path = self.pool_input / "pools" / (r["cid"] + ".npz")
                assert sha(path) == r["pool_sha256"]
                with np.load(path, allow_pickle=False) as z:
                    pids = [
                        x.decode() if isinstance(x, bytes) else str(x)
                        for x in z["program_sha256"]
                    ]
                    gates.append(
                        [bool(z["gate"][pids.index(x)]) for x in packet["pids"][i]]
                    )
            packet.update(X=X, gate=np.asarray(gates, bool), meta=meta)
            self.packets[name] = packet
        self.cids = [r["cid"] for r in self.meta["contexts"]]
        self.families = [r["family_id"] for r in self.meta["contexts"]]
        # Pure frozen inference. No fit, optimizer, candidate outcome or writer state enters prediction.
        self.predictions = {}
        for name, packet in self.packets.items():
            self.predictions[name] = {}
            X = packet["X"].reshape(-1, 568)
            for seed, (cell, rank, veto) in self.models.items():
                sc = rank.predict(
                    X, num_iteration=cell["selected_n_estimators"], num_threads=1
                ).reshape(91, 8)
                p = veto.predict(X, num_threads=1).reshape(91, 8)
                margin = (
                    math.inf
                    if cell["strict_margin"] == "inf"
                    else float(cell["strict_margin"])
                )
                self.predictions[name][seed] = frozen_picks(sc, p, margin)
        self.frozen_picks_sha256 = hashlib.sha256(
            json.dumps(
                {
                    n: {str(k): v.tolist() for k, v in p.items()}
                    for n, p in self.predictions.items()
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def score_round(self, round_path, p2_verdict="PENDING"):
        r = json.loads(Path(round_path).read_text())
        assert (
            r["completed_round"]
            and r["dope_sha256"] == DOPE
            and r["dope_checksum_match"]
        )
        assert r["evaluation_identity"] == IDENTITY and r["trainable_params"] == 29211
        for key, c in r["parameter_counts"].items():
            assert c["trainable_params"] == (
                29211 if key in ["adapter_10tok", "__global_total__"] else 0
            )
        order = self.meta["live_dev_order"]
        u = r["dev_utility"]["per_context"]
        ids = r["dev_utility"]["selected_programs"]
        assert len(order) == len(u) == len(ids) == 91
        lookup = {c: (p, v) for c, p, v in zip(order, ids, u)}
        inc = np.asarray([lookup[c][1] for c in self.cids])
        incgate = []
        for c in self.cids:
            with np.load(
                self.pool_input / "pools" / (c + ".npz"), allow_pickle=False
            ) as z:
                pids = [
                    p.decode() if isinstance(p, bytes) else str(p)
                    for p in z["program_sha256"]
                ]
                j = pids.index(lookup[c][0])
                gate = bool(z["gate"][j])
                value = -float(z["fitness"][j]) if gate else -2.0
            assert value == lookup[c][1]
            incgate.append(gate)
        panels = {}
        for name, predictions in self.predictions.items():
            packet = self.packets[name]
            seeds = {}
            for seed, picks in predictions.items():
                values = packet["utility"][np.arange(91), picks]
                gate = packet["gate"][np.arange(91), picks]
                seeds[str(seed)] = {
                    **compare(inc, values, self.families, incgate, gate),
                    "picks": [
                        {
                            "cid": c,
                            "incumbent_pid": lookup[c][0],
                            "shadow_pid": str(packet["pids"][i, picks[i]]),
                            "slot": int(picks[i]),
                            "incumbent_utility": float(inc[i]),
                            "shadow_utility": float(values[i]),
                            "invalid": bool(not gate[i]),
                        }
                        for i, c in enumerate(self.cids)
                    ],
                }
            panels[name] = {
                "status": "SCORED",
                "seeds": seeds,
                "candidate_pool_scope": "r882 frozen top8"
                if name.startswith("P1")
                else "CPU-reproducible r3175 top8; native reference mismatch90/91; exploratory only",
            }
        return {
            "schema": "jope.h278.shadow-round/v1",
            "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "round": r["round"],
            "completed_round": True,
            "status": "SCORED",
            "mode": "READ_ONLY_SHADOW",
            "source_round_sha256": sha(round_path),
            "model_manifest_sha256": self.verification["manifest"]["sha256"],
            "model_manifest_VersionId": self.verification["manifest"]["version_id"],
            "model_source_commit": self.manifest["source_commit"],
            "DOPE_sha256": DOPE,
            "frozen_picks_sha256": self.frozen_picks_sha256,
            "live_incumbent_best_round": r["adapter_admission"]["best_round"],
            "live_incumbent_utility": r["dev_utility"]["mean"],
            "evaluation_identity": IDENTITY,
            "panels": panels,
            "H278_P2_verdict": p2_verdict,
            "fit_calls": 0,
            "weight_updates": 0,
            "promotion_wiring": False,
            "adoption_authorized": False,
            "decision_owner": "watcher",
        }


def main():
    p = argparse.ArgumentParser()
    for x in [
        "export",
        "verification",
        "old-packet",
        "p2-packet",
        "pool-input",
        "run",
        "output",
        "p2-metrics",
    ]:
        p.add_argument("--" + x, type=Path, required=True)
    p.add_argument("--writer-pid", type=int, required=True)
    p.add_argument("--writer-created", type=float, required=True)
    p.add_argument("--first-round", type=int, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--max-rounds", type=int, default=40)
    a = p.parse_args()
    assert (
        not a.output.resolve().is_relative_to(a.run.resolve())
        and 0 < a.max_rounds <= 40
    )
    import psutil

    writer = psutil.Process(a.writer_pid)
    assert writer.create_time() == a.writer_created
    frozen = FrozenExport(
        a.export, a.verification, a.old_packet, a.p2_packet, a.pool_input
    )
    metrics = json.loads(a.p2_metrics.read_text())
    assert metrics["source_commit"] == frozen.manifest["source_commit"]
    p2verdict = metrics["verdict"]
    rows = []
    write(
        a.output / "LOAD_VALIDATION.json",
        {
            "all3_native_text_model_pairs_loaded": True,
            "all_exported_feature_packets_align": True,
            "packet_alignment": frozen.packet_alignment,
            "fit_calls": 0,
            "source_commit": frozen.manifest["source_commit"],
            "frozen_picks_sha256": frozen.frozen_picks_sha256,
            "H278_P2_verdict": p2verdict,
            "promotion_wiring": False,
        },
    )
    while time.time() < a.deadline_epoch and len(rows) < a.max_rounds:
        own = psutil.Process()
        if own.num_threads() > 4 or own.memory_info().rss > 2 * 2**30:
            raise RuntimeError("shadow resource ceiling")
        if (
            not writer.is_running()
            or writer.status() == psutil.STATUS_ZOMBIE
            or writer.create_time() != a.writer_created
        ):
            break
        seen = {r["round"] for r in rows}
        for path in sorted(
            (
                p
                for p in a.run.glob("round-*.json")
                if p.stem.removeprefix("round-").isdigit()
            ),
            key=lambda p: int(p.stem.split("-")[1]),
        ):
            n = int(path.stem.split("-")[1])
            if n < a.first_round or n in seen:
                continue
            r = frozen.score_round(path, p2verdict)
            write(a.output / f"round-{n}.json", r)
            rows.append(r)
            if len(rows) == 1:
                write(
                    a.output / "FIRST_NON_NULL.json",
                    {"round": n, "path": str(a.output / f"round-{n}.json")},
                )
            gate_rows = [
                {
                    "round": x["round"],
                    "status": x["status"],
                    "completed_round": True,
                    "seeds": x["panels"]["P1_R882_TOP8"]["seeds"],
                }
                for x in rows
            ]
            write(a.output / "ADVISORY_GATE.json", advisory_gate(gate_rows, p2verdict))
            print(
                json.dumps(
                    {
                        "round": n,
                        "status": "SCORED",
                        "H278_P2_verdict": p2verdict,
                        "P1_seed_readout": {
                            s: {
                                k: v
                                for k, v in t.items()
                                if k
                                in [
                                    "family_median_delta",
                                    "Elo_lift",
                                    "Elo_lift_ci95",
                                    "shadow_invalid_picks",
                                    "incumbent_invalid_picks",
                                    "harmed_families",
                                ]
                            }
                            for s, t in r["panels"]["P1_R882_TOP8"]["seeds"].items()
                        },
                    }
                ),
                flush=True,
            )
            if len(rows) >= a.max_rounds:
                break
        time.sleep(15)
    write(
        a.output / "WATCH_STOP.json",
        {
            "rounds": [r["round"] for r in rows],
            "reason": "bound_or_writer_exit",
            "adoption_authorized": False,
        },
    )


if __name__ == "__main__":
    main()
