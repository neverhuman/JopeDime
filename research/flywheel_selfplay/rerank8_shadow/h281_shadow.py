"""H281 GPU-exact P2 shadow, with unique-pool evidence rather than repeated-round credit."""

from pathlib import Path
import argparse
import hashlib
import json
import signal
import time
import numpy as np
from h278_shadow import FrozenExport, sha, write, IDENTITY, DOPE
from h278_metrics import frozen_picks
from pool_evidence import UniquePoolEvidence, pool_sha256

GPU_PANEL = "P2_GPU_EXACT_R3175_TOP8"


class GPUExactExport(FrozenExport):
    def add_gpu_packet(self, folder, verification):
        folder = Path(folder)
        v = json.loads(Path(verification).read_text())
        assert (
            v["all5_exact_VersionId_SHA_verified"]
            and v["H281_verdict"] == "ADOPT_SHADOW_PASS"
        )
        artifacts = folder.parent / "artifacts"
        for rec in v["objects"]:
            p = artifacts / Path(rec["s3_uri"]).name
            assert sha(p) == rec["sha256"]
        meta = json.loads((folder / "packet.json").read_text())
        assert sha(folder / "dev-packet.npz") == meta["packet_sha256"]
        assert [r["cid"] for r in meta["contexts"]] == self.cids
        with np.load(folder / "dev-packet.npz", allow_pickle=False) as z:
            packet = {k: z[k].copy() for k in z.files}
        assert packet["X"].shape == (91, 8, 568) and np.isfinite(packet["X"]).all()
        self.packets[GPU_PANEL] = {**packet, "meta": meta}
        self.predictions[GPU_PANEL] = {}
        for seed, (cell, rank, veto) in self.models.items():
            X = packet["X"].reshape(-1, 568)
            scores = rank.predict(
                X, num_iteration=cell["selected_n_estimators"], num_threads=1
            ).reshape(91, 8)
            probs = veto.predict(X, num_threads=1).reshape(91, 8)
            margin = (
                float("inf")
                if cell["strict_margin"] == "inf"
                else float(cell["strict_margin"])
            )
            self.predictions[GPU_PANEL][seed] = frozen_picks(scores, probs, margin)
        self.frozen_picks_sha256 = hashlib.sha256(
            json.dumps(
                {
                    n: {str(k): v.tolist() for k, v in p.items()}
                    for n, p in self.predictions.items()
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        expected = json.loads((artifacts / "p2-001-p2_picks.json").read_text())
        byseed = {r["seed"]: r["picks"] for r in expected}
        utilities = []
        invalid = []
        for seed in [278101, 278102, 278103]:
            picks = self.predictions[GPU_PANEL][seed]
            assert picks.tolist() == byseed[seed]
            utilities.append(
                np.stack(
                    [packet["utility"][:, 0], packet["utility"][np.arange(91), picks]],
                    axis=-1,
                )
            )
            invalid.append(
                np.stack(
                    [~packet["gate"][:, 0], ~packet["gate"][np.arange(91), picks]],
                    axis=-1,
                )
            )
        with np.load(artifacts / "p2-001-p2_panel.npz", allow_pickle=False) as z:
            assert np.array_equal(np.asarray(utilities), z["utility"])
            assert np.array_equal(np.asarray(invalid), z["invalid"])
        self.h281_verification = v
        self.gpu_meta = meta
        return {
            "H281_reference_picks_exact": 273,
            "H281_p2_utility_panel_bit_exact": True,
            "H281_p2_invalid_panel_bit_exact": True,
            "H281_verdict": "ADOPT_SHADOW_PASS",
            "GPU_used_here": False,
            "fit_calls": 0,
            "packet_sha256": meta["packet_sha256"],
            "H281_manifest": v["manifest"],
        }

    def revised_round(self, path, run):
        r = super().score_round(path, "ADOPT_SHADOW_PASS")
        br = r["live_incumbent_best_round"]
        accepted = json.loads((Path(run) / f"round-{br}.json").read_text())
        assert accepted["adapter_admission"]["accepted"]
        inc = accepted.get(
            "candidate_parameter_sha256",
            accepted["adapter_admission"].get("candidate_parameter_sha256"),
        )
        pool = self.packets[GPU_PANEL]
        r["pool_sha256"] = pool_sha256(
            self.cids,
            pool["pids"].tolist(),
            self.gpu_meta["pool_anchor_parameters_sha256"],
            inc,
            IDENTITY,
            DOPE,
        )
        r.update(
            schema="jope.h281.unique-pool-shadow-round/v1",
            scoring_status="SCORED",
            gate_panel=GPU_PANEL,
            live_incumbent_parameters_sha256=inc,
            pool_anchor_parameters_sha256=self.gpu_meta[
                "pool_anchor_parameters_sha256"
            ],
            historical_H278_CPU_P2_verdict="P2_BLOCKED",
            P2_qualification_source="H281 GPU-exact r3175 replay",
            H281_manifest_VersionId=self.h281_verification["manifest"]["version_id"],
            H281_manifest_sha256=self.h281_verification["manifest"]["sha256"],
            pool_hash_policy="semantic candidate membership plus anchor/incumbent parameter identities; no round/time/serialization/order noise",
        )
        r["panels"][GPU_PANEL]["candidate_pool_scope"] = (
            "H281 GPU-exact r3175 top8;91/91 stored native baseline picks/utilities"
        )
        return r


def combined_size(roots):
    return sum(
        p.stat().st_size
        for root in roots
        if root.exists()
        for p in root.rglob("*")
        if p.is_file()
    )


def main():
    p = argparse.ArgumentParser()
    for x in [
        "export",
        "verification",
        "old-packet",
        "p2-packet",
        "pool-input",
        "gpu-packet",
        "h281-verification",
        "run",
        "output",
        "state",
        "legacy-dir",
    ]:
        p.add_argument("--" + x, type=Path, required=True)
    p.add_argument("--writer-pid", type=int, required=True)
    p.add_argument("--writer-created", type=float, required=True)
    p.add_argument("--deadline-epoch", type=float, required=True)
    p.add_argument("--max-records", type=int, default=180)
    a = p.parse_args()
    assert 0 < a.max_records <= 180 and not a.output.resolve().is_relative_to(
        a.run.resolve()
    )
    import psutil

    writer = psutil.Process(a.writer_pid)
    assert writer.create_time() == a.writer_created
    f = GPUExactExport(
        a.export, a.verification, a.old_packet, a.p2_packet, a.pool_input
    )
    regression = f.add_gpu_packet(a.gpu_packet, a.h281_verification)
    write(a.state.parent / "H281_REGRESSION.json", regression)
    rows = sorted(
        (
            p
            for p in a.output.glob("round-*.json")
            if p.stem.removeprefix("round-").isdigit()
        ),
        key=lambda p: int(p.stem.split("-")[1]),
    )
    committed = [
        json.loads(p.read_text())
        for p in rows
        if json.loads(p.read_text()).get("schema")
        == "jope.h281.unique-pool-shadow-round/v1"
    ]
    ledger = UniquePoolEvidence.replay(committed)
    # First run migrates only preserved copies, never editing evidence in place without a recovery route.
    for old in sorted(
        a.legacy_dir.glob("round-*.json"), key=lambda p: int(p.stem.split("-")[1])
    ):
        previous = json.loads(old.read_text())
        n = previous["round"]
        destination = a.output / old.name
        if (
            destination.exists()
            and json.loads(destination.read_text()).get("schema")
            == "jope.h281.unique-pool-shadow-round/v1"
        ):
            continue
        r = f.revised_round(a.run / old.name, a.run)
        for panel in previous["panels"]:
            assert r["panels"][panel] == previous["panels"][panel]
        r.update(
            record_mode="RETROSPECTIVE_CONTRACT_REVISION",
            legacy_record_sha256=sha(old),
            legacy_record_path=str(old),
        )
        ledger.annotate(r, historical=True)
        write(destination, r)
    write(a.state, ledger.state())
    write(a.output / "ADVISORY_GATE.json", ledger.gate())
    write(
        a.output / "CONTRACT_STATUS.json",
        {
            "H278_P2_verdict": "ADOPT_SHADOW_PASS",
            "P2_qualification_source": "H281",
            "historical_H278_CPU_P2_verdict": "P2_BLOCKED",
            "H280": "PENDING_QUALIFICATION",
            "H282": "FRESH_POOLS_EXPECTED",
            "pool_gate_policy": "unique scored pool hashes only; repeated frozen panels add no evidence",
            "source": "h281_shadow",
            "adoption_authorized": False,
            "promotion_wiring": False,
            "serving_wiring": False,
        },
    )
    stop = False

    def stopping(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, stopping)
    signal.signal(signal.SIGINT, stopping)
    while not stop and time.time() < a.deadline_epoch:
        own = psutil.Process()
        if own.num_threads() > 2 or own.memory_info().rss > 2 * 2**30:
            raise RuntimeError("shadow resource ceiling")
        if (
            combined_size([a.old_packet, a.output.parent, a.gpu_packet.parent])
            >= 98_000_000
        ):
            raise RuntimeError("combined shadow size ceiling; no deletion")
        if (
            not writer.is_running()
            or writer.status() == psutil.STATUS_ZOMBIE
            or writer.create_time() != a.writer_created
        ):
            break
        done = {
            int(p.stem.split("-")[1])
            for p in a.output.glob("round-*.json")
            if p.stem.removeprefix("round-").isdigit()
        }
        if len(done) >= a.max_records:
            break
        first = (
            max(
                [int(p.stem.split("-")[1]) for p in a.legacy_dir.glob("round-*.json")],
                default=0,
            )
            + 1
        )
        for path in sorted(
            (
                p
                for p in a.run.glob("round-*.json")
                if p.stem.removeprefix("round-").isdigit()
            ),
            key=lambda p: int(p.stem.split("-")[1]),
        ):
            n = int(path.stem.split("-")[1])
            if n < first or n in done:
                continue
            r = f.revised_round(path, a.run)
            r["record_mode"] = "LIVE_SHADOW"
            ledger.annotate(r)
            write(a.output / path.name, r)
            write(a.state, ledger.state())
            write(a.output / "ADVISORY_GATE.json", ledger.gate())
            first_live = a.state.parent / "FIRST_GPU_EXACT_LIVE.json"
            if not first_live.exists():
                write(
                    first_live,
                    {
                        "round": n,
                        "pool_sha256": r["pool_sha256"],
                        "counts_toward_gate": r["counts_toward_gate"],
                    },
                )
            print(
                json.dumps(
                    {
                        "round": n,
                        "pool_sha256": r["pool_sha256"],
                        "counts_toward_gate": r["counts_toward_gate"],
                        "status": r["status"],
                        "H278_P2_verdict": r["H278_P2_verdict"],
                        "gate": ledger.gate(),
                    }
                ),
                flush=True,
            )
            if stop:
                break
        time.sleep(10)
    write(
        a.state.parent / "WATCH_STOP.json",
        {
            "reason": "bound_writer_exit_or_own_stop",
            "gate": ledger.gate(),
            "adoption_authorized": False,
        },
    )


if __name__ == "__main__":
    main()
