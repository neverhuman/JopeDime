"""One frozen-model attempt: native identity, revealed invalid scores, FIT/B lock, A/Val."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import numpy as np
import torch
import analysis as stat
from parents import ROOT, h188, h192, h177
from treatment import invalid_share, revealed_invalid, veto_pick

HERE = Path(__file__).resolve().parent
sha, save = h192.sha, h192.save


def forbidden_training(*args, **kwargs):
    raise RuntimeError("H193 forbids optimizer construction/backward")


def verify(pins):
    for kind in ("inputs", "assets", "parent_sources", "source_sha256"):
        for name, digest in pins[kind].items():
            path = (
                ROOT / name
                if kind == "parent_sources"
                else HERE / name
                if kind == "source_sha256"
                else Path(name)
            )
            if sha(path) != digest:
                raise RuntimeError(f"{kind} pin mismatch: {path}")
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package " + name)


class Engine(h192.Engine):
    """Native forward/batch/load/check inherited; new output is only revealed invalid share."""

    def check(self):
        super().check()
        if (
            self.rt.tensor_hash(h192.adapter_state(self.net, self.names))
            != self.initial_hash
        ):
            raise RuntimeError("adapter state drift")

    def probabilities(self, rows):
        self.check()
        groups = {}
        result = {}
        for row in rows:
            cid = row["context_sha256"]
            groups.setdefault(len(self.pools[cid]["fitness"]), []).append(cid)
        with torch.inference_mode():
            for k, cids in sorted(groups.items()):
                for start in range(0, len(cids), 8):
                    batch = cids[start : start + 8]
                    kwargs, tokens, masks, _, _ = self.batch(
                        batch, [np.arange(k) for _ in batch]
                    )
                    if any(cid not in self.cached for cid in batch):
                        anchor = self.anchored(kwargs)
                        for i, cid in enumerate(batch):
                            self.cached[cid] = tuple(t[i].clone() for t in anchor)
                    prior = torch.stack([self.cached[c][1] for c in batch])
                    static = h188.static_terms(
                        self.net,
                        kwargs["prog_412"],
                        kwargs["sos_prior"],
                        tokens,
                        masks,
                        prior,
                    )
                    indicators = torch.as_tensor(
                        np.stack(
                            [
                                revealed_invalid(
                                    self.pools[c]["fitness"][:4],
                                    self.pools[c]["gate"][:4],
                                )
                                for c in batch
                            ]
                        )
                    )
                    p = invalid_share(
                        self.net.adapter_10tok,
                        static,
                        kwargs["scale"],
                        kwargs["revealed_indices"],
                        kwargs["revealed_raw_f"],
                        kwargs["revealed_gate"],
                        indicators,
                    )
                    for i, cid in enumerate(batch):
                        result[cid] = p[i].numpy().copy()
        self.check()
        return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if (
        os.uname().nodename.split(".")[0] not in ("xbabe2", "xbabe1")
        or os.getpriority(os.PRIO_PROCESS, 0) != 19
        or torch.cuda.is_initialized()
    ):
        raise RuntimeError("authorized CPU host/nice19 required")
    output = args.output.resolve()
    output.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    launch = json.loads(args.launch.read_text())
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
    ).strip()
    h188.validate_launch(
        launch,
        head,
        sha(HERE / "manifest.json"),
        sha(HERE / "PREREG.md"),
        sha(HERE / "fold_manifest.json"),
        smoke=args.smoke,
    )
    rt = h192.runtime()
    train, val = h192.cohorts(rt, json.loads((HERE / "provenance.json").read_text()))
    if (
        stat.fold_manifest(train)
        != json.loads((HERE / "fold_manifest.json").read_text())["reshuffles"]
    ):
        raise RuntimeError("fold manifest mismatch")
    allowed = (
        set(pins["inputs"])
        | set(pins["assets"])
        | {str(args.launch.resolve())}
        | {
            str(rt.INPUT / "pools" / (r["context_sha256"] + ".npz"))
            for r in train + val
        }
    )
    audit = rt.Audit(allowed, output)
    sys.addaudithook(audit.check)
    started = time.monotonic()

    def interrupted(signum, frame):
        raise RuntimeError(f"signal {signum}; no scientific retry")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        net, dope_guard, names, frozen, fused = rt.build(pins["checkpoint"])
        initial_hash = rt.tensor_hash(h192.adapter_state(net, names))
        for parameter in net.parameters():
            parameter.requires_grad_(False)
        torch.optim.Optimizer.__init__ = forbidden_training
        torch.autograd.backward = forbidden_training
        rt.check_frozen(net, dope_guard, names, frozen)
        if args.smoke:
            save(
                output / "smoke.json",
                dict(
                    status="PASS",
                    contexts_scored=0,
                    optimizer_updates=0,
                    backward_calls=0,
                    adapter_sha256=initial_hash,
                    dope_sha256=rt.DOPE,
                ),
            )
            return
        engine = Engine(net, rt, names, frozen, dope_guard, fused, train + val)
        engine.initial_hash = initial_hash
        engine.check()
        engine.load(val)
        baseline, raw_val = engine.evaluate(val, native=True, prove_identity=True)
        ref = json.loads(Path(pins["reference"]).read_text())
        base_values = np.asarray([baseline[r["context_sha256"]] for r in val])
        if (
            not np.array_equal(base_values, ref["utility"]["per_context"])
            or float(base_values.mean()) != 1.1377941898164525
        ):
            raise RuntimeError("native91 baseline mismatch")
        save(
            output / "baseline.json",
            dict(
                status="PASS",
                contexts=91,
                bit_exact=True,
                mean=float(base_values.mean()),
            ),
        )
        np.savez_compressed(output / "VAL-native-costs.npz", **raw_val)
        print("BASELINE PASS 91/91 bit-exact", flush=True)
        engine.check()
        engine.load(train)
        native, raw_train = engine.evaluate(train, native=True, prove_identity=True)
        np.savez_compressed(output / "TRAIN-native-costs.npz", **raw_train)
        engine.check()
        probabilities = engine.probabilities(train)
        np.savez_compressed(output / "TRAIN-invalid-share.npz", **probabilities)
        utilities = {
            cid: np.where(pool["gate"], -pool["fitness"], -2.0)
            for cid, pool in engine.pools.items()
        }
        for row in train:
            cid = row["context_sha256"]
            pick, count = veto_pick(raw_train[cid], probabilities[cid], None)
            if (
                pick != int(np.argmin(raw_train[cid]))
                or utilities[cid][pick] != native[cid]
                or count["vetoed"]
                or count["fallback"]
            ):
                raise RuntimeError("theta=inf native identity")
        save(
            output / "inf-identity.json",
            dict(status="PASS", contexts=len(train), bit_exact_picks_and_utility=True),
        )
        print("TRAIN INF IDENTITY PASS 1219/1219", flush=True)
        lock, bpanel = stat.stage1(train, probabilities, raw_train, utilities)
        np.savez_compressed(output / "B-panel.npz", utility=bpanel)
        lock.update(
            fold_manifest_sha256=sha(HERE / "fold_manifest.json"),
            adapter_sha256=initial_hash,
            b_panel_sha256=sha(output / "B-panel.npz"),
            native_costs_sha256=sha(output / "TRAIN-native-costs.npz"),
            invalid_share_sha256=sha(output / "TRAIN-invalid-share.npz"),
        )
        lockpath = output / "veto_selection_lock.json"
        save(lockpath, lock)
        guard = stat.SelectionGuard()
        guard.lock(lockpath)
        print("LOCK PASS all15 B choices", flush=True)
        engine.check()
        a, counts_a = stat.stage2(
            train, raw_train, probabilities, utilities, guard, "A"
        )
        save(
            output / "A-stage.json",
            dict(selection_lock_sha256=sha(lockpath), counts=counts_a),
        )
        print("A COMPLETE", flush=True)
        guard.require()
        engine.check()
        pval = engine.probabilities(val)
        np.savez_compressed(output / "VAL-invalid-share.npz", **pval)
        v, counts_v = stat.stage2(val, raw_val, pval, utilities, guard, "VAL")
        save(
            output / "VAL-stage.json",
            dict(selection_lock_sha256=sha(lockpath), counts=counts_v),
        )
        np.savez_compressed(
            output / "final-panels.npz",
            B=bpanel,
            A=a,
            VAL=v,
            train_context_ids=np.asarray([r["context_sha256"] for r in train]),
            val_context_ids=np.asarray([r["context_sha256"] for r in val]),
            readout_arms=np.asarray(stat.READOUT_ARMS),
        )
        a_summary = stat.summarize(train, a)
        v_summary = stat.summarize(val, v)
        admission = stat.select_a(train, a)
        receipt = json.loads(
            (
                ROOT
                / "research/flywheel_selfplay/h178_pooled_calibration/h178_receipt.json"
            ).read_text()
        )
        decision = stat.verdict(admission, a_summary, receipt, sha(Path(h177.__file__)))
        engine.check()
        guard.require()
        metrics = dict(
            wave="H193",
            status="COMPLETE",
            verdict=decision["verdict"],
            decision=decision,
            optimizer_updates=0,
            backward_calls=0,
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            adapter_sha256_before=initial_hash,
            adapter_sha256_after=initial_hash,
            seeds=stat.SEEDS,
            fold_seeds=stat.FOLD_SEEDS,
            readout_arms=stat.READOUT_ARMS,
            pooled_A_admission=admission,
            A_readout=a_summary,
            DL_Val_descriptive=v_summary,
            selection_lock_sha256=sha(lockpath),
            selection_counts={
                label: sum(c["selected_index"] == i for c in lock["cells"].values())
                for i, label in enumerate(("INF", "Q90", "Q75", "Q50"))
            },
            theta_by_cell={key: c["theta"] for key, c in lock["cells"].items()},
            grid_by_cell={key: c["grid"] for key, c in lock["cells"].items()},
            val_grids=lock["val_grids"],
            val_theta=lock["val_theta"],
            veto_counts_A=counts_a,
            veto_counts_VAL=counts_v,
            limitations=[
                "Zero updates; native costs and revealed invalid shares are identical across seeds. FIT quantiles and selected policies can differ; three positive seeds are not independent replication.",
                "Each cell excludes its own A outcomes; families recur in other FIT/B roles and were historically exposed.",
                "Bootstrap conditions on fitted thresholds and locked policies; no reselection inside bootstrap or new empirical H178 FWER calibration.",
                "VETO is the internal H191 KXSEL alias; DL-Val role/p-value fields remain descriptive only.",
                "Invalid share is a kernel-weighted score, not a calibrated probability.",
            ],
            serving_promotion=False,
        )
        if len(json.dumps(metrics).encode()) >= 100000:
            raise RuntimeError("metrics >=100KB")
        save(output / "metrics.json", metrics)
        save(
            output / "audit.json",
            dict(
                status="PASS",
                **audit.receipt(),
                optimizer_updates=0,
                backward_calls=0,
                cuda_initialized=torch.cuda.is_initialized(),
            ),
        )
        print("COMPLETE", metrics["verdict"], flush=True)
    except BaseException as error:
        save(
            output / "failure.json",
            dict(
                status="INCOMPLETE",
                error=repr(error),
                optimizer_updates=0,
                backward_calls=0,
                audit=audit.receipt(),
                elapsed_seconds=time.monotonic() - started,
            ),
        )
        raise


if __name__ == "__main__":
    main()
