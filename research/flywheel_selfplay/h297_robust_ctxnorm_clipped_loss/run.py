"""H297 single-attempt paired trajectories; one terminal dev score per candidate."""

import json
import time
import numpy as np
import torch
import runtime as x
from trajectory import batches, fit
from analysis import ARMS, SEEDS, summarize


def main():
    assert json.loads((x.ROOT / "regression.json").read_text())["status"] == "PASS"
    plan_path = x.ROOT / "plan.json"
    plan = json.loads(plan_path.read_text())
    plan_sha = x.sha(plan_path)
    pub = json.loads((x.ROOT / "published.json").read_text())
    assert pub["plan_sha256"] == plan_sha and pub["source_is_ancestor"]
    assert x.sha(x.ROOT / "monitor.py") == plan["monitor_sha256"]
    for name, h in plan["source_sha256"].items():
        assert x.sha(x.ROOT / "source" / name) == h
    assert not (x.ROOT / "RUN_STARTED.json").exists()
    x.save(
        x.ROOT / "RUN_STARTED.json",
        dict(
            epoch=time.time(), plan_sha256=plan_sha, prereg_commit=pub["source_commit"]
        ),
    )
    selected, contract = x.verify_inputs()
    net, ckpt = x.build()
    baseline = {n: t.detach().clone() for n, t in net.state_dict().items()}
    tokens = torch.load(x.ROOT / "tokens.pt", map_location="cpu", weights_only=True)
    train = x.load_pools(selected["train"])
    train_tokens = {c: tokens[c] for c in train}
    dev = None
    report = x.reporter("H297")
    results = []
    (x.ROOT / "candidates").mkdir(exist_ok=False)
    (x.ROOT / "batches").mkdir(exist_ok=False)
    for seed in SEEDS:
        paired = batches(seed, 8, selected["train"], train, report)
        np.savez_compressed(
            x.ROOT / "batches" / f"{seed}.npz",
            **{
                k: v
                for j, (c, i) in enumerate(paired)
                for k, v in [(f"cids_{j}", np.asarray(c)), (f"indices_{j}", i)]
            },
        )
        initial = None
        for arm in ARMS[1:]:
            artifact = x.ROOT / "candidates" / f"{arm}-{seed}.pt"
            receipt = fit(
                net,
                ckpt,
                baseline,
                train,
                train_tokens,
                selected["train"],
                seed,
                arm,
                8,
                paired,
                artifact,
            )
            if initial is None:
                initial = receipt["initial_adapter_sha256"]
            assert receipt["initial_adapter_sha256"] == initial
            # Reload check uses TRAIN only; dev remains terminal-only.
            net.eval()
            c = paired[0][0][0]
            ix = paired[0][1][0]
            with torch.no_grad():
                before = x.predict(net, train, train_tokens, [c], [ix])[
                    "total_cost"
                ].clone()
            saved = torch.load(artifact, map_location="cpu", weights_only=True)
            net.load_state_dict(saved["state_dict"], strict=True)
            with torch.no_grad():
                after = x.predict(net, train, train_tokens, [c], [ix])["total_cost"]
            assert torch.equal(before, after)
            assert x.guard.snapshot_dope_weights(net).composite_sha256 == x.DOPE
            if dev is None:
                dev = x.load_pools(selected["dev"])
            result = x.evaluate(net, selected["dev"], dev, tokens, report)
            result.update(
                seed=seed,
                arm=arm,
                fit=receipt,
                reload_bit_exact=True,
                plan_sha256=plan_sha,
                terminal_scores=1,
            )
            x.save(x.ROOT / "candidates" / f"{arm}-{seed}-result.json", result)
            results.append(result)
            x.event(
                "terminal_result",
                seed=seed,
                arm=arm,
                mean=result["mean"],
                lift=result["mean"] - x.BASE,
            )
    for name, h in plan["source_sha256"].items():
        assert x.sha(x.ROOT / "source" / name) == h
    baseline_result = json.loads((x.ROOT / "regression-baseline.json").read_text())
    metrics, panel = summarize(
        results, baseline_result, [r["family_id"] for r in selected["dev"]]
    )
    metrics.update(
        source_commit=pub["source_commit"],
        plan_sha256=plan_sha,
        evaluation_identity=x.IDENTITY,
        baseline_mean=x.BASE,
        dope_sha256=x.DOPE,
        trainable_parameters=29211,
        seeds=list(SEEDS),
        terminal_candidate_scores=9,
        forbidden_cohort_reads=0,
        live_changes=False,
        finished_epoch=time.time(),
        fit_receipts=[r["fit"] for r in results],
    )
    np.savez_compressed(
        x.ROOT / "panel.npz",
        panel=panel,
        seeds=SEEDS,
        arms=ARMS,
        cids=[r["context_sha256"] for r in selected["dev"]],
    )
    x.save(x.ROOT / "result-full.json", metrics)
    x.event("H297_complete", verdict=metrics["verdict"])


if __name__ == "__main__":
    main()
