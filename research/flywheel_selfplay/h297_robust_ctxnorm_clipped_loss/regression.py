"""Required r3175 and t1845 control regressions; no new experimental arm."""

import json
import time
import torch
import runtime as x
from trajectory import batches, fit


def main():
    if (x.ROOT / "REGRESSION_STARTED.json").exists():
        assert (x.ROOT / "regression-baseline.json").exists() and not (
            x.ROOT / "regression-control.pt"
        ).exists()
        assert "optimizer_update" not in (x.ROOT / "events.jsonl").read_text()
        x.save(
            x.ROOT / "REGRESSION_RESUME.json",
            dict(
                epoch=time.time(),
                reason="correct overstrict unique-context count assertion; t1845 uses8190 family-blocked occurrences per update,1219 unique contexts; zero updates before resume",
            ),
        )
    else:
        x.save(x.ROOT / "REGRESSION_STARTED.json", dict(epoch=time.time()))
    selected, contract = x.verify_inputs()
    net, ckpt = x.build()
    baseline = {n: t.detach().clone() for n, t in net.state_dict().items()}
    tokens = torch.load(x.ROOT / "tokens.pt", map_location="cpu", weights_only=True)
    assert set(tokens) == {
        r["context_sha256"] for r in selected["train"] + selected["dev"]
    }
    dev_pools = x.load_pools(selected["dev"])
    report = x.reporter("regression-control")
    result = (
        json.loads((x.ROOT / "regression-baseline.json").read_text())
        if (x.ROOT / "regression-baseline.json").exists()
        else x.evaluate(net, selected["dev"], dev_pools, tokens, report)
    )
    expected = json.loads((x.ROOT / "reference/round-3175.json").read_text())[
        "dev_utility"
    ]
    assert result["mean"] == expected["mean"] == x.BASE
    assert (
        result["selected_programs"] == expected["selected_programs"]
        and result["per_context"] == expected["per_context"]
    )
    x.save(x.ROOT / "regression-baseline.json", result)
    x.event(
        "baseline_regression_PASS", identity=x.IDENTITY, mean=result["mean"], choices=91
    )
    train = x.load_pools(selected["train"])
    train_tokens = {c: tokens[c] for c in train}
    paired = batches(184501, 2, selected["train"], train, report)
    expected_batches = [
        (
            "a9e83cfe8954e235e190dd5108976c75c93ffe8e401304fdce78c1c04d87347a",
            "a46602bc2b92c4f26dce372337df7088dcb5126833cf84cc997e30ad95bebb2e",
        ),
        (
            "ef37e0c735f30284d0daf89783f3a29a652c48ac98568962f236c1fee8eec668",
            "021d0d2fd23ef9115eaf98414bef5451af2fcea9c639a52e04d5ccb646df506c",
        ),
    ]
    import hashlib

    for (c, ix), (order_sha, slate_sha) in zip(paired, expected_batches):
        assert hashlib.sha256(json.dumps(c).encode()).hexdigest() == order_sha
        assert hashlib.sha256(ix.tobytes()).hexdigest() == slate_sha
    out = x.ROOT / "regression-control.pt"
    rec = fit(
        net,
        ckpt,
        baseline,
        train,
        train_tokens,
        selected["train"],
        184501,
        "CONTROL",
        2,
        paired,
        out,
        regression=True,
    )
    actual = x.evaluate(net, selected["dev"], dev_pools, tokens, report)
    expected = json.loads((x.ROOT / "reference/control_v2-184501.json").read_text())
    assert actual["mean"] == expected["mean"] == 1.1354387884248622, (
        actual["mean"],
        expected["mean"],
    )
    assert (
        actual["selected_programs"] == expected["selected_programs"]
        and actual["per_context"] == expected["per_context"]
    )
    x.save(x.ROOT / "regression-control.json", dict(actual, fit=rec))
    x.save(
        x.ROOT / "regression.json",
        dict(
            status="PASS",
            identity=x.IDENTITY,
            baseline_mean=x.BASE,
            baseline_choices_exact=91,
            control_seed=184501,
            control_mean=actual["mean"],
            control_choices_exact=91,
            optimizer_updates=2,
            source_sha256={f.name: x.sha(f) for f in (x.ROOT / "source").glob("*.py")},
            dope_sha256=x.DOPE,
            trainable_parameters=29211,
            epoch=time.time(),
        ),
    )
    x.event("regressions_PASS", control_mean=actual["mean"])


if __name__ == "__main__":
    main()
