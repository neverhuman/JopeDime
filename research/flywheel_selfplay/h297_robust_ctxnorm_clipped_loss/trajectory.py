"""Paired TRAIN-only trajectories; no development scoring inside this module."""

import copy
import hashlib
import json
import numpy as np
import torch
from loss import training_loss
import runtime as x

REFERENCE_LR = 3.6880929895544625e-05
REFERENCE_SIGMA = 2.7751362402065425e-05


def batches(seed, steps, rows, pools, report, logical_round=None):
    np.random.seed(seed)
    result = []
    logical_round = seed if logical_round is None else logical_round
    for step in range(steps):
        order_seed = x.batch_seed(seed, logical_round, step)
        cids = report.training_order(order_seed, 8192, False)
        assert (
            set(cids) == {r["context_sha256"] for r in rows}
            and 1219 <= len(cids) <= 8192
        )
        ix = np.stack([x.training_indices(pools[c], 192) for c in cids])
        result.append((cids, ix))
        x.event(
            "batch_frozen",
            seed=seed,
            step=step,
            order_sha256=hashlib.sha256(json.dumps(cids).encode()).hexdigest(),
            slate_sha256=hashlib.sha256(ix.tobytes()).hexdigest(),
        )
    return result


def fit(
    net,
    ckpt,
    baseline,
    pools,
    tokens,
    train_rows,
    seed,
    arm,
    steps,
    paired_batches,
    out,
    regression=False,
):
    trainable = {n: p for n, p in net.named_parameters() if p.requires_grad}
    params = list(trainable.values())
    assert sum(p.numel() for p in params) == 29211
    allowed = {r["context_sha256"] for r in train_rows}
    assert set(pools) == allowed
    net.load_state_dict(baseline, strict=True)
    frozen = lambda: {n: t for n, t in net.state_dict().items() if n not in trainable}  # noqa: E731
    frozen_sha = x.state_sha(frozen())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
    opt.load_state_dict(copy.deepcopy(ckpt["optimizer"]))
    torch.manual_seed(seed + 999)
    if regression:
        restart = x.warm_restart_adapter(net, opt, seed, seed)
        assert restart["lr"] == REFERENCE_LR and restart["sigma"] == REFERENCE_SIGMA
    else:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        with torch.no_grad():
            for p in params:
                p.add_(
                    torch.randn(
                        p.shape, dtype=p.dtype, device=p.device, generator=generator
                    ),
                    alpha=REFERENCE_SIGMA,
                )
        for group in opt.param_groups:
            group["lr"] = REFERENCE_LR
    initial_sha = x.state_sha(trainable)
    before = max(int(v["step"]) for v in opt.state.values() if "step" in v)
    receipts = []
    for step, (cids, ix) in enumerate(paired_batches):
        assert set(cids) == allowed
        assert x.guard.snapshot_dope_weights(net).composite_sha256 == x.DOPE
        net.train()
        opt.zero_grad(set_to_none=True)
        total = 0.0
        N = len(cids)
        for off in range(0, N, 32):
            cc = cids[off : off + 32]
            ii = ix[off : off + 32]
            pred = x.predict(net, pools, tokens, cc, list(ii))["total_cost"].float()
            target = torch.as_tensor(
                np.stack([pools[c]["fitness"][j] for c, j in zip(cc, ii)]),
                dtype=torch.float32,
            )
            gate = torch.as_tensor(
                np.stack([pools[c]["gate"][j] for c, j in zip(cc, ii)])
            )
            target = torch.where(gate, target, torch.full_like(target, 2.0))
            loss = training_loss(pred, target, arm)
            assert torch.isfinite(loss)
            weight = len(cc) / N
            (loss * weight).backward()
            total += float(loss.detach()) * weight
        assert all(
            p.grad is None for n, p in net.named_parameters() if n not in trainable
        )
        norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        assert np.isfinite(norm)
        opt.step()
        assert (
            x.state_sha(frozen()) == frozen_sha
            and x.guard.snapshot_dope_weights(net).composite_sha256 == x.DOPE
        )
        rec = dict(
            step=step + 1,
            loss=total,
            grad_norm=norm,
            dope_sha256=x.DOPE,
            adapter_sha256=x.state_sha(trainable),
        )
        receipts.append(rec)
        x.event("optimizer_update", seed=seed, arm=arm, **rec)
    after = max(int(v["step"]) for v in opt.state.values() if "step" in v)
    assert after - before == steps == len(paired_batches)
    final_sha = x.state_sha(trainable)
    assert final_sha != initial_sha
    assert not out.exists()
    torch.save(
        dict(
            state_dict=net.state_dict(),
            optimizer=opt.state_dict(),
            seed=seed,
            arm=arm,
            dope_sha256=x.DOPE,
        ),
        out,
    )
    receipt = dict(
        seed=seed,
        arm=arm,
        updates=steps,
        optimizer_step_before=before,
        optimizer_step_after=after,
        initial_adapter_sha256=initial_sha,
        final_adapter_sha256=final_sha,
        checkpoint=str(out.relative_to(x.ROOT)),
        checkpoint_sha256=x.sha(out),
        lr=REFERENCE_LR,
        sigma=REFERENCE_SIGMA,
        frozen_sha256=frozen_sha,
        update_receipts=receipts,
    )
    x.save(out.with_suffix(".json"), receipt)
    x.event(
        "candidate_frozen_before_dev",
        seed=seed,
        arm=arm,
        checkpoint_sha256=receipt["checkpoint_sha256"],
    )
    return receipt
