"""H236 optimizer-only continuation; all arms retain the exact H170 CTRL loss."""

import math
import torch

ARMS = ("CONTROL", "SF", "SF_LR3")
CONTROL_LR, CONTROL_WD = 1e-4, 1e-4


class ScheduleFreeAdamW(torch.optim.Optimizer):
    """Scalar-loop SF-AdamW: beta1 interpolation, two state tensors, decay at y.

    Equation-level implementation of Defazio et al. (2024), official scalar path
    at 70785b53e778d0e872c0bbb75ff4ee54ee10c291, inner_momentum=0,
    r=0, warmup_steps=4, weight_lr_power=2. No third iterate buffer.
    """

    def __init__(self, params, lr):
        super().__init__(
            params,
            dict(
                lr=lr,
                betas=(0.9, 0.999),
                weight_decay=CONTROL_WD,
                k=0,
                train_mode=False,
                weight_sum=0.0,
                lr_max=0.0,
                scheduled_lr=0.0,
            ),
        )

    @torch.no_grad()
    def _mode(self, training):
        for group in self.param_groups:
            if group["train_mode"] != training:
                beta1 = group["betas"][0]
                for p in group["params"]:
                    if "z" in self.state[p]:
                        p.lerp_(
                            self.state[p]["z"], 1 - beta1 if training else 1 - 1 / beta1
                        )
                group["train_mode"] = training
        return self

    def train(self):
        return self._mode(True)

    def eval(self):
        return self._mode(False)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                raise RuntimeError("SF step requires optimizer.train()")
            k, (beta1, beta2) = group["k"], group["betas"]
            lr = group["lr"] * min(1.0, (k + 1) / 4)
            group["scheduled_lr"] = lr
            group["lr_max"] = max(lr, group["lr_max"])
            weight = group["lr_max"] ** 2
            group["weight_sum"] += weight
            c = weight / group["weight_sum"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if "z" not in state:
                    state.update(z=p.detach().clone(), exp_avg_sq=torch.zeros_like(p))
                v, z, grad = state["exp_avg_sq"], state["z"], p.grad
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                grad.div_(v.div(1 - beta2 ** (k + 1)).sqrt_().add_(1e-8))
                grad.add_(p, alpha=group["weight_decay"])
                p.lerp_(z, c).add_(grad, alpha=lr * (beta1 * (1 - c) - 1))
                z.add_(grad, alpha=-lr)
            group["k"] += 1


def make_optimizer(arm, params):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    if arm == "CONTROL":
        return torch.optim.AdamW(
            params, lr=CONTROL_LR, betas=(0.9, 0.999), weight_decay=CONTROL_WD
        )
    return ScheduleFreeAdamW(params, CONTROL_LR * (3 if arm == "SF_LR3" else 1))


def settings(arm, step):
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or update")
    lr = (
        CONTROL_LR
        if arm == "CONTROL"
        else CONTROL_LR * (3 if arm == "SF_LR3" else 1) * min(1.0, step / 4)
    )
    return lr, 0.9, 0.999, CONTROL_WD


def optimizer_mode(optimizer, training):
    if isinstance(optimizer, ScheduleFreeAdamW):
        optimizer.train() if training else optimizer.eval()


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return ctrl_loss(pred, fitness, gate)


def gradient_norm(params):
    """None contributes zero to diagnostics; native optimizer None semantics are retained."""
    grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.0


def adapter_drift(initial, final):
    """All 29,211 coordinates, including inactive/unchanged tensors; float64 L2."""
    if set(initial) != set(final):
        raise ValueError("adapter state inventory differs")
    return math.sqrt(
        math.fsum(
            float((final[n].double() - initial[n].double()).square().sum())
            for n in initial
        )
    )


def adapter_forward(
    net,
    prog,
    sos,
    tokens,
    mask,
    scale,
    rev,
    raw,
    gate,
    anchor_total,
    anchor_prior,
    return_hidden=False,
):
    import torch.nn.functional as F

    ad = net.adapter_10tok
    pooled, query, semantic, adjustment = ad.encode_multimodal_context_and_candidates(
        mm_tokens_10x32=tokens,
        mm_mask_10=mask,
        prog_412=prog,
        sos_prior=sos,
        active_modality_override_mask=None,
    )
    static = dict(
        eff_prior=anchor_prior + adjustment,
        mm_prior_adj=adjustment,
        sem24=semantic,
        cand_q=query,
        cand_q_norm=F.normalize(query, p=2, dim=-1, eps=1e-8),
        ctx_pooled=pooled,
    )
    delta = ad.compute_adapter_delta_from_static_terms(
        static_terms=static,
        scale=scale,
        revealed_indices=rev,
        revealed_raw_f=raw,
        revealed_gate=gate,
    )
    pred = (anchor_total + delta).float()
    # H235: the per-candidate query embedding feeds the log-variance head; the mean prediction is unchanged.
    return (pred, query) if return_hidden else pred


def validate_launch(
    launch, head, manifest_sha256, prereg_sha256, fold_sha256, smoke=False
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest binding")
    if (
        not launch.get("merged_source_verified")
        or launch.get("source_commit") != head
        or type(launch.get("review_id")) is not int
        or launch["review_id"] <= 0
        or type(launch.get("prereg_pr")) is not int
        or launch["prereg_pr"] <= 0
        or launch.get("prereg_sha256") != prereg_sha256
        or launch.get("fold_manifest_sha256") != fold_sha256
    ):
        raise RuntimeError("external merged/reviewed launch binding")
