"""H251 arms for the r882 adapter continuation: exact H170 CTRL AdamW (CONTROL) and first-order Reptile across TRAIN
families (Nichol et al. 2018): inner plain SGD on ONE family, then the outer interpolation theta <- theta + eps (phi -
theta). REPTILE uses k = 5 inner steps per outer step; REPTILE_K1 (k = 1) is the mechanism control that reduces to
scaled per-family SGD. Every arm uses the exact inherited H170 CTRL loss, 8 x 24 batches, the global clip 1.0 and 40
gradient evaluations per cell. All hyperparameters are frozen here before any outcome read (hw_t1200)."""

import math
import torch

ARMS = ("CONTROL", "REPTILE", "REPTILE_K1")
CONTROL_LR, CONTROL_WD = (
    1e-4,
    1e-4,
)  # merged H170 CTRL: AdamW lr=1e-4, wd=1e-4, betas (0.9, 0.999)
BETAS = (0.9, 0.999)
INNER_LR = (
    10 * CONTROL_LR
)  # hw_t1200: plain SGD, lr = H170 lr x 10 = 1e-3, no momentum, no weight decay
META_EPS = 0.5  # constant outer step size
INNER_STEPS = {"REPTILE": 5, "REPTILE_K1": 1}
GRADIENT_EVALS = (
    40  # compute-matched: 40 inner gradient evaluations per cell for every arm
)


def position(arm, step):
    """1-based (outer step, inner step) of gradient evaluation `step` for a Reptile arm."""
    k = INNER_STEPS[arm]
    if not 1 <= step <= GRADIENT_EVALS:
        raise ValueError("gradient evaluation out of range")
    return (step - 1) // k + 1, (step - 1) % k + 1


class Reptile:
    """First-order Reptile over the 47 adapter tensors. On the first inner step of an outer step theta is copied; each
    inner step is one plain SGD step on the current family's minibatch; after k inner steps the parameters are set to
    theta + eps (phi - theta). Parameters with a None gradient are skipped by SGD (and so stay at theta)."""

    def __init__(self, params, k, lr=INNER_LR, eps=META_EPS):
        self.params = list(params)
        if k < 1 or lr <= 0 or not 0 < eps <= 1:
            raise ValueError("invalid Reptile hyperparameters")
        self.k, self.eps, self.inner, self.theta, self.outer_updates = (
            k,
            eps,
            0,
            None,
            0,
        )
        self.sgd = torch.optim.SGD(
            self.params,
            lr=lr,
            momentum=0.0,
            dampening=0.0,
            weight_decay=0.0,
            nesterov=False,
        )
        self.param_groups = self.sgd.param_groups

    def zero_grad(self, set_to_none=True):
        self.sgd.zero_grad(set_to_none=set_to_none)

    def filter_gradients(self):
        return None

    @torch.no_grad()
    def step(self):
        """One inner SGD step; returns True when it completed an outer (meta) update."""
        if self.inner == 0:
            self.theta = [p.detach().clone() for p in self.params]
        self.sgd.step()
        self.inner += 1
        if self.inner < self.k:
            return False
        for p, theta in zip(self.params, self.theta, strict=True):
            p.copy_(theta + self.eps * (p - theta))
        self.inner, self.theta = 0, None
        self.outer_updates += 1
        return True


class ControlAdamW(torch.optim.AdamW):
    """The exact H170 CTRL AdamW, with a no-op gradient filter."""

    def __init__(self, params):
        super().__init__(params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD)

    def filter_gradients(self):
        return None

    def step(self):
        super().step()
        return True


def make_optimizer(arm, params):
    """CONTROL: exact H170 CTRL AdamW. REPTILE / REPTILE_K1: first-order Reptile with k = 5 / 1 inner SGD steps."""
    params = list(params)
    if arm == "CONTROL":
        return ControlAdamW(params)
    if arm in INNER_STEPS:
        return Reptile(params, INNER_STEPS[arm])
    raise ValueError("unknown arm")


def settings(arm, step):
    """The applied update settings at gradient evaluation 1..40 (constant hyperparameters)."""
    if arm not in ARMS or not 1 <= step <= GRADIENT_EVALS:
        raise ValueError("unknown arm or update")
    if arm == "CONTROL":
        return dict(
            optimizer="AdamW",
            lr=CONTROL_LR,
            beta1=BETAS[0],
            beta2=BETAS[1],
            weight_decay=CONTROL_WD,
            inner_steps=None,
            outer_step=step,
            inner_step=None,
            meta_eps=None,
            meta_update=True,
        )
    outer, inner = position(arm, step)
    return dict(
        optimizer="SGD",
        lr=INNER_LR,
        beta1=None,
        beta2=None,
        weight_decay=0.0,
        inner_steps=INNER_STEPS[arm],
        outer_step=outer,
        inner_step=inner,
        meta_eps=META_EPS,
        meta_update=inner == INNER_STEPS[arm],
    )


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    """H251 changes only the update rule: every arm uses the inherited H146 CTRL loss."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return ctrl_loss(pred, fitness, gate)


def gradient_norm(params):
    """None contributes zero to diagnostics; native AdamW None semantics are retained."""
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
    return (anchor_total + delta).float()


def validate_launch(
    launch, head, manifest_sha256, prereg_sha256, fold_sha256, smoke=False
):
    if launch.get("manifest_sha256") != manifest_sha256:
        raise RuntimeError("launch manifest binding")
    if smoke:
        return
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
