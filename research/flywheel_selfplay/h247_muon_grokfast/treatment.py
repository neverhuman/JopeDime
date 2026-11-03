"""H247 update-geometry arms for the r882 adapter continuation: exact H170 CTRL AdamW (CONTROL), Muon (orthogonalized
Nesterov momentum on 2-D adapter matrices, Jordan et al. 2024) and Grokfast-EMA (slow-gradient amplification on the
CTRL AdamW recipe, Lee et al. 2024). Every arm uses the exact inherited H170 CTRL loss, batches and data order. All
hyperparameters are frozen here before any outcome read (hw_t0530)."""

import math
import torch

ARMS = ("CONTROL", "MUON", "GROKFAST")
CONTROL_LR, CONTROL_WD = (
    1e-4,
    1e-4,
)  # merged H170 CTRL: AdamW lr=1e-4, wd=1e-4, betas (0.9, 0.999)
BETAS = (0.9, 0.999)
MUON_LR, MUON_MOMENTUM, MUON_WD = (
    2e-4,
    0.95,
    1e-4,
)  # decoupled wd as CTRL (lr*wd = 2e-8 per step)
NS_COEFFS, NS_STEPS, NS_EPS = (
    (3.4445, -4.7750, 2.0315),
    5,
    1e-7,
)  # Jordan 2024 quintic Newton-Schulz
GROK_ALPHA, GROK_LAMBDA = 0.98, 2.0


def newton_schulz5(g):
    """Approximate the orthogonal polar factor U V^T of a 2-D matrix with 5 quintic Newton-Schulz iterations
    (Jordan 2024 coefficients), in float32 on CPU; tall matrices are processed transposed."""
    if g.dim() != 2:
        raise ValueError("2-D gradient required")
    a, b, c = NS_COEFFS
    x = g.float() / (torch.linalg.matrix_norm(g.float()) + NS_EPS)
    tall = x.shape[0] > x.shape[1]
    if tall:
        x = x.T
    for _ in range(NS_STEPS):
        s = x @ x.T
        x = a * x + (b * s + c * (s @ s)) @ x
    return x.T if tall else x


class Muon(torch.optim.Optimizer):
    """Muon for 2-D matrices: buf <- mu buf + g; nesterov direction d = g + mu buf; O = NS5(d);
    p <- p (1 - lr wd); p <- p - lr sqrt(max(1, rows / cols)) O. Parameters with a None gradient are skipped."""

    def __init__(
        self, params, lr=MUON_LR, momentum=MUON_MOMENTUM, weight_decay=MUON_WD
    ):
        params = list(params)
        if any(p.dim() != 2 for p in params) or lr <= 0 or not 0 <= momentum < 1:
            raise ValueError("Muon takes 2-D parameters and valid hyperparameters")
        super().__init__(
            params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("H247 Muon takes no closure")
        for group in self.param_groups:
            lr, mu, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(p)
                buf = state["momentum_buffer"]
                buf.mul_(mu).add_(p.grad)
                direction = p.grad.add(buf, alpha=mu)
                update = newton_schulz5(direction).to(p.dtype)
                scale = math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
                p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr * scale)


class Composite:
    """One optimizer facade over CTRL AdamW (1-D/3-D tensors) and Muon (2-D matrices): param_groups lists the AdamW
    group first, so traces report the CTRL settings; Muon's group carries its own lr/momentum."""

    def __init__(self, adamw, muon):
        self.adamw, self.muon = adamw, muon
        self.param_groups = adamw.param_groups + muon.param_groups

    def zero_grad(self, set_to_none=True):
        self.adamw.zero_grad(set_to_none=set_to_none)
        self.muon.zero_grad(set_to_none=set_to_none)

    def filter_gradients(self):
        return None

    def step(self):
        self.adamw.step()
        self.muon.step()


class GrokfastAdamW(torch.optim.AdamW):
    """CTRL AdamW with the Grokfast-EMA gradient filter (Lee et al. 2024, gradfilter_ema): on the first step the EMA
    is initialised to the gradient; then ema <- alpha ema + (1 - alpha) g and g <- g + lambda ema. The runner calls
    filter_gradients() after backward and BEFORE the inherited global clip at 1.0. None gradients are skipped."""

    def __init__(self, params, alpha=GROK_ALPHA, lamb=GROK_LAMBDA):
        super().__init__(params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD)
        self.alpha, self.lamb, self.ema = alpha, lamb, {}

    @torch.no_grad()
    def filter_gradients(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                if id(p) not in self.ema:
                    self.ema[id(p)] = p.grad.detach().clone()
                ema = self.ema[id(p)]
                ema.mul_(self.alpha).add_(p.grad, alpha=1 - self.alpha)
                p.grad.add_(ema, alpha=self.lamb)


class ControlAdamW(torch.optim.AdamW):
    """The exact H170 CTRL AdamW, with a no-op gradient filter."""

    def __init__(self, params):
        super().__init__(params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD)

    def filter_gradients(self):
        return None


def make_optimizer(arm, params):
    """CONTROL: exact H170 CTRL AdamW. MUON: Muon on every 2-D adapter matrix, CTRL AdamW on all other tensors.
    GROKFAST: CTRL AdamW with the Grokfast-EMA filter. Settings constant over the 40 updates."""
    params = list(params)
    if arm == "CONTROL":
        return ControlAdamW(params)
    if arm == "MUON":
        matrices = [p for p in params if p.dim() == 2]
        others = [p for p in params if p.dim() != 2]
        return Composite(ControlAdamW(others), Muon(matrices))
    if arm == "GROKFAST":
        return GrokfastAdamW(params)
    raise ValueError("unknown arm")


def settings(arm, step):
    """The applied (lr, beta1, beta2, weight_decay, muon_lr, grok_alpha, grok_lambda) at update 1..40 (constant)."""
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or update")
    return (
        CONTROL_LR,
        BETAS[0],
        BETAS[1],
        CONTROL_WD,
        MUON_LR if arm == "MUON" else None,
        GROK_ALPHA if arm == "GROKFAST" else None,
        GROK_LAMBDA if arm == "GROKFAST" else None,
    )


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    """H247 changes only the update geometry: every arm uses the inherited H146 CTRL loss."""
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
