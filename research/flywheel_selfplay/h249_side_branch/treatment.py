"""H249 zero-init low-rank residual side branch on a frozen r882 adapter (LoRA-style, Hu et al. 2021) vs exact H170
CTRL full-adapter continuation:
- CONTROL: exact H170 CTRL AdamW (lr=1e-4, betas=(0.9, 0.999), decoupled wd=1e-4) over all 47 adapter tensors (29,211 params).
- SIDE4 (primary, rank r=4, alpha=4.0, 6,232 trainable params) and SIDE16 (secondary, rank r=16, alpha=16.0, 24,928 trainable
  params): freeze all 47 r882 adapter tensors (requires_grad=False) and attach a zero-initialised low-rank residual branch
  delta = (B @ A) * (alpha / r) to each of the 16 2-D adapter matrices W (out x in), with A ~ Kaiming-uniform(a=sqrt(5))
  (seeded), B = 0, alpha = r, trained with AdamW (lr=1e-4, betas=(0.9, 0.999), decoupled wd=1e-2 on branch only).
All arms use the identical H170 CTRL loss, minibatches, and data order. At update 0 (B = 0), SIDE4 and SIDE16 are
bit-identical to r882 on Native91 (hw_t1035)."""

import contextlib
import math
import torch

ARMS = ("CONTROL", "SIDE4", "SIDE16")
CONTROL_LR, CONTROL_WD = 1e-4, 1e-4
SIDE_LR, SIDE_WD = 1e-4, 1e-2
BETAS = (0.9, 0.999)
RANKS = {"CONTROL": None, "SIDE4": 4, "SIDE16": 16}
ALPHAS = {"CONTROL": None, "SIDE4": 4.0, "SIDE16": 16.0}
TRAINABLE_PARAMS = {"CONTROL": 29211, "SIDE4": 6232, "SIDE16": 24928}


class ControlAdamW(torch.optim.AdamW):
    """Exact H170 CTRL AdamW over all 47 r882 adapter parameters (lr=1e-4, betas=(0.9, 0.999), wd=1e-4)."""

    def __init__(self, params):
        super().__init__(params, lr=CONTROL_LR, betas=BETAS, weight_decay=CONTROL_WD)

    def filter_gradients(self):
        return None


class SideBranchAdamW(torch.optim.AdamW):
    """AdamW on low-rank side-branch factors (A, B) only (lr=1e-4, betas=(0.9, 0.999), decoupled wd=1e-2)."""

    def __init__(self, params):
        super().__init__(params, lr=SIDE_LR, betas=BETAS, weight_decay=SIDE_WD)

    def filter_gradients(self):
        return None


class SideBranch:
    """Zero-initialised low-rank residual side branch on all 2-D matrices in `initial`.
    For each 2-D matrix W_0 (out x in), adds delta = (B @ A) * (alpha / r), where A ~ Kaiming-uniform(a=sqrt(5)) (seeded),
    B = 0, and alpha = r."""

    def __init__(self, initial, rank, alpha=None, seed=0):
        if rank not in (4, 16):
            raise ValueError("SideBranch rank must be 4 or 16")
        self.rank = int(rank)
        self.alpha = float(rank if alpha is None else alpha)
        if self.alpha != float(self.rank):
            raise ValueError("H249 fixes alpha = r")
        self.scale = self.alpha / self.rank
        self.initial = {
            n: t.detach().clone().requires_grad_(False) for n, t in initial.items()
        }
        self.matrix_names = tuple(n for n, t in self.initial.items() if t.dim() == 2)
        if not self.matrix_names:
            raise ValueError("at least one 2-D matrix required")
        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        self.pairs = {}
        params = []
        for n in self.matrix_names:
            out_d, in_d = self.initial[n].shape
            a = torch.nn.Parameter(torch.empty((self.rank, in_d), dtype=torch.float32))
            torch.nn.init.kaiming_uniform_(a, a=math.sqrt(5), generator=gen)
            b = torch.nn.Parameter(torch.zeros((out_d, self.rank), dtype=torch.float32))
            self.pairs[n] = (a, b)
            params.extend([a, b])
        self.params = tuple(params)

    def parameters(self):
        return self.params

    def parameter_names_and_tensors(self):
        out = []
        for n in self.matrix_names:
            a, b = self.pairs[n]
            out.append((f"{n}.lora_A", a))
            out.append((f"{n}.lora_B", b))
        return out

    def delta(self, name):
        a, b = self.pairs[name]
        return (b @ a) * self.scale

    @contextlib.contextmanager
    def applied(self, net):
        """Temporarily bind each 2-D parameter slot on `net` to the differentiable expression W_0 + (B @ A) * (alpha / r)
        during forward, restoring the frozen nn.Parameter W_0 on exit."""
        saved = []
        try:
            for n in self.matrix_names:
                mod_path, _, attr = n.rpartition(".")
                mod = net.get_submodule(mod_path) if mod_path else net
                saved.append((mod, attr, mod._parameters[attr]))
                mod._parameters[attr] = self.initial[n] + self.delta(n)
            yield
        finally:
            for mod, attr, orig in saved:
                mod._parameters[attr] = orig

    def merged_state(self):
        """Return a detached state dict of all 47 adapter tensors with W_0 + (B @ A) * (alpha / r) merged into 2-D matrices."""
        out = {}
        with torch.no_grad():
            for n, w0 in self.initial.items():
                if n in self.pairs:
                    out[n] = (w0 + self.delta(n).detach()).clone()
                else:
                    out[n] = w0.clone()
        return out

    def branch_tensors(self):
        """Detached numpy-ready dict of all 32 (A, B) factor tensors."""
        out = {}
        with torch.no_grad():
            for n, (a, b) in self.pairs.items():
                out[f"{n}.lora_A"] = a.detach().clone()
                out[f"{n}.lora_B"] = b.detach().clone()
        return out

    def branch_norm(self):
        """Float64 L2 norm across all 32 (A, B) branch factor tensors."""
        return math.sqrt(
            math.fsum(float(p.detach().double().square().sum()) for p in self.params)
        )


def make_optimizer(arm, params):
    """Build the arm optimizer over `params`."""
    params = list(params)
    if arm == "CONTROL":
        return ControlAdamW(params)
    if arm in ("SIDE4", "SIDE16"):
        return SideBranchAdamW(params)
    raise ValueError("unknown arm")


def setup_arm(arm, net, names, initial, seed=0):
    """Configure `net` requires_grad flags, construct `SideBranch` (or None for CONTROL), and return
    `(branch, trainable_params, named_trainable, optimizer)`."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    name_set = set(names)
    if arm == "CONTROL":
        for n, p in net.named_parameters():
            p.requires_grad_(n in name_set)
        named = [(n, p) for n, p in net.named_parameters() if n in name_set]
        params = [p for _, p in named]
        return None, params, named, ControlAdamW(params)
    for _, p in net.named_parameters():
        p.requires_grad_(False)
    branch = SideBranch(initial, rank=RANKS[arm], alpha=ALPHAS[arm], seed=seed)
    named = branch.parameter_names_and_tensors()
    params = list(branch.parameters())
    return branch, params, named, SideBranchAdamW(params)


def settings(arm, step):
    """The applied (lr, beta1, beta2, weight_decay, rank, alpha, trainable_params) at update 1..40 (constant)."""
    if arm not in ARMS or not 1 <= step <= 40:
        raise ValueError("unknown arm or update")
    wd = CONTROL_WD if arm == "CONTROL" else SIDE_WD
    lr = CONTROL_LR if arm == "CONTROL" else SIDE_LR
    return (
        lr,
        BETAS[0],
        BETAS[1],
        wd,
        RANKS[arm],
        ALPHAS[arm],
        TRAINABLE_PARAMS[arm],
    )


def arm_loss(arm, ctrl_loss, pred, fitness, gate):
    """Every H249 arm uses the exact inherited H146/H170 CTRL loss."""
    if arm not in ARMS:
        raise ValueError("unknown arm")
    return ctrl_loss(pred, fitness, gate)


def gradient_norm(params):
    """None contributes zero to diagnostics; native AdamW None semantics are retained."""
    grads = [p.grad.detach().reshape(-1) for p in params if p.grad is not None]
    return float(torch.linalg.vector_norm(torch.cat(grads))) if grads else 0.0


def adapter_drift(initial, final):
    """All 29,211 coordinates of the (merged) adapter state, including inactive/unchanged tensors; float64 L2."""
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
    branch=None,
):
    import torch.nn.functional as F

    cm = branch.applied(net) if branch is not None else contextlib.nullcontext()
    with cm:
        ad = net.adapter_10tok
        pooled, query, semantic, adjustment = (
            ad.encode_multimodal_context_and_candidates(
                mm_tokens_10x32=tokens,
                mm_mask_10=mask,
                prog_412=prog,
                sos_prior=sos,
                active_modality_override_mask=None,
            )
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
