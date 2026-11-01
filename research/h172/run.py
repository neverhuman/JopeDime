"""Single H172 wave: Anchored convex last-layer refit (L-BFGS) + family bagging on frozen r882 penultimate features."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analysis as stat  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, obj):
    Path(p).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def verify(pins):
    for kind in ("inputs", "assets"):
        for p, h in pins[kind].items():
            if sha(p) != h:
                raise RuntimeError("identity mismatch: " + p)
    for p, h in pins["source_sha256"].items():
        if sha(HERE / p) != h:
            raise RuntimeError("source identity mismatch: " + p)
    for name, version in pins["packages"].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError("package version: " + name)


def parent_executor():
    parent = HERE.parent / "flywheel_selfplay/h146_lookahead_exec"
    sys.path.insert(0, str(parent))
    spec = importlib.util.spec_from_file_location("h172_parent", parent / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module, sys.modules["runtime"]


def resolve_slate_target_tensor(fitness, gate):
    """Resolve H59/H140 effective regression cost target: valid clamped to [-4.0, 2.0], failed-gate mapped to 2.0."""
    import torch

    return torch.where(gate, fitness.clamp(-4.0, 2.0), torch.full_like(fitness, 2.0))


def per_context_live_losses(pred, target, p_target=None):
    """Exact per-context H140/H146 live adapter loss vector (shape [B]) for pred, target of shape [B, K]."""
    import torch.nn.functional as F

    if p_target is None:
        p_target = F.softmax(-target / 0.2, dim=-1)
    return F.smooth_l1_loss(pred, target, reduction="none").mean(1) + 0.1 * F.kl_div(
        F.log_softmax(-pred / 0.2, dim=-1),
        p_target,
        reduction="none",
    ).sum(1)


def sanitize_and_verify_head_grads(params, names):
    """Handle any parameter with grad=None (e.g. parent_delta_net.3) by zero-filling and verify finiteness."""
    import torch

    missing = []
    grads = []
    for name, p in zip(names, params, strict=True):
        if p.grad is None:
            missing.append(name)
            grads.append(torch.zeros_like(p.detach().reshape(-1)))
        else:
            if not torch.isfinite(p.grad).all():
                raise RuntimeError(f"nonfinite gradient in {name}")
            grads.append(p.grad.detach().reshape(-1))
    flat = torch.cat(grads, dim=0)
    return float(torch.linalg.vector_norm(flat).cpu()), missing


def build_size_batches(cids, feature_cache, target_override=None, context_weights=None):
    """Group contexts by candidate pool size M_c into contiguous float64 tensors for vectorized L-BFGS."""
    import torch
    import torch.nn.functional as F

    if context_weights is None:
        w_map = {c: 1.0 / len(cids) for c in cids}
    else:
        w_map = {c: float(w) for c, w in zip(cids, context_weights, strict=True)}

    by_size = {}
    for c in cids:
        w_c = w_map[c]
        if w_c <= 0.0:
            continue
        k = int(feature_cache[c]["c0_np"].shape[0])
        by_size.setdefault(k, []).append((c, w_c))

    batches = []
    for k, items in sorted(by_size.items()):
        cs = [c for c, _ in items]
        ws = torch.tensor([w for _, w in items], dtype=torch.float64)
        c0 = torch.stack([feature_cache[c]["c0_64"] for c in cs], dim=0)
        x48 = torch.stack([feature_cache[c]["x48_64"] for c in cs], dim=0)
        if target_override is None:
            tgt = torch.stack([feature_cache[c]["target_64"] for c in cs], dim=0)
            ptgt = torch.stack([feature_cache[c]["p_target_64"] for c in cs], dim=0)
        else:
            tgt_list = []
            ptgt_list = []
            for c in cs:
                f_ov, g_ov = target_override[c]
                f_t = torch.as_tensor(f_ov, dtype=torch.float64)
                g_t = torch.as_tensor(g_ov, dtype=torch.bool)
                t_c = resolve_slate_target_tensor(f_t, g_t)
                tgt_list.append(t_c)
                ptgt_list.append(F.softmax(-t_c / 0.2, dim=-1))
            tgt = torch.stack(tgt_list, dim=0)
            ptgt = torch.stack(ptgt_list, dim=0)
        batches.append((c0, x48, tgt, ptgt, ws))
    return batches


def solve_anchored_convex_head(
    size_batches,
    lam,
    init_delta99=None,
    max_iter=stat.LBFGS_MAX_ITER,
    grad_tol=stat.LBFGS_GRAD_TOL,
):
    """
    Solve the strictly convex anchored output-layer refit:
        min_{delta_W in R^99} sum_{c in S} w_c * l_c(c_r882(c) + X_c * delta_w48, t_c) + lam * ||delta_W||_2^2
    over the full 99-parameter H132 output-layer subset:
      - [0:24]  mm_prior_out.2.weight (24)
      - [24:25] mm_prior_out.2.bias (1)
      - [25:49] traj_delta_head.2.weight (24)
      - [49:50] traj_delta_head.2.bias (1)
      - [50:98] parent_delta_net.3.weight (48)
      - [98:99] parent_delta_net.3.bias (1)
    """
    import torch

    if lam is None or math.isinf(lam):
        zero99 = np.zeros(stat.H132_OUTPUT_TOTAL_PARAMS, dtype=np.float64)
        return dict(
            delta99=zero99,
            delta48=np.zeros(stat.ACTIVE_RANKING_WEIGHT_PARAMS, dtype=np.float64),
            l2_drift=0.0,
            mm_weight_l2_drift=0.0,
            traj_weight_l2_drift=0.0,
            bias_and_parent_l2_drift=0.0,
            grad_norm=0.0,
            closure_evals=0,
            initial_obj=0.0,
            final_obj=0.0,
            converged=True,
        )

    if lam <= 0.0:
        raise ValueError(f"positive anchor lambda required, got {lam}")

    if init_delta99 is None:
        d99 = torch.zeros(
            stat.H132_OUTPUT_TOTAL_PARAMS, dtype=torch.float64, requires_grad=True
        )
    else:
        d99 = torch.tensor(
            np.asarray(init_delta99, dtype=np.float64),
            dtype=torch.float64,
            requires_grad=True,
        )

    lam_f = float(lam)
    evals = [0]

    def compute_objective_and_grad():
        if d99.grad is not None:
            d99.grad.zero_()
        dw_mm = d99[0:24]
        db_mm = d99[24:25]
        dw_traj = d99[25:49]
        db_traj = d99[49:50]
        dw48 = torch.cat([dw_mm, dw_traj], dim=0)
        # Bias terms cancel identically under within-slate mean-centering: (z + b) - mean(z + b) == z - mean(z).
        zero_bias_effect = 0.0 * (db_mm.sum() + db_traj.sum())
        fit_loss = torch.zeros((), dtype=torch.float64)
        for c0, x48, tgt, ptgt, ws in size_batches:
            pred = c0 + torch.matmul(x48, dw48) + zero_bias_effect
            ctx_losses = per_context_live_losses(pred, tgt, ptgt)
            fit_loss = fit_loss + torch.sum(ws * ctx_losses)
        obj = fit_loss + lam_f * torch.sum(d99 * d99)
        obj.backward()
        return obj

    with torch.enable_grad():
        init_obj = float(compute_objective_and_grad().detach().cpu())
        init_gn = float(torch.linalg.vector_norm(d99.grad.detach()).cpu())
        if init_gn > grad_tol:
            remaining = int(max_iter)
            for _ in range(3):
                if remaining <= 0:
                    break
                step_budget = min(remaining, 200)
                opt = torch.optim.LBFGS(
                    [d99],
                    lr=1.0,
                    max_iter=step_budget,
                    tolerance_grad=grad_tol * 0.25,
                    tolerance_change=0.0,
                    history_size=20,
                    line_search_fn="strong_wolfe",
                )

                def closure():
                    evals[0] += 1
                    return compute_objective_and_grad()

                opt.step(closure)
                remaining -= step_budget
                compute_objective_and_grad()
                gn = float(torch.linalg.vector_norm(d99.grad.detach()).cpu())
                if gn <= grad_tol:
                    break
        final_obj = float(compute_objective_and_grad().detach().cpu())
        final_gn = float(torch.linalg.vector_norm(d99.grad.detach()).cpu())

    if not math.isfinite(final_obj) or not math.isfinite(final_gn):
        raise RuntimeError("nonfinite L-BFGS solution")
    if final_gn > grad_tol:
        raise RuntimeError(
            f"L-BFGS gradient norm {final_gn:.3e} exceeded tolerance {grad_tol:.3e} at lambda={lam}"
        )

    d99_np = d99.detach().cpu().numpy().copy()
    # Enforce exact analytical zero on zero-gradient coordinates (biases 24, 49 and parent_delta_net 50:99)
    bias_parent_idx = [24, 49] + list(range(50, 99))
    bias_parent_drift = float(np.linalg.norm(d99_np[bias_parent_idx]))
    if bias_parent_drift > 1e-9:
        raise RuntimeError(
            f"unexpected non-zero drift {bias_parent_drift:.3e} on zero-gradient output coordinates"
        )
    d99_np[bias_parent_idx] = 0.0
    d48_np = np.concatenate([d99_np[0:24], d99_np[25:49]], axis=0)
    return dict(
        delta99=d99_np,
        delta48=d48_np,
        l2_drift=float(np.linalg.norm(d99_np)),
        mm_weight_l2_drift=float(np.linalg.norm(d99_np[0:24])),
        traj_weight_l2_drift=float(np.linalg.norm(d99_np[25:49])),
        bias_and_parent_l2_drift=bias_parent_drift,
        grad_norm=final_gn,
        closure_evals=int(evals[0]),
        initial_obj=init_obj,
        final_obj=final_obj,
        converged=bool(final_gn <= grad_tol),
    )


def solve_lambda_path(size_batches):
    """Solve the 6-point descending lambda grid with warm starts from lambda=1e4 down to 1e-1."""
    solutions = {"inf": solve_anchored_convex_head(size_batches, math.inf)}
    prev_d99 = None
    for label in stat.FINITE_LAMBDA_LABELS:
        lam = stat.LAMBDA_BY_LABEL[label]
        sol = solve_anchored_convex_head(size_batches, lam, init_delta99=prev_d99)
        solutions[label] = sol
        prev_d99 = sol["delta99"]
    return solutions


def evaluate_head_on_rows(rows, feature_cache, pools, delta48):
    """Evaluate a refit 48D active ranking head displacement delta48 across rows using frozen r882 penultimate features."""
    d48 = np.asarray(delta48, dtype=np.float32)
    is_zero = bool(np.all(d48 == 0.0))
    values = []
    choices = []
    invalid = []
    cids = []
    for r in rows:
        cid = r["context_sha256"]
        fc = feature_cache[cid]
        if is_zero:
            cost = fc["c0_np"]
        else:
            cost = fc["c0_np"] + fc["x48_np"] @ d48
        if not np.isfinite(cost).all():
            raise RuntimeError("nonfinite predictions")
        idx_min = int(np.argmin(cost))
        p = pools[cid]
        valid = bool(p["gate"][idx_min])
        u = float(-p["fitness"][idx_min] if valid else -2.0)
        values.append(u)
        choices.append(idx_min)
        invalid.append(not valid)
        cids.append(cid)
    return dict(values=values, choices=choices, invalid=invalid, context_ids=cids)


def head_state_tensors(base_head_tensors, delta99):
    """Reconstruct the 6 H132 output-layer tensors at displacement delta99 around r882."""
    d99 = np.asarray(delta99, dtype=np.float32)
    return {
        "adapter_10tok.mm_prior_out.2.weight": (
            base_head_tensors["adapter_10tok.mm_prior_out.2.weight"]
            + d99[0:24].reshape(1, 24)
        ),
        "adapter_10tok.mm_prior_out.2.bias": (
            base_head_tensors["adapter_10tok.mm_prior_out.2.bias"]
            + d99[24:25].reshape(1)
        ),
        "adapter_10tok.traj_delta_head.2.weight": (
            base_head_tensors["adapter_10tok.traj_delta_head.2.weight"]
            + d99[25:49].reshape(1, 24)
        ),
        "adapter_10tok.traj_delta_head.2.bias": (
            base_head_tensors["adapter_10tok.traj_delta_head.2.bias"]
            + d99[49:50].reshape(1)
        ),
        "adapter_10tok.parent_delta_net.3.weight": (
            base_head_tensors["adapter_10tok.parent_delta_net.3.weight"]
            + d99[50:98].reshape(1, 48)
        ),
        "adapter_10tok.parent_delta_net.3.bias": (
            base_head_tensors["adapter_10tok.parent_delta_net.3.bias"]
            + d99[98:99].reshape(1)
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir()  # Exclusive, no result/cache reuse or automatic retry.
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    parent, rt = parent_executor()
    import torch
    import torch.nn.functional as F

    train, val = parent.selected()
    if len({r["family_id"] for r in train}) != 205:
        raise RuntimeError("TRAIN family inventory")
    groups = stat.folds(train)
    if groups != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
        raise RuntimeError("fold identity")
    allowed = set(pins["inputs"])
    if args.launch is not None:
        allowed.add(str(args.launch.resolve()))
    allowed.update(
        str(rt.INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val
    )
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    lbfgs_solves = 0
    started = time.monotonic()
    try:
        net, guard, names, frozen, fused = rt.build(pins["checkpoint"])
        census = rt.dropout_census(net)
        if (
            census["adapter_dropout_modules"]
            or any(m["p"] != 0 for m in census["adapter_attention_dropout"])
            or census["effective_encoder"] != "_sync_free_encode_context_and_candidates"
            or census["adapter_training"]
            or census["dope_training"]
        ):
            raise RuntimeError("effective inference/training mode changed")
        rt.check_frozen(net, guard, names, frozen)
        param_dict = dict(net.named_parameters())
        params_check = [param_dict[n] for n in names]
        trainable_params = sum(p.numel() for p in params_check)
        if trainable_params != 29211:
            raise RuntimeError(f"expected 29211 adapter params, got {trainable_params}")
        head_params = [param_dict[n] for n in stat.H132_OUTPUT_TENSORS]
        head_param_count = sum(p.numel() for p in head_params)
        if head_param_count != stat.H132_OUTPUT_TOTAL_PARAMS:
            raise RuntimeError(
                f"expected {stat.H132_OUTPUT_TOTAL_PARAMS} H132 output-layer params, got {head_param_count}"
            )

        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        base_head_tensors = {
            n: initial[n].detach().cpu().numpy().astype(np.float32).copy()
            for n in stat.H132_OUTPUT_TENSORS
        }

        if args.smoke:
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    wave="H172",
                    optimizer_updates=0,
                    lbfgs_solves=0,
                    scored_contexts=0,
                    adapter_tensors=len(names),
                    adapter_parameters=trainable_params,
                    h132_output_tensors=list(stat.H132_OUTPUT_TENSORS),
                    h132_output_total_parameters=stat.H132_OUTPUT_TOTAL_PARAMS,
                    h132_output_weight_parameters=stat.H132_OUTPUT_WEIGHT_PARAMS,
                    h132_output_bias_parameters=stat.H132_OUTPUT_BIAS_PARAMS,
                    active_ranking_weight_parameters=stat.ACTIVE_RANKING_WEIGHT_PARAMS,
                    arms=list(stat.ARMS),
                    lambda_grid=list(stat.LAMBDA_GRID),
                    lambda_labels=list(stat.LAMBDA_LABELS),
                    bag_count=stat.BAG_COUNT,
                    lbfgs_max_iter=stat.LBFGS_MAX_ITER,
                    lbfgs_grad_tol=stat.LBFGS_GRAD_TOL,
                    census=census,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    packages=pins["packages"],
                ),
            )
            return

        launch = json.loads(args.launch.read_text())
        actual_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=HERE, text=True
        ).strip()
        if (
            launch["manifest_sha256"] != sha(HERE / "manifest.json")
            or not launch["merged_source_verified"]
            or launch.get("source_commit") != actual_head
        ):
            raise RuntimeError(
                "merged preregistration and exact HEAD commit match required"
            )

        ref = json.loads(Path(pins["reference"]).read_text())
        contract = ref["evaluation_contract"]
        expected = [
            dict(
                cid=r["context_sha256"],
                pool_sha256=r["pool_sha256"],
                training_table_sha256=r["train_only_jtt1_sha256"],
            )
            for r in val
        ]
        if (
            contract["contexts"] != expected
            or hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
            != pins["evaluation_identity"]
        ):
            raise RuntimeError("evaluation identity")

        pools = {}
        projection = fused.get_w1_sign_matrix() / np.sqrt(412.0)

        def load_pool_rows(cohort_rows):
            for r in cohort_rows:
                c = r["context_sha256"]
                if c in pools:
                    continue
                p = rt.INPUT / "pools" / f"{c}.npz"
                if sha(p) != r["pool_sha256"]:
                    raise RuntimeError("pool hash mismatch: " + c)
                with np.load(p, allow_pickle=False) as z:
                    progs = np.asarray(z["program"], dtype=np.float32)
                    pools[c] = dict(
                        program=progs,
                        proj16=(progs.astype(np.float64) @ projection).astype(
                            np.float32
                        ),
                        fitness=np.asarray(z["fitness"], dtype=np.float64),
                        gate=np.asarray(z["gate"], dtype=bool),
                    )

        # Load ONLY the 91 DL-Val pools first so the 91/91 baseline gate executes before touching TRAIN pools.
        load_pool_rows(val)
        device = torch.device("cpu")
        with np.load(rt.CACHE, allow_pickle=False) as z:
            ids = [str(x) for x in z["cids"]]
            tok = np.asarray(z["tokens_10x32"], dtype=np.float32)
            mask = np.asarray(z["mask_10"], dtype=bool)
            tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32, device=device)
                for i, c in enumerate(ids)
            }
            masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool, device=device)
                for i, c in enumerate(ids)
            }

        ad = net.adapter_10tok
        s_mm = float(torch.tanh(ad.mm_prior_scale.detach()).cpu())
        s_traj = float(torch.tanh(ad.traj_attn_scale.detach()).cpu())
        w_mm_0 = ad.mm_prior_out[2].weight.detach().float()
        b_mm_0 = ad.mm_prior_out[2].bias.detach().float()
        w_traj_0 = ad.traj_delta_head[2].weight.detach().float()
        b_traj_0 = ad.traj_delta_head[2].bias.detach().float()

        feature_cache = {}

        def precompute_penultimate_features(cohort_rows, verify_native_bit_exact=False):
            """Precompute frozen r882 penultimate features (Phi_mm, Phi_traj) and native c_r882 on full pools."""
            cohort_groups = {}
            for r in cohort_rows:
                c = r["context_sha256"]
                cohort_groups.setdefault(len(pools[c]["fitness"]), []).append(c)

            with torch.no_grad():
                for size, cids in sorted(cohort_groups.items()):
                    for start in range(0, len(cids), 8):
                        batch = cids[start : start + 8]
                        b = len(batch)
                        k = size
                        progs = torch.as_tensor(
                            np.stack([pools[c]["program"] for c in batch]),
                            dtype=torch.float32,
                            device=device,
                        )
                        proj = torch.as_tensor(
                            np.stack([pools[c]["proj16"] for c in batch]),
                            dtype=torch.float32,
                            device=device,
                        )
                        tok_b = torch.stack([tokens[c] for c in batch])
                        mask_b = torch.stack([masks[c] for c in batch])
                        rev_idx = (
                            torch.arange(min(4, k), device=device)
                            .reshape(1, -1)
                            .expand(b, -1)
                        )
                        rev_raw_f = torch.as_tensor(
                            np.stack([pools[c]["fitness"][: min(4, k)] for c in batch]),
                            dtype=torch.float32,
                            device=device,
                        )
                        rev_gate = torch.as_tensor(
                            np.stack([pools[c]["gate"][: min(4, k)] for c in batch]),
                            dtype=torch.float32,
                            device=device,
                        )
                        scale_b = torch.full(
                            (b,), 0.4208, dtype=torch.float32, device=device
                        )
                        sos_b = torch.zeros(
                            (b, k, 2), dtype=torch.float32, device=device
                        )

                        captured = {}

                        def hook_mm(_m, inp, _out):
                            captured["phi_mm"] = inp[0].detach().clone()

                        def hook_traj(_m, inp, _out):
                            captured["phi_traj"] = inp[0].detach().clone()

                        hk1 = ad.mm_prior_out[2].register_forward_hook(hook_mm)
                        hk2 = ad.traj_delta_head[2].register_forward_hook(hook_traj)
                        try:
                            out_pred = ad.predict_costs(
                                ctx_feat=torch.zeros(
                                    (b, 584), dtype=torch.float32, device=device
                                ),
                                prog_412=progs,
                                proj_16=proj,
                                base_prior_cost=torch.zeros(
                                    (b, k), dtype=torch.float32, device=device
                                ),
                                scale=scale_b,
                                sos_prior=sos_b,
                                mm_tokens_10x32=tok_b,
                                mm_mask_10=mask_b,
                                revealed_indices=rev_idx,
                                revealed_raw_f=rev_raw_f,
                                revealed_gate=rev_gate,
                            )
                        finally:
                            hk1.remove()
                            hk2.remove()

                        scores_t = out_pred["total_cost"].detach().float()
                        if verify_native_bit_exact:
                            native_t = parent.predict(
                                net,
                                pools,
                                tokens,
                                masks,
                                batch,
                                [np.arange(size) for _ in batch],
                            )
                            if not torch.equal(scores_t, native_t):
                                raise RuntimeError(
                                    "step-0 raw cost mismatch vs native r882"
                                )

                        phi_mm_b = captured["phi_mm"].float()  # (B, K, 24)
                        phi_traj_b = captured["phi_traj"].float()  # (B, K, 24)
                        # Verify linear head readout matches ad.mm_prior_out[2] and ad.traj_delta_head[2]
                        raw_mm = F.linear(phi_mm_b, w_mm_0, b_mm_0).squeeze(-1)
                        mm_adj_chk = s_mm * (raw_mm - raw_mm.mean(dim=-1, keepdim=True))
                        if not torch.allclose(
                            mm_adj_chk, out_pred["mm_prior_adj"].float(), atol=1e-6
                        ):
                            raise RuntimeError("mm_prior_adj penultimate hook mismatch")
                        raw_traj = F.linear(phi_traj_b, w_traj_0, b_traj_0).squeeze(-1)
                        traj_adj_chk = s_traj * (
                            raw_traj - raw_traj.mean(dim=-1, keepdim=True)
                        )
                        if not torch.allclose(
                            traj_adj_chk, out_pred["traj_attn_adj"].float(), atol=1e-6
                        ):
                            raise RuntimeError(
                                "traj_attn_adj penultimate hook mismatch"
                            )

                        x_mm_b = s_mm * (phi_mm_b - phi_mm_b.mean(dim=1, keepdim=True))
                        x_traj_b = s_traj * (
                            phi_traj_b - phi_traj_b.mean(dim=1, keepdim=True)
                        )
                        x48_b = torch.cat([x_mm_b, x_traj_b], dim=-1)  # (B, K, 48)

                        for i_b, cid in enumerate(batch):
                            c0_f32 = scores_t[i_b].cpu().numpy().copy()
                            x48_f32 = x48_b[i_b].cpu().numpy().copy()
                            f_t = torch.as_tensor(
                                pools[cid]["fitness"], dtype=torch.float64
                            )
                            g_t = torch.as_tensor(pools[cid]["gate"], dtype=torch.bool)
                            tgt_64 = resolve_slate_target_tensor(f_t, g_t)
                            ptgt_64 = F.softmax(-tgt_64 / 0.2, dim=-1)
                            feature_cache[cid] = dict(
                                c0_np=c0_f32,
                                x48_np=x48_f32,
                                c0_64=torch.as_tensor(c0_f32, dtype=torch.float64),
                                x48_64=torch.as_tensor(x48_f32, dtype=torch.float64),
                                target_64=tgt_64,
                                p_target_64=ptgt_64,
                            )
            rt.check_frozen(net, guard, names, frozen)

        # 1. Execute 91/91 DL-Val baseline gate BEFORE opening any TRAIN pools.
        precompute_penultimate_features(val, verify_native_bit_exact=True)
        baseline = evaluate_head_on_rows(
            val,
            feature_cache,
            pools,
            np.zeros(stat.ACTIVE_RANKING_WEIGHT_PARAMS, dtype=np.float32),
        )
        sanity = parent.baseline_gate(baseline["values"], ref)
        sanity["raw_cost_bit_exact_91_91"] = True
        save(out / "baseline.json", dict(**baseline, sanity=sanity))
        print("BASELINE PASS", sanity, flush=True)

        # Verify one forward+backward probe on the 6 H132 output-layer tensors to confirm 07:26Z None-grad handling.
        with torch.enable_grad():
            for n, p in net.named_parameters():
                p.requires_grad_(n in stat.H132_OUTPUT_TENSORS)
                if p.grad is not None:
                    p.grad = None
            cid0 = val[0]["context_sha256"]
            p0 = torch.as_tensor(pools[cid0]["program"][:24]).unsqueeze(0)
            j0 = torch.as_tensor(pools[cid0]["proj16"][:24]).unsqueeze(0)
            t0 = tokens[cid0].unsqueeze(0)
            m0 = masks[cid0].unsqueeze(0)
            r_idx0 = torch.arange(4).reshape(1, -1)
            rf0 = torch.as_tensor(
                pools[cid0]["fitness"][:4], dtype=torch.float32
            ).unsqueeze(0)
            rg0 = torch.as_tensor(
                pools[cid0]["gate"][:4], dtype=torch.float32
            ).unsqueeze(0)
            f0 = torch.as_tensor(
                pools[cid0]["fitness"][:24], dtype=torch.float32
            ).unsqueeze(0)
            g0 = torch.as_tensor(pools[cid0]["gate"][:24], dtype=torch.bool).unsqueeze(
                0
            )
            out0 = ad.predict_costs(
                ctx_feat=torch.zeros((1, 584)),
                prog_412=p0,
                proj_16=j0,
                base_prior_cost=torch.zeros((1, 24)),
                scale=torch.full((1,), 0.4208),
                sos_prior=torch.zeros((1, 24, 2)),
                mm_tokens_10x32=t0,
                mm_mask_10=m0,
                revealed_indices=r_idx0,
                revealed_raw_f=rf0,
                revealed_gate=rg0,
            )
            probe_loss = rt.loss(out0["total_cost"], f0, g0)
            probe_loss.backward()
            _, none_grad_tensors = sanitize_and_verify_head_grads(
                [param_dict[n] for n in stat.H132_OUTPUT_TENSORS],
                stat.H132_OUTPUT_TENSORS,
            )
            for p in net.parameters():
                p.grad = None
            for n, p in net.named_parameters():
                p.requires_grad_(
                    n.startswith("adapter_10tok.")
                    and not n.startswith("adapter_10tok.anchor.")
                )
        rt.check_frozen(net, guard, names, frozen)

        # 2. Now load the 1,219 TRAIN pools and precompute their frozen r882 penultimate features once.
        load_pool_rows(train)
        precompute_penultimate_features(train, verify_native_bit_exact=False)

        train_index = {r["context_sha256"]: i for i, r in enumerate(train)}
        primary = np.full((len(stat.SEEDS), len(train), len(stat.ALL_ARMS)), np.nan)
        secondary = np.full((len(stat.SEEDS), 5, len(val), len(stat.ALL_ARMS)), np.nan)
        unconstrained_primary = np.full(
            (len(stat.SEEDS), len(train), len(stat.ALL_ARMS)), np.nan
        )
        unconstrained_secondary = np.full(
            (len(stat.SEEDS), 5, len(val), len(stat.ALL_ARMS)), np.nan
        )
        best_finite_primary = np.full(
            (len(stat.SEEDS), len(train), len(stat.ALL_ARMS)), np.nan
        )
        best_finite_secondary = np.full(
            (len(stat.SEEDS), 5, len(val), len(stat.ALL_ARMS)), np.nan
        )

        # Full lambda path panels across the 205-family OOF TRAIN and 91-ctx DL-Val
        lambda_path_oof_train = {
            lbl: np.full(len(train), np.nan, dtype=np.float64)
            for lbl in stat.LAMBDA_LABELS
        }
        lambda_path_val = {
            lbl: np.full((5, len(val)), np.nan, dtype=np.float64)
            for lbl in stat.LAMBDA_LABELS
        }
        lambda_path_drifts = {lbl: [] for lbl in stat.LAMBDA_LABELS}
        lambda_path_grad_norms = {lbl: [] for lbl in stat.LAMBDA_LABELS}
        lambda_path_inner_cv_gains = {lbl: [] for lbl in stat.LAMBDA_LABELS}

        selections = []
        cell_records = []
        strict_drifts = {arm: [] for arm in stat.ARMS}
        uncon_drifts = {arm: [] for arm in stat.ARMS}
        finite_drifts = {arm: [] for arm in stat.ARMS}

        for outer in range(5):
            fit, assess = stat.partition(train, outer, groups=groups)
            fit_ids = [r["context_sha256"] for r in fit]
            excluded = {r["context_sha256"] for r in assess + val}
            assert not (set(fit_ids) & excluded)

            ai_idx = [train_index[r["context_sha256"]] for r in assess]
            r882_assess_eval = evaluate_head_on_rows(
                assess,
                feature_cache,
                pools,
                np.zeros(stat.ACTIVE_RANKING_WEIGHT_PARAMS, dtype=np.float32),
            )
            r882_fit_eval = evaluate_head_on_rows(
                fit,
                feature_cache,
                pools,
                np.zeros(stat.ACTIVE_RANKING_WEIGHT_PARAMS, dtype=np.float32),
            )

            # Solve the 6 full-fit lambda heads on fit (164 families) once for this outer fold
            fit_batches = build_size_batches(fit_ids, feature_cache)
            full_fit_sols = solve_lambda_path(fit_batches)
            lbfgs_solves += len(stat.FINITE_LAMBDA_LABELS)

            full_fit_assess_evals = {"inf": r882_assess_eval}
            full_fit_val_evals = {"inf": baseline}
            for lbl in stat.LAMBDA_LABELS:
                sol = full_fit_sols[lbl]
                if lbl != "inf":
                    full_fit_assess_evals[lbl] = evaluate_head_on_rows(
                        assess, feature_cache, pools, sol["delta48"]
                    )
                    full_fit_val_evals[lbl] = evaluate_head_on_rows(
                        val, feature_cache, pools, sol["delta48"]
                    )
                lambda_path_oof_train[lbl][ai_idx] = full_fit_assess_evals[lbl][
                    "values"
                ]
                lambda_path_val[lbl][outer, :] = full_fit_val_evals[lbl]["values"]
                lambda_path_drifts[lbl].append(sol["l2_drift"])
                lambda_path_grad_norms[lbl].append(sol["grad_norm"])

            for si, seed in enumerate(stat.SEEDS):
                cell_tick = time.monotonic()
                for panel_p, panel_s in (
                    (primary, secondary),
                    (unconstrained_primary, unconstrained_secondary),
                    (best_finite_primary, best_finite_secondary),
                ):
                    panel_p[si, ai_idx, 0] = r882_assess_eval["values"]
                    panel_p[si, ai_idx, 1] = r882_assess_eval["values"]
                    panel_s[si, outer, :, 0] = baseline["values"]
                    panel_s[si, outer, :, 1] = baseline["values"]

                strict_drifts["CTRL"].append(0.0)
                uncon_drifts["CTRL"].append(0.0)
                finite_drifts["CTRL"].append(0.0)

                # --- Step A: Seed-driven inner 5-fold family CV on FIT split (164 families) for REFIT_CV ---
                icv_splits = stat.inner_cv_folds(fit, seed, outer, n_inner=5)
                fit_idx_map = {r["context_sha256"]: idx for idx, r in enumerate(fit)}
                icv_oof_values = {
                    lbl: [None] * len(fit) for lbl in stat.FINITE_LAMBDA_LABELS
                }
                icv_grad_norms = []
                for m, (itr_rows, iva_rows, _val_fams) in enumerate(icv_splits):
                    itr_ids = [r["context_sha256"] for r in itr_rows]
                    iva_ids = {r["context_sha256"] for r in iva_rows}
                    assert not (set(itr_ids) & iva_ids)
                    assert not (set(itr_ids) & excluded)
                    itr_batches = build_size_batches(itr_ids, feature_cache)
                    icv_sols = solve_lambda_path(itr_batches)
                    lbfgs_solves += len(stat.FINITE_LAMBDA_LABELS)
                    for lbl in stat.FINITE_LAMBDA_LABELS:
                        icv_grad_norms.append(icv_sols[lbl]["grad_norm"])
                        ev = evaluate_head_on_rows(
                            iva_rows, feature_cache, pools, icv_sols[lbl]["delta48"]
                        )
                        for r_v, u_v in zip(iva_rows, ev["values"], strict=True):
                            icv_oof_values[lbl][fit_idx_map[r_v["context_sha256"]]] = (
                                u_v
                            )

                for lbl in stat.FINITE_LAMBDA_LABELS:
                    assert all(v is not None for v in icv_oof_values[lbl])

                cv_decision = stat.select_lambda(
                    r882_fit_eval["values"], icv_oof_values, fit, seed, outer
                )
                for lbl in stat.LAMBDA_LABELS:
                    lambda_path_inner_cv_gains[lbl].append(
                        cv_decision["candidates"][lbl]["delta"]
                    )

                # --- Step B: REFIT_BAG (mean head of 8 family-bootstrap resamples of FIT split) ---
                # Solve 8-bag refits for the distinct finite lambdas needed by strict, unconstrained, and best_finite
                needed_bag_labels = {
                    cv_decision["selected_lambda"],
                    cv_decision["unconstrained_selected_lambda"],
                    cv_decision["best_finite_lambda"],
                }
                bag_heads_by_label = {
                    "inf": dict(
                        delta99=np.zeros(
                            stat.H132_OUTPUT_TOTAL_PARAMS, dtype=np.float64
                        ),
                        delta48=np.zeros(
                            stat.ACTIVE_RANKING_WEIGHT_PARAMS, dtype=np.float64
                        ),
                        l2_drift=0.0,
                        per_bag_l2_drifts=[0.0] * stat.BAG_COUNT,
                        max_grad_norm=0.0,
                    )
                }
                finite_needed = [
                    lbl for lbl in stat.FINITE_LAMBDA_LABELS if lbl in needed_bag_labels
                ]
                if finite_needed:
                    bag_d99_lists = {lbl: [] for lbl in finite_needed}
                    bag_gns = {lbl: [] for lbl in finite_needed}
                    for b_idx in range(stat.BAG_COUNT):
                        w_ctx, _ = stat.family_bootstrap_context_weights(
                            fit, seed, outer, b_idx
                        )
                        bag_batches = build_size_batches(
                            fit_ids, feature_cache, context_weights=w_ctx
                        )
                        prev_b = None
                        for lbl in finite_needed:
                            sol_b = solve_anchored_convex_head(
                                bag_batches,
                                stat.LAMBDA_BY_LABEL[lbl],
                                init_delta99=prev_b,
                            )
                            lbfgs_solves += 1
                            prev_b = sol_b["delta99"]
                            bag_d99_lists[lbl].append(sol_b["delta99"])
                            bag_gns[lbl].append(sol_b["grad_norm"])
                    for lbl in finite_needed:
                        mean_d99 = np.mean(np.stack(bag_d99_lists[lbl], axis=0), axis=0)
                        mean_d48 = np.concatenate(
                            [mean_d99[0:24], mean_d99[25:49]], axis=0
                        )
                        bag_heads_by_label[lbl] = dict(
                            delta99=mean_d99,
                            delta48=mean_d48,
                            l2_drift=float(np.linalg.norm(mean_d99)),
                            per_bag_l2_drifts=[
                                float(np.linalg.norm(x)) for x in bag_d99_lists[lbl]
                            ],
                            max_grad_norm=float(max(bag_gns[lbl])),
                        )

                bag_decision = dict(
                    selected_lambda=cv_decision["selected_lambda"],
                    selected_lambda_value=cv_decision["selected_lambda_value"],
                    selected_gain=cv_decision["selected_gain"],
                    unconstrained_selected_lambda=cv_decision[
                        "unconstrained_selected_lambda"
                    ],
                    unconstrained_selected_lambda_value=cv_decision[
                        "unconstrained_selected_lambda_value"
                    ],
                    unconstrained_selected_gain=cv_decision[
                        "unconstrained_selected_gain"
                    ],
                    best_finite_lambda=cv_decision["best_finite_lambda"],
                    best_finite_lambda_value=cv_decision["best_finite_lambda_value"],
                    best_finite_gain=cv_decision["best_finite_gain"],
                )

                # --- Step C: REFIT_SHUF (within-family target-permuted placebo on FIT split) ---
                shuf_targets = {}
                for cid in fit_ids:
                    f_shuf, g_shuf, _ = stat.permute_targets_within_family(
                        pools[cid]["fitness"], pools[cid]["gate"], seed, outer, cid
                    )
                    shuf_targets[cid] = (f_shuf, g_shuf)

                shuf_icv_oof_values = {
                    lbl: [None] * len(fit) for lbl in stat.FINITE_LAMBDA_LABELS
                }
                shuf_grad_norms = []
                for m, (itr_rows, iva_rows, _val_fams) in enumerate(icv_splits):
                    itr_ids = [r["context_sha256"] for r in itr_rows]
                    itr_shuf_batches = build_size_batches(
                        itr_ids, feature_cache, target_override=shuf_targets
                    )
                    shuf_icv_sols = solve_lambda_path(itr_shuf_batches)
                    lbfgs_solves += len(stat.FINITE_LAMBDA_LABELS)
                    for lbl in stat.FINITE_LAMBDA_LABELS:
                        shuf_grad_norms.append(shuf_icv_sols[lbl]["grad_norm"])
                        # Score against unpermuted inner-validation targets on I_{k,s,m}
                        ev_shuf = evaluate_head_on_rows(
                            iva_rows,
                            feature_cache,
                            pools,
                            shuf_icv_sols[lbl]["delta48"],
                        )
                        for r_v, u_v in zip(iva_rows, ev_shuf["values"], strict=True):
                            shuf_icv_oof_values[lbl][
                                fit_idx_map[r_v["context_sha256"]]
                            ] = u_v

                shuf_decision = stat.select_lambda(
                    r882_fit_eval["values"], shuf_icv_oof_values, fit, seed, outer
                )

                # Fit full-FIT shuffled head for needed finite lambdas
                needed_shuf_labels = {
                    shuf_decision["selected_lambda"],
                    shuf_decision["unconstrained_selected_lambda"],
                    shuf_decision["best_finite_lambda"],
                }
                shuf_full_batches = build_size_batches(
                    fit_ids, feature_cache, target_override=shuf_targets
                )
                shuf_heads_by_label = {
                    "inf": dict(
                        delta99=np.zeros(
                            stat.H132_OUTPUT_TOTAL_PARAMS, dtype=np.float64
                        ),
                        delta48=np.zeros(
                            stat.ACTIVE_RANKING_WEIGHT_PARAMS, dtype=np.float64
                        ),
                        l2_drift=0.0,
                        grad_norm=0.0,
                    )
                }
                prev_shuf = None
                for lbl in stat.FINITE_LAMBDA_LABELS:
                    if lbl in needed_shuf_labels:
                        sol_s = solve_anchored_convex_head(
                            shuf_full_batches,
                            stat.LAMBDA_BY_LABEL[lbl],
                            init_delta99=prev_shuf,
                        )
                        lbfgs_solves += 1
                        prev_shuf = sol_s["delta99"]
                        shuf_heads_by_label[lbl] = sol_s
                        shuf_grad_norms.append(sol_s["grad_norm"])

                # --- Step D: Lock inner-CV decisions before outer assessment evaluation ---
                cell_decisions = {
                    "CTRL": dict(
                        selected_lambda="inf",
                        selected_lambda_value=None,
                        selected_gain=0.0,
                        unconstrained_selected_lambda="inf",
                        unconstrained_selected_lambda_value=None,
                        unconstrained_selected_gain=0.0,
                        best_finite_lambda="inf",
                        best_finite_lambda_value=None,
                        best_finite_gain=0.0,
                    ),
                    "REFIT_CV": cv_decision,
                    "REFIT_BAG": bag_decision,
                    "REFIT_SHUF": shuf_decision,
                }
                sel_path = out / f"A{outer}-{seed}-selection.json"
                save(
                    sel_path,
                    dict(
                        outer_fold=outer,
                        seed=seed,
                        fit_families=sorted({r["family_id"] for r in fit}),
                        assess_families=sorted({r["family_id"] for r in assess}),
                        fit_context_ids=fit_ids,
                        fit_family_ids=[r["family_id"] for r in fit],
                        r882_fit_values=r882_fit_eval["values"],
                        refit_cv_inner_oof_values=icv_oof_values,
                        refit_shuf_inner_oof_values=shuf_icv_oof_values,
                        selected=cell_decisions,
                    ),
                )
                sel_lock_sha = sha(sel_path)
                selections.append(cell_decisions)

                # --- Step E: Evaluate locked heads on outer assessment fold A_k and DL-Val ---
                arm_sources = {
                    "REFIT_CV": (
                        cv_decision,
                        lambda lbl: full_fit_sols[lbl],
                    ),
                    "REFIT_BAG": (
                        bag_decision,
                        lambda lbl: bag_heads_by_label[lbl],
                    ),
                    "REFIT_SHUF": (
                        shuf_decision,
                        lambda lbl: shuf_heads_by_label[lbl],
                    ),
                }

                head_shas = {}
                assess_choices = {}
                cell_drifts = {
                    "CTRL": dict(strict=0.0, unconstrained=0.0, best_finite=0.0)
                }
                snapshot_shas = {}

                for arm_idx, arm in enumerate(
                    ("REFIT_CV", "REFIT_BAG", "REFIT_SHUF"), 2
                ):
                    dec, getter = arm_sources[arm]
                    head_shas[arm] = {}
                    assess_choices[arm] = {}
                    cell_drifts[arm] = {}
                    snapshot_shas[arm] = {}

                    for mode_key, dec_field, p_mat, s_mat, drift_list in (
                        (
                            "strict",
                            "selected_lambda",
                            primary,
                            secondary,
                            strict_drifts[arm],
                        ),
                        (
                            "unconstrained",
                            "unconstrained_selected_lambda",
                            unconstrained_primary,
                            unconstrained_secondary,
                            uncon_drifts[arm],
                        ),
                        (
                            "best_finite",
                            "best_finite_lambda",
                            best_finite_primary,
                            best_finite_secondary,
                            finite_drifts[arm],
                        ),
                    ):
                        lbl = dec[dec_field]
                        h_obj = getter(lbl)
                        d99 = h_obj["delta99"]
                        d48 = h_obj["delta48"]
                        l2_d = float(h_obj["l2_drift"])
                        drift_list.append(l2_d)
                        cell_drifts[arm][mode_key] = l2_d

                        snap_file = out / f"A{outer}-{seed}-{arm}-{mode_key}.npz"
                        np.savez(
                            snap_file,
                            **head_state_tensors(base_head_tensors, d99),
                        )
                        snap_sha = sha(snap_file)
                        snapshot_shas[arm][mode_key] = snap_sha
                        head_shas[arm][mode_key] = hashlib.sha256(
                            np.asarray(d99, dtype=np.float64).tobytes()
                        ).hexdigest()

                        ev_a = evaluate_head_on_rows(assess, feature_cache, pools, d48)
                        ev_v = evaluate_head_on_rows(val, feature_cache, pools, d48)
                        p_mat[si, ai_idx, arm_idx] = ev_a["values"]
                        s_mat[si, outer, :, arm_idx] = ev_v["values"]
                        assess_choices[arm][mode_key] = ev_a["choices"]

                rt.check_frozen(net, guard, names, frozen)
                cell_wall = time.monotonic() - cell_tick
                cell_rec = dict(
                    outer_fold=outer,
                    seed=seed,
                    fit_context_count=len(fit),
                    assess_context_count=len(assess),
                    fit_family_count=164,
                    assess_family_count=41,
                    none_grad_tensors=none_grad_tensors,
                    selection_lock_sha256=sel_lock_sha,
                    decisions=cell_decisions,
                    head_l2_drift=cell_drifts,
                    head_sha256=head_shas,
                    snapshot_sha256=snapshot_shas,
                    assess_choices=assess_choices,
                    max_icv_grad_norm=float(max(icv_grad_norms)),
                    max_shuf_grad_norm=float(max(shuf_grad_norms)),
                    bag_details={
                        lbl: dict(
                            l2_drift=bag_heads_by_label[lbl]["l2_drift"],
                            per_bag_l2_drifts=bag_heads_by_label[lbl][
                                "per_bag_l2_drifts"
                            ],
                            max_grad_norm=bag_heads_by_label[lbl]["max_grad_norm"],
                        )
                        for lbl in bag_heads_by_label
                    },
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    wall_seconds=cell_wall,
                )
                cell_records.append(cell_rec)
                save(out / f"A{outer}-{seed}.json", cell_rec)
                print(
                    "CELL COMPLETE",
                    f"A{outer}-{seed}",
                    "lbfgs_solves",
                    lbfgs_solves,
                    "strict_picks",
                    {a: cell_decisions[a]["selected_lambda"] for a in stat.ARMS},
                    "uncon_picks",
                    {
                        a: cell_decisions[a]["unconstrained_selected_lambda"]
                        for a in stat.ARMS
                    },
                    "finite_picks",
                    {a: cell_decisions[a]["best_finite_lambda"] for a in stat.ARMS},
                    "seconds",
                    round(cell_wall, 2),
                    flush=True,
                )

        if (
            np.isnan(primary).any()
            or np.isnan(secondary).any()
            or np.isnan(unconstrained_primary).any()
            or np.isnan(unconstrained_secondary).any()
            or np.isnan(best_finite_primary).any()
            or np.isnan(best_finite_secondary).any()
        ):
            raise RuntimeError("incomplete evaluation panel")

        np.savez(
            out / "panels.npz",
            primary=primary,
            secondary=secondary,
            unconstrained_primary=unconstrained_primary,
            unconstrained_secondary=unconstrained_secondary,
            best_finite_primary=best_finite_primary,
            best_finite_secondary=best_finite_secondary,
            lambda_path_oof_train=np.stack(
                [lambda_path_oof_train[lbl] for lbl in stat.LAMBDA_LABELS], axis=-1
            ),
            lambda_path_val=np.stack(
                [lambda_path_val[lbl] for lbl in stat.LAMBDA_LABELS], axis=-1
            ),
        )

        def minus2_summary(panel_3d):
            out_m2 = {}
            for idx, arm in enumerate(stat.ALL_ARMS):
                counts = [
                    int(np.sum(panel_3d[si, :, idx] <= -2.0 + 1e-9))
                    for si in range(len(stat.SEEDS))
                ]
                out_m2[arm] = dict(
                    per_seed=counts,
                    median=float(np.median(counts)),
                    mean=float(np.mean(counts)),
                )
            return out_m2

        drift_summary = {
            arm: dict(
                strict_l2_median=float(np.median(strict_drifts[arm])),
                strict_l2_mean=float(np.mean(strict_drifts[arm])),
                strict_l2_max=float(np.max(strict_drifts[arm])),
                unconstrained_l2_median=float(np.median(uncon_drifts[arm])),
                unconstrained_l2_mean=float(np.mean(uncon_drifts[arm])),
                best_finite_l2_median=float(np.median(finite_drifts[arm])),
                best_finite_l2_mean=float(np.mean(finite_drifts[arm])),
            )
            for arm in stat.ARMS
        }

        primary_summary = stat.summarize(
            train,
            primary,
            minus2_counts=minus2_summary(primary),
            drift_by_arm=drift_summary,
        )
        secondary_mean = secondary.mean(axis=1)
        secondary_summary = stat.summarize(
            val,
            secondary_mean,
            minus2_counts=minus2_summary(secondary_mean),
            drift_by_arm=drift_summary,
        )

        uncon_primary_summary = stat.summarize(
            train,
            unconstrained_primary,
            minus2_counts=minus2_summary(unconstrained_primary),
            drift_by_arm=drift_summary,
        )
        uncon_secondary_mean = unconstrained_secondary.mean(axis=1)
        uncon_secondary_summary = stat.summarize(
            val,
            uncon_secondary_mean,
            minus2_counts=minus2_summary(uncon_secondary_mean),
            drift_by_arm=drift_summary,
        )

        finite_primary_summary = stat.summarize(
            train,
            best_finite_primary,
            minus2_counts=minus2_summary(best_finite_primary),
            drift_by_arm=drift_summary,
        )
        finite_secondary_mean = best_finite_secondary.mean(axis=1)
        finite_secondary_summary = stat.summarize(
            val,
            finite_secondary_mean,
            minus2_counts=minus2_summary(finite_secondary_mean),
            drift_by_arm=drift_summary,
        )

        # Build descriptive full lambda-path summary across the 7 grid points
        r882_train_fams, r882_train_vec = stat.family_vector(
            train, lambda_path_oof_train["inf"]
        )
        r882_val_fams, r882_val_vec = stat.family_vector(
            val, lambda_path_val["inf"].mean(axis=0)
        )
        lpath_names = ("R882_inf",) + tuple(
            f"LAM_{lbl}" for lbl in stat.FINITE_LAMBDA_LABELS
        )
        lpath_reordered_train = np.stack(
            [
                np.stack(
                    [lambda_path_oof_train["inf"]]
                    + [lambda_path_oof_train[lbl] for lbl in stat.FINITE_LAMBDA_LABELS],
                    axis=-1,
                )
                for _ in stat.SEEDS
            ],
            axis=0,
        )
        lpath_elo_train = stat.elo_ratings(
            train,
            lpath_reordered_train,
            draws=1000,
            seed=stat.BOOT_SEED + 500,
            arm_names=lpath_names,
        )

        lambda_path_summary = {}
        for idx_l, lbl in enumerate(stat.LAMBDA_LABELS):
            t_vals = lambda_path_oof_train[lbl]
            v_vals = lambda_path_val[lbl].mean(axis=0)
            _, t_vec = stat.family_vector(train, t_vals)
            _, v_vec = stat.family_vector(val, v_vals)
            if lbl == "inf":
                t_boot = dict(
                    delta=0.0,
                    ci_low_95=0.0,
                    ci_high_95=0.0,
                    p_one_sided=1.0,
                    draws=10000,
                )
                v_boot = dict(
                    delta=0.0,
                    ci_low_95=0.0,
                    ci_high_95=0.0,
                    p_one_sided=1.0,
                    draws=10000,
                )
                elo_key = "R882_inf"
            else:
                t_boot = stat.bootstrap_family(
                    t_vec - r882_train_vec,
                    draws=10000,
                    seed=stat.BOOT_SEED + 600 + idx_l,
                )
                v_boot = stat.bootstrap_family(
                    v_vec - r882_val_vec,
                    draws=10000,
                    seed=stat.BOOT_SEED + 700 + idx_l,
                )
                elo_key = f"LAM_{lbl}"
            lambda_path_summary[lbl] = dict(
                lambda_value=stat.LAMBDA_BY_LABEL[lbl],
                oof_train_equal_family_mean=float(t_vec.mean()),
                oof_train_raw_context_mean=float(t_vals.mean()),
                oof_train_delta_vs_r882=t_boot["delta"],
                oof_train_ci_low_95=t_boot["ci_low_95"],
                oof_train_ci_high_95=t_boot["ci_high_95"],
                oof_train_p_one_sided=t_boot["p_one_sided"],
                oof_train_elo=lpath_elo_train[elo_key]["elo"],
                oof_train_elo_ci_low_95=lpath_elo_train[elo_key]["ci_low_95"],
                oof_train_elo_ci_high_95=lpath_elo_train[elo_key]["ci_high_95"],
                oof_train_minus2_count=int(np.sum(t_vals <= -2.0 + 1e-9)),
                dl_val_equal_family_mean=float(v_vec.mean()),
                dl_val_raw_context_mean=float(v_vals.mean()),
                dl_val_delta_vs_r882=v_boot["delta"],
                dl_val_ci_low_95=v_boot["ci_low_95"],
                dl_val_ci_high_95=v_boot["ci_high_95"],
                dl_val_minus2_count=int(np.sum(v_vals <= -2.0 + 1e-9)),
                inner_cv_gain_median=float(np.median(lambda_path_inner_cv_gains[lbl])),
                inner_cv_gain_mean=float(np.mean(lambda_path_inner_cv_gains[lbl])),
                head_l2_drift_median=float(np.median(lambda_path_drifts[lbl])),
                head_l2_drift_mean=float(np.mean(lambda_path_drifts[lbl])),
                head_l2_drift_max=float(np.max(lambda_path_drifts[lbl])),
                max_grad_norm=float(np.max(lambda_path_grad_norms[lbl])),
            )

        pseudo_report = stat.pseudoreplication_report(cell_records)

        advance = any(
            primary_summary["arms"][a]["advance"] for a in stat.TREATMENT_ARMS
        )
        rule_neutral = all(
            c[a]["selected_lambda"] == "inf" for c in selections for a in stat.ARMS
        )
        metrics = dict(
            wave="H172",
            status="COMPLETE",
            verdict="ADVANCE" if advance else "NO_ADVANCE",
            rule_neutral=rule_neutral,
            placebo_advance_leak_flag=primary_summary["placebo_advance_leak_flag"],
            h145_train_1219_205fam=primary_summary,
            dl_val_91_19fam=secondary_summary,
            primary=primary_summary,
            secondary_dl_val=secondary_summary,
            unconstrained_best_lambda_h145_train_1219_205fam=uncon_primary_summary,
            unconstrained_best_lambda_dl_val_91_19fam=uncon_secondary_summary,
            best_finite_lambda_h145_train_1219_205fam=finite_primary_summary,
            best_finite_lambda_dl_val_91_19fam=finite_secondary_summary,
            lambda_path_summary=lambda_path_summary,
            pseudoreplication_check=pseudo_report,
            drift_summary=drift_summary,
            parameter_inventory=dict(
                adapter_total_tensors=len(names),
                adapter_total_parameters=trainable_params,
                h132_output_tensors=list(stat.H132_OUTPUT_TENSORS),
                h132_output_total_parameters=stat.H132_OUTPUT_TOTAL_PARAMS,
                h132_output_weight_parameters=stat.H132_OUTPUT_WEIGHT_PARAMS,
                h132_output_bias_parameters=stat.H132_OUTPUT_BIAS_PARAMS,
                active_ranking_weight_tensors=list(stat.ACTIVE_RANKING_WEIGHT_TENSORS),
                active_ranking_weight_parameters=stat.ACTIVE_RANKING_WEIGHT_PARAMS,
                none_grad_tensors=none_grad_tensors,
            ),
            selections=selections,
            lbfgs_solves=lbfgs_solves,
            optimizer_updates=lbfgs_solves,
            cells=len(selections),
            baseline=sanity,
            gpu_lock_held=False,
            source_commit=launch["source_commit"],
            manifest_sha256=launch["manifest_sha256"],
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            historical_r882_train_exposure=True,
            serving_promotion=False,
            wall_seconds=time.monotonic() - started,
        )
        save(out / "metrics.json", metrics)
    finally:
        import resource
        import torch

        save(
            out / "audit.json",
            dict(
                **audit.receipt(),
                lbfgs_solves=lbfgs_solves,
                optimizer_updates=lbfgs_solves,
                wall_seconds=time.monotonic() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                threads_at_end=torch.get_num_threads(),
                cuda_initialized=torch.cuda.is_available()
                and torch.cuda.is_initialized(),
            ),
        )


if __name__ == "__main__":
    main()
