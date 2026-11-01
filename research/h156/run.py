"""Single H156 continuation wave: 5-fold family cross-fit early stopping + capacity sweep {0.5x, 1x, 2x}."""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np
import analysis as stat

HERE = Path(__file__).resolve().parent


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
    spec = importlib.util.spec_from_file_location("h156_parent", parent / "runner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.path.pop(0)
    return module, sys.modules["runtime"]


def configure_adapter_width(net, r882_sd, names, width_mult, seed):
    import torch
    import torch.nn as nn

    ad = net.adapter_10tok
    ad.prog_412_bottleneck = nn.Sequential(
        nn.Linear(412, 16), nn.LayerNorm(16), nn.SiLU()
    )
    ad.cand_query_proj = nn.Sequential(nn.Linear(66, 32), nn.LayerNorm(32), nn.SiLU())
    ad.ctx_bilinear_u = nn.Linear(32, 16, bias=False)
    ad.prog_bilinear_v = nn.Linear(24, 16, bias=False)
    ad.mm_prior_out = nn.Sequential(nn.Linear(50, 24), nn.SiLU(), nn.Linear(24, 1))
    ad.traj_delta_head = nn.Sequential(nn.Linear(67, 24), nn.SiLU(), nn.Linear(24, 1))
    ad.parent_delta_net = nn.Sequential(
        nn.Linear(114, 48), nn.LayerNorm(48), nn.SiLU(), nn.Linear(48, 1)
    )
    net.load_state_dict(r882_sd, strict=True)
    if abs(width_mult - 1.0) > 1e-9:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + int(round(width_mult * 1000)))
        d_bot = int(round(16 * width_mult))
        d_bi = int(round(16 * width_mult))
        d_mm = int(round(24 * width_mult))
        d_traj = int(round(24 * width_mult))
        d_par = int(round(48 * width_mult))
        with torch.no_grad():
            w_bot0 = ad.prog_412_bottleneck[0].weight.detach().cpu()
            b_bot0 = ad.prog_412_bottleneck[0].bias.detach().cpu()
            w_bot1 = ad.prog_412_bottleneck[1].weight.detach().cpu()
            b_bot1 = ad.prog_412_bottleneck[1].bias.detach().cpu()
            w_cq0 = ad.cand_query_proj[0].weight.detach().cpu()
            b_cq0 = ad.cand_query_proj[0].bias.detach().cpu()
            w_cq1 = ad.cand_query_proj[1].weight.detach().cpu()
            b_cq1 = ad.cand_query_proj[1].bias.detach().cpu()
            w_bi_u = ad.ctx_bilinear_u.weight.detach().cpu()
            w_bi_v = ad.prog_bilinear_v.weight.detach().cpu()
            w_mm0 = ad.mm_prior_out[0].weight.detach().cpu()
            b_mm0 = ad.mm_prior_out[0].bias.detach().cpu()
            w_mm2 = ad.mm_prior_out[2].weight.detach().cpu()
            b_mm2 = ad.mm_prior_out[2].bias.detach().cpu()
            w_tr0 = ad.traj_delta_head[0].weight.detach().cpu()
            b_tr0 = ad.traj_delta_head[0].bias.detach().cpu()
            w_tr2 = ad.traj_delta_head[2].weight.detach().cpu()
            b_tr2 = ad.traj_delta_head[2].bias.detach().cpu()
            w_pa0 = ad.parent_delta_net[0].weight.detach().cpu()
            b_pa0 = ad.parent_delta_net[0].bias.detach().cpu()
            w_pa1 = ad.parent_delta_net[1].weight.detach().cpu()
            b_pa1 = ad.parent_delta_net[1].bias.detach().cpu()
            w_pa3 = ad.parent_delta_net[3].weight.detach().cpu()
            b_pa3 = ad.parent_delta_net[3].bias.detach().cpu()

            ad.prog_412_bottleneck = nn.Sequential(
                nn.Linear(412, d_bot), nn.LayerNorm(d_bot), nn.SiLU()
            )
            ad.cand_query_proj = nn.Sequential(
                nn.Linear(50 + d_bot, 32), nn.LayerNorm(32), nn.SiLU()
            )
            ad.cand_query_proj[1].weight.copy_(w_cq1)
            ad.cand_query_proj[1].bias.copy_(b_cq1)
            ad.ctx_bilinear_u = nn.Linear(32, d_bi, bias=False)
            ad.prog_bilinear_v = nn.Linear(24, d_bi, bias=False)
            ad.mm_prior_out = nn.Sequential(
                nn.Linear(34 + d_bi, d_mm), nn.SiLU(), nn.Linear(d_mm, 1)
            )
            ad.traj_delta_head = nn.Sequential(
                nn.Linear(67, d_traj), nn.SiLU(), nn.Linear(d_traj, 1)
            )
            ad.parent_delta_net = nn.Sequential(
                nn.Linear(114, d_par),
                nn.LayerNorm(d_par),
                nn.SiLU(),
                nn.Linear(d_par, 1),
            )

            if width_mult > 1.0:
                eps_s = 1e-3
                ad.prog_412_bottleneck[0].weight.copy_(
                    torch.cat([w_bot0, w_bot0], dim=0)
                )
                ad.prog_412_bottleneck[0].bias.copy_(torch.cat([b_bot0, b_bot0], dim=0))
                ad.prog_412_bottleneck[1].weight.copy_(
                    torch.cat([w_bot1, w_bot1], dim=0)
                )
                ad.prog_412_bottleneck[1].bias.copy_(torch.cat([b_bot1, b_bot1], dim=0))

                w_cq_b = w_cq0[:, 48:64]
                e_cq = torch.randn(w_cq_b.shape, generator=g, dtype=w_cq0.dtype) * eps_s
                ad.cand_query_proj[0].weight.copy_(
                    torch.cat(
                        [
                            w_cq0[:, :48],
                            0.5 * w_cq_b + e_cq,
                            0.5 * w_cq_b - e_cq,
                            w_cq0[:, 64:],
                        ],
                        dim=1,
                    )
                )
                ad.cand_query_proj[0].bias.copy_(b_cq0)

                ad.ctx_bilinear_u.weight.copy_(torch.cat([w_bi_u, w_bi_u], dim=0))
                ad.prog_bilinear_v.weight.copy_(torch.cat([w_bi_v, w_bi_v], dim=0))

                w_mm_b = w_mm0[:, 32:48]
                e_bi = torch.randn(w_mm_b.shape, generator=g, dtype=w_mm0.dtype) * eps_s
                w_mm_in = torch.cat(
                    [
                        w_mm0[:, :32],
                        0.5 * w_mm_b + e_bi,
                        0.5 * w_mm_b - e_bi,
                        w_mm0[:, 48:],
                    ],
                    dim=1,
                )
                ad.mm_prior_out[0].weight.copy_(torch.cat([w_mm_in, w_mm_in], dim=0))
                ad.mm_prior_out[0].bias.copy_(torch.cat([b_mm0, b_mm0], dim=0))
                e_mm2 = torch.randn(w_mm2.shape, generator=g, dtype=w_mm2.dtype) * eps_s
                ad.mm_prior_out[2].weight.copy_(
                    torch.cat([0.5 * w_mm2 + e_mm2, 0.5 * w_mm2 - e_mm2], dim=1)
                )
                ad.mm_prior_out[2].bias.copy_(b_mm2)

                ad.traj_delta_head[0].weight.copy_(torch.cat([w_tr0, w_tr0], dim=0))
                ad.traj_delta_head[0].bias.copy_(torch.cat([b_tr0, b_tr0], dim=0))
                e_tr2 = torch.randn(w_tr2.shape, generator=g, dtype=w_tr2.dtype) * eps_s
                ad.traj_delta_head[2].weight.copy_(
                    torch.cat([0.5 * w_tr2 + e_tr2, 0.5 * w_tr2 - e_tr2], dim=1)
                )
                ad.traj_delta_head[2].bias.copy_(b_tr2)

                ad.parent_delta_net[0].weight.copy_(torch.cat([w_pa0, w_pa0], dim=0))
                ad.parent_delta_net[0].bias.copy_(torch.cat([b_pa0, b_pa0], dim=0))
                ad.parent_delta_net[1].weight.copy_(torch.cat([w_pa1, w_pa1], dim=0))
                ad.parent_delta_net[1].bias.copy_(torch.cat([b_pa1, b_pa1], dim=0))
                e_pa3 = torch.randn(w_pa3.shape, generator=g, dtype=w_pa3.dtype) * eps_s
                ad.parent_delta_net[3].weight.copy_(
                    torch.cat([0.5 * w_pa3 + e_pa3, 0.5 * w_pa3 - e_pa3], dim=1)
                )
                ad.parent_delta_net[3].bias.copy_(b_pa3)
            else:
                imp_bot = (
                    w_bot0.norm(dim=1) * w_bot1.abs() * w_cq0[:, 48:64].norm(dim=0)
                )
                idx_bot = (
                    torch.topk(imp_bot, k=d_bot, largest=True).indices.sort().values
                )
                ad.prog_412_bottleneck[0].weight.copy_(w_bot0[idx_bot])
                ad.prog_412_bottleneck[0].bias.copy_(b_bot0[idx_bot])
                ad.prog_412_bottleneck[1].weight.copy_(w_bot1[idx_bot])
                ad.prog_412_bottleneck[1].bias.copy_(b_bot1[idx_bot])
                ad.cand_query_proj[0].weight.copy_(
                    torch.cat(
                        [w_cq0[:, :48], w_cq0[:, 48 + idx_bot], w_cq0[:, 64:]], dim=1
                    )
                )
                ad.cand_query_proj[0].bias.copy_(b_cq0)

                imp_bi = (
                    w_bi_u.norm(dim=1)
                    * w_bi_v.norm(dim=1)
                    * w_mm0[:, 32:48].norm(dim=0)
                )
                idx_bi = torch.topk(imp_bi, k=d_bi, largest=True).indices.sort().values
                ad.ctx_bilinear_u.weight.copy_(w_bi_u[idx_bi])
                ad.prog_bilinear_v.weight.copy_(w_bi_v[idx_bi])

                w_mm0_in = torch.cat(
                    [w_mm0[:, :32], w_mm0[:, 32 + idx_bi], w_mm0[:, 48:]], dim=1
                )
                imp_mm = w_mm0_in.norm(dim=1) * w_mm2[0].abs()
                idx_mm = torch.topk(imp_mm, k=d_mm, largest=True).indices.sort().values
                ad.mm_prior_out[0].weight.copy_(w_mm0_in[idx_mm])
                ad.mm_prior_out[0].bias.copy_(b_mm0[idx_mm])
                ad.mm_prior_out[2].weight.copy_(w_mm2[:, idx_mm])
                ad.mm_prior_out[2].bias.copy_(b_mm2)

                imp_tr = w_tr0.norm(dim=1) * w_tr2[0].abs()
                idx_tr = (
                    torch.topk(imp_tr, k=d_traj, largest=True).indices.sort().values
                )
                ad.traj_delta_head[0].weight.copy_(w_tr0[idx_tr])
                ad.traj_delta_head[0].bias.copy_(b_tr0[idx_tr])
                ad.traj_delta_head[2].weight.copy_(w_tr2[:, idx_tr])
                ad.traj_delta_head[2].bias.copy_(b_tr2)

                imp_pa = w_pa0.norm(dim=1) * w_pa1.abs() * w_pa3[0].abs()
                idx_pa = torch.topk(imp_pa, k=d_par, largest=True).indices.sort().values
                ad.parent_delta_net[0].weight.copy_(w_pa0[idx_pa])
                ad.parent_delta_net[0].bias.copy_(b_pa0[idx_pa])
                ad.parent_delta_net[1].weight.copy_(w_pa1[idx_pa])
                ad.parent_delta_net[1].bias.copy_(b_pa1[idx_pa])
                ad.parent_delta_net[3].weight.copy_(w_pa3[:, idx_pa])
                ad.parent_delta_net[3].bias.copy_(b_pa3)

    dev = next(net.adapter_10tok.anchor.parameters()).device
    ad.to(dev)
    net.eval()
    for name, p in net.named_parameters():
        p.requires_grad_(
            name.startswith("adapter_10tok.")
            and not name.startswith("adapter_10tok.anchor.")
        )
    cur_names = [n for n, p in net.named_parameters() if p.requires_grad]
    if sorted(cur_names) != sorted(names):
        raise RuntimeError("adapter parameter tensor names changed")
    param_map = dict(net.named_parameters())
    params = [param_map[n] for n in names]
    init_state = {n: net.state_dict()[n].detach().cpu().clone() for n in names}
    init_vec = torch.cat([init_state[n].reshape(-1) for n in names])
    return params, init_state, init_vec


def forward_adapter_with_anchor(
    net,
    prog_t,
    sos_t,
    tok_t,
    mask_t,
    scale_t,
    rev_idx,
    rev_raw_f,
    rev_gate,
    anchor_total,
    anchor_eff_prior,
):
    import torch.nn.functional as F

    ad = net.adapter_10tok
    ctx_pooled, cand_q, sem24, mm_prior_adj = (
        ad.encode_multimodal_context_and_candidates(
            mm_tokens_10x32=tok_t,
            mm_mask_10=mask_t,
            prog_412=prog_t,
            sos_prior=sos_t,
            active_modality_override_mask=None,
        )
    )
    static = {
        "eff_prior": anchor_eff_prior + mm_prior_adj,
        "mm_prior_adj": mm_prior_adj,
        "sem24": sem24,
        "cand_q": cand_q,
        "cand_q_norm": F.normalize(cand_q, p=2, dim=-1, eps=1e-8),
        "ctx_pooled": ctx_pooled,
    }
    delta = ad.compute_adapter_delta_from_static_terms(
        static_terms=static,
        scale=scale_t,
        revealed_indices=rev_idx,
        revealed_raw_f=rev_raw_f,
        revealed_gate=rev_gate,
    )
    return (anchor_total + delta).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--launch", type=Path)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--gpu-lock-held", action="store_true")
    args = ap.parse_args()
    out = args.output.resolve()
    out.mkdir()
    pins = json.loads((HERE / "manifest.json").read_text())
    verify(pins)
    parent, rt = parent_executor()
    import torch

    train, val = parent.selected()
    if len({r["family_id"] for r in train}) != 205:
        raise RuntimeError("TRAIN family inventory")
    groups = stat.folds(train)
    if groups != json.loads((HERE / "fold_manifest.json").read_text())["families"]:
        raise RuntimeError("fold identity")
    allowed = set(pins["inputs"])
    allowed.update(
        str(rt.INPUT / "pools" / f"{r['context_sha256']}.npz") for r in train + val
    )
    audit = rt.Audit(allowed, out)
    sys.addaudithook(audit.check)
    updates = 0
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
        initial = {n: t.detach().clone() for n, t in net.state_dict().items()}
        width_counts = {}
        for arm in stat.WIDTH_ARMS:
            params_w, _, _ = configure_adapter_width(
                net, initial, names, stat.WIDTH_MULTS[arm], stat.SEEDS[0]
            )
            width_counts[arm] = sum(p.numel() for p in params_w)
            rt.check_frozen(net, guard, names, frozen)
        configure_adapter_width(net, initial, names, 1.0, stat.SEEDS[0])
        rt.check_frozen(net, guard, names, frozen)
        if args.smoke:
            save(
                out / "smoke.json",
                dict(
                    status="PASS",
                    optimizer_updates=0,
                    scored_contexts=0,
                    adapter_tensors=len(names),
                    trainable_parameters=width_counts["W10"],
                    width_parameter_counts=width_counts,
                    census=census,
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    packages=pins["packages"],
                ),
            )
            return

        launch = json.loads(args.launch.read_text())
        if (
            launch["manifest_sha256"] != sha(HERE / "manifest.json")
            or not launch["merged_source_verified"]
        ):
            raise RuntimeError("merged preregistration required")
        if (
            launch["source_commit"]
            != __import__("subprocess")
            .check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True)
            .strip()
        ):
            raise RuntimeError("launch source HEAD mismatch")

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
        for row in train + val:
            cid = row["context_sha256"]
            path = rt.INPUT / "pools" / f"{cid}.npz"
            if rt.sha(path) != row["pool_sha256"]:
                raise RuntimeError("pool identity")
            with np.load(path, allow_pickle=False) as z:
                program = z["program"].astype(np.float32)
                pools[cid] = dict(
                    program=program,
                    proj16=(program.astype(np.float64) @ projection).astype(np.float32),
                    fitness=z["fitness"].astype(np.float64),
                    gate=z["gate"].astype(bool),
                )
        use_cuda = bool(args.gpu_lock_held and torch.cuda.is_available())
        if use_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            device = torch.device("cuda:0")
            net.to(device)
        else:
            device = torch.device("cpu")

        with np.load(rt.CACHE, allow_pickle=False) as z:
            ids = list(map(str, z["cids"]))
            if set(ids) != set(pools) or len(ids) != len(pools):
                raise RuntimeError("feature cache closure")
            tok, mask = z["tokens_10x32"], z["mask_10"]
            tokens = {
                c: torch.as_tensor(tok[i], dtype=torch.float32, device=device)
                for i, c in enumerate(ids)
            }
            masks = {
                c: torch.as_tensor(mask[i], dtype=torch.bool, device=device)
                for i, c in enumerate(ids)
            }

        # Precompute frozen anchor terms per cohort (12 MB total) for exact zero-diff evaluation
        eval_cache = {}

        def precompute_cohort(cohort_rows):
            with torch.no_grad():
                cohort_groups = {}
                for r in cohort_rows:
                    c = r["context_sha256"]
                    cohort_groups.setdefault(len(pools[c]["fitness"]), []).append(c)
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
                        a_out = net.adapter_10tok.anchor.predict_costs(
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
                            revealed_indices=rev_idx,
                            revealed_raw_f=rev_raw_f,
                            revealed_gate=rev_gate,
                            enable_prior_correction=True,
                            enable_struct_kernel=True,
                            enable_cross_attn=True,
                            enable_ucb=False,
                        )
                        a_tot = a_out["total_cost"].detach().float()
                        a_eff = a_out["eff_prior"].detach().float()
                        for i_b, c in enumerate(batch):
                            eval_cache[c] = (
                                a_tot[i_b].clone(),
                                a_eff[i_b].clone(),
                                rev_raw_f[i_b].clone(),
                                rev_gate[i_b].clone(),
                            )

        def evaluate(rows):
            groups_r = {}
            result = {}
            for r in rows:
                c = r["context_sha256"]
                groups_r.setdefault(len(pools[c]["fitness"]), []).append(c)
            with torch.no_grad():
                for size, cids in sorted(groups_r.items()):
                    for start in range(0, len(cids), 8):
                        batch = cids[start : start + 8]
                        b = len(batch)
                        k = size
                        prog_b = torch.as_tensor(
                            np.stack([pools[c]["program"] for c in batch]),
                            dtype=torch.float32,
                            device=device,
                        )
                        a_tot_b = torch.stack([eval_cache[c][0] for c in batch])
                        a_eff_b = torch.stack([eval_cache[c][1] for c in batch])
                        rev_raw_b = torch.stack([eval_cache[c][2] for c in batch])
                        rev_gate_b = torch.stack([eval_cache[c][3] for c in batch])
                        tok_b = torch.stack([tokens[c] for c in batch])
                        mask_b = torch.stack([masks[c] for c in batch])
                        rev_idx = (
                            torch.arange(min(4, k), device=device)
                            .reshape(1, -1)
                            .expand(b, -1)
                        )
                        scale_b = torch.full(
                            (b,), 0.4208, dtype=torch.float32, device=device
                        )
                        sos_b = torch.zeros(
                            (b, k, 2), dtype=torch.float32, device=device
                        )
                        scores = (
                            forward_adapter_with_anchor(
                                net,
                                prog_b,
                                sos_b,
                                tok_b,
                                mask_b,
                                scale_b,
                                rev_idx,
                                rev_raw_b,
                                rev_gate_b,
                                a_tot_b,
                                a_eff_b,
                            )
                            .cpu()
                            .numpy()
                        )
                        if not np.isfinite(scores).all():
                            raise RuntimeError("nonfinite predictions")
                        for cid, cost in zip(batch, scores, strict=True):
                            idx_min = int(np.argmin(cost))
                            p = pools[cid]
                            valid = bool(p["gate"][idx_min])
                            result[cid] = dict(
                                choice=idx_min,
                                utility=float(
                                    -p["fitness"][idx_min] if valid else -2.0
                                ),
                                invalid=not valid,
                            )
            rt.check_frozen(net, guard, names, frozen)
            return dict(
                values=[result[r["context_sha256"]]["utility"] for r in rows],
                choices=[result[r["context_sha256"]]["choice"] for r in rows],
                invalid=[result[r["context_sha256"]]["invalid"] for r in rows],
                context_ids=[r["context_sha256"] for r in rows],
            )

        precompute_cohort(val)
        baseline = evaluate(val)
        sanity = parent.baseline_gate(baseline["values"], ref)
        save(out / "baseline.json", dict(**baseline, sanity=sanity))
        print("BASELINE PASS", sanity, flush=True)
        for outer in range(5):
            precompute_cohort(stat.partition(train, outer)[2])

        train_index = {r["context_sha256"]: i for i, r in enumerate(train)}
        primary = np.full((len(stat.SEEDS), len(train), len(stat.ALL_ARMS)), np.nan)
        secondary = np.full((len(stat.SEEDS), 5, len(val), len(stat.ALL_ARMS)), np.nan)

        selections = []
        for outer in range(5):
            fit, select, assess = stat.partition(train, outer)
            fit_ids = {r["context_sha256"] for r in fit}
            excluded = {r["context_sha256"] for r in select + assess + val}
            assert not fit_ids & excluded
            orders = {c: np.argsort(pools[c]["fitness"]) for c in fit_ids}
            r882_sel_scores = evaluate(select)
            r882_assess_scores = evaluate(assess)
            ai_idx = [train_index[r["context_sha256"]] for r in assess]

            for si, seed in enumerate(stat.SEEDS):
                primary[si, ai_idx, 0] = r882_assess_scores["values"]
                secondary[si, outer, :, 0] = baseline["values"]
                cell_tick = time.monotonic()
                plan = stat.batch_plan(fit_ids, orders, seed)
                if len(plan) != 40:
                    raise RuntimeError("40-update plan required")

                # Precompute frozen anchor terms for the 40 minibatches of this (outer, seed)
                batch_cache = []
                with torch.no_grad():
                    for step, (stream, cids, ixs) in enumerate(plan, 1):
                        if not set(cids) <= fit_ids or set(cids) & excluded:
                            raise RuntimeError("minibatch family leakage")
                        b = len(cids)
                        k = ixs[0].shape[0]
                        progs = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["program"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        proj = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["proj16"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        tok_b = torch.stack([tokens[c] for c in cids])
                        mask_b = torch.stack([masks[c] for c in cids])
                        rev_idx = (
                            torch.arange(min(4, k), device=device)
                            .reshape(1, -1)
                            .expand(b, -1)
                        )
                        rev_raw_f = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["fitness"][ix[: min(4, k)]]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        rev_gate = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["gate"][ix[: min(4, k)]]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        fitness = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["fitness"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.float32,
                            device=device,
                        )
                        gate = torch.as_tensor(
                            np.stack(
                                [
                                    pools[c]["gate"][ix]
                                    for c, ix in zip(cids, ixs, strict=True)
                                ]
                            ),
                            dtype=torch.bool,
                            device=device,
                        )
                        scale_b = torch.full(
                            (b,), 0.4208, dtype=torch.float32, device=device
                        )
                        sos_b = torch.zeros(
                            (b, k, 2), dtype=torch.float32, device=device
                        )
                        a_out = net.adapter_10tok.anchor.predict_costs(
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
                            revealed_indices=rev_idx,
                            revealed_raw_f=rev_raw_f,
                            revealed_gate=rev_gate,
                            enable_prior_correction=True,
                            enable_struct_kernel=True,
                            enable_cross_attn=True,
                            enable_ucb=False,
                        )
                        batch_cache.append(
                            (
                                step,
                                stream,
                                cids,
                                ixs,
                                progs,
                                sos_b,
                                tok_b,
                                mask_b,
                                scale_b,
                                rev_idx,
                                rev_raw_f,
                                rev_gate,
                                a_out["total_cost"].detach().float(),
                                a_out["eff_prior"].detach().float(),
                                fitness,
                                gate,
                            )
                        )

                arm_snapshots = {}
                arm_sel_scores = {}
                arm_losses = {}
                arm_drifts = {}

                for arm in stat.WIDTH_ARMS:
                    wm = stat.WIDTH_MULTS[arm]
                    params, init_arm_state, init_arm_vec = configure_adapter_width(
                        net, initial, names, wm, seed
                    )
                    rt.check_frozen(net, guard, names, frozen)
                    optimizer = torch.optim.AdamW(params, lr=1e-4, weight_decay=1e-4)
                    snapshots, sel_scores, losses = {}, [], []

                    def take_snapshot(step_num, arm_label=arm):
                        st = {
                            n: net.state_dict()[n].detach().cpu().clone() for n in names
                        }
                        snapshots[step_num] = st
                        np.savez_compressed(
                            out / f"A{outer}-{seed}-{arm_label}-step{step_num}.npz",
                            **{n: t.numpy() for n, t in st.items()},
                        )
                        if arm_label == "W10" and step_num == 0:
                            sel_scores.append(r882_sel_scores)
                        else:
                            sel_scores.append(evaluate(select))

                    take_snapshot(0)
                    for (
                        step,
                        stream,
                        cids,
                        ixs,
                        progs,
                        sos_b,
                        tok_b,
                        mask_b,
                        scale_b,
                        rev_idx,
                        rev_raw_f,
                        rev_gate,
                        a_tot_b,
                        a_eff_b,
                        fitness,
                        gate,
                    ) in batch_cache:
                        torch.manual_seed(stream)
                        optimizer.zero_grad(set_to_none=True)
                        pred = forward_adapter_with_anchor(
                            net,
                            progs,
                            sos_b,
                            tok_b,
                            mask_b,
                            scale_b,
                            rev_idx,
                            rev_raw_f,
                            rev_gate,
                            a_tot_b,
                            a_eff_b,
                        )
                        objective = rt.loss(pred, fitness, gate)
                        if not torch.isfinite(objective):
                            raise RuntimeError("nonfinite objective")
                        objective.backward()
                        torch.nn.utils.clip_grad_norm_(
                            params, 1.0, error_if_nonfinite=True
                        )
                        optimizer.step()
                        updates += 1
                        rt.check_frozen(net, guard, names, frozen)
                        losses.append(float(objective.detach()))
                        if step in stat.STEPS:
                            take_snapshot(step)

                    drift = float(
                        torch.linalg.vector_norm(
                            torch.cat(
                                [
                                    dict(net.named_parameters())[n]
                                    .detach()
                                    .cpu()
                                    .reshape(-1)
                                    for n in names
                                ]
                            )
                            - init_arm_vec
                        )
                    )
                    if drift <= 0:
                        raise RuntimeError("no optimizer parameter drift")
                    arm_snapshots[arm] = snapshots
                    arm_sel_scores[arm] = sel_scores
                    arm_losses[arm] = losses
                    arm_drifts[arm] = drift

                # Selection on inner selector fold B locked BEFORE outer fold A assessment
                r882_sel_vals = np.asarray(r882_sel_scores["values"], dtype=np.float64)
                arm_panels = {
                    arm: np.array(
                        [r["values"] for r in arm_sel_scores[arm]], dtype=np.float64
                    ).T
                    for arm in stat.WIDTH_ARMS
                }
                decisions = {}
                for ai, arm in enumerate(stat.WIDTH_ARMS):
                    decisions[arm] = stat.choose(
                        select,
                        arm_panels[arm],
                        stat.BOOT_SEED + 100 * outer + 10 * si + ai,
                        r882_col=r882_sel_vals,
                    )
                decisions["XFIT_JOINT"] = stat.choose_joint(
                    select,
                    arm_panels,
                    r882_sel_vals,
                    stat.BOOT_SEED + 100 * outer + 10 * si + 9,
                )

                cellname = f"A{outer}-{seed}"
                lockpath = out / f"{cellname}-selection.json"
                save(lockpath, decisions)

                # Evaluate selected (arm, step) pairs on outer assessment fold A and DL-Val
                needed_pairs = {("W10", 0)}
                for arm in stat.WIDTH_ARMS:
                    st_pick = decisions[arm]["selected_step"]
                    needed_pairs.add(("W10", 0) if st_pick == 0 else (arm, st_pick))
                j_arm = decisions["XFIT_JOINT"]["selected_arm"]
                j_step = decisions["XFIT_JOINT"]["selected_step"]
                needed_pairs.add(("W10", 0) if j_step == 0 else (j_arm, j_step))

                assessment = {("W10", 0): r882_assess_scores}
                val_assessment = {("W10", 0): baseline}
                for arm_k, step_k in sorted(needed_pairs):
                    if (arm_k, step_k) in assessment:
                        continue
                    configure_adapter_width(
                        net, initial, names, stat.WIDTH_MULTS[arm_k], seed
                    )
                    with torch.no_grad():
                        for n, p in net.named_parameters():
                            if n in arm_snapshots[arm_k][step_k]:
                                p.copy_(arm_snapshots[arm_k][step_k][n].to(p.device))
                    assessment[(arm_k, step_k)] = evaluate(assess)
                    val_assessment[(arm_k, step_k)] = evaluate(val)

                configure_adapter_width(net, initial, names, 1.0, seed)
                for col, arm in enumerate(stat.WIDTH_ARMS, start=1):
                    st_pick = decisions[arm]["selected_step"]
                    key = ("W10", 0) if st_pick == 0 else (arm, st_pick)
                    primary[si, ai_idx, col] = assessment[key]["values"]
                    secondary[si, outer, :, col] = val_assessment[key]["values"]

                j_key = ("W10", 0) if j_step == 0 else (j_arm, j_step)
                primary[si, ai_idx, 4] = assessment[j_key]["values"]
                secondary[si, outer, :, 4] = val_assessment[j_key]["values"]

                cell = dict(
                    outer_fold=outer,
                    selection_fold=(outer + 1) % 5,
                    seed=seed,
                    updates_per_arm=40,
                    total_cell_updates=120,
                    fit_contexts=sorted(fit_ids),
                    selector_contexts=[r["context_sha256"] for r in select],
                    assessment_contexts=[r["context_sha256"] for r in assess],
                    losses=arm_losses,
                    adapter_l2_drift=arm_drifts,
                    snapshot_sha256={
                        arm: {
                            str(s): rt.tensor_hash(v)
                            for s, v in arm_snapshots[arm].items()
                        }
                        for arm in stat.WIDTH_ARMS
                    },
                    selection_lock_sha256=sha(lockpath),
                    decisions=decisions,
                    assessment={f"{k[0]}_step{k[1]}": v for k, v in assessment.items()},
                    frozen_sha256=frozen,
                    dope_sha256=rt.DOPE,
                    wall_seconds=time.monotonic() - cell_tick,
                )
                save(out / f"{cellname}.json", cell)
                selections.append(decisions)
                print(
                    "CELL COMPLETE",
                    cellname,
                    "updates",
                    updates,
                    "picks",
                    {
                        **{w: decisions[w]["selected_step"] for w in stat.WIDTH_ARMS},
                        "XFIT_JOINT": (
                            decisions["XFIT_JOINT"]["selected_arm"],
                            decisions["XFIT_JOINT"]["selected_step"],
                        ),
                    },
                    "seconds",
                    round(cell["wall_seconds"], 2),
                    flush=True,
                )

        train_summary = stat.summarize(train, primary)
        val_summary = stat.summarize(val, secondary.mean(axis=1))
        metrics = dict(
            wave="H156",
            status="COMPLETE",
            verdict=stat.verdict(train_summary, val_summary, selections),
            h145_train_1219_205fam=train_summary,
            dl_val_91_19fam=val_summary,
            primary=train_summary,
            secondary_dl_val=val_summary,
            width_parameter_counts=width_counts,
            selections=selections,
            optimizer_updates=updates,
            cells=len(selections),
            baseline=sanity,
            gpu_lock_held=bool(args.gpu_lock_held),
            source_commit=launch["source_commit"],
            manifest_sha256=sha(HERE / "manifest.json"),
            dope_sha256=rt.DOPE,
            frozen_sha256=frozen,
            historical_r882_train_exposure=True,
            serving_promotion=False,
            wall_seconds=time.monotonic() - started,
        )
        if updates != 1800 or len(selections) != 15:
            raise RuntimeError("incomplete wave")
        np.savez_compressed(out / "panels.npz", primary=primary, secondary=secondary)
        verify(pins)
        save(out / "metrics.json", metrics)
    finally:
        save(
            out / "audit.json",
            dict(
                **audit.receipt(),
                optimizer_updates=updates,
                wall_seconds=time.monotonic() - started,
                peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * 1024,
                threads_at_end=len(list(Path("/proc/self/task").iterdir())),
                cuda_initialized=torch.cuda.is_initialized(),
            ),
        )


if __name__ == "__main__":
    main()
