"""Phase C: recover the scrambled chimera's deployed (multiquery) performance by fine-tuning
ONLY expert A's icl_attention (the CONCERNING K/V layers the scramble degraded), with a
MAX-DISTANCE regularizer that keeps those tensors far from the ORIGINAL weights.

LOCK  = every clean-scrambled tensor (A's MLP/decoder/stage-1-2, ALL of B) — stays at cos≈0.
UNLOCK= A.icl_blocks.*.icl_attention.* (+ trunk) — re-learns a good KV-head-0 for the sliced
        test path while the regularizer holds it far from origin.

Reuses finetune_e2e's live forward/eval. Experts loaded from the SCRAMBLED checkpoints via
TABPFN_CLS_CKPT / TABICL_CLS_WEIGHTS; the ORIGINAL A ckpt is the distance reference.
"""
import argparse, os, time, sys
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.expanduser("~/fable_god"))
from python.fusion import experts as E
from python.fusion.model import AttnFusion
from python.fusion.data import iter_packed_shards, load_packed_shard, group_by_shape
from python.fusion.finetune_e2e import pfn_forward, icl_forward, build_rows, evaluate, seg_mean

KV_TAGS = ("q_projection", "k_projection", "v_projection", "out_projection", "softmax_scaling")


def is_kv(name):
    return "icl_attention" in name and any(t in name for t in KV_TAGS)


def cos_to_ref(p, ref):
    a = p.flatten().double(); b = ref.flatten().double()
    return (a @ b) / (a.norm() * b.norm() + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion-ckpt", default=os.path.expanduser("~/fable_god/chimera_classification/chimera_best.pt"))
    ap.add_argument("--orig-a", default=os.path.expanduser("~/fable_god/weights/tabpfn-v3-classifier-v3_20260417_binary.ckpt"))
    ap.add_argument("--lr-kv", type=float, default=2e-5)
    ap.add_argument("--lr-fusion", type=float, default=1e-4)
    ap.add_argument("--lam-dist", type=float, default=0.5)      # max-distance regularizer weight
    ap.add_argument("--cos-cap", type=float, default=0.20)      # penalize cos-to-original above this
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--shards", type=int, default=40)
    ap.add_argument("--eval-every", type=int, default=40)
    ap.add_argument("--out", default=os.path.expanduser("~/fable_god/weights/shuffled/expert_a_recovered.ckpt"))
    args = ap.parse_args()
    print(f'[rank {os.environ.get("RANK",0)}] main entered', flush=True)
    import torch.distributed as dist
    ddp = "RANK" in os.environ
    rank = int(os.environ.get("RANK", 0)); world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if ddp:
        dist.init_process_group("nccl"); torch.cuda.set_device(local)
    dev = torch.device(f"cuda:{local}"); is_main = rank == 0

    def allreduce_grads(params):
        if not ddp:
            return
        for p in params:
            if p.grad is not None:
                dist.all_reduce(p.grad); p.grad /= world

    pfn = E.TabPFNv3ClsBinary(dev)                              # SCRAMBLED A (env)
    icl = E.TabICLCls(dev)                                      # SCRAMBLED B (env)
    fusion = AttnFusion(n_experts=2, out_dim=2, gate_meta_dim=8, conf_dim=3, d=256, rank=128,
                        d_model=512, n_layers=8, n_heads=8, p=0.0).to(dev)
    fusion.load_state_dict(torch.load(args.fusion_ckpt, map_location=dev))

    # selective unlock: only A's icl_attention; freeze everything else in both experts
    orig_ck = torch.load(args.orig_a, map_location="cpu", weights_only=False)["state_dict"]
    kv, refs = [], {}
    for name, p in pfn.model.named_parameters():
        if is_kv(name):
            p.requires_grad_(True); kv.append((name, p)); refs[name] = orig_ck[name].to(dev)
        else:
            p.requires_grad_(False)
    for p in icl.model.parameters():
        p.requires_grad_(False)
    for p in fusion.parameters():
        p.requires_grad_(True)
    kv_params = [p for _, p in kv]
    print(f"UNLOCK {len(kv)} A.icl_attention tensors ({sum(p.numel() for p in kv_params):,} params) + trunk; "
          f"everything else LOCKED at max distance", flush=True)

    raw = []
    for sh in list(iter_packed_shards("real_packed", "train"))[:args.shards]:
        try:
            raw += load_packed_shard(sh)
        except Exception:
            pass
    groups = [g for g in group_by_shape(raw, ctx_buckets=(128,)) if len(g) >= 2]   # ctx=128 -> fits memory
    rng = np.random.default_rng(0); rng.shuffle(groups)
    nval = max(4, len(groups) // 6)
    val_g, all_train = groups[:nval], groups[nval:]
    train_g = all_train[rank::world]                             # DDP: shard train groups per rank
    rng = np.random.default_rng(rank + 1)                        # per-rank sampling
    if is_main:
        print(f"{sum(len(g) for g in all_train)} train / {sum(len(g) for g in val_g)} val tasks | "
              f"{world}-GPU DDP, {len(train_g)} groups/rank", flush=True)

    def kv_cos():
        with torch.no_grad():
            return float(np.mean([float(cos_to_ref(p, refs[n])) for n, p in kv]))

    base_auc, base_win = (evaluate(pfn, icl, fusion, val_g, dev) if is_main else (0.0, 0.0))
    if is_main:
        print(f"SCRAMBLED FLOOR (val): AUC {base_auc:.4f} win% {100*base_win:.0f} | KV cos-to-orig {kv_cos():.3f}", flush=True)
    opt = torch.optim.AdamW([{"params": kv_params, "lr": args.lr_kv},
                             {"params": list(fusion.parameters()), "lr": args.lr_fusion}], weight_decay=0)
    best = base_auc; t0 = time.time()
    trainable = kv_params + list(fusion.parameters())
    for step in range(1, args.steps + 1):
        pfn.model.train(); fusion.train()
        grp = train_g[int(rng.integers(0, len(train_g)))]        # one same-shape group (this rank's shard)
        pl, ph = pfn_forward(pfn, grp, dev); il, ih = icl_forward(icl, grp, dev)
        lg, hd, gm, cf, y, seg = build_rows(pl, ph, il, ih, grp, dev)
        out = fusion(lg, hd, gm, cf); final = out["final"].float()
        ce = seg_mean(F.cross_entropy(final, y, reduction="none"), seg, int(seg.max().item()) + 1).mean()
        dreg = torch.stack([F.relu(cos_to_ref(p, refs[n]).float() - args.cos_cap) for n, p in kv]).mean()
        loss = ce + args.lam_dist * dreg
        opt.zero_grad(); loss.backward()
        allreduce_grads(trainable)                               # DDP: average grads across ranks
        torch.nn.utils.clip_grad_norm_(trainable, 1.0); opt.step()
        if is_main and step % args.eval_every == 0:
            va, vw = evaluate(pfn, icl, fusion, val_g, dev)
            best = max(best, va)
            tag = "RECOVERED" if va >= base_auc - 1e-3 else "below floor"
            print(f"step {step:4d} | loss {float(loss):.4f} ce {float(ce):.4f} dist {float(dreg):.4f} | "
                  f"val AUC {va:.4f} (floor {base_auc:.4f}) win% {100*vw:.0f} | KV cos {kv_cos():.3f} {tag} | "
                  f"{step/(time.time()-t0):.1f} it/s", flush=True)
    if is_main:
        ck = torch.load(os.environ["TABPFN_CLS_CKPT"], map_location="cpu", weights_only=False)
        sd = pfn.model.state_dict()
        ck["state_dict"] = {k: sd[k].cpu() for k in sd}
        torch.save(ck, args.out)
        torch.save(fusion.state_dict(), args.out.replace("expert_a_recovered.ckpt", "trunk_recovered.pt"))
        print(f"\nsaved recovered A -> {args.out} | final KV cos-to-orig {kv_cos():.3f} best val {best:.4f}", flush=True)
    if ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
