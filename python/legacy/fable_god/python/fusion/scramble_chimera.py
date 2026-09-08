"""Driver: produce a MAX-DISTANCE scramble of BOTH classification experts (move EVERY
weight matrix) and emit a lock/unlock manifest for the recovery trainer.

Ops (see weight_symmetry_cls.py):
  A: icl-MLP, stage-1/2 MLP, decoder-head, global-E   -> CLEAN (behavior-preserving; LOCK)
     icl-attention head-perm + O/V                     -> CONCERNING (degrades sliced-KV; UNLOCK)
  B: MLP, head, O/V                                     -> CLEAN (LOCK)
     global channel-perm                                -> moves decoder/norms/embeds but not yet
                                                            behavior-exact -> UNLOCK (train to recover)

Saves scrambled A .ckpt + B .safetensors, and manifest.json listing which expert tensors are
LOCKED (verified clean, stay at max distance) vs UNLOCKED (trained on 8 GPUs to recover).
"""
import os, sys, json, argparse, torch
from safetensors.torch import load_file, save_file
sys.path.insert(0, os.path.expanduser("~/fable_god"))
from python.fusion import weight_symmetry_cls as W


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-ckpt", default=os.path.expanduser("~/fable_god/weights/tabpfn-v3-classifier-v3_20260417_binary.ckpt"))
    ap.add_argument("--b-weights", default=os.path.expanduser("~/fable_god/weights/tabicl-classifier-v2-20260212.safetensors"))
    ap.add_argument("--out", default=os.path.expanduser("~/fable_god/weights/shuffled"))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ov-mode", default="orthogonal")
    ap.add_argument("--b-global", action="store_true", help="also apply B's global perm (moves decoder/norms; needs training to recover)")
    ap.add_argument("--a-global", action="store_true", help="also apply A's global perm (moves embeds/norms; permutes hidden -> needs trunk retrain)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True); s = args.seed

    # ---- Expert A: CLEAN (icl-mlp, stage12-mlp, decoder, global-E) then CONCERNING (icl heads + O/V) ----
    ck = torch.load(args.a_ckpt, map_location="cpu", weights_only=False)
    origA = {k: t.clone() for k, t in ck["state_dict"].items()}
    A = W.a_permute_icl_mlp(origA, s)
    A = W.a_permute_stage12_mlp(A, s + 1)
    A = W.a_permute_decoder_heads(A, s + 2)
    idxA = torch.arange(512)
    if args.a_global:
        A, idxA = W.a_global_e_permute(A, s + 3)                 # CLEAN logits; permutes hidden by idxA
    A_clean = {k for k in origA if not torch.equal(origA[k], A[k])}   # everything clean so far -> LOCK
    A = W.a_permute_icl_heads(A, s + 4)                          # CONCERNING
    A = W.a_ov_transform_icl(A, s + 5, mode=args.ov_mode)        # CONCERNING
    A_unlock = sorted({k for k in origA if not torch.equal(origA[k], A[k])} - A_clean)  # only K/V moved by concerning
    ck["state_dict"] = A
    torch.save(ck, os.path.join(args.out, "expert_a_shuffled.ckpt"))

    # ---- Expert B: CLEAN (mlp, heads, O/V). Optional GLOBAL (moves decoder/norms/embeds; not yet
    #      behavior-exact -> UNLOCK + train). Default off to keep B behavior-preserving. ----
    origB = load_file(args.b_weights)
    B = W.b_permute_mlp(origB, s)
    B = W.b_permute_heads(B, s + 1)
    B = W.b_ov_transform(B, s + 2, mode=args.ov_mode)
    B_clean = {k for k in origB if not torch.equal(origB[k], B[k])}
    idxB = torch.arange(512)
    B_unlock = []
    if args.b_global:
        B, idxB = W.b_global_permute(B, s + 3)
        B_unlock = sorted({k for k in origB if not torch.equal(origB[k], B[k])} - B_clean)
    save_file({k: t.contiguous() for k, t in B.items()},
              os.path.join(args.out, "expert_b_shuffled.safetensors"), metadata={"format": "pt"})

    manifest = dict(idxA=idxA.tolist(), idxB=idxB.tolist(),
                    A_locked=sorted(A_clean), A_unlock=A_unlock,
                    B_locked=sorted(B_clean), B_unlock=B_unlock)
    json.dump(manifest, open(os.path.join(args.out, "manifest.json"), "w"))

    # ---- report ----
    def rep(tag, o, sd):
        moved = [k for k in o if o[k].is_floating_point() and not torch.equal(o[k], sd[k])]
        d = W.dist_metrics(o, sd, moved)
        unmoved = sum(o[k].numel() for k in o if o[k].is_floating_point() and torch.equal(o[k], sd[k]))
        print(f"{tag}: {len(moved)} tensors moved ({d['n']:,} params) cos={d['cos']:.3f} relL2={d['rel_l2']:.2f} | {unmoved:,} params UNMOVED")
    rep("Expert A (all ops)", origA, A)
    rep("Expert B (all ops)", origB, B)
    print(f"LOCK (clean): A {len(A_clean)} + B {len(B_clean)} tensors | UNLOCK (train): A {len(A_unlock)} + B {len(B_unlock)} tensors")
    print(f"saved -> {args.out}/{{expert_a_shuffled.ckpt, expert_b_shuffled.safetensors, manifest.json}}")


if __name__ == "__main__":
    main()
