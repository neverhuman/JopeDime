#!/usr/bin/env python3
"""TA_Hyperion_v8 stage 1: frozen Hyperion v7 trunk -> per-row 512-d embeddings for every stored context (GPU, cached).

Pinned v7 bundle (weights sha256 05c17edd..., manifest d9bb9b8d...) loaded STRICT via the pinned v7 binding
(hyperion_v7_model.load_v7). Recipe = the v7 production recipe (per-column train q01/q99 uint8 quantisation, all six ws_list
scales on the 24x24 canvas, embed_only, per-cell mean -> train-cell per-dimension minmax -> row mean) with two documented
cuts for throughput: ONE sweep (identity order; production uses 3) and inputs capped at NH rows x PHV features (the stored
fixed-seed row subsample and the XGBoost-importance-ordered features, both fit on outer-train rows). fp16 autocast.
Inputs are outer-train rows only (no labels are read here). Output: DIR/hv7rows.npy (sum_n x 512 f16) + hv7rows_index.json.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ta_common as C

V7 = "/home/ubuntu/scratch/jopedime-dl-training-v1/build-v2/HYPERION-V7"
NH = 512


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def encode_rows(model, cfg, X, torch, batch=512):
    ws_list, P, D = tuple(cfg["ws_list"]), int(cfg["max_patch"]), int(cfg["emb_dim"])
    pad, qs, mp = int(cfg["pad_u8"]), float(cfg["quant_scale"]), int(cfg["min_patch"])
    h, w = X.shape
    lo = np.percentile(X, cfg["q_low"], axis=0).astype("float32"); hi = np.percentile(X, cfg["q_high"], axis=0).astype("float32")
    span = np.where(hi - lo < 1e-9, 1, hi - lo); u = np.clip((X - lo) / span * 255, 0, 255).astype("uint8")
    sums = torch.zeros((h, w, D), device="cuda", dtype=torch.float32); count = torch.zeros((h, w), device="cuda", dtype=torch.float32)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        for ws in ws_list:
            xp = np.pad(u, ((0, (-h) % ws), (0, (-w) % ws)), constant_values=pad)
            nr, nc = xp.shape[0] // ws, xp.shape[1] // ws
            tiles = xp.reshape(nr, ws, nc, ws).transpose(0, 2, 1, 3).reshape(nr * nc, ws, ws)
            canv = np.full((nr * nc, P, P), pad, "uint8"); canv[:, :ws, :ws] = tiles
            mask = np.zeros((P, P), "float32"); mask[:ws, :ws] = 1
            ct = torch.from_numpy(canv).cuda(); mt = torch.from_numpy(mask).cuda()
            zs = []
            for s in range(0, len(ct), batch):
                xt = ct[s:s + batch].float().div_(255).mul_(2).sub_(1).mul_(qs)
                pv = torch.full((len(xt),), ws, device="cuda", dtype=torch.long)
                zs.append(model.embed_only(xt, mt.expand(len(xt), P, P), pv - mp, pv).float())
            z = torch.cat(zs).view(nr, nc, D)
            # every cell of tile (r,c) receives z[r,c]; crop padding
            full = z.repeat_interleave(ws, 0).repeat_interleave(ws, 1)[:h, :w]
            sums += full; count += 1
    cells = sums / count[:, :, None]
    vmin = cells.amin((0, 1)); vmax = cells.amax((0, 1)); sp = vmax - vmin
    cells = torch.where(sp[None, None, :] < 1e-9, torch.zeros_like(cells), (cells - vmin) / torch.where(sp < 1e-9, torch.ones_like(sp), sp))
    return cells.mean(1).half().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    import torch
    sys.path.insert(0, os.path.join(V7, "tools"))
    import hyperion_v7_model as H
    model, cfg, rep = H.load_v7(os.path.join(V7, "weights/model.safetensors"), os.path.join(V7, "weights/manifest.json"))
    _ef = model.emb_head.forward
    def _emb_fp32(g, p_val):          # the hard-gated head index-puts into an fp32 buffer: run it outside autocast
        with torch.autocast("cuda", enabled=False):
            return _ef(g.float(), p_val)
    model.emb_head.forward = _emb_fp32
    idx = json.load(open(os.path.join(a.data, "index.json"))); X = np.load(os.path.join(a.data, "X.npy"), mmap_mode="r")
    ctxs = idx["contexts"][: a.limit] if a.limit else idx["contexts"]
    g = C.Guard(log, min_free_mib=2500)
    out, recs, t0 = [], [], time.time(); off = 0
    for i, r in enumerate(ctxs):
        ts = time.time()
        Xc = np.asarray(X[r["x_off"]: r["x_off"] + r["n"] * r["p"]], dtype=np.float32).reshape(r["n"], r["p"])[:NH, :C.PHV]
        E = encode_rows(model, cfg, Xc, torch)
        out.append(E); recs.append({"cid": r["cid"], "off": off, "n": int(E.shape[0]), "p_used": int(Xc.shape[1])}); off += E.shape[0]
        g.step(time.time() - ts)
        if i % 200 == 0:
            log("hv7rows", i, "/", len(ctxs), f"{(i + 1) / (time.time() - t0):.2f} ctx/s", "peak_mib", torch.cuda.max_memory_allocated() >> 20)
    np.save(os.path.join(a.data, "hv7rows.npy"), np.concatenate(out))
    json.dump({"contexts": recs, "recipe": {"sweeps": 1, "nh": NH, "phv": C.PHV, "ws_list": cfg["ws_list"], "autocast": "fp16",
               "weights_sha256": H.WEIGHTS_SHA256, "manifest_sha256": H.MANIFEST_SHA256, "load_report": {k: v for k, v in rep.items() if k != "unused_non_embedding_heads"}},
               "gpu": C.gpu_info(), "seconds": time.time() - t0, "guard_paused_s": g.paused_s}, open(os.path.join(a.data, "hv7rows_index.json"), "w"))
    log("DONE", len(recs), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
