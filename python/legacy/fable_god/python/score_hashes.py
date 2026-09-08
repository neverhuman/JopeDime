"""Score the trained fusion + frozen experts on the 230 hyperion binary TEST
datasets (raw data at ~/quant/binary/<hash>/), emit per-dataset AUC JSONs for the
Elo bridge. Also emits my-protocol expert columns (tabpfn/tabicl_MINE) so the Elo
run can cross-check protocol drift vs the pack's precomputed expert columns.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from python.fusion import experts as E
from python.fusion.model import AttnFusion
from python.fusion.data import load_dataset, gate_meta_vector

QBIN = os.path.expanduser("~/quant/binary")


def _auc(p, y):
    y = np.asarray(y).astype(int)
    return roc_auc_score(y, p) if len(np.unique(y)) > 1 else 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.expanduser("~/fable_god/cache/fusion/FUSION_BEST_0p9119.pt"))
    ap.add_argument("--out-dir", default=os.path.expanduser("~/fable_god/cache/fusion"))
    args = ap.parse_args()
    dev = torch.device("cuda")

    idx = pd.read_parquet("/home/ubuntu/hyperion/pack/router_bin_index.parquet")
    sel = idx[(idx.task == "bin") & (idx.is_real) & (idx.has_router) & (idx.split == "test")]
    hashes = sel.hash.astype(str).to_numpy()

    pfn = E.TabPFNv3ClsBinary(dev); icl = E.TabICLCls(dev)
    fusion = AttnFusion(n_experts=2, out_dim=2, gate_meta_dim=8, conf_dim=3, p=0.0).to(dev)
    fusion.load_state_dict(torch.load(args.ckpt, map_location=dev)); fusion.eval()

    fused, o_pfn, o_icl, o_orac = {}, {}, {}, {}
    ok = skip = 0
    for h in hashes:
        d = os.path.join(QBIN, h)
        try:
            ds = load_dataset(d)
            op, oi = pfn.infer(ds), icl.infer(ds)
            logits = torch.stack([op.logits, oi.logits], 1).to(dev)         # (M,2,2)
            hidden = torch.stack([op.hidden, oi.hidden], 1).to(dev)         # (M,2,512)
            conf = torch.stack([op.conf, oi.conf], 1).to(dev)              # (M,2,3)
            gm = torch.from_numpy(gate_meta_vector(ds)).to(dev)
            with torch.no_grad():
                out = fusion(logits, hidden, gm.unsqueeze(0).expand(logits.shape[0], -1), conf)
            y = ds.yte
            pf = torch.softmax(logits, -1)[:, 0, 1].cpu().numpy()
            pi = torch.softmax(logits, -1)[:, 1, 1].cpu().numpy()
            ff = torch.softmax(out["final"], -1)[:, 1].cpu().numpy()
            ap_, ai_ = _auc(pf, y), _auc(pi, y)
            fused[h] = _auc(ff, y); o_pfn[h] = ap_; o_icl[h] = ai_
            o_orac[h] = max(ap_, ai_)
            ok += 1
        except Exception as e:
            skip += 1
            if skip <= 5:
                print(f"skip {h}: {type(e).__name__}: {e}", flush=True)
        if (ok + skip) % 40 == 0:
            print(f"{ok+skip}/{len(hashes)} ({ok} ok)", flush=True)

    json.dump(fused, open(os.path.join(args.out_dir, "tabfusion_auc.json"), "w"))
    json.dump({"tabpfn_MINE": o_pfn, "tabicl_MINE": o_icl, "TabFUSION_ORACLE_MINE": o_orac},
              open(os.path.join(args.out_dir, "tabfusion_extra_auc.json"), "w"))
    fv = np.array(list(fused.values()))
    pv = np.array(list(o_pfn.values())); iv = np.array(list(o_icl.values()))
    ov = np.array(list(o_orac.values()))
    print(f"\nscored {ok}/{len(hashes)} (skip {skip}) | single-split mean AUC: "
          f"fused {fv.mean():.4f}  tabpfn {pv.mean():.4f}  tabicl {iv.mean():.4f}  "
          f"oracle {ov.mean():.4f}", flush=True)
    win = (fv >= np.maximum(pv, iv) - 1e-9).mean()
    print(f"fusion >= best-expert on {100*win:.0f}% of the 230 test datasets", flush=True)


if __name__ == "__main__":
    main()
