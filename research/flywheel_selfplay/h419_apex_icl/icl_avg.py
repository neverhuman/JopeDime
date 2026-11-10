#!/usr/bin/env python3
"""H419 addendum 3: mean of the workers' model weights for one averaging round (CPU).
  icl_avg.py --out FILE in1.pt in2.pt ...   prints {"n", "dist": [relative L2 distance of each input to the mean]}"""
import json
import sys

import torch

out = sys.argv[sys.argv.index("--out") + 1]
files = [f for f in sys.argv[1:] if f not in ("--out", out)]
sds = [torch.load(f, map_location="cpu", weights_only=False)["model"] for f in files]
mean, num, den = {}, [0.0] * len(sds), 0.0
for k, v in sds[0].items():
    if v.is_floating_point():
        m = torch.stack([s[k].float() for s in sds]).mean(0)
        den += float((m.double() ** 2).sum())
        for i, s in enumerate(sds):
            num[i] += float(((s[k].float() - m).double() ** 2).sum())
        mean[k] = m.to(v.dtype)
    else:
        mean[k] = v
torch.save({"model": mean}, out)
print(json.dumps({"n": len(sds), "dist": [round((x / max(den, 1e-30)) ** 0.5, 6) for x in num]}))
