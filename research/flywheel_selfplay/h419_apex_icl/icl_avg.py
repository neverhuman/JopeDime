#!/usr/bin/env python3
"""H419 addendum 3: mean of the workers' model weights for one averaging round (CPU).
  icl_avg.py --out FILE in1.pt in2.pt ...   prints {"n", "dist": [relative L2 distance of each input to the mean]}

Addendum 5, outer momentum (SlowMo, Wang et al. 2019; the scheme DiLoCo builds on):
  icl_avg.py --out FILE --slowmo STATE.pt --beta B in1.pt in2.pt ...
With x_t the weights the group started the round from and avg the workers' mean, the returned weights are
  x_{t+1} = avg + B * (x_t - x_{t-1})
(heavy-ball momentum on the outer step; B = 0 is the plain mean). STATE.pt keeps x_t and the last outer step and is
replaced each round. The first round with a new STATE returns the plain mean. Non-floating-point entries are copied.
Extra output: "beta", "step_ratio" = |x_{t+1} - x_t| / |avg - x_t|, "step_cos" = cosine between this round's mean
displacement (avg - x_t) and the previous outer step, "disp" = |avg - x_t| / |avg|. With B = 0 the returned weights
are bit-identical to the plain mean and the extra numbers are measurements only."""
import json
import os
import sys

import torch

argv = sys.argv[1:]


def opt(name, default=None):
    if name in argv:
        i = argv.index(name)
        v = argv[i + 1]
        del argv[i:i + 2]
        return v
    return default


out = opt("--out")
state_path = opt("--slowmo")
beta = float(opt("--beta", "0"))
files = list(argv)
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
res = {"n": len(sds), "dist": [round((x / max(den, 1e-30)) ** 0.5, 6) for x in num]}
if state_path:
    st = torch.load(state_path, map_location="cpu", weights_only=False) if os.path.exists(state_path) else None
    new, step, a2, b2, dot, l2 = {}, {}, 0.0, 0.0, 0.0, 0.0
    for k, v in mean.items():
        if not v.is_floating_point() or st is None:
            new[k] = v
            if v.is_floating_point():
                step[k] = torch.zeros_like(v, dtype=torch.float32)
            continue
        prev, last = st["x"][k].float(), st["step"][k].float()
        d = v.float() - prev
        x1 = v.float() + beta * last if beta else v.float()
        step[k] = x1 - prev
        a2 += float((d.double() ** 2).sum()); b2 += float((step[k].double() ** 2).sum())
        dot += float((d.double() * last.double()).sum()); l2 += float((last.double() ** 2).sum())
        new[k] = x1.to(v.dtype) if beta else v
    torch.save({"x": {k: v for k, v in new.items() if v.is_floating_point()}, "step": step}, state_path + ".tmp")
    os.replace(state_path + ".tmp", state_path)
    mean = new
    if st is not None:
        res.update(beta=beta, step_ratio=round((b2 / max(a2, 1e-30)) ** 0.5, 4),
                   step_cos=round(dot / max((a2 * l2) ** 0.5, 1e-30), 4) if l2 > 0 else None,
                   disp=round((a2 / max(den, 1e-30)) ** 0.5, 6))
    else:
        res.update(beta=0.0, step_ratio=1.0, step_cos=None, disp=None)
torch.save({"model": mean}, out)
print(json.dumps(res))
