#!/usr/bin/env python3
"""H419 shape stress: one training step at the largest shapes the trainer can draw (peak memory, kernel limits)."""
import sys, time, torch
from icl_model import ICL
from icl_prior import Prior
size, tokens, bmax, rowtok = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dev = torch.device("cuda"); torch.backends.cuda.matmul.allow_tf32 = True
m = ICL(size).to(dev); opt = torch.optim.AdamW(m.parameters(), lr=1e-5); p = Prior(dev, 3)
for N in (1024, 1536):
    for Fn in (3, 12, 32, 100):
        for n_ctx in (96, N - max(64, N // 8)):
            B = max(2, min(bmax, tokens // (N * (Fn + 4)), (rowtok // N) if rowtok else bmax))
            torch.cuda.reset_peak_memory_stats(); t = time.time()
            x, y, _ = p.sample(B, N, Fn)
            lg, ym, ys = m(x, y[:, :n_ctx], n_ctx)
            loss = m.loss(lg, y[:, n_ctx:], ym, ys).mean()
            opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize()
            print(f"{size} N={N} F={Fn} n_ctx={n_ctx} B={B} loss={float(loss):.3f} peak_gb={torch.cuda.max_memory_allocated()/2**30:.2f} s={time.time()-t:.2f}", flush=True)
print("STRESS_OK")
