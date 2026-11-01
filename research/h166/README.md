# H166: C-Mixup (Label-Similarity Mixup) Regularized `r882` Adapter Continuation

- **Status**: `PREREGISTERED` (pre-execution; awaiting preregistration merge before single bounded CPU wave execution)
- **Owner**: `gpu_tokbench_agy`
- **Authority**: `/home/ubuntu/nudges/hwave_t0125_h166_tokbench_agy.md`
- **Claim receipt**: `/home/ubuntu/jope_plan/ns/receipts/H166_claim.txt`
- **Preregistration**: [`PREREG.md`](PREREG.md)

## Overview

`H166` tests **C-Mixup** (Yao et al. 2022 label-similarity mixup) on `r882` (`29,211`-parameter `adapter_10tok`) across the `1,219`-context / `205`-family real S3 `TRAIN` cohort (`H145`) using a `3`-seed (`166101..166103`) × `5`-fold family-grouped cross-fit harness (`FOLD_SEED = 166140`, `BOOT_SEED = 166141`):

- **`CTRL`**: Unmixed live AdamW (`lr = 1e-4`, `weight_decay = 1e-4`, `40` updates, snapshots at `(0, 10, 20, 30, 40)`).
- **`MIX02`**: Slate-level same-shape C-Mixup with $\alpha = 0.2$ ($\lambda \sim \operatorname{Beta}(0.2, 0.2)$) and Gaussian target-similarity partner sampling ($P(j \mid i) \propto \exp(-\|y_{c_i} - y_{c_j}\|_2^2 / (2\sigma_k^2))$, where $\sigma_k$ is the fit-fold median pairwise Euclidean target distance).
- **`MIX04`**: Slate-level same-shape C-Mixup with $\alpha = 0.4$ ($\lambda \sim \operatorname{Beta}(0.4, 0.4)$) and the same Gaussian target-similarity partner kernel.
- **`RANDMIX02`**: Uniform-partner mechanism control ($\alpha = 0.2$, sharing identical $\lambda \sim \operatorname{Beta}(0.2, 0.2)$ draws with `MIX02` and sampling $j \neq i$ uniformly within the minibatch).

## Verification Commands

```bash
python3 -B research/h166/test_h166.py
python3 -B research/h166/monitor.py --custody /home/ubuntu/jope_plan/evidence/H166/attempt1 --smoke
```
