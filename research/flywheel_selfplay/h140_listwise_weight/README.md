# H140 — Argmin-Aligned Listwise-KL Weight Sweep (`L0.1`, `L0.5`, `L1.0`, `L1.0_nofail`) vs `r882`

- **Status**: `PREREGISTERED` (0 real optimizer updates executed prior to preregistration merge)
- **Owner**: `rd_argon` (`[ARGON]`, `h140_rd_argon_t1800`)
- **Preregistration**: [`PREREG.md`](PREREG.md)
- **Fold Manifest (`fold_seed = 140140`)**: [`fold_manifest_140140.json`](fold_manifest_140140.json) (`0b4aa9b485f6eaec27f95b5242e48fcc6b47e918420c2689b7121705959de482`)
- **Pinned Executor**: `/home/ubuntu/jope-rescue-20260921/h140-listwise-weight/run_h140_cpu.py` (`fee6b6f56245d41274124d5875abe82f978a24341561c1837fdce27f52bdbf7e`)

## Summary

Tests whether increasing the listwise KL weight ($\lambda \in \{0.5, 1.0\}$) or masking gate-failing candidates out of the listwise KL softmax (`L1.0_nofail`) during a paired 40-step AdamW continuation from a SHA-pinned copy of `round-882.pt` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`) across 3 seeds (`140101, 140102, 140103`) improves 91-context validation argmin utility (`evaluation_identity = e79a7715...`) over both `L0.1` control and `r882` (`1.1377942`).
