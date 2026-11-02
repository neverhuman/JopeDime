# H212 — frozen-r882 candidate-pool width sweep

**Status: INCOMPLETE; no efficacy verdict.** The preregistration is in [PREREG.md](PREREG.md). The single registered attempt ran from merged source `b5519c31df3601858ccd9b7ba83fba3a929fa7c3` on xbabe2, CPU-only, nice 19.

The frozen r882 validation regression passed exactly: 91/91 contexts, mean `1.1377941898164525`, maximum error `0.0`. The zero-update/DOPE freeze receipt also passed (`optimizer_updates=0`, `CUDA_initialized=false`, DOPE SHA unchanged). Before the first proposal call, the runner verified the first TRAIN input and stopped because the H58 compressed payload decompressed to bytes whose SHA did not match the selected TRAIN record. The compressed object SHA matched the H58 gzip manifest, while the H50 selected source table SHA matched the selected record. This is an input provenance mismatch, not an efficacy result.

No proposal was generated, no candidate was Rust-evaluated, no treatment utility was scored, and no candidate panels exist. The attempt is recorded as `INCOMPLETE`; there is no `NO_ADVANCE` efficacy verdict and no promotion. The one-attempt budget is consumed. Any further science run requires a new H-wave assignment after the H58 cache seam is repaired and revalidated.

See [metrics.json](metrics.json) for exact hashes and attempt receipts. Raw evidence remains at `/home/ubuntu/jope-data/h212/attempt-001/`; no protected, sealed, holdout, or meta-test payload was read.
