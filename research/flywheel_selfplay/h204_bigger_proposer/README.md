# H204: Bigger Proposer (`128 -> 256 -> 512`) with Research-Only `DOPE` Unfreeze

- **Wave**: `H204`
- **Owner**: `gpu_tokbench_agy` (`[TOKBENCH-AGY]`)
- **Host / GPU**: `xbabe2` (`NVIDIA GeForce RTX 3090`, `cuda:0`)
- **Authority**: `"Jepson via Grok Bot, card answer 8:22 PM MT" (2026-11-01, t11s3): "Start H205 and H204 now with those unblocks; H206 after xbabe3 disk is cleared."`

## Files

- [`PREREG.md`](PREREG.md): Preregistered scientific hypothesis, controls (`R882_CONT`, `H180_RAND`), Stage-1/Stage-2 gates, and resource rules.
- [`analysis.py`](analysis.py): Constants, family-cluster bootstrap CI (`10,000` resamples), `H178` 205-family OOF selector, Holm-Bonferroni correction, and `91/91` `r882` parity check (`1.1377941898164525`).
- [`labeler.py`](labeler.py): Bounded local LRU cache (`<= 20 GiB`), disk budget guard (`<= 93.8%` used, `>= 40 GiB` free), `JRB-01` admission check, and `8–16` candidates/context exact Rust CV cost labeler (`run_outer_test=False`).
- [`trainer.py`](trainer.py): `WiderBlockFormerProposer` (`128 -> 256 -> 512`), research-only `DOPE` unfreeze gate (`JOPE_RESEARCH_DOPE_UNFREEZE=1` + nonempty `JOPE_RESEARCH_RUN_ID`), differentiable `DOPE` -> Stage-1B `tokens_10x32` bridge, and Stage-1 training/evaluation loop.
- [`monitor.py`](monitor.py): Resource monitor (`nice 19`, thread rule (b) `<= 4` numeric threads with CUDA runtime/driver threads exempt, RSS `<= 6 GiB`, cache `<= 20 GiB`, disk `<= 93.8%`) and `launch.json` writer.
- [`test_h204.py`](test_h204.py): Unit tests covering preregistration invariants, research-only `DOPE` unfreeze vs. serving/flywheel refusal, `<= 20 GiB` LRU cache eviction, numeric thread rule (b), `WiderBlockFormerProposer` widths `128/256/512`, and `91/91` parity verification.

## Verification

```bash
pytest -q research/flywheel_selfplay/h204_bigger_proposer/test_h204.py
```
