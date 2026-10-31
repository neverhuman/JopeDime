# H77 — Long Direct-Utility Evolution Strategies (`ES` vs `ESH`, 24 Generations, Pop 64)

- **Status**: `STOPPED pre-outcome, infeasible-as-preregistered`
- **Preregistration**: [`PREREG.md`](./PREREG.md) (`SHA256: 37910c1d4afde9687cb4de2bb6456795fd264d0c839e6711425ec9517be4b7e1`, frozen; not amended post-hoc)
- **Execution Commit**: `6b828216c5f3bc7c725dbd5f7dca43d19092deb9` (PR #863 + PR #875 cap-accounting hardening)
- **Successor Wave**: [`research/flywheel_selfplay/h80_long_es/`](../h80_long_es/) (`H80`, fresh seeds `(21, 22, 23)`, `16` antithetic pairs / `pop = 32` $\times$ `16` generations, calibrated a priori from the non-outcome H77 timing cell)

---

## 1. Pre-Outcome Cap Stop Summary

Per [`PREREG.md`](./PREREG.md) §3.4 and §5, the H77 coordinator (`run_wave.py`) executed the non-outcome `1`-generation `TIMING` cell (`timing/ES/seed18`, `sigma = 0.001`, `32` antithetic pairs / `population = 64`, `164` Fold-A families / `1,001` contexts) under `~/jope_plan/qualification.lock` via the FIFO ticket queue (`research/flywheel_selfplay/qualification_lock_fifo.py`).

Immediately after the `TIMING` cell completed and released `~/jope_plan/qualification.lock`, the coordinator evaluated the preregistered `9,000.0 s` (`2.5 h`) cumulative GPU cell-wall cap and halted cleanly before starting any outcome cell:

```text
RuntimeError: H77 projected GPU cell wall (20475.7s) exceeds cap (9000.0s)
```

No post-hoc amendment was made to H77's frozen horizon (`24` generations), population (`64`), or GPU wall cap (`9,000.0 s`).

---

## 2. Non-Outcome Timing Measurements (`/home/ubuntu/jope-data/h77/h77-wave/timing/`)

| Metric | Value |
| --- | --- |
| Timing cell | `timing/ES/seed18` (`1` generation, `32` antithetic pairs / `pop = 64`, `sigma = 0.001`) |
| Fold-A workload | `164` families, `1,001` contexts (`seed = 4900`) |
| `generation_wall_seconds` (`1` gen, `64` perturbations) | `120.445342 s` (`1.8820 s` per perturbation) |
| Runner `wall_seconds` | `120.450693 s` |
| Coordinator cell wall (incl. subprocess startup, model/data load, staging) | `170.716799 s` (`~50.27 s` setup overhead) |
| FIFO `lock_wait_seconds` | `1.263351 s` |
| Preregistered outcome workload | `7` cells (`3 ES + 3 ESH + 1 NULL`) $\times$ `24` generations = `168` generations (`+ 2` gen equiv buffer) |
| `projected_total_seconds` | `20,475.708172 s` (`5.69 h`) |
| `cap_seconds` | `9,000.0 s` (`2.50 h`) |
| `fits_within_cap` | `false` |

---

## 3. Firewall & Leakage Verification (Pre-Outcome Stop)

- **`outcome_cells_started`**: `0`
- **`fold_b_scored`**: `false` (`0` Fold-B candidate or baseline reads)
- **`dl_val_reads`**: `0` (`validation_payloads_opened = false`)
- **`holdout_payloads_opened`**: `0`
- **`outer_rows_delivered`**: `0`
- **`dope_sha256_before` / `dope_sha256_after`**: `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` (unchanged)
- **Outcome / Elo**: Not applicable (`STOPPED pre-outcome, infeasible-as-preregistered`).
- **FIFO Queue Hygiene**: Confirmed zero stale `H77` tickets remain in `~/jope_plan/qualification.queue/`.
