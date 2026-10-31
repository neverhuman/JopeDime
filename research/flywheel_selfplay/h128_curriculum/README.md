# H128 — Loss-Ordered Self-Paced Curriculum (`CURRIC` & `ANTI` vs `CONTROL`)

**NO_ADMISSION** (`candidate_breakthrough = false`). CPU-only offline `r882` fine-tuning under [PREREG.md](PREREG.md) (`sha256 = 7cecc7fd36978dd27caddde708ef926f8679f3cd67b8a23ef8366fdf792f4dac`), preregistered in [PR #1116](https://github.com/neverhuman/JopeDime/pull/1116) (`ffea7c80e5`) before curriculum ranking or execution. Status: `COMPLETE`; actual optimizer updates: `360` (`9` cells $\times$ `40` AdamW updates). No serving promotion.

## Hypothesis & Setup

- **Hypothesis**: Every recent `40`-step AdamW fine-tune from `r882` has declined on fresh family-held-out `DL-Train` assessment (`H125` trained `CONTROL` `-0.063180`, `H126` trained `CONTROL` `-0.045027` vs unchanged `r882`). `H128` tested whether an **easy-to-hard loss-ordered self-paced curriculum (`CURRIC`)** — sampling at update $u \in \{0, \dots, 39\}$ only fit-fold contexts in the easiest $q_u = 0.5 + 0.5 u / 39 \in [0.5, 1.0]$ fraction of frozen `r882` baseline per-context losses — prevents high-loss contexts from destabilizing early updates, alongside a **hard-to-easy directional/placebo control (`ANTI`)** and uniform sampling (`CONTROL`).
- **Partition & Pairing**: Fresh 5-fold family-grouped `DL-Train` split (`fold_seed = 128076`, `fold_manifest_sha256 = 010e5c3857acf7b7e62ed2da10bc5f332bf7e65fcd62f82a7d80ea995e5b7e8c`):
  - **Assessment (`fold 0`)**: `41` held-out families (`244` contexts), strictly excluded from curriculum ranking and training.
  - **Fit (`folds 1..4`)**: `164` training families (`975` contexts).
  - **Seeds**: `76, 77, 78`, paired across `CONTROL`, `CURRIC`, and `ANTI` (`3 x 3 = 9` cells). Every cell starts from its own fresh SHA-verified copy of `r882` (`0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64`); `DOPE` (`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`) is frozen and verified before and after every cell; only the `29,211`-parameter `adapter_10tok` trains.

## Pre-Update Gates

1. **Unchanged-`r882` Regression-First Gate (`PASS`, `0` updates)**:
   - Two independent fresh `r882` copies achieved exact-zero (`0.0`) max absolute prediction and utility difference across all `244` `fold 0` assessment pools (`r882_fold0_unrev_family_utility = 0.950413909355`, `r882_fold0_unrev_context_utility = 0.936791994174`, `r882_fold0_native_family_utility = 0.999963131394`) and exact-zero (`0.0`) `CONTROL` loss and gradient difference on the first `seed = 76` minibatch.
   - Receipt: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H128/h128_cpu_run_20261031T1735Z/regression_receipt.json` (`VersionId = HgNsGca0hOL8pXxKhIs4CzlQglyLr_DL`).
2. **Frozen Fit-Fold Baseline Loss Ranking (`PASS`, `0` updates)**:
   - Evaluated unchanged `r882` across all `975` fit-fold (`folds 1..4`) contexts under `torch.no_grad()` (`assessment_contexts_touched = 0`, `adapter_mutated = false`, `sorted_fit_by_loss_sha256 = 434b109f3d7639084c75bb2daec67373a57b47ba48fd0ff043eea95411401d92`):
     - Quantiles across `975` fit contexts: `min = 0.629900`, `q10 = 0.831451`, `q25 = 0.941659`, `median = 1.534114`, `mean = 4.216280`, `q75 = 4.276989`, `q90 = 9.290837`, `max = 53.568588`.
     - Eligible pool size rises from $K_0 = 488$ ($q_0 = 0.50$) to $K_{39} = 975$ ($q_{39} = 1.00$).
   - Receipt: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H128/h128_cpu_run_20261031T1735Z/curriculum_ranking_manifest.json` (`VersionId = XV6RsMqY6jycYK5QrYlPpWgho2I.0P0l`).

## Outcome & Metrics Summary

Unchanged `r882` `fold 0` assessment utility: **`0.950414`** (`41` families) / **`0.936792`** (`244` contexts). Full per-cell and bootstrap metrics are in [metrics.json](metrics.json).

### 1. Paired Comparison vs `CONTROL` (`fold 0`, `41` Families)

| Arm | Role | Seed `fold 0` family utilities (`76, 77, 78`) | Median seed gain vs `CONTROL` | Family-bootstrap `LCB95` (`95% CI`) | Pooled paired Elo vs `CONTROL=1500` (`95% CI`) | BT Elo (`CONTROL=1500`) | Max seed family harm ($\Delta < -0.01$) | Holm ($\alpha=0.05$) | Verdict |
|---|---|---|---:|---|---|---:|---:|---|---|
| `CONTROL` | Baseline | `0.897008, 0.898376, 0.899375` | `0.000000` | n/a | `1500.0` (ref) | `1500.00` | n/a | ref | BASELINE |
| `CURRIC` | Easy-to-hard treatment | `0.890335, 0.897510, 0.890048` | `-0.006673` | `-0.015152` (`[-0.017537, +0.004499]`) | `1451.67` (`[1379.59, 1519.79]`) | `1496.28` | `19.51%` (`8/41`) | fail (`p=0.8517 > 0.050`) | **NO_ADMISSION** |
| `ANTI` | Hard-to-easy placebo | `0.924489, 0.930834, 0.931746` | `+0.032371` | `+0.009963` (`[+0.006986, +0.060543]`) | `1536.86` (`[1480.21, 1595.55]`) | `1507.68` | `7.32%` (`3/41`) | pass vs `CONTROL` (`p=0.0026 <= 0.025`), fail vs `r882` | **PLACEBO_ONLY / NO_ADMISSION** |

### 2. Paired Comparison vs Unchanged `r882` (`fold 0`, `41` Families)

| Arm | Median seed gain vs `r882` | Family-bootstrap `LCB95` (`95% CI`) | Pooled paired Elo vs `r882=1500` (`95% CI`) | BT Elo (`r882=1500`) | Max seed family harm ($\Delta < -0.01$) | Adapter L2 drift from `r882` (`76, 77, 78`) |
|---|---:|---|---|---:|---:|---|
| `CONTROL` | `-0.052038` | `-0.082428` (`[-0.088564, -0.020637]`) | `1373.19` (`[1262.69, 1466.00]`) | `1474.43` | `58.54%` (`24/41`) | `0.4401, 0.4404, 0.4410` |
| `CURRIC` | `-0.060079` | `-0.086886` (`[-0.093442, -0.027525]`) | `1350.11` (`[1235.24, 1445.90]`) | `1470.70` | `58.54%` (`24/41`) | `0.4647, 0.4643, 0.4676` |
| `ANTI` | `-0.019580` | `-0.046073` (`[-0.050877, +0.008600]`) | `1369.96` (`[1271.28, 1454.55]`) | `1482.11` | `48.78%` (`20/41`) | `0.4092, 0.4202, 0.3926` |

### Key Mechanistic Findings

1. **Easy-to-hard curriculum (`CURRIC`) is decisively refuted (`CURRIC - ANTI` median `-0.034154`, `95% CI [-0.068501, -0.009787]`)**:
   - Sampling low-loss contexts (`first5_sampled_frozen_loss = 1.01–1.05` in `CURRIC`) where `r882` already ranks top candidates accurately causes AdamW to over-fit smooth-L1 score margins on already-solved contexts, **increasing** parameter L2 drift from `r882` (`0.4643–0.4676` vs `0.4401–0.4410` in `CONTROL`) and worsening `fold 0` utility (`-0.006673` vs `CONTROL`, `-0.060079` vs `r882`).
2. **Hard-to-easy sampling (`ANTI`) cuts `40`-step AdamW regression vs `r882` by more than half (`-0.019580` vs `-0.052038` in `CONTROL`)**:
   - Focusing early updates on the highest-loss fit-fold contexts (`first5_sampled_frozen_loss = 8.13–9.18` in `ANTI`) produces **smaller** L2 parameter drift from `r882` (`0.3926–0.4202`) and beats `CONTROL` across all 3 seeds (`+0.027481, +0.032458, +0.032371`, `LCB95 = +0.009963`, `3/41` = `7.32%` harmed families vs `CONTROL`, pooled paired Elo `1536.86`).
   - However, `ANTI` was preregistered as the directional/placebo control and still declines against unchanged `r882` (`median_seed_gain = -0.019580`, `95% CI [-0.050877, +0.008600]`, `48.78%` harmed families vs `r882`), so `candidate_breakthrough = false` and overall verdict is `NO_ADMISSION`.

## Resource & Governance Audit

- **CPU-only execution (`xbabe2`)**: `nice 15`, `4` Torch/math threads (`OMP/MKL/NUMEXPR=4, OPENBLAS=1`), `0.0` GPU seconds, no `qualification.lock`, `0` reads of `DL-Val`, outer-test, sealed meta-test, or live-loop state.
- **Observed bounds**: peak RSS `4,850.89 MiB` (`5.087 GB <= 6 GB`), `MemAvailable` `66.38–72.59 GiB` (`>= 30 GB`), `load1` `37.62–56.93` (`<= 86.7`), disk `91.43%–91.92%` (`<= 94%`).
- **Create-only S3 receipts (`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H128/h128_cpu_run_20261031T1735Z/`)**:
  - `h128_summary_v2.json`: `VersionId = WU2whfsEHRO8IIwWqy4Z7RIdql8maSAd`, `sha256 = faaf55c6850bb84ddb92adb0293118b941909fb2e2450a3973ed791c5d1defdd`
  - `regression_receipt.json`: `VersionId = HgNsGca0hOL8pXxKhIs4CzlQglyLr_DL`, `sha256 = 68c8cc7750583e05419fe3c858f71a2bc6293da31fba9b80a1a8cf5c8347838b`
  - `curriculum_ranking_manifest.json`: `VersionId = XV6RsMqY6jycYK5QrYlPpWgho2I.0P0l`, `sha256 = c89d492eeac65d550b865b259294a61fd23f5019f3eb706829d81582d9d6fc48`
  - `fold_manifest_128076.json`: `VersionId = 24gh0dt0l0BcCDTjlByu0DbDn4pE2IPr`, `sha256 = 010e5c3857acf7b7e62ed2da10bc5f332bf7e65fcd62f82a7d80ea995e5b7e8c`
  - `run_h128_cpu.py`: `VersionId = 6vHP3u3GKo4DQkPHr8CmJRFdApLO1G9B`, `sha256 = e116de3eef5eb2af8db9bc30647f415380b2ff74007322a5e1e4d13f5b010269`
  - Per-cell receipts `cells/{CONTROL,CURRIC,ANTI}_seed{76,77,78}.json`: all `9` uploaded create-only with verified `VersionId`s in [metrics.json](metrics.json).

## Verdict & Next Step

- **Verdict**: `NO_ADMISSION` (`CURRIC` rejected; `r882` remains the unpromoted reference).
- **Next step**: Do not use easy-to-hard loss curricula (`CURRIC`), which amplify parameter drift on already-solved contexts; if selective context sampling is revisited, pair hard-context focus (`ANTI`) with explicit `r882` L2-anchoring or early stopping so updates cannot drift `0.40` L2 units away from `r882`.
