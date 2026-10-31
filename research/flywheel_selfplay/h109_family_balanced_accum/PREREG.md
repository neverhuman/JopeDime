# H109 — family-balanced accumulation and warmup/cosine restarts

Status: CPU preparation only. No timing, outcome, GPU, FIFO ticket, or qualification is authorized by this preregistration.

## Hypothesis and estimand

H97's EMA candidate improved over CONTROL but remained below r882. H73 found family-size concentration in the training objective. H109 tests whether sampling uniformly over families and then uniformly over contexts, with four accumulated microbatches per update, improves candidate generation relative to the frozen r882 checkpoint. The primary estimand is each candidate arm's paired equal-family utility gain over r882 on a fresh family-held-out fold. CONTROL is a matched training diagnostic, not the primary reference.

## Arms and matched training budget

| Arm | Sampler | Optimizer schedule |
|---|---|---|
| CONTROL | Uniform over DL-Train contexts | Flat AdamW, LR 1e-4 |
| FBAL | Uniform family, then uniform context within family | Flat AdamW, LR 1e-4 |
| FBAL+WCR | Uniform family, then uniform context within family | 5% linear warmup, then cosine schedule with restarts, T0 = one third of the 40-update budget |

All arms receive 40 optimizer updates. Each update averages four microbatches of 16 examples (effective batch 64); all other optimizer, loss, clipping, and initialization settings are matched. FBAL+WCR warmup is two updates (`ceil(0.05 * 40)`). The restart period is represented as 40/3 updates to preserve the preregistered one-third budget exactly. Seeds 39, 40, and 41 are paired across arms. Every cell starts from an independent copy of the exact r882 checkpoint; only the 29,211 adapter parameters may train. DOPE remains frozen at `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.

## Data boundary and fold

Use only the admitted real H50 DL-Train rows. Before any outcome execution, derive five deterministic, whole-family folds (fold seed 10900); folds 1–4 are the training roster and fold 5 is the untouched within-DL-Train scoring roster. No H50 DL-Val data, outer test, sealed, holdout, or meta-test data may be opened. This fold is prospective relative to H109 but is not an untouched external cohort: r882 has already seen DL-Train. Record exact row/family roster identities and the fold-assignment hash. The CPU dry run uses synthetic in-memory rows only.

## Analysis and admission

For each of FBAL and FBAL+WCR, compare the candidate to r882 on the same held-out DL-Train families. Report paired equal-family utility mean and median, two-sided 95% family-cluster bootstrap interval for the median, one-sided 95% LCB for the mean, fraction of families harmed by more than 0.01, and family-grouped Bradley–Terry Elo median with bootstrap CI and LCB95 (r882 anchored at 1500; Elo is secondary to raw utility). Use 10,000 paired family-cluster bootstrap draws, seed 20261031. Apply Holm correction across the three registered arm-vs-r882 hypotheses (CONTROL included in family so the confirmatory family is fully represented). A candidate advances only when the strict corrected mean LCB95 exceeds 0.001, family median is nonnegative, harm fraction is at most 10%, and its Elo evidence is reported; exact ties fail. Report all arms and seeds, including failures; no best-seed or best-fold selection.

The fold and scoring protocol is preparation only. GPU work remains separately gated until H99, H100, and H105 have cleared the FIFO and the owner issues explicit authorization. Do not create a waiter or ticket. Never access the xbabe1 live loop, write S3, alter DOPE, or promote a model from this CPU preparation.

## Staging and recovery

Each cell and retry uses a unique `scratch/{arm}-seed{seed}-attemptN/` directory. Copy the read-only r882 source into that directory and verify its pinned SHA256 before training. Never chmod, mutate, or hardlink the canonical source. An existing cell-attempt scratch is a hard error; recovery must use a new attempt number and preserve the previous attempt.
