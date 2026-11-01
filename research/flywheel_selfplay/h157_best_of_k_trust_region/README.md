# H157: best-of-8 trust-region proposals with inner-family admission

## Hypothesis

Selecting the best of eight H142 trust-region proposals by an inner TRAIN-family bootstrap would identify an adapter continuation that improves on frozen `r882` on the 91-context `DL-Val` panel.

## Setup

- Three fresh seeds: `157101–157103`; 1,080 optimizer updates across the preregistered TR1 and eight BOK8 candidates.
- Strict inner admission used the preregistered TRAIN-family split (41 inner-select and 164 fit families). The primary readout was equal-family utility on 91 `DL-Val` contexts from 19 families; Holm correction covered BOK8 and TR1.
- CPU-only run on `xbabe2`, source `2f37146e2d415513007154c1a37c8514da54a95e` (PR #1337), one attempt. `DOPE` remained frozen at SHA-256 `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`.
- The frozen `r882` baseline matched all 91 reference contexts exactly (mean `1.1377941898164525`; reference and independent-copy maximum absolute error `0`).

## Outcome

The result is `RULE_NEUTRAL`. Both primary treatments had per-seed equal-family gain `[0, 0, 0]`, median `0.0`, and 95% bootstrap interval `[0.0, 0.0]`; Holm-adjusted `p=1.0`. For the descriptive best-of-8 inner result, the maximum candidate inner median was `0.0` on each seed (median `0.0`): all eight candidates tied, so the strict selector retained `r882` for every seed. There was no inner-fold separation for best-of-K to exploit.

Elo versus `r882=1500.0` was `1500.0` for TR1 and BOK8 (three-seed median; 95% interval `[1500.0, 1500.0]` for each). The baseline arm was also `1500.0`. No arm was admitted and no serving promotion is supported.

## Verdict and next step

Close H157 as a null result. Any successor should first change proposal generation enough to produce measurable inner-fold separation under the existing strict gate; do not rerun this unchanged candidate roster.

## Evidence and custody

Compact metrics are in [`metrics.json`](metrics.json). The 27 candidate NPZ files (3,250,077 bytes) are in noncanonical staging custody at `s3://veox-jopedime-learning-use2/staging/hwave/H157/attempt-001/`; each remote SHA-256 checksum, metadata hash, and object length was read back and matched the local file. Receipt SHA-256: `abb5b99dff062aab47aa88cc47b7477611740d8f6efe38f2a6b77033583c36d2`. The run audit records 1,080 updates, zero protected reads, no refused reads, and no CUDA initialization. Raw NPZ files are not stored in Git.
