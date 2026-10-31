# H61 adapter checkpoint-selection and strict-margin admission (S0–S3)

## Hypothesis

The plateau comes from the selection step. Replacing the live native top-1 tie-accepting rule (S0) with strict unrev-masked top-1 (S1), softmin expected utility with inner-CV tau/margin (S2), or a family-bootstrap LCB rule (S3) would admit better adapters from the same candidate stream.

## Setup

Each seed (12/13/14) shares one paired update stream from an r882 copy: 20 rounds × k=2 AdamW steps, batch 16, lr 1e-4, wd 1e-4, identical to the live step. Each rule makes its accept/reject decision on identical candidate weights. Fold A has 164 families / 1,001 contexts; fold B has 41 families / 218 contexts, scored in unrev mode (4..P−1). Source `db8faaa1` (PR #816), run `~/jope-data/h61/h61-wave`. DOPE `c51a56d0` was checked every round (0 mismatches), trainable = 29,211, holdout/outer = 0, DL-Val not opened.

## Outcome

| Arm | Accepted rounds (3 seeds × 20) | Best per-round selection gain (s12/s13/s14) | Fold-B gain vs r882 | Median Elo | Admitted |
|---|---:|---|---:|---:|---|
| S0 native, ties accepted | 0/60 | −0.0015 / −0.0019 / −0.0019 | 0.0000 (exact tie, 0/41 non-zero families) | 1500.0 | No |
| S1 strict unrev top-1 | 0/60 | −0.0006 / −0.0022 / −0.0021 | 0.0000 | 1500.0 | No |
| S2 softmin + inner CV | 0/60 | −0.0020 / −0.0026 / −0.0021 | 0.0000 | 1500.0 | No |
| S3 family-bootstrap LCB | 0/60 | −0.0024 / −0.0046 / −0.0044 | 0.0000 | 1500.0 | No |

Median per-round selection gain was −0.02 to −0.04 for every arm. The final selected adapter equals the restored r882 copy (sha `b9ba619e…`) in all 12 cells.

## Verdict

H61 is a degenerate null, and an informative one. Across 240 accept/reject decisions, no candidate from the live update step beat the r882 incumbent under any rule, including the permissive S0. Changing the selection rule cannot move the plateau when the candidate pool contains no improving candidates. This revises H55's verdict: the bottleneck is the candidate generator (2 AdamW steps at lr 1e-4 from r882 always land in a worse basin), not selection noise or tie acceptance. The Elo of 1500 is an exact tie by construction, not evidence of parity.

Red-team: metric direction checked (`utility = -fitness`; gain = candidate − incumbent; negative = worse). No leakage (fold B not scored during training, families disjoint). No tie acceptance occurred because nothing tied or won. The H61 run used the PR #816 immediate-retry lock helper, which starved H57 by about 25 min. The FIFO fix (#832) applies to later runs, and outcomes are unaffected.

## Next

Stop spending GPU on selection-rule variants. Put effort into candidate generation: H76 (SAM/ASAM vs K8 steps, running), and H77 (long ES from H57's positive-mean signal, with CLIP-loss-free direct utility).
