# H82: fresh-seed SAM/ASAM versus more AdamW steps

H82 tests whether flatness-aware updates or more AdamW steps improve the exposed
r882 adapter on the reused development panel. [PREREG.md](PREREG.md) was committed
before execution. H76 remains incomplete; its outcome cells were not read or
reused. This is a conditional cached-pool dev experiment, not an unseen-family
or serving qualification.

All four arms start from a verified r882 copy and train only the 29,211 non-anchor
adapter parameters, with DOPE c51a56d0 frozen. Seeds 24–26, 20 rounds, batch 16,
constant learning rate 1e-4, weight decay 1e-4 and the H59 CONTROL loss are fixed. CONTROL/SAM/ASAM use
40 AdamW updates (SAM rho .05, ASAM rho .5); K8 uses 160. SAM/ASAM use 80
backward passes, CONTROL 40 and K8 160. Sampling and update code
are reused directly from H76. The roster is 1,219 DL-Train contexts/205 families,
with 91 terminal dev contexts/19 disjoint families, unrevealed-only selection.
Each CONTROL terminal cell additionally scores its original r882 adapter and
restores the candidate afterward.

## Primary readout

No arm clears the admission gate. ASAM has the highest median Elo against
fresh CONTROL, but its primary gain interval includes negative values. SAM’s
one-sided gain lower bound is zero; K8’s entire gain interval is negative.

Per-seed gains compare equal-family val_reg against same-seed fresh CONTROL.
The primary is their median, with 10,000 paired family-cluster bootstrap draws
retaining all three seed vectors. CI95 is two-sided; LCB95 is the one-sided 5th
percentile. Admission requires LCB95 strictly > 0.001546 and median >= 0; ties fail.
Three contrasts are nominal; the JSON also retains Bonferroni LCB diagnostics.

| Arm | Median gain | CI95 | LCB95 | Admitted |
|---|---:|---|---:|---|
| ASAM | 0.033488 | [-0.062702, 0.132609] | -0.046023 | False |
| CONTROL | 0.000000 | [0.000000, 0.000000] | 0.000000 | False |
| K8 | -0.106780 | [-0.194176, -0.028665] | -0.176202 | False |
| SAM | 0.003377 | [0.000000, 0.007452] | 0.000000 | False |

## Family Elo

Each comparison separately anchors its reference at 1500. Per-seed family wins
count 1, ties 0.5; the reported median is across three seed Elo ratings. CI95
resamples whole families with all seeds paired. Endpoint infinities are literal
limiting ratings, not finite estimates. H76-style pooled Elo remains separately
labeled in [metrics.json](metrics.json).

| Arm | Median Elo vs CONTROL | CI95 | Median Elo vs r882 | CI95 |
|---|---:|---|---:|---|
| ASAM | 1634.3 | [1500.0, 1790.8] | 1518.3 | [1386.4, 1656.0] |
| CONTROL | 1500.0 | [1500.0, 1500.0] | 1344.0 | [1209.2, 1463.3] |
| K8 | 1321.1 | [1128.2, 1481.7] | 1321.1 | [1128.2, 1481.7] |
| SAM | 1536.7 | [1500.0, 1555.3] | 1365.7 | [1241.5, 1481.7] |

| Arm | Seed | Train reg | Val reg | Train−val gap | Val gain vs CONTROL | Elo vs CONTROL | Elo vs r882 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ASAM | 24 | 1.142865 | 1.127586 | 0.015279 | 0.033488 | 1593.6 | 1518.3 |
| ASAM | 25 | 1.145940 | 1.103144 | 0.042796 | 0.016199 | 1634.3 | 1500.0 |
| ASAM | 26 | 1.145017 | 1.129163 | 0.015855 | 0.043718 | 1634.3 | 1555.3 |
| CONTROL | 24 | 1.114472 | 1.094098 | 0.020374 | 0.000000 | 1500.0 | 1365.7 |
| CONTROL | 25 | 1.114909 | 1.086945 | 0.027965 | 0.000000 | 1500.0 | 1344.0 |
| CONTROL | 26 | 1.105959 | 1.085445 | 0.020514 | 0.000000 | 1500.0 | 1296.7 |
| K8 | 24 | 1.029281 | 0.987318 | 0.041962 | -0.106780 | 1321.1 | 1321.1 |
| K8 | 25 | 1.020844 | 0.992637 | 0.028208 | -0.094308 | 1406.4 | 1365.7 |
| K8 | 26 | 1.008339 | 0.965287 | 0.043052 | -0.120158 | 1321.1 | 1321.1 |
| SAM | 24 | 1.111743 | 1.097475 | 0.014269 | 0.003377 | 1536.7 | 1386.4 |
| SAM | 25 | 1.117572 | 1.087166 | 0.030406 | 0.000222 | 1518.3 | 1365.7 |
| SAM | 26 | 1.108085 | 1.095037 | 0.013048 | 0.009592 | 1536.7 | 1344.0 |

## Execution, verdict and next step

All 12 fixed outcome cells completed. Charged cell wall was 976.62s of 2700s, across 15 attempts including 2 unsuccessful attempts (load voids plus one fatal setup fault). Admitted arms: none. No serving promotion is performed. These settings did not clear the preregistered gate; retain r882 and investigate a different prospectively registered intervention.

CPU [PR #897](https://github.com/neverhuman/JopeDime/pull/897) merged before launch.
The [operator repair](ops/README.md), merged in
[PR #960](https://github.com/neverhuman/JopeDime/pull/960), retained numerical
worker `037f914a59aabd34279000c40288315dc00500d8` and recorded launcher
`614e15c09b25fd91c37eb125c06d3149616bda64` separately.
Nice 19, OMP/MKL 4, two load <= 85 checks 30 s apart,
FIFO H82 tickets behind older W1/H80, per-attempt release, RSS < 6 GiB, RAM >= 16 GiB
and disk guards were enforced. Pre-outcome host-load aborts could retry at most
twice per cell. SAM seed 24 attempt 2 then failed before model construction because
attempt 1 left a read-only staging copy. Its fatal diagnosis and wall charge stay
in the ledger. A separately reviewed operator launcher preserved/hash-verified
that duplicate input, cleaned it, and used only final physical attempt 3; there
was no fourth attempt. Its reviewed revision/hash and conservatively charged
operator downtime are recorded separately from the unchanged worker revision.
The original 2700 s cell cap and 3 h wait allowance stayed in force. Void weights
were never compared, all wall remained charged, and no computed outcome was
retried. Successful receipts verify changed adapters, exactly 29,211 trainable parameters, frozen
DOPE before/after, fixed checkpoint/input pins and a common source identity.
No sealed/holdout/outer payload, live learner or xbabe0/3 work was involved.
No other job was paused or killed.

Raw receipts, phases, snapshots and budget/status logs are under
`~/jope-data/h82/h82-wave/`. CPU proof is retained under
`~/jope_plan/evidence/H82/cpu/`; closeout artifacts were generated from the
complete machine receipt without editing outcome values.
