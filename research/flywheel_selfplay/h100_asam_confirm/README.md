# H100: ASAM confirmation and drift control

**COMPLETE — NO_ADMIT.** All twelve fixed outcome cells finished on 2026-10-31
at 18:47Z. No candidate passed the preregistered utility-margin, Holm and harm
gates. Retain unchanged r882. [metrics.json](metrics.json) contains the full
precision native metrics and provenance; [PREREG.md](PREREG.md) is unchanged.

## Hypothesis and fixed setup

Test whether the earlier ASAM signal survives fresh optimization seeds and differs
from reducing adapter drift. Seeds **39–41** pair four arms: CONTROL (LR 1e-4),
ASAM05 (rho 0.5), ASAM10 (rho 1.0), and DRIFT_MATCH (CONTROL LR
**8.465252628604803e-5**). The [historical H82 calibration](calibration.json)
fixed this LR before H100: L2 ratio 0.8465252628604802, without fresh-seed tuning.

Each cell started from an offline r882 copy and trained exactly **29,211 adapter
parameters**, 20 rounds × 2 updates, batch 16 / slate 192, AdamW weight decay 1e-4 and
gradient clip 1. CONTROL/DRIFT_MATCH use 40 backwards; ASAM uses 80 with matched
stochastic state across its two passes. DOPE remained
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` before
and after all thirteen training cells, including the discarded timing cell.

Real-S3 H50 inputs comprise **1,219 DL-Train contexts / 205 families** and the
reused **91 DL-Val contexts / 19 disjoint families**. Terminal scoring uses
unrevealed candidates 4..P-1. These are conditional results on an exposed dev
panel. Binary is NOT_RUN; there is no serving promotion or new qualification.
The earlier [CPU dry run](cpu_dry_run.json) remains preparation evidence.

## Outcome

Elo is family-grouped, with each comparison reference anchored at 1500. Intervals
are the native **95% common-family bootstrap** intervals for the median across
three seeds. Utility is clipped `-fitness` for passing candidates (gate failures
map to -2), so larger utility wins. Exact zero gain gives a half-win Elo tie.

| Arm | Elo vs CONTROL [CI95] | Elo vs r882 [CI95] | Median utility gain vs CONTROL | LCB95 | Harm vs r882 |
| --- | --- | --- | --- | --- | --- |
| CONTROL | 1500.00 [1500.00, 1500.00] | 1321.14 [1172.18, 1444.68] | +0.000000 | +0.000000 | 7/19 (36.8%) |
| ASAM05 | 1634.32 [1500.00, 1827.82] | 1518.30 [1386.40, 1634.32] | +0.017743 | -0.078750 | 4/19 (21.1%) |
| ASAM10 | 1655.99 [1518.30, 1871.77] | 1536.71 [1406.37, 1678.86] | +0.018566 | -0.079279 | 4/19 (21.1%) |
| DRIFT_MATCH | 1536.71 [1481.70, 1613.60] | 1406.37 [1270.39, 1518.30] | +0.023804 | +0.000086 | 5/19 (26.3%) |

All candidate LCBs fall below the strict **0.001546** margin. All three Holm
decisions fail (adjusted tail area 0.8051194881). Candidate harm is4/19,4/19 and
5/19 families, above the **10%** limit (at most1/19). CONTROL is a diagnostic
reference. ASAM's higher Elo over the trained CONTROL coexists with broad
utility intervals and excessive harm relative to r882; no arm is admitted.
The original 10,000 draws / RNG 100000, paired family weights, margin, strict Holm
boundaries and harm rule were replayed exactly from the saved receipts.

DRIFT_MATCH achieved mean adapter L2 **0.3760463125**, versus ASAM05
**0.3655750442** (ratio **1.0286432798**, within the preregistered 0.90–1.10 band).
The ASAM05-minus-DRIFT_MATCH **90%** utility interval is
**[-0.0967037297,+0.0692854625]**, outside the required equivalence band
[-0.001546,+0.001546]. Equivalence fails; the drift-control explanation remains
unresolved. The calibrated LR was not retuned.

## Seed and integrity audit

Every arm has three distinct final adapter hashes, final losses, per-family and
per-context utility vectors, and top-four selection maps. Some W/L/T totals
repeat: ASAM10 seeds 40/41 are 12/3/4 against CONTROL; DRIFT_MATCH seeds 40/41 are
3/15/1. Full per-seed W/L/T counts are in `metrics.json`. The source consumes
each fresh seed in its batch RNG and paired stochastic operations. These three
optimization replicates share the same 19-family assessment; the bootstrap keeps
all seed vectors together and the effective assessment size remains 19 families.
Same-seed r882 reference utilities match across all four arms.

All twelve outcomes have changed adapter parameters, 40 updates each, matching
source/checkpoint/roster identities, finite unrevealed utilities and zero
holdout/outer-row deliveries. A CPU-only replay of the frozen native statistical
functions matched **every aggregate**, including both Elo references/CIs, utility
LCBs, Holm, harm and the drift interpretation. It imported no model or Torch and
ran no new model scoring, training or GPU work.

## Timing, abort and readout binding

The complete ledger retains **14 attempts / 945.7571616434 GPU seconds** against
the unchanged 2700-second cap: 12 outcomes, one discarded ASAM10/seed39 train-only
timing cell (**68.8755788582s; 40 updates**), and the CONTROL/seed39 loading-stage
disk abort (**15.9451317261s; zero optimizer updates**). The explicit owner t1130
re-arm resumed CONTROL/seed39 as attempt 2, with the original timing and all old
charges preserved. The final panel has 480 updates; timing adds 40, for 520 actual
updates and 800 backwards in total. Timing exports no validation/top-four output.
There was no automatic retry, budget reset or extra experiment.

After writing aggregate metrics, the authorized operator finalized its last
administrative state from `FIFO_REARM` to `COMPLETE`. That sole field caused the
first CPU publication to report INVALID/UNKNOWN. The
[publisher correction](https://github.com/neverhuman/JopeDime/pull/1160) accepts
only that forward state change; every other ledger field remains exact. The
corrected merged-source readout is **COMPLETE/NO_ADMIT**. The original invalid
report and all 15 raw receipt hashes remain unchanged outside Git.

The owner t1130 re-arm used fair per-cell leases behind H105/H99 NULL, strict
disk <94%, MemAvailable >=30 GiB, nice 19 and CPU/thread guards. The historical
timing/abort charges remain attributable to their earlier original limits.
Raw receipts stay outside Git; the small summary binds their SHA256 identities,
frozen runtime source, unchanged preregistration and merged publisher source.

Next step: close H100 with **no admitted candidate** and keep r882. This closeout
authorizes no further training, retuning or serving promotion. A CPU readout can
be reproduced with the [terminal publisher](../h99_h100_readout/README.md) into a
fresh output directory outside the run root.
