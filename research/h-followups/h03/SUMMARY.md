# H03 readout-aware summaries: evidence summary

## Preregistered contrasts and verdict

The intended follow-up is a family-grouped DL-Val comparison of the H03
readout-aware treatment against current and frozen A0, reporting 97.5% confidence
intervals for reduction in Elo loss. The treatment, fitted model, prediction
arrays, and matched current/A0 predictions needed to execute that comparison
are not available in the pinned H03 source. The implementation therefore
reports the existing aggregate only; it does not constitute the preregistered
fresh comparison.

Historical H03 used 1,594 contexts across 379 families and reports 30 fits.
Its saved aggregate gives:

| Historical contrast | Reduction | 97.5% CI | Interpretation |
| --- | ---: | ---: | --- |
| Readout-aware function vs current | +4.556% | [+0.018%, +9.506%] | Positive historical interval; exploratory only |
| Readout-aware features + trained function vs frozen A0 | -0.790% | [-4.727%, +3.165%] | No established lift over frozen A0 |
| Full function vs frozen A0 | -1.380% | [-5.195%, +2.419%] | No established lift over frozen A0 |

**Verdict: `NOT_REPLICATED`.** This means no fresh replication was run: the
required reproducible treatment and comparator inputs are missing. It does not
mean the historical effect was refuted. The historical study itself concluded
that it found no established lift over stronger frozen A0 or benefit when the
features were added to A0.

## Provenance and scope

The values above are copied from
[`hypotheses-results.json`](../../../docs/research/studies/dope-neural-ranking-20261021/early-history/hypotheses-results.json),
whose H03 entry pins source artifact
`4da33283fc32b609166f2a59bfbf93407f26428921f3f881915d32405a87df20`. Its
[`source receipt`](../../../docs/research/studies/dope-neural-ranking-20261021/early-history/source-receipt.json)
describes the artifact as saved report/aggregate/curve snapshots and explicitly
records no arrays or model. These intervals are historical reported intervals,
not recomputed from family-level observations here.

No DL-Val payload, DL-Holdout payload, sealed meta-test payload, or DOPE model
was loaded for this summary. No fit or scoring run was performed. The helper
`run.py --dry-run` validates the pinned aggregate and prints the same historical
values; without `--dry-run` it refuses to proceed because no fresh replay is
supported.
