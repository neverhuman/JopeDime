# H52 summary

**Hypothesis:** Cost-focused ranking, explicit gate penalties, anchoring, or a within-family target permutation can improve argmin utility over regression controls.

**Setup:** Offline r882, frozen DOPE, 1,219 DL-Train contexts / 205 families, five family-grouped folds, seeds 12–14. Twenty-seven cells: three matched CONTROL configurations, FOCUS tau 0.1/0.5, GATE, ANCHOR lambda 1/10, and NULL. Admission was pooled median and family-cluster bootstrap LB95 both strictly above 0.001. No holdout reads.

| Arm | Mean gain vs matched control | Candidate mean vs r882 | Median / LB95 | Terminal DL-Val | Elo |
|---|---:|---:|---:|---|---|
| FOCUS tau 0.1 | +0.0206 | −0.0092 | 0 / 0 | Not read | N/A |
| FOCUS tau 0.5 | +0.0248 | −0.0051 | 0 / 0 | Not read | N/A |
| GATE | +0.0251 | −0.0048 | 0 / 0 | Not read | N/A |
| NULL placebo | +0.0220 | −0.0079 | 0 / 0 | Not read | N/A |
| ANCHOR lambda 1 | +0.00015 | −0.0796 | 0 / 0 | Not read | N/A |
| ANCHOR lambda 10 | +0.00006 | −0.0797 | 0 / 0 | Not read | N/A |

The positive FOCUS/GATE paired means are also present in NULL, while the matched `k=16/lr=1e-4` control is about −0.030 vs the live-sized `k=2` control. This points to harm from the higher-step control rather than a treatment-specific gain. Every candidate configuration remains below r882 on continuity means.

**Verdict:** No arm beats r882; FOCUS/GATE vs-control mean gains are explained by the NULL placebo. All pooled family medians and LB95 values are zero, so the preregistered H52 rule admits no candidate. No terminal validation was read; Elo is N/A.

**Next step:** H53 learn-from-zero pilot. H54 fold-clean baseline is proposed only. See [README](README.md) for per-seed, bootstrap, and red-team details and [metrics](metrics.json) for the compact record.
