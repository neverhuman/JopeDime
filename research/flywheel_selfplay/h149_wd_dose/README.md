# H149 — stronger weight-decay dose from r882

**NO_ADVANCE.** Nine cells completed with 360 actual optimizer updates. [PREREG.md](PREREG.md) was reviewed and merged before scored inference or training. [metrics.json](metrics.json) records every primary/secondary contrast, efficacy gate and dose disposition. No serving promotion.

Hypothesis: H144's small doses may have been unable to affect adapter continuation. H149 fixed AdamW weight decay at CTRL=0.0001, WD_5=5 and WD_25=25, retaining learning rate 0.0001, 40 steps, batch 16, slate 192, gradient clip 1, H144's loss/sampler/model and r882 initialization. Seeds 149101–149103, family fold seed 149140 and bootstrap seed 149141 were fixed prospectively. Fit used 1,011 TRAIN contexts in 164 families; the remaining 41 TRAIN families were excluded from fit. Only 47 adapter tensors (29,211 parameters) trained; DOPE c51a56d0 and all other parameters/buffers stayed byte-identical.

Primary: equal context weighting within each of 19 DL-Val families, equal families within seed, median across three seeds. Shared family bootstrap with 10,000 draws; percentile 95% intervals; Holm over both non-control hypotheses. All six gates must pass: median gain >0.001, CI lower bound >0, at least two positive seeds, top-three-family positive-gain share <=0.8, Holm-adjusted p <0.05, and valid dose. The 91-context baseline mean is not substituted for the equal-family baseline. Repeatedly exposed DL-Val supports exploratory evidence, not fresh or sealed qualification.

| Arm | Median gain vs r882 | 95% CI | Gain vs CTRL | Elo vs r882 | Seeds >r882 | Invalid picks | Holm p | Decision |
|---|---:|---|---:|---:|---:|---|---:|---|
| CTRL | -0.04490044 | [-0.11158115, -0.00322302] | +0.00000000 | 1386.53 | 0/3 | 1,0,1 | — | control |
| WD_5 | -0.03649827 | [-0.09522582, -0.00200713] | +0.00840217 | 1406.47 | 0/3 | 1,0,1 | 1.0 | NO_ADVANCE |
| WD_25 | -0.03601419 | [-0.11004206, +0.00837851] | +0.00764572 | 1463.33 | 0/3 | 1,1,1 | 1.0 | NO_ADVANCE |

The manipulation statistic is displacement from initialization, not weight norm or an efficacy measure. An arm is DOSE_INVALID when its median displacement lies within the inclusive 95%–105% CTRL band; a larger displacement only clears this check, it does not establish benefit. Both doses cleared the prespecified drift manipulation check; efficacy still requires every primary gate. Invalid hypotheses remain in the fixed Holm family.

| Arm | Adapter L2 drift by seed | Median drift | Absolute relative difference from CTRL | Dose status |
|---|---|---:|---:|---|
| CTRL | 0.44712731, 0.44197899, 0.44783950 | 0.44712731 | reference | control |
| WD_5 | 0.49561808, 0.49246055, 0.49619010 | 0.49561808 | 10.845% | DOSE_VALID |
| WD_25 | 1.63289189, 1.63418400, 1.63339567 | 1.63339567 | 265.309% | DOSE_VALID |

Both doses cleared the manipulation check. Neither dose beat r882 on any of the three primary seeds. WD_5's median gain versus CTRL was +0.00840217 (95% CI [-0.00013568, +0.02534614]); WD_25's was +0.00764572 (95% CI [-0.08424755, +0.09064663]). Both intervals span zero.

The full 1,219-context, 205-family TRAIN secondary includes fit families and historical r882 exposure. It is **TRAIN-exposed/descriptive**, not out-of-fold or independent generalization evidence; it did not affect admission.

| Arm | Median TRAIN gain vs r882 | 95% CI | Gain vs CTRL | Elo vs r882 | Invalid picks |
|---|---:|---|---:|---:|---|
| CTRL | -0.04176565 | [-0.07198804, -0.01354186] | +0.00000000 | 1436.60 | 27,27,26 |
| WD_5 | -0.03451093 | [-0.06183146, -0.00765597] | +0.00603329 | 1455.70 | 23,27,25 |
| WD_25 | -0.00406515 | [-0.02708867, +0.02177572] | +0.03838122 | 1493.22 | 18,19,21 |

On the descriptive TRAIN panel, WD_25 recovered +0.03838122 versus CTRL (95% CI [+0.01164904, +0.06556654]), while its gain versus r882 remained -0.00406515 (95% CI [-0.02708867, +0.02177572]). This comparison does not change the primary admission verdict.

Baseline matched 91/91 utilities exactly, with zero context/mean error: context mean 1.1377941898, equal-family mean 1.1536152770. Baseline invalid counts: 0/91 validation and 17/1219 TRAIN. Full CTRL-relative Elo and its intervals are in metrics; Elo anchors each comparator at 1500. Ties are not wins.

Executed once on xbabe2 from merged source `9122ff702e388f6b8f5991937f77cebe0266f7d6`, nice 19 and one numerical thread. Sampled supervisor+worker peak RSS 3,330,871,296B, max 4 OS threads, minimum free disk 682,701,852,672B, max owned scratch 257,404,336B; 0 pressure pauses. Runtime 3719.2s. No GPU/qualification lock/live writes. Exact input hashes and read allowlist remained enforced, with no protected reads. No retry or outcome-selected dose/seed/step change. A later standing 95% disk-stop rule was added through an external guard during the run, without changing the pinned executor. Resource maxima above include that guard; its start time and the earlier free-space-only enforcement are recorded in the custody addendum.

Admission evidence: 25/25 data-free controls, Ruff/format/hash/link/registration/count checks, zero-update smoke and exact-head same-agent review passed; required ARGON read-only review preceded merge. Unchanged scientific paths are covered by the 16-function [parent-parity.json](parent-parity.json) receipt. Independent scalar replay reproduced all six primary contrasts (10,000 draws each; tolerance1e-14), the drift gate/Holm/decisions, complete panels, cell identities, paired plans and 360 updates. Every saved adapter hash matched; an independent float64 displacement calculation matched the recorded norm within 5.6e-8. Between 23,335 and 23,546 parameter coordinates changed per cell. Full hosted CI was not run under master section 3; no product qualification claim.

Custody: `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H149/astra-t2030/manifest.json`, version `FelDLOLtQKNJ5O50HaEqcGlm9HYCHV7T`, SHA-256 `574f6f35a67630db27bfdc45b60ba47fe116db39371b949afd777b69b2b88297`. [custody.json](custody.json) binds create-only numerical-input and result archives; exact-version readback and 1397 member hashes verified recovery before owned bulk cleanup. Original H50/H58 derived-object versions remain unknown; the exact used numerical closure is preserved. External model implementation files are hash-pinned and are not bundled in the numerical archives; postprocessing and guard helpers are preserved as evidence. Shared inputs are retained.

Next: close H149 without promotion; any further dose, schedule or mechanism needs a separately preregistered wave. H144's original metrics/verdict remain intact; its index caveat is retrospective.
