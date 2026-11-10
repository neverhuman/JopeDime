# H253: V-REx across FIT families — COMPLETE / NO_ADVANCE

Hypothesis: penalizing variance of FIT-family mean H170 losses yields r882 adapter updates that transfer across held-out families and clear the DL-Val91 native utility margin. Preregistered in PR #2107, protocol `144f045c585000b1eb7955eeb1fbf1eaf16b4e6f`. Actual execution source `1e2518e260a44db2b5460266619163fc4bca257f`, exact reviewed PR #2108 head. Source was unmerged when launched and was subsequently merged at 13:07:13Z as `344faf823d57fdd9663550662c43f3eb93183ed5`; the whole tree matches. Original launch-time flags and later readback are preserved as dated evidence. Publication routes to **v1/next**, with main held and full hosted CI deferred per owner CI_FREEZE split; this is local scientific evidence, not release/serving qualification.

CONTROL / VREX_B1 / VREX_B10: exact H170 scalar CTRL loss and AdamW lr=1e-4, wd=1e-4, betas=(.9,.999), clip=1.0, 40 updates/cell. The treatments add 1x /10x detached FIT normalization times population variance of four family risks. Every paired batch has four distinct FIT families, two contexts each, 4 reveals + 20 sampled candidates per context; same `(seed,fold,step)` stream across arms. Normalization is reciprocal median across FIT contexts of full-pool candidate-loss variance at frozen r882, floor=1e-12, once per outer fold shared across seeds/arms. Formula/floor were frozen before outcomes, with all numeric scales/variances retained. No annealing, tuning or extra arm.

205 TRAIN families / 1,219 contexts, 91 DL-Val contexts / 19 disjoint families; fold seed 253140 shared across seeds 253101–253103. Each outer A and next B selector are family-disjoint from 123 FIT families; B strict selection accepts earliest best steps 0/10/20/30/40 only for gain>.001 and family LCB>0, otherwise r882. **All 15 decisions fsync-locked globally before trained A/DL-Val assessment.** No optimizer updates followed the lock. Native91 passed first, all 91 exact, mean 1.1377941898164525 and cached/native bit equivalence. Then 3 bounded smoke updates, then **1,800 primary updates**, 15 cells/arm, all 47 adapter tensors / 29,211 coordinates inventoried. Frozen DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and every non-adapter state stayed bitwise unchanged. Same corpus/fold and possibly identical fallback predictions are paired repetitions, not independent dataset replications; the repeatedly exposed development panel is not a sealed confirmation set.

Primary is the DL-Val **context-mean** per seed, five cells averaged, then seed median; 10,000 family-cluster draws retain within-family context counts. Holm m2 includes only VREX_B1 and VREX_B10 vs r882. ADVANCE requires gain>+.002, LCB95>0, Holm<.05, at least 2/3 positive seeds; ties reject. Harm is family delta<−.0001 per seed, descriptive. Source formulas retain epsilon=1e-12 at numerical ties.

| Arm vs r882 | Seed-median gain [CI95] | Positive seeds | Holm p | Harm families by seed |
|---|---|---:|---:|---|
| CONTROL | +0.000000000 [+0.000000000, +0.000000000] | 0/3 | descriptive | [0, 0, 0] |
| VREX_B1 | +0.000000000 [+0.000000000, +0.000000000] | 0/3 | 1.0 | [0, 0, 0] |
| VREX_B10 | -0.005281298 [-0.020292376, +0.002618498] | 0/3 | 1.0 | [0, 5, 5] |

**Verdict: NO_ADVANCE.** Pooled TRAIN OOF below is secondary, with 205 equally weighted families; it cannot replace the sole DL-Val decision.

| Arm vs r882 | Seed-median gain [CI95] | Positive seeds | Holm p | Harm families by seed |
|---|---|---:|---:|---|
| CONTROL | +0.000000000 [+0.000000000, +0.000000000] | 0/3 | descriptive | [0, 0, 0] |
| VREX_B1 | +0.000000000 [+0.000000000, +0.000000000] | 0/3 | 1.0 | [0, 0, 0] |
| VREX_B10 | +0.000000000 [-0.001024781, +0.002561080] | 1/3 | 1.0 | [0, 10, 9] |

Unconstrained B-best and step 40 are report-only. Neither can replace strict selection:

| Arm | Unconstrained DL-Val gain [CI95] | Step 40 DL-Val gain [CI95] | Strict VREX minus CONTROL [CI95] |
|---|---|---|---|
| CONTROL | -0.001059042 [-0.002965807, +0.000128027] | -0.022468367 [-0.052729387, -0.002722561] | reference |
| VREX_B1 | -0.020816432 [-0.063592825, +0.002334727] | -0.031461550 [-0.092773699, +0.003405551] | +0.000000000 [+0.000000000, +0.000000000] |
| VREX_B10 | -0.029770594 [-0.090893118, +0.008451300] | -0.038233053 [-0.116052457, +0.012696476] | -0.005281298 [-0.020292376, +0.002618498] |

| Arm | Strict DL-Val Elo vs r882=1500 [CI95] | Archived-only r3175=1500 [CI95] | Step 40 L2 drift median [min,max] |
|---|---|---|---|
| r882 | 1500.000000 [1500.000000, 1500.000000] | 1500.000000 [1489.036249, 1510.963751] | 0 (frozen) |
| CONTROL | 1500.000000 [1500.000000, 1500.000000] | 1500.000000 [1489.036249, 1510.963751] | 0.35756150264885833 [0.3446880597835098,0.3732158226307788] |
| VREX_B1 | 1500.000000 [1500.000000, 1500.000000] | 1500.000000 [1489.036249, 1510.963751] | 0.20596707708158662 [0.15415181933382827,0.25895148955194186] |
| VREX_B10 | 1500.000000 [1485.377889, 1518.283715] | 1500.000000 [1485.286349, 1525.619933] | 0.2738785898813463 [0.2336977829742005,0.31207418355649014] |

Elo uses canonical H148 400-iteration MM/prior.01 with 1,000 shared family draws. Plateau utility 1.1389572 is historical r3175; the separately pinned H201 archived vector matches evaluation identity/order and all 91 r882 utilities. The archived mean is 1.1389572403748425. H226 CPU replay remains archive-parity FAIL (mean 1.1366829009431036), excluded here; archived-only Elo does not qualify fresh CPU r3175 or remove winner-selection bias. No new r3175 model/native evaluation occurred.

| Arm | Median DL-Val context utility | Minus plateau1.1389572 | Median FIT-family risk variance, step 1 / step 40 |
|---|---:|---:|---|
| r882 | 1.1377941898 | -0.0011630102 | frozen reference |
| CONTROL | 1.1377941898 | -0.0011630102 | 2.532576560974121 / 6.767180919647217 |
| VREX_B1 | 1.1377941898 | -0.0011630102 | 2.532576560974121 / 6.701221942901611 |
| VREX_B10 | 1.1325128915 | -0.0064443085 | 2.532576560974121 / 6.690528392791748 |

Full paired per-family/context loss trajectories for every update, all 40-point across-cell medians, beta scales, all-arm L2drifts, selection counts and diagnostic TRAIN/step 40 contrasts are retained in metrics/custody. Different steps use different frozen batches; the trajectory is descriptive. Selected step counts: `{"CONTROL": {"0": 15}, "VREX_B1": {"0": 15}, "VREX_B10": {"0": 13, "40": 2}}`. All loss/variance changes are training diagnostics; no utility, speed, AUC or serving claim is inferred from them.

Context index 27 in the pinned DL-Val order is included throughout: cid `43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`, family `fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f`, r882 utility `1.1226949552792034`. Every strict/step 40 per-seed utility and invalid-count contribution across five cells is in compact metrics. No cliff exclusion. Identical fallback vectors/zero-width bootstrap intervals do not establish model equivalence; full duplicate/pseudoreplication inventories remain archived.

Initial process run-001 exited 1 after 2.277 s on a configuration-only tuple/dictionary schedule API mismatch, before runtime/model/Native91/optimizer; 0 scientific calls/updates. A data-free fixture preserves the failure; the dictionary reader correction retained the exact frozen schedule. A separate fixture corrected the family-mean label while primary calculations stayed unchanged. All scientific runner code after admission, treatment functions and estimator calculations retain AST parity with the merged preregistration. Reviewed local runtime was admitted under the already merged protocol and current main hold with honest distinct source identities. Run-002 was the first scientific invocation, with no cell reuse, post-Native retry, reroll or budget/deadline reset. Both raw attempts and all receipts are preserved.

Read-only replay PASS: the 1,800-update journal and 3 smoke updates, 45 arm-cell traces, 180 snapshot ZIP CRCs and hashes/inventory for 47 tensors, initial state/L2 drift, family roles and sampler/scale arithmetic, nativeutility from stored choices/poolfitness+gate, strict selector/global-lock ordering, complete panel assembly, independent primary/TRAIN/diagnostic bootstrap+Holm, canonical Elo arithmetic, context 27, source/input/parent hashes and every guard transition. Frozen full-pool FIT prediction costs were not saved: normalization replay verifies stored per-context variances, median/reciprocal and FIT IDs; it does not rerun a model to recreate those costs.

Worker exit 0; sampled peak aggregate RSS `3361906688` bytes, maximum aggregate threads `3`, minimum available RAM `48877412352` bytes, minimum disk `709317324800` bytes. Resource guard PASS; `2` pause/resume events. Continuous final RSS maximum was not recorded and remains null; sampled extrema are named as such. CPU 40–43 / nice 19 / one math thread / H239 A1 bounds retained, no GPU or other-job controls. Compute stop 15:30Z / closeout 16:15Z budgets not reset.

Custody `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H253/run-002/H253-execution-output.zip`, VersionId `jKUUlx4ev6KvcWT3pwsSu8IF0GjmbxL9`, SHA256 `c7715bd9dc53c04af64b8884c200d8071447579f3392a70fa09bd5cc0511debb`; manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H253/run-002/H253-terminal-manifest.json`, VersionId `9h9VcSAdYe0h.vZNuE6HTdaLJMAg3ZsE`. Exact-version byte replay plus all `413` member hashes/sizes and ZIPCRC PASS. Raw roots `/home/ubuntu/jope-data/h253`, `/home/ubuntu/jope_plan/evidence/H253/res08-t1225` retained. Compact metrics: 63,628 bytes (<100 KB); full rawmetrics SHA256 `e3a4c0e5e09c3e477d7711328df6e05adb14f9bbbdc872830e3e9146d0c3dab8`.

26 data-free tests/Ruff/format/JSON/schema/source+registry pins/diff/actual static coverage passed at the byte-identical execution source. Whole v1/next hosted CI is NOT_RUN, deferred by owner until PR #2071 merges; critical CI/config/records/global README glance files are not edited in this routing. Same-agent exact-head review is not independent/human approval. Publication is queued on v1/next, not a claim of main acceptance or serving promotion. [Frozen protocol and dated pre-native addendum](PREREG.md).

Next step: retain r882; this V-REx wave does not qualify promotion. No follow-on compute is authorized.
