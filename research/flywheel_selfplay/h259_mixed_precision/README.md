# H259: mixed-precision r882 continuation — COMPLETE / NO_ADVANCE

Hypothesis: bf16 forward/backward rounding or unbiased stochastic bf16 adapter writeback regularizes exact H170 continuation enough to transfer to held-out families. Secondary hypothesis: bounded numerical drift without nonfinite values on the offline adapter copy. Frozen preregistration `531ab2ccc83ac81466b7362702ea8241802d0424` was pushed to **v1/next at 2026-11-03T15:10:01Z**, before the 15:15Z cutoff and before outcomes. Main is held and whole hosted CI deferred by the owner split. No source/model/serving qualification is inferred from publication.

CONTROL is exact H170 scalar loss, fp32 adapter/AdamW. BF16_AMP (primary) uses CPU bf16 autocast only for cached adapter forward; fp32 master adapter weights, fp32 scalar loss and fp32 AdamW moments. Backward runs outside autocast and follows saved forward operator dtypes. BF16_SR (secondary) stores authoritative adapter weights in bf16, uses differentiable fp32 parameter views inside CPU bf16 autocast for LayerNorm/MHA compatibility, accumulates gradients into bf16 storage, clips fp32 work gradients, computes fp32 AdamW updates and stochastically rounds writeback. The fp32 rounding residual is discarded every step. Both treatments preserve fp32 frozen DOPE/anchor and **fp32 strict B selection and A/DL-Val scoring**. No GradScaler or precision fallback.

SR uses adjacent representable bf16 values, high16 plus Bernoulli(low16/65536), a dedicated CPU generator with SHA256(`H259:bf16-sr:seed:fold:step`) low31bit seed, fixed named-parameter tensor order and setup step 0 / updates 1–40. Original fp32 r882 remains selector snapshot0 and the exact strict fallback. Before outcomes, data-free tests established exact mean on all 65,536 uniform draws for fixed positive/negative/subnormal cases, signed-zero/exact values, deterministic streams, finite bounds and Toy LayerNorm/MHA backward/dtypes. Pre-writeback tensors are not saved, so this is not a second replay of every rounding event.

All arms: AdamW lr1e-4, wd1e-4, betas(.9,.999), eps1e-8, global clip 1, 40 updates/cell. Exact H170 uniform sampler picks 8 FIT contexts without replacement, top-fitness 4 reveals + 20 sampled candidates per context. Every arm shares each fresh seed/fold/step batch. No V-REx, loss normalization or family-balanced sampler is added. 205 TRAIN families / 1,219 contexts, 91 DL-Val contexts / 19 disjoint families, fold seed 259140 shared across seeds 259101–259103, boot 259150. Each outerA and nextB selector are excluded from 123 FIT families. Strict B selection accepts earliest best step 0/10/20/30/40 only for gain>.001 and family LCB95>0, else r882. **All 15 decisions fsync-locked globally before trained A/DL-Val assessment**, with no updates afterward.

Native91 passed first: 91/91 exact, mean 1.1377941898164525 and cached/native evaluation bit equality. Then 3 bounded smoke updates and 1,800 primary updates, 15 cells/arm. All 47 adapter tensors / 29,211 coordinates inventoried, DOPE `c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814` and every non-adapter state bitwise frozen. One scientific invocation, no post-Native retry, reroll, resume, tuning or deadline reset.

Primary: DL-Val context-mean per seed after averaging five cells, then three-seed median; 10,000 family-cluster draws retain within-family context counts. Holm m2 covers only BF16_AMP/BF16_SR vs r882. ADVANCE needs gain>+.002, LCB95>0, Holm<.05 and **3/3 positive seeds**; ties reject, with epsilon 1e-12 at numerical boundaries. Same corpus/fold and repeated development panel do not constitute independent dataset replications or sealed confirmation. Family harm means delta<−.0001 per seed, descriptive; exact IDs are archived.

| Arm vs r882 | Seed-median gain [CI95] | Positive seeds | Holm p | Harm families by seed |
|---|---|---:|---:|---|
| CONTROL | +0.000000000 [+0.000000000, +0.000000000] | 0/3 | descriptive | [0, 0, 0] |
| BF16_AMP | +0.000000000 [+0.000000000, +0.000000000] | 0/3 | 1.0 | [0, 0, 0] |
| BF16_SR | +0.000000000 [+0.000000000, +0.000000000] | 0/3 | 1.0 | [0, 0, 0] |

**Verdict: NO_ADVANCE.** The strict selector retained original r882 in all 45 arm-cell choices; the zero gains describe that fallback policy, not equivalence of the trained models. TRAIN strict OOF below is secondary, with 205 equally weighted families; it cannot replace the DL-Val gate.

| Arm vs r882 | Seed-median gain [CI95] | Positive seeds | Holm p | Harm families by seed |
|---|---|---:|---:|---|
| CONTROL | +0.000000000 [+0.000000000, +0.000000000] | 0/3 | descriptive | [0, 0, 0] |
| BF16_AMP | +0.000000000 [+0.000000000, +0.000000000] | 0/3 | 1.0 | [0, 0, 0] |
| BF16_SR | +0.000000000 [+0.000000000, +0.000000000] | 0/3 | 1.0 | [0, 0, 0] |

Unconstrained B-best and step40 are diagnostic only:

| Arm | Unconstrained DL-Val gain [CI95] | Step40 DL-Val gain [CI95] | Strict ARM minus CONTROL [CI95] |
|---|---|---|---|
| CONTROL | -0.003053060 [-0.007517642, +0.000333759] | -0.039657190 [-0.085828129, -0.008610981] | reference | reference |
| BF16_AMP | -0.003053060 [-0.007517642, +0.000333759] | -0.039657190 [-0.084491710, -0.008610981] | +0.000000000 [+0.000000000, +0.000000000] | +0.001726795 [+0.000000000, +0.005418563] |
| BF16_SR | -0.003135988 [-0.007545568, +0.000326133] | -0.040951580 [-0.082383418, -0.009276935] | +0.000000000 [+0.000000000, +0.000000000] | -0.000929595 [-0.004083636, +0.000846921] |

| Arm | Strict DL-Val Elo vs r882=1500 [CI95] | Archived-only r3175=1500 [CI95] | Median context utility |
|---|---|---|---:|
| r882 | 1500.000000 [+1500.000000000, +1500.000000000] | 1500.000000 [+1489.036249127, +1510.963750873] | 1.1377941898 |
| CONTROL | 1500.000000 [+1500.000000000, +1500.000000000] | 1500.000000 [+1489.036249127, +1510.963750873] | 1.1377941898 |
| BF16_AMP | 1500.000000 [+1500.000000000, +1500.000000000] | 1500.000000 [+1489.036249127, +1510.963750873] | 1.1377941898 |
| BF16_SR | 1500.000000 [+1500.000000000, +1500.000000000] | 1500.000000 [+1489.036249127, +1510.963750873] | 1.1377941898 |

Elo: canonical H148 400-iteration MM/prior .01, 1,000 shared family draws. Archived r3175 mean 1.1389572403748425 comes from pinned H201 vector with matched evaluation identity/order and all 91 r882 values. H226 fresh CPU r3175 parity remains FAIL (1.1366829009431036) and excluded. Archived-only Elo does not qualify a fresh CPU comparator or correct winner-selection bias. No r3175 model/native call.

| Arm | Active adapter storage | AdamW moments | Recorded forward NaN / inf | Weight/state NaN / inf | Max writeback quantization L2 | Step40 r882 L2 median [min,max] |
|---|---|---|---|---|---:|---|
| CONTROL | ['torch.float32'] | ['torch.float32'] | 0 / 0 | 0 / 0 | 0.000000000 | 0.373786879 [0.360482923,0.386913985] |
| BF16_AMP | ['torch.float32'] | ['torch.float32'] | 0 / 0 | 0 / 0 | 0.000000000 | 0.373667244 [0.360295025,0.386276129] |
| BF16_SR | ['torch.bfloat16'] | ['torch.float32'] | 0 / 0 | 0 / 0 | 0.030721863 | 0.404210541 [0.390731643,0.416427293] |

Numerical stability: `{"all_forward_gradient_weight_state_finite": true, "bound_pass": true, "maximum_paired_l2": 0.15825630040133842, "paired_l2_reporting_bound": 1.0, "qualification": "LOCAL_ONLY; not serving or speed qualification"}`. All forward/gradient/weight/state finite checks passed. The preregistered matched CONTROL-vs-precision L2 reporting bound 1.0 is descriptive and separate from the utility gate; it does not establish serving safety or a speed advantage. Every paired step 0/10/20/30/40 drift, per-step quantization error/SRseed and dtype/state diagnostics is archived; compact metrics include all per-cell paired distances and 40-point median loss trajectories. Different steps use different frozen batches, so training loss trajectories are descriptive. Selected counts: `{"BF16_AMP": {"0": 15}, "BF16_SR": {"0": 15}, "CONTROL": {"0": 15}}`. No AUC or speed/tok-s verdict.

Context index 27 is included: cid `43022ae6a8bc83e9e25ea4a2882389152856226f5b2bdd4668db07f2f510ae91`, family `fam-9f67f1449b77152cbf20a6c82926f90496dacd7f05a222376edf0fdfb45f667f`, r882 utility `1.1226949552792034`. Compact metrics retain strict/step40 per-seed utilities and invalid counts across five cells. No cliff exclusion. Identical strict fallback vectors and zero-width intervals do not establish equivalence; full duplicate inventories remain archived.

Read-only replay PASS: all 1,800 journal updates / 3 smoke, 45 arm-cell traces / 180 snapshot CRC + 47 tensor hashes, stored choice utility joined to source fitness/gate, uniform H170 sampler / FIT roles, independent strict selector and globally locked order, full panel reconstruction, independently recomputed context-weighted DL-Val and equal-family TRAIN bootstrap / Holm / 3-positive gate, canonical Elo, context 27, initial fp32 state / SR setup rounding and all snapshot L2, source/input/parent hashes, and every guard transition. No model/native/optimizer construction or update during replay. Forward activation/gradient tensors and unrounded fp32 SR work values are not persisted; replay verifies algorithm tests, finite/dtype checks, exact RNG seeds, bf16 checkpoint representability and paired snapshot drift, rather than a second training execution.

Worker exit 0, sampled peak aggregate RSS `1869422592` bytes, maximum aggregate threads `3`, minimum available RAM `66897911808` bytes, minimum disk free `672062631936` bytes, `0` cooperative guard events. Continuous final peakRSS was not recorded and remains null. CPU 40–43 / nice 19 / math 1 / RSS 4 GiB / H239 A1; swap-full is not an abort condition, pause/resume thresholds retained. Program/projection arrays are evicted between FIT/assessment cells with malloc_trim while compact fp32 anchors are retained; all caches released before final statistics. Compute 16:45Z / closeout 17:30Z fixed.

Custody `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H259/run-001/H259-execution-output.zip`, VersionId `tURzMhppAV6xUQfAeV2sCImR_E__s6IU`, SHA256 `fb548052e8cc48e49ccc1311d5f2a4abb75eb916df1b19cd2bf355278712b483`; manifest `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H259/run-001/H259-terminal-manifest.json`, VersionId `5aLWPKLUGAV82LDkFzXPVW_mEZc0HiNE`. Exact-VersionId bytes of both objects and every `318` ZIP member size/hash/CRC PASS. Raw roots `/home/ubuntu/jope-data/h259`, `/home/ubuntu/jope_plan/evidence/H259/res08-t1435` retained. Compact metrics 58343 bytes < 100 KB; raw metrics SHA256 `9718f9bcd8bc2d291d6ea4df48ab58316a7b2232046061e9846159719addbcb6`.

21 data-free targeted cases/Ruff/format/JSON/schema/pins/diff and exact-head same-agent review apply to the byte-identical source; this review is not independent or human approval. Whole hosted CI remains NOT_RUN per owner split, no main/fastlane/PR2071/critical CI/records/schema/rootREADME changes. Publication on v1/next does not assert main acceptance. [Frozen protocol](PREREG.md).

Next step: retain r882; no promotion or further compute from this wave.
