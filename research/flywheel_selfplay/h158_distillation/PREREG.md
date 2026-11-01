# H158: R782-to-r882 listwise distillation with a context veto

Owner rd_astra [ASTRA]; authority `hwave_t2300_h158_rd_astra.md` (2026-11-01).
Status: preregistered, not executed. OPUS cross-review precedes merge and one run.
No scientific outcomes were produced while selecting this protocol.

Hypothesis: transfer R782's ranking distribution into the r882 adapter at low learning rate; dropping the teacher term on fit contexts where its full-pool argmin fails the native gate may avoid continuation harm. This is function-space distillation, distinct from H152 weight averaging. Historical H139/H145 TRAIN advantages and H152 validity cliffs motivate it; this is not independent confirmation of those exposed-panel findings. A veto only affects fit-context loss, not evaluation or deployment, and cannot guarantee a valid pick on a new context.

## Locked experiment

| Arm | KD weight a | Context veto | Optimizer updates |
|---|---:|---|---:|
| R882 | — | unchanged reference | 0 |
| CTRL | 0 | no teacher term | 40 per seed |
| KD03 | 0.3 | yes | 40 per seed |
| KD07 | 0.7 | yes | 40 per seed |
| KD07_NOVETO | 0.7 | no | 40 per seed |

Three seeds 158101–158103. Exact total **480 updates**, resolving the dispatch's approximate “600”: R882 is unchanged, so only four arms train. One attempt, no retry/resume, dose/step/seed selection or early efficacy peeking. Each cell resets all student tensors to byte-exact r882. AdamW lr=1e-5 (H142 TR_LR learning rate only), weight_decay=1e-4, 40 steps, batch16, slate192, clip norm1. Only47 adapter tensors/29,211 parameters optimize. All other state including buffers/DOPE remains byte-identical, checked every update and after scoring. Teacher R782 is eval-mode, all requires_grad false, no optimizer, no_grad forward; complete teacher state hash must match before/after caching. Teacher/student non-adapter tensors must match exactly.

H151's live objective is unchanged: per context, SmoothL1(beta1) mean over candidates plus0.1*KL(softmax(-target/.2) || softmax(-student/.2)), target=clamp(raw fitness,-4,2) for valid candidates and cost2 for gate failures. Student loss is (1-a)*live + a*KL(softmax(-teacher/.2) || softmax(-student/.2)); no temperature-squared multiplier. On a vetoed context the loss is **full live loss**, not (1-a)*live. Contexts are equally weighted within the minibatch, including duplicates.

Veto is fixed from teacher **full-pool** inference on fit contexts: argmin uses NumPy's first-index tie rule; veto iff that candidate's gate boolean is false, rather than comparing a potentially valid numeric utility to -2. Entire teacher term for that context is dropped. No candidate labels are relabeled or removed. Teacher KL targets are separately computed for the **exact same sampled192 candidates and first4 revealed inputs** passed to the student; slicing full-pool scores would not preserve those inputs and is forbidden. Cache40 paired teacher batches per seed, reuse across arms, preserve cache bytes/hashes. Veto count and per-cell sampled veto exposures are reported; if no fit context is vetoed, do not claim a test of veto benefit.

## Inputs, folds and boundaries

[manifest.json](manifest.json) pins checkpoint/model/source/package hashes, [input-provenance.json](input-provenance.json) pins exact S3 checkpoint versions. Student round882 SHA0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64; teacher round782 SHA427b0d4608e94f60f5f97d79b5c1c4ce9a7efd59d17e7853f714e901f3a4af0c. Original H50/H58 derived-object versions remain unknown; their exact used closure is hashed and archived, not substituted with catalog data. Offline owned copies only; no live-loop directory access.

Sort205 TRAIN families, permute with NumPy default_rng(158140), split five equal41-family folds. Fold0 assesses (260 contexts), folds1–4 fit (959 contexts/164 families); [fold_manifest.json](fold_manifest.json) fixes the actual roster and counts. Validation91 contexts/19 families stays outside the fit/teacher closure. Original TRAIN/inner regression metadata, family/context disjointness, all1,310 pool hashes and feature-cache exact ID closure are checked. No sealed, meta-test, external holdout or outer-test rows. Teacher only reads fit-context feature/fitness/gate inputs. Historical r782/r882 exposure means even the held-family assess fold is held only from **this continuation**, not from original training.

Paired batches reuse H151's fixed stream:20 rounds, generate8 draws per round and retain first2; per step16 context draws with replacement, first48 fitness-ordered candidates plus144 draws from the remainder. All arms share batches, ordered indices and RNG streams for each seed. H149's exact-baseline evaluator is retained; no proposal/pool/evaluator changes.

## Baseline and admission

Before teacher scored inference or any update, r882 must reproduce **all91 utilities bit-for-bit**, zero mean error versus1.1377941898164525, with identical evaluation-contract identity. A failed baseline is INCOMPLETE, never NO_ADVANCE. Constructor-only smoke may build both networks and AdamW/trigger lazy framework JIT imports, but must perform0 forwards/0updates/0scored contexts. Smoke is not the one-shot scientific attempt.

Primary: within each seed, equal context weights within each of19 DL-Val families, then equal family means; median across the three seeds. Bootstrap seed158141 produces10,000 shared family-index draws, retaining all three seed panels on each draw; report percentile95%CI, per-seed gains, invalid picks, and Elo vs r882=1500. Tied families score0.5 for descriptive Elo; gains tied at0 or margin do not advance.

Multiplicity family is exactly the **three KD arms**, including NOVETO; CTRL is descriptive and ineligible. Sort their one-sided bootstrap p=(1+#draws<=0)/10001 ascending (arm name breaks exact ties). At rank r=1..3, Holm p is running max of min(1,(4-r)*p). Define the registered two-sided Holm-step-down lower bound as percentile0.025/(4-r) of that arm's bootstrap gains. Its entire preceding prefix must have lower bound>0 and adjustedp<.05. An arm advances only when its median gain is strictly>+0.001 **and** this Holm lower-bound prefix passes. These definitions are explicit before outcomes; ordinary95%CIs are also retained. Missing/nonfinite cells/hypotheses are INCOMPLETE, not dropped or imputed. Positive-seed counts and top3 positive-family concentration are reported diagnostically; no extra posthoc gates. Overall ADVANCE iff an eligible KD arm passes; otherwise complete run NO_ADVANCE. No serving promotion is authorized by this experiment.

Secondary: fold0 assess (260/41) and full1,219/205 TRAIN panel, all TRAIN-exposed/descriptive. Preserve per-context utilities/choices/invalid flags/IDs, per-seed contrasts vs r882 and CTRL, and invalid counts. Full TRAIN results never override primary. Compare KD07 versus NOVETO descriptively; any future mechanism/adoption test needs new preregistration.

## Execution and evidence

CPU xbabe2 only, nice19, numerical thread1, at most4 total OS threads including supervisor/descendants, aggregate RSS<6,000,000,000B; terminal stop on load1>85, MemAvailable<16GiB, user-visible used/(used+available)>=95%, free<40,000,000,000B, owned scratch>=3GB, or wall>9000s. xbabe0/xbabe3/GPU/qualification.lock forbidden. Pressure abort consumes the scientific attempt; no pause/resume. Python audit denies protected paths/unlisted jope-data reads/known input writes, while allowing framework /tmp JIT stubs; it is application evidence, not an OS sandbox. Constructor and audit negative controls must pass before launch.

Every cell saves all47 adapter tensors plus per-context panels, losses, drift, initial/source/teacher/frozen hashes and paired-plan identity. Save teacher-fit veto panel, exact cached teacher batches, baseline/assess/full TRAIN panels, final small metrics, audit and continuous resources. Verify exact merged bytes before launch, independent numerical replay after run. Create-only S3 archive inputs/results; verify exact VersionId downloads and every member hash before scoped owned-bulk cleanup after closeout merge. No large Git objects. Targeted local checks, self-review and OPUS review; no hosted CI under current owner policy. Latest t2300 authorizes author squash/admin merge with exact-head guard after checks, without changing protection settings.
