# H75 CPU point scoring support

Standalone support for ASTRA's H75 score interface and OPUS's runner, in scoring/ only. This computes **point diagnostics**, not bootstrap intervals, the seven-contrast conjunction, an experiment result or a promotion. The unrelated H75 W4 shard is preserved.

Input is one JSON packet with `manifest` and `rows`. The caller freezes a `jope.h75-closed-loop.score-admission/v1` manifest containing explicit boolean `fixture_only`, `run_id`, `source_git_sha`, `prereg_sha256`, `execution_manifest_sha256`, frozen `dope_sha256`, `panel`, and `roster`. Each roster cell has `context_sha256`, resolved `family_id`, `fold`0..4, `seed`701/702/703, `role: DL-Train`, and paired `evaluation_identity_sha256`. All three seeds are required for every context; families cannot cross folds. Confirmed-U8 cells also pin the ordered three `confirmation_cut_sha256s`.

Rows follow `jope.h75-closed-loop.score/v1` in the H75 INTERFACES contract: execution identities, panel/policy, context/family/fold/seed/role, selected program SHA, evaluation identity, status and raw utility; confirmed-U8 rows include the exact three cut identities. Legacy uses A,B,C,D,replay_B,replay_D; confirmed_U8 uses A,B,C,D,G,R. Reject duplicates/conflicts, mixed panels, foreign roster/role/identity, unresolved families and nonfinite utilities. Missing cells or failed/timeout/unknown execution return **INCOMPLETE**, counts and no Elo/utility metrics. Completed rows require finite utility; no missing values are replaced with zero.

Seeds are averaged within context **before** the pinned repository Bradley–Terry helper sees context records. Each family then has equal weight across its contexts. A is anchored1500, tie1e-4, prior.01,400iterations. A is the preregistered r882 continuity control; prior exposure remains conditional. W/T/L count context-mean outcomes, not independent families. The existing helper rounds Elo to two decimals; unrounded input/context/family utility means are retained. Context-mean `val_reg_inner` is reported separately from equal-family mean utility; confirmed_U8 is never relabelled `val_reg_inner`. Raw utility and Elo can disagree.

Use the same-source repository helper `python/src/jope/north_star/elo_tracker.py`, SHA256 `9ece98326c32453179b0c765eb45944b68ca9de944d2828a1d42dee1401d1835`. Exact bytes are hashed before module execution; its data loaders and baseline entry point are never called. NumPy is required. From the repository root:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 python3 -B -m unittest discover -s research/flywheel_selfplay/h75_closed_loop_evolution/scoring -p test_score_panel.py -v
nice -n 10 python3 -B research/flywheel_selfplay/h75_closed_loop_evolution/scoring/score_panel.py packet.json --elo-source python/src/jope/north_star/elo_tracker.py
```

Tests use tiny fabricated development utility records, no dataset/model reads, fits, network, GPU or qualification lock. CLI records packet/admission/helper/scorer hashes and scorer Git HEAD; these do not prove a clean checkout or upstream provenance. `H75_ELO_SOURCE` can point tests to a byte-identical reference source.

**Integration remaining:** OPUS supplies a frozen complete authoritative roster and validates upstream role/alias, source and evaluation identities, actual legacy gate/failure mapping, and the preselected program plus all three unrounded confirmation utilities. This helper validates joined declared identities; it cannot prove how the manifest/utility was produced or detect a fabricated family assignment already present in it. OPUS must select the registered fold/track/contrast and prevent leakage/reranking. The receipt always declares bootstrap NOT_COMPUTED (0 of10000 required paired-family draws, seed20261031) and admission=null. Preregistered shared-index family intervals and the scientific decision remain unimplemented by this bounded shard.


## Paired-family bootstrap CPU API

`bootstrap.family_bootstrap(packet, reference)` extends the same admitted score packet/point report with descriptive paired-family intervals. Use `score_panel.load_reference` for the pinned helper. Defaults are10000 draws, seed20261031, linear percentile quantiles, mean LCB alpha.05/7 and median95 intervals. The seven registered contrasts remain partitioned3 legacy/4 confirmed_U8; their selection and scientific conjunction are unchanged. Every contrast within a panel uses the same sorted whole-family index matrix; seeds stay averaged within context and all contexts remain together. Panels with different cohorts have distinct family-order hashes; their Elo scales are not transferable.

Elo resamples use the canonical400 MM updates in a bounded NumPy batch from equal-family pair totals (same A1500/tie1e-4/prior.01), checked against the actual pinned helper on explicit cluster resamples. Receipts retain the original point W/T/L, NumPy version, family order, shared index hash, source-helper hash and resampling settings. Original family IDs are never invented or changed. Family bounds205/20 and draws<=10000 limit batch memory; set OMP_NUM_THREADS/OPENBLAS_NUM_THREADS/MKL_NUM_THREADS=1 and use nice10.

Incomplete execution produces no intervals and zero draws. Reduced draw counts or altered seeds are labelled SMOKE_ONLY. The fixed-settings state is COMPLETE_DESCRIPTIVE_BOOTSTRAP; **admission remains null**. Caller provenance, program/confirmation binding, paired outcome completeness and runner integration remain separate obligations. These repeatedly used development cohorts and r882 exposure do not become untouched-family confirmation. The original point-only `score_panel.score` API and its explicit NOT_COMPUTED bootstrap state remain unchanged.

CPU fixtures: from the repository root, run the same unittest command above with `-p test_bootstrap.py`. This covers paired weighting/seed averaging, reference-solver parity, repeatability, incomplete/refusal states and the prescribed10000-draw synthetic case. No dataset/model/GPU/fit/qualification-lock operation is part of this module or its tests.
