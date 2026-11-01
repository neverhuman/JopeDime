# REL-07 M5 causal wiring

M5 must exercise the actual frozen CPU ranker and the admission path on serving-shaped 4+8 slates. A mocked cost vector or the synthetic program-channel prep test cannot do that. Reuse the existing external `ranker::adapter::real_cpu_parity::m5_serving_causal_measurements` test with its hash-pinned 50-context TRAIN fixture, frozen adapter, actual `CpuRoundAdmission::evaluate_round`, and explicit execution of the normally ignored fixture test. Add separate 4+8 measurements and provider assertions, so changes on 20-candidate slates cannot satisfy M5. Apply the bounded deterministic −6/0/+6 interventions only to landmark-derived conditioning slots 0/1/7/9 while preserving masks and DOPE-only slots 2/4/5; independently reverse revealed fitness. The provider requires both perturbations to change at least one actual admitted set and a fixed first-four admission control to disagree. Measurement execution succeeds when the complete fixture is measured; it does not itself grant M5. A report with zero causal changes is retained with `passed:false`, and the provider refuses PASS. The existing cost tolerances and original fixture bytes stay intact. This is a controlled causal test at the ranker's conditioning input, not raw landmark extractor parity, model-quality evidence or a serving promotion.

The provider verifies a non-development full candidate and its source pin, requires the exact clean checkout and unchanged candidate before/after testing, and runs the one exact real fixture test with `--ignored --exact --show-output`. It requires one collected/executed/passed case with no failures, skips or exclusions and a fresh external native report containing all 50 serving-shaped trials and positive causal controls. Report and input hashes accompany the candidate/source-bound result. Symlinks/non-files, missing fixture inputs, source/artifact drift, zero/wrong/duplicate summaries, stale results and missing/failed causal reports refuse PASS. A local disconnected-admission mutant must produce measurements that the same gate validator rejects; no mutant or production runtime switch is shipped.

Runner argv is `python3 -B deploy/qualification/v1.0/providers/m5.py --golden-dir ABSOLUTE_GOLDEN_DIR --bundle-dir ABSOLUTE_BUNDLE_DIR`, timeout 3600 seconds, plus the standard five `JOPE_REL07_*` bindings. The independently pinned original fixture index is `27905f29703431bedacac90eed1bd54257fd0c59bdf24b56605d0b9416ae1393`; the frozen `jope.safetensors` SHA-256 is `ce39c75f37624e64dfbf89e6493adbf74811a5248f9355997161491e93a132ef`. No default host paths or replacement corpus are inferred. On xbabe1 these existing read-only inputs are under `/home/ubuntu/jope-data/gov-03/priority-cpu-packet-27905f297034/{golden,bundle}`. Provider controls use synthetic files/children only and are not actual M5 qualification. The real native selector and disconnected mutant run separately on xbabe1, nice 10, at most 8 build jobs and math threads one, reusing its target.

Acceptance: `nice -n 10 python3 -B deploy/qualification/v1.0/test_provider_m5 -v`, the exact Rust selector above with the explicit fixture bindings, Ruff/format, diff and the five records checks. Source implementation, controlled fixture proof, candidate gate execution, RNK-10/Q1 scientific qualification and release signing remain separate. No sealed/holdout input, GPU, speed verdict, training or signing is involved; no GitHub CI is dispatched under the owner ruling. Gate evidence context is `status/RES-07.md` and `status/RNK-10.md`; those records cannot substitute for executing this candidate gate.

Initial real CPU probes at sources `0c6fb19ec27e8ef333cbfbb011dc8b35b4b9b138` and `6b5e92e733948bb75f1e7b8309ec49cb4a24b68c` failed the stronger 4+8 landmark assertion, first under negation and then under the bounded interventions. Those failures are preserved. The final provider retains complete measurements and rejects this missing causal effect; no M5 release PASS, weight repair or quality grant is inferred from implementation tests.


## Discovery input binding

The release runner discovers M5 as `python3 -B providers/m5.py` without extra
arguments. M5 therefore accepts the existing `JOPE_GOLDEN_DIR` and
`JOPE_PARITY_BUNDLE_DIR` environment bindings as defaults for `--golden-dir` and
`--bundle-dir`. Explicit flags take precedence, and either source may supply
each path. Missing bindings refuse before candidate verification or child
execution. Absolute paths, non-symlink regular files and the exact TRAIN-index
and frozen-weight hashes remain mandatory. No input is downloaded or selected
automatically. The causal report predicates and result publication are unchanged;
the observed 0/50 landmark-causality result remains unqualified.

Data-free CLI controls reproduce the runner's bare invocation through the real
context parser and M5 checks. They cover environment-only and mixed/explicit
inputs, precedence, missing inputs and invalid paths. Synthetic child receipts
prove orchestration, not model causality or a release-candidate PASS.
