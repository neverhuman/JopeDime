# Optional landmarks: trained candidate qualification

The first dropout-trained checkpoint has verified Rust CPU/CUDA execution and preliminary development results. It is **unqualified**: the remaining quality, class preparation, product-consumer and live-flywheel requirements prevent promotion. The released checkpoint and packaged default remain separate.

`ecded0225fca7c6e13c413b0e39b1040bf95f7694940b2809f1ce2ce2abf7bd9` was trained from released `d0b69603…` for eight epochs, 360 optimizer updates and 714 training cells (319 regression, 199 binary, 196 multiclass). All 56 tensors changed and all 444,168 parameters are finite. Forty active encoder tensors changed, with L2 movement 2.295971159. These checks establish an actual update, not predictive improvement.

The same candidate pin produces 629 coordinates, with its first 512 used as Dope512, for every mask. Four identical operator inputs match the Python CPU reference within 1.907348633e-6 on both Rust CPU and CUDA (tolerance 3e-5). Actual Rust native sessions additionally pass on regression, binary and three-class inputs: observed masks 0→8→17→22, both direct and learned Jope consumers, device residency, independent exported storage and exact terminal/full equality. Missing native measurements remain missing. The learned CUDA context differs from the scalar CPU oracle by at most 7.390975952e-6. These are execution tests; the components are test inputs and do not measure policy quality.

Explicit artifact qualification fails when the candidate or governed Jope bundle is absent. It is intentionally ignored by the ordinary suite because those artifacts require separate custody. The learned Jope bundle is not included in the appliance package.

## Paired development stage

All 112 requested development datasets are accounted for. A safe, role-filtered study completed 102 datasets (41 regression, 17 binary, 44 multiclass); ten multiclass tables exceeding 64 features are explicitly unavailable. The study covers none/macro/full, both released/candidate pins and seeds 20261024/20261025: 1,224 completed rows and 120 unavailable rows, with zero evaluator failures, timeouts or source drift. Native reports are genuine cached measurements tied to FIT rows; no new native report was synthesized.

| Task | Paired datasets | Released none / macro / full R | Candidate none / macro / full R | Candidate none/full, 95% CI |
| --- | ---: | --- | --- | --- |
| Regression | 41 | 0.367472 / 0.405391 / 0.402708 | 0.386742 / 0.404376 / 0.401258 | 0.9638 [0.8863, 1.0317] |
| Binary | 17 | 0.471328 / 0.535202 / 0.510756 | 0.561817 / 0.532127 / 0.533637 | 1.0528 [0.9131, 1.2602] |
| Multiclass | 44 | 0.912056 / 0.885092 / 0.889386 | 0.897760 / 0.892712 / 0.902286 | 0.9950 [0.9624, 1.0205] |

Paired seeds are averaged within each dataset before aggregation and 2,000 paired dataset bootstrap draws. Regression none/full improves from .9125 to .9638 at the point estimate; binary improves from .9228 to 1.0528; multiclass changes from 1.0255 to .9950. The intervals do not establish the retention target across tasks. Candidate full/released full is .9964 [.9649,1.0295] for regression, 1.0448 [.8841,1.2936] for binary and 1.0145 [.9909,1.0391] for multiclass. Paired feature-importance checkpoint deltas have intervals crossing zero; undefined correlations remain null.

The existing metric clips regression R² to [-1,1] and floors it at zero before retention; binary uses AUROC and multiclass uses accuracy. Retention also applies chance adjustment, denominator floors and [0,1.25] clipping. Floors affect 24/41 regression, 11/17 binary and 1/44 multiclass datasets. Published score fields therefore are not unbounded raw scores. Absolute strength and these floor flags must accompany relative ratios. The study is exploratory development evaluation, with no sealed qualification or Jope DL/search ablation.

The parallel all-arm study agrees exactly on the 1,224 comparable rows. Its additional wide multiclass rows need correction: synthesis truncates to 64 features while the evaluator receives wider TEST inputs, and caught feature-count errors become chance scores. A zero table-error counter does not establish valid measurements for those rows. Keep them unavailable until one declared, FIT-only feature policy is applied consistently and evaluator failures propagate. Separately, the shared harness main has a mixed-role manifest-builder hazard; the independent study bypassed that builder. Actual read provenance for the parallel invocation must be reviewed by its owner.

## Remaining work and owners

1. **Claude/AGY, source custody resolved:** the original source commit `4e095b74` and all five pre-run hashes were recovered exactly. The owner archived the complete19-file import closure; an independent review verified every member against Git, receipt hashes and the extracted tree. Keep post-training integration `52619c78` separate. Helper use at launch is owner-attested and corroborated by pre-launch history, rather than an independently recorded pre-run helper hash. Historical receipts remain unchanged.
2. **Claude-Code, Dope integration:** publish a distinct lossless class-preparation lineage for the trained product. Preserve released full-path cap32 semantics and frozen outputs; add new K3/K32/K33, fractional/close-label, permutation and unknown-label parity cases. All masks must select the same preparation/checkpoint policy.
3. **Claude-Code, qualification:** fix wide-table and per-auditor failure propagation, enforce role filtering before blob opens, then retain the complete none/macro/larger/full/random/N100/200/300/500/1000 comparisons with paired uncertainty and per-task counts. N300 remains the agreed default; this study does not qualify it.
4. **Codex, consumers:** expose the optional learned consumer through the product flags/events using verified scoped foundation/target components. Run matched direct/learned/component-control predictive ablations. The governed Gen31 decoder supports binary/regression; genuine multiclass readout remains required.
5. **Flywheel owner + Codex:** source admission/snapshot regressions pass (26 tests), but no deployed Rust flywheel was found on this host at 17:19 UTC. Establish a production evidence-authority binding and fresh runtime/progress proof. Source tests cannot satisfy live-green. Retired services remain untouched.
6. **Performance owners:** complete matched weight-cold/resident CPU/CUDA measurements after the final preparation code and checkpoint are pinned. Earlier GPU cold cells failed the 5× gate; numerical parity and package smoke do not change that result.

## Evidence custody

The host-local study is `/home/ubuntu/jope_plan/evidence/candidate-dev112/stage1-20261027`. Its summary SHA-256 is `8cec31f96fbd7bc3e68643645f6b0790ab31d6c55aeecefd3ae47e719b05d0ee`; per-arm result SHA is `9de9229b667d2ac21b6f9de68be6c8958cd21478b49b0def7594ae8618ef94ed`; paired FI SHA is `9f22458bd4385779eeb7114a5bf9e6b32bf7d4b90767ef01bcabfd7a807e10b9`; verification SHA is `2a9409ff0013e1579e0bd6581ee502c42ff31a8e2d19ad7f1622ba47f8e5aba2`. The [machine-readable qualification record](optional-landmarks.json) retains per-task means, intervals and exclusions.

Rust evidence is in this lane’s `target/landmarks/`: `candidate-cpu-parity.log`, `candidate-cuda-parity.log` and `candidate-native-cuda-parity.log`; the last has SHA `63c9c43d23268db7752880d22745c0ef15c2de625878862ccdcdd643cccad025`. `candidate-native-cuda-window.json` binds test binary, source hashes, the admitted GPU window and its release. Training receipt SHA is `5be7ca4319bb87b56f9a3c9a3b51f5b468b2df737ebec8b7f8c33df6836f5874`; the provenance companion SHA is `26789e70427f1e2d9a97251907e74f819a5cdf620df9e70ea88417356527b699`.

The verified source archive has SHA `ed992dbc878ea0262ddab22e5c7e416fb522abe177d72db939f2aea937334c29`; its companion receipt has SHA `856b341a30c6aa0deda35e6abc93ec3134c85118226c420cfd172b049db80342`. Independent verification SHA is `d6c1b4cccad167e30f9b9f487cfb704301155a2e99d07960299b0b695039c62a`. Source-byte custody is resolved with the helper attestation limit stated above; this does not qualify the checkpoint or replace data admission.
