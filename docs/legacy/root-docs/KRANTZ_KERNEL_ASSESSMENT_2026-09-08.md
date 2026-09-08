**Assessment: useful for qualification and search fidelity; insufficient evidence for a new production architecture.**

The strongest fit is to make each approximation state the quantity it preserves,
the conditions under which it is trusted, and its fallback. This complements the
kernel upgrade: type/shape admission answers whether a program can execute, while
approximation evidence answers whether replacing it with something cheaper is
acceptable. These must remain separate contracts. Neither makes an empirical
predictive claim into a proof.

I reviewed the supplied research claims against the cited sources and the local
code. I inspected relevant sections of Krantz, not every chapter or derivation.
His scaling procedure starts from the governing equations and auxiliary conditions;
it cannot be transferred to arbitrary ML merely by inventing ratios. Yip already
generated simplified physical models with applicability constraints. Callaham et al.
identify local balances in equation space. The proposed reusable learned solver
and failure detector would need its own evidence and novelty argument.
[Krantz, sections 2.3–2.4](https://ndl.ethernet.edu.et/bitstream/123456789/27730/1/13.pdf),
[Yip, 1993](https://cdn.aaai.org/AAAI/1993/AAAI93-095.pdf),
[Callaham et al., 2021](https://www.nature.com/articles/s41467-021-21331-z).

| Proposal | Fit to this project | Action justified now |
|---|---|---|
| Assumption-carrying algorithms | High. Cheap numerical solvers, feature omissions, and fitness screening all make different approximation claims. | Add falsification fixtures and specify evidence independently of executable admission. Require total diagnostic/correction/fallback costs before adoption. |
| Computational regime representation | High as an evaluation objective. The policy should distinguish target mechanisms, not only dataset marginals. | Add mechanism holdouts and matched-marginal task pairs to future encoder qualification; first compare with existing supervised contexts and ordinary meta-features. |
| Boundary-layer reasoning | Moderate. Escalating selected candidates or blocks is plausible. | Audit missed high-value candidates and whole-pipeline consistency. Do not add a learned router before demonstrating localized errors and net savings. |
| Similar miniature experiments | Useful, conditional. Smaller screening datasets can erase rare interactions, minority labels or forecast histories. | Predeclare which rankings/failure boundaries should transfer and test them at full fidelity. No small fixture substitutes for eight-GPU recovery or the integrated campaign. |
| Regime-controlled self-improvement | Later. Current defects in representation and proposal admission already explain wasted work. | Measure valid unique executions and externally verified gains per total resource budget first. Compare any later controller with a strong adaptive scheduler. |

Existing precedents support the cautions in the proposal. Mixture-of-Depths is an
example of selective computation, not evidence that our failures are localized.
μP transfers hyperparameters under a particular parametrization; it does not prove
that arbitrary small GP campaigns predict large ones. The agent-scaling paper's
current v3 reports task-dependent architecture outcomes and limited predictive
fit, not universal scale similarity. Two-timescale GAN convergence likewise does
not establish convergence of this evolutionary/neural system.
[Mixture-of-Depths](https://arxiv.org/abs/2404.02258),
[μTransfer](https://arxiv.org/abs/2203.03466),
[agent scaling v3](https://arxiv.org/abs/2512.08296v3),
[TTUR](https://arxiv.org/abs/1706.08500).

**The immediate issue is screening, not the runtime scheduler.**

`vendor/feat-math/src/evolve.rs::search_surrogate` replaces the learner with ridge
and caps feature parameters. Its implicit assumption is that this score preserves
enough ordering of the original candidates. A ridge solution error bound cannot
justify that assumption: replacing a nonlinear learner changes the learning
procedure. The first useful check is actual-head evaluation of a reproducibly
sampled set of rejected candidates, stratified by learner and task mechanism.
Selection probabilities must be saved; otherwise an audit concentrated near the
cutoff cannot estimate overall false rejection. Compare within identical task,
split, evaluator, metric and fidelity contexts. Include audit computation in the
fixed wall-time comparison and retain exploration promotions.

This is specific to the legacy search path. The inspected parallel teacher runs
candidate evaluation in a separate process with a memory limit and timeout and
reads its actual RMSE. It should not be described as universally using ridge
substitution. A future screening change must identify the exact consumer affected.

The neural regime encoder should be tested with identical feature marginals and
balanced label marginals but different target mechanisms (for example, a balanced
binary grid with `y=x0` versus `y=x0*x1`). These are training/development labels;
prediction-time checks must never inspect evaluation targets. A rank or covariance
descriptor already available in the existing encoder is a strong baseline. A new
embedding is useful only if it improves transfer under mechanism holdouts and pays
for its cost.

**A small falsification experiment was added, without changing production.**

`research/algorithm-kernel-regimes/assumption_probe.py` evaluates the supplied
perturbation bound in exact rational arithmetic. Two diagonal problems have the
same omitted-term norm, `9.99e-7`. In the well-conditioned system the parameter
error is about `1e-6`; in the sensitive system it is exactly `999`. The relative
operator bound correctly distinguishes them. Another fixture has an accurate
approximation despite `eta >= 1`: failure of this sufficient condition means
**abstain**, not that the approximation is necessarily bad.

This is a correctness/counterexample fixture, not a discovered algorithm, a
floating-point certificate, a speedup, or evidence of learned transfer. The
implementation uses exact diagonal solves solely to make the assumptions
independently checkable. A practical solver would need a cheap conservative norm
estimate, rounding-error accounting, an error bound on the required output
(parameter error alone does not establish task quality), and measured fallback
cost. Computing a dense inverse to certify a cheap solve is not justified.

A second probe executes the repository's actual ridge and histogram-boosted
learners on ten paired seeds, using identical feature rows for two different
target mechanisms. The settings were saved in
`research/algorithm-kernel-regimes/screening-manifest-v1.json` before execution.

| Target mechanism | Ridge mean MSE | Histogram boosting mean MSE | Winner across seeds |
|---|---:|---:|---|
| `y = x0` | 0.00012905 | 0.00057611 | Ridge, 10/10 |
| `y = x0*x1` | 0.11208378 | 0.01295267 | Histogram boosting, 10/10 |

Both candidate pipelines collapse to the same ridge surrogate. This makes the
missing distinction concrete: ranking the actual learners requires information
that their identical surrogate cannot express. It does **not** measure campaign
false rejection, show that existing exploration fails, or qualify a replacement
screening policy. The targets' marginals are not matched, so this is also not the
stronger representation test proposed above. The debug-build timings are
descriptive; no throughput claim follows from them. Raw observations and the
summary are saved as `screening-probe-v1.json` and `screening-summary-v1.json` in
the same research directory. Reproduce with the `screening_regime_probe` Rust
example and that manifest.

**The cost comparison also changed one implementation choice.**

The expression microbenchmark fixes ten seeds, four widths, two formulas, 128
rows and eight repetitions in `expression-manifest-v1.json`. It compares scalar
interpretation and register execution from the same source, checks every output
bit, warms both paths, and alternates measurement order. At width 512, hoisting
summaries for `(x-row_mean)/(row_std+0.1)` gives a median ratio of about 56×.
At width 4, register execution of `x+0.1` takes about 1.41× as long as direct
interpretation. Raw measurements and summaries are retained in this directory.

Consequently, the legacy map/derive integration retains direct evaluation for
formulas without row summaries. This is an exact execution choice based on
observable syntax, not a learned validity detector. It demonstrates why the
optimization's applicability matters, without claiming a new research result.
The narrow-row measurements last less than a millisecond per case and are
timing-sensitive. This benchmark is not a frozen historical control, does not
establish the whole-search 5% throughput gate, and does not qualify fast numerics.

**What changes in the upgrade design.**

When an approximate backend, solver or screening policy is introduced, its
qualification receipt should bind the exact reference computation, preserved
quantity, scope, evidence kind, diagnostic inputs, abstention condition and
fallback identity. Formal checker output, empirical calibration and untested
hypotheses need distinct states. A failed monitor must not become a fabricated
fitness score, and a fallback must preserve the original objective and state/RNG
contract. Evidence and measurement metadata belong outside the evolvable fitness
definition. Receipt creation alone does not qualify anything.

The evaluator-guided search precedent in AlphaEvolve is relevant; the recent
self-authored-verification preprint reinforces the need for an independent
acceptance signal in the settings it studied. Neither is a reason to alter the
held-out metrics or expose final tests to an adaptive verifier.
[AlphaEvolve](https://arxiv.org/abs/2506.13131),
[self-authored verification preprint](https://arxiv.org/html/2607.24300v1).

No new production model, learned controller, grammar tokens or remote services
were introduced because of these ideas. The decision is to adopt the sharper
falsification and cost-accounting requirements, then require evidence before
expanding the algorithm or policy architecture.
