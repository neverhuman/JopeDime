# H71 — W5 parent-child value and bounded evolutionary lookahead

**State: gated CPU preparation only.** This directory records the W5 design
and an offline plan validator. It does not read datasets or outcome records,
build folds, fit or score a model, or run any study cell. No result, Elo, gate
outcome, or advance decision is claimed.

## Authority and prerequisites

The authority is `origin/main` at `35f49f70ce7011adcf800876972c677efa3b9a36`,
file `research/flywheel_selfplay/rd_breakthrough/PROPOSALS.md`, SHA-256
`f0526f20699dea3fe5259d10eee30483ac3c52655249906fd1ce79dd8e193e50`. W5 is
ordered after W1, W2, W3, W4, and W6. It is gated on an independent provenance
audit: admitted exact same-context, same-evaluator parent-child pairs must
carry verified operator and parent IDs and original unrounded outcomes. Do not
infer parentage from similar ASTs. The current prep does not assert that such
pairs exist or that the audit passed. No fitting, pairing, fold construction,
S3 access, or upstream outcome import is authorized by this skeleton.

The runner is a design-plan validator only. Its output always says
`execution_admitted: false`, `provenance_status: not_checked`, and zero fits.
Any later data-backed work needs a separate owner-approved continuation after
the provenance audit passes, a refreshed source and input freeze, and a valid
resource/lock slot. The W5 ceilings remain 2 GPU-hours, 4,800 combined native
calls, and four CPU-hours; a ceiling is not a budget authorization.

## Common experimental contract: advance gate (verbatim)

> The proposed advance gate is a positive adjusted family-mean lower bound above **+.001**, median family delta ≥0, and at most 10% of families losing more than .01, against the strongest matched control. A **+5 Elo** point gain is a practical target, not a forecast or substitute for raw utility. For search-primary waves also report legacy selection; a search-only gain is called a search gain. Before launch, a frozen fit-only pilot estimates detectable-effect curves and runtime. If the budget cannot identify the intended contrast, report an informative interval or defer the study; low power is not proof of a null. The historical live 1.13779 is a repeatedly selected incumbent value, not an independent target distribution. Rejection counts do not establish a capability ceiling, and no wave is judged solely by exceeding that scalar. No development gate authorizes serving promotion.

## W5 — Parent-child value and bounded evolutionary lookahead (verbatim)

- **Hypothesis:** a context-conditioned transition critic can select useful edits better than absolute child ranking, and a cheap screen can allocate full evaluations more effectively.
- **Arms:** r882 absolute-cost ranking, program-feature GBM transition baseline, neural parent-child head, and parent/operator-shuffled placebo. Primary is neural versus GBM transition at the same parent/proposal set; a separately declared screening contrast holds the selector fixed. Do not combine both changes in one unidentifiable arm.
- **Protocol:** seeds 701–703, five family folds, 40 fixed updates on exact same-context, same-evaluator parent-child pairs. Operator/parent IDs and original unrounded outcomes must join; do not infer parenthood from similar ASTs. For prospective search, four full-fit opportunities/context; screening gets a fixed share of the same CPU-time allowance. Full-evaluate a randomized sample of screen rejects to measure missed improvements. Count all screen and full fits, failures and wall time.
- **Metrics / success / kill:** Elo and raw utility after verified search, confirmed improvement per fit and per second. Advance on the common gate and ≥20% more confirmed improvements per total evaluation cost. Kill if shuffled controls match, pair coverage is inadequate, or apparent gain is entirely extra compute. The historical 511/512 confirmation rate is selected precision, not a recall estimate.
- **Data / budget:** admitted training pairs from existing custody or a separately prepared clean export under the approved S3 prefix; absent provenance blocks fitting. Any strat300 tables must prove they use only the target's original training rows. ≤2 GPU-h, ≤4,800 combined native calls, four CPU-hours.
- **Support:** I-A4, GT-AR6 and I-AR4; existing transition head is unused by the current fused objective, but wiring it is not itself a quality result.

No H71 code creates model behavior or changes the W5 protocol. Future execution must preserve the 1,219 DL-Train context / 205-family closure, original labels and row envelopes, alias-resolved five-family folds with seed 4900, experiment seeds 701–703, frozen r882 continuity control, and matched fold-excluded control from the common contract. Zero new DL-Val, outer-test, or sealed reads are permitted.
