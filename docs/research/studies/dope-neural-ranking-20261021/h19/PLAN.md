# H19 — qualify the next neural-state experiment

Objective: determine whether H18's independent candidate-score correction box
or its full-slate ranking loss is a plausible bottleneck. H18 is complete and
unchanged. This is a new TRAIN-only diagnostic, with no model fitting.

Read first: RTK.md, current human instructions, CORRECTED_PLAN and amendments,
RESEARCH_LOOP, GIT_RESEARCH_HANDOFF, H18 critical report and separate later replay,
NEXT_MECHANISM_REVIEW.txt, and INDEPENDENT_GEOMETRY_REVIEW.md when present.
Root owns producer/protocol/launch/results/publication; the independent reviewer
owns verify_geometry.py. The persistence and S3 packets have disjoint write
scopes. Product/Dope source, existing experiments, their outputs, CI, runner
configuration, serving policy and other owners' files are read-only.

Scope: six frozen, verified H18 non-state bases, two folds and three seeds.
For each model read only its own fold's original TRAIN slates (FIT/SELECT roles),
matching TRAIN-only preprocessing and fixed score-scale. Do not read the model's
opposite-fold pools, CAL, OLDDEV, reservations or outer outcomes. Both folds'
TRAIN unions cover the existing850 contexts/162 components; this is no new
independent validation or S3 admission. Reuse archived outcomes; zero GP,
baseline, Dope, neural or synthetic optimizer updates.

Compute policy: no deadline, RuntimeMaxSec, wall budget or time-forecast refusal.
Use one idle owned GPU,4CPU,8GiB RAM, swap0,64tasks, CUDA30%, RAM4GiB/VRAM2GiB
reserves, temperature<85C, hostCPU<70 and evidence<4GiB. Stop for resource
ownership/health failure, source/hash/role drift, invalid arithmetic or mismatch;
preserve failed receipts. Do not change scientific settings to repair runtime.

Before numerical execution, freeze producer/verifier/source hashes and protocol,
append evidence claim under the product coordination lock, prove no active
overlapping numerical worker/GPU process, and read actual source versions.
Current native Dope55fb1313 remains fixed; no extraction/refitting is needed.

Producer: replay full TRAIN slates under eval/inference_mode and FP32, preserving
canonical original program-SHA ordering. For B=float32(.1*TRAIN-score-scale),
compute upper=fl(base+B), lower=fl(base-B). The independent-box winning envelope
contains candidate j when its upper exceeds max(lower), or equals it and its
canonical index is no later than the first maximizer of lower. This is a safe
box envelope, not the set the shared adapter weights can necessarily realize.
Also record the closed real gap<=2B proxy. FP32 ties must not be treated as
real-arithmetic strict exclusions. Check CUDA/NumPy endpoint arithmetic.

Measure envelope size, top32 omissions/extras, top1/top2 gap, probability mass
outside the envelope, and L1 score-coordinate ranking-gradient share outside it.
Gradient uses centered regret and includes minimum-CE only for nonconstant,
nonsingleton slates. A zero-gradient fraction is reported as0 with an explicit
zero-gradient flag/count; this is not a parameter-gradient attribution.
Record base raw fitness, full-slate and envelope oracle minima/headroom, base
failures and whether any global-minimum tie is in the envelope. Oracle
headroom is an upper limit, never achieved lift. No recipe/K/bound is tuned here.

Save six full private score/source arrays, per-query geometry, immutable execution
and protocol receipts, hashes, progress, source/monitor controls, results and all
failures. Compare all six TRAIN32 selected-fitness panels and fixed score-scales
against original receipts. Independent NumPy QA recomputes box/ties, gradients,
source/role bindings and equal-component aggregates, without Torch or fitting.
Completion requires2550 model-query rows,850 distinct contexts,162 components,
all six monitor checks, source hashes and independent QA. Report partial/failure
literally if any requirement is unmet. Record measured costs without time caps.

Decision after this TRAIN-only diagnosis: if the box excludes most useful
TRAIN alternatives, prioritize a separately frozen correction-capacity control
before shortlist-loss changes. If it preserves meaningful TRAIN headroom while
the full loss spends score-gradient outside it, prioritize a loss on the exact
box envelope rather than arbitrary top32. A follow-on experiment still needs
matched trained/random controls, complete full-pool evaluation, TRAIN-only
selection and independent replay. Never select a favorable H18 seed/checkpoint.

Research basis: Mandi et al., ICML2022,
https://proceedings.mlr.press/v162/mandi22a.html motivates solution-subset ranking
losses, not this specific box or a Dope benefit. PyTorch documentation requires
eval mode separately from inference_mode:
https://docs.pytorch.org/docs/stable/notes/autograd . These motivate controls;
the mathematics and measured evidence determine this diagnosis.

Persistence: follow PERSISTENCE_AND_CHECKIN_PLAN.md. Create a compact public
study results projection and source-bound research Git branch; keep weights,
source identities and candidate arrays outside product Git. Product import
remains queued behind foreign ownership/PR411. Update dashboard and existing
30-minute heartbeat with blocked status, actual S3 admissions/next source gate,
and the highest-value next experiment/branch decision after completion.
