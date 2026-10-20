# Three-branch research campaign

This keeps the completed native-feedback trial and adds two child concepts. The
first wave is **historical S3 replay on actual frozen native/Kumo features**. It
can test ranking and acquisition hypotheses, but cannot establish fresh raw-data
GP lift or production self-improvement. Original results and failures stay intact.

| Branch | Concept | Assigned execution |
| --- | --- | --- |
| `codex/native-feedback-upgrade-20261020` | Composite ranking control and standard result contract | Paired control on each admitted node |
| `codex/research-listwise-20261020` | Top-weighted, tie-group listwise loss | xbabe2 GPU only |
| `codex/research-uncertainty-20261020` | Parent bootstrap ensemble and Thompson acquisition | xbabe3 GPU only |

Both children start from the current branch's new campaign foundation, which
descends from `0cc177c3ba34f69e4ed94e9bb03aca2d69741c31`. Creating branch refs
does not create an extra checkout or claim another owner's GPU. One writer uses
the existing source capsule; bundles deploy exact source outside the product
checkout. PR387 remains with its existing owner and is the only integration PR.

## Scientific rationale

Ranking surrogates are relevant to choosing the best programs, rather than
predicting absolute scores. The 2025 [Ranking Trick](https://proceedings.mlr.press/v293/vazquez25a.html)
and [Ranking over Regression](https://arxiv.org/abs/2410.09290) motivate testing
this distinction; their results do not imply a JopeDime improvement. The second
concept tests whether a [deep ensemble](https://arxiv.org/abs/1612.01474) with
parent bootstrap and posterior sampling chooses more useful feedback than the
existing deterministic mean-plus-standard-deviation rule. This is a research
adaptation, not a reproduction or a claim of calibrated Bayesian uncertainty.

Read `../README.md`, `../NEXT_WAVE.md`, canonical AGENTS/MASTER_PLAN/claims and
current Admiral decisions first. `protocol.py` defines immutable standardized
results; `runner.py` performs the bounded CUDA study only after external resource
admission. `admission.py` is a **readback gate, not a launcher or governor**.
Its source and the actual external route require independent review before use.
It creates no unit, guard, root staging, process, signal or GPU allocation.

## Frozen first wave

- All inputs bind compressed S3 archive, decoded/member provenance, features,
  original split, source, four native control predictions and both checkpoints.
- Train on the original 713 training contexts. Use the previously unused 209
  calibration contexts as a **development** assessment; do not evaluate or select
  on the old 214 assessment contexts. Both native models historically saw
  all contexts as TRAIN, so this is not unseen-family backbone generalization.
- Each concept has continuation, uniform feedback and targeted feedback arms;
  same three member seeds, initial state, optimizer, contexts/updates and logical
  outcome budget. Run the composite-control concept with the same protocol on
  each node before its candidate. Charge repeated control computation explicitly.
- Start with two uniform cached TRAIN outcomes/context, then three waves of two
  additional outcomes and 192 updates per member/wave. Complete sequences and
  contexts stay intact; microbatch accumulation preserves the effective batch.
- Primary: uniform-feedback minus continuation parent-mean normalized quality
  at budget4, with a 99% parent bootstrap interval and a practical margin0.005.
  Across concepts, candidate versus composite is a separate paired secondary
  comparison only when exact evaluation identities match. Seeds/members/folds
  are not independent samples. Do not select checkpoints on development scores.
- Report teacher CE, composite training objective, selection regret, incumbent
  beating rate, useful throughput, CPU/GPU costs, actual allocated/reserved peak
  VRAM and failures as separate quantities. No fresh GP fit or HWM promotion is
  inferred from replay; `fresh_gp_fits` stays0.

## Execution and resource ownership

The latest human request authorizes this new research source/campaign. It does
not revive expired operational-helper phases, an unadopted inert candidate or
another owner's R4 job. xbabe1/R4 and xbabe0/Jeryu remain outside this campaign.

At source publication both GPUs are idle with **resource qualification pending**.
The existing Admiral/resource owner must supply an exact current protected job
admission, actual independent control-tree/continuous CPU70 qualification,
assigned CPU cores with sibling conflicts resolved, kernel CPU/RAM/swap/tasks
bounds, storage/temperature/VRAM guards, whole-job accounting and owned-only
custody. A root-owned JSON file alone is not proof; its source-bound artifacts
must substantiate the actual qualified mechanism. No new provider/capacity.

On each node, one admitted numerical job at a time; no CPU-heavy parallel
preprocessing. Keep CPU<=70% total host with margin, target60/hold65 and external
1-second peak accounting. Stop/reduce only this job if foreign work rises. Node3
currently has about19GiB root free and remains on storage hold: choose an existing
approved filesystem with headroom; never delete foreign data to make room.

Tune microbatch sizes on TRAIN inputs before outcomes, with a fixed effective
batch and at least2GiB VRAM reserve. Record useful examples/second and both CUDA
allocator peaks. Do not reserve memory merely to fill a GPU. Failed/OOM attempts
remain charged and immutable; no automatic retry or undocumented batch change.

The executable shape after actual admission is:

```
python3 -I runner.py --plan /absolute/frozen-plan.json \
  --inputs /absolute/exact-restored-inputs --out /absolute/new-run \
  --grant /absolute/owner-protected-current-grant.json
```

Do not execute that shape from a template grant. Before launch freeze `source`
and input hashes in the plan, independently review this source and the external
resource route, then bind the plan digest in the owner's grant. The script
refuses CPU fallback, reused output directories and missing/stale admissions.

## Continuous research cadence

The existing20-minute heartbeat owns the three-branch registry. On each check,
verify actual optimizer-step/weight/checkpoint change, loss denominators, guard
freshness and costs. A running process or GPU occupancy is not sufficient.
An idle GPU has an exact next admitted job or a named owner/blocker.

At termination, preserve complete or failed raw output/checkpoints to private
exact-version S3, fresh-GET/restore every member, validate `result.json`, and
compare only identical `comparison_identity` objects. Publish a research card
including negative/inconclusive findings and propose the next frozen hypothesis.
Development reuse must be counted as selection exposure. Before another wave,
freeze a new immutable plan and renew exact job/resource ownership; do not change
an old result, extend an old deadline or silently rerun an exhausted experiment.

Fresh GP/native backbone training is a subsequent wave after current TRAIN
rights/raw S3 VersionIds, authenticated native cut/evaluator/vocabulary/incumbent,
teaching admission, selected native runtime and matched cost protocol close.
Neither this historical development study nor the original adapter result closes
those dependencies. Production promotion always needs a separate decision.
