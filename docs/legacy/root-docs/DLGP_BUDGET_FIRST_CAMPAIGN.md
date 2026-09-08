# Budget-first DLGP campaign

This campaign targets at least 7 robust wins on the frozen 22-dataset
development cohort. Development is explicitly benchmark-adaptive across
rounds. The disjoint 20-dataset confirmation cohort is the only cohort on
which the protocol may be described as frozen before campaign test opening.
The public benchmark was not historically unknown.

The machine-readable preregistration is
`manifests/dlgp-budget-first/protocol.v1.json`. The existing six
`jope_dlgp_best5` commands remain unchanged; the budget-first contracts are a
layer around their artifacts, not a seventh public command or a production
pointer path.

## Preserved round-2 evidence

The 88 round-2 results and configs, 440 cached sealed scores, verdict,
baselines, priors, proposal manifest, job specs, train bindings, binary digest,
and exact strategy-screen source were promoted from temporary storage to:

```text
artifacts/dlgp-budget-first/round2/
  62d250dd6d9600b526ec9513eaa44bfd01b007b060feaf6622d02daec4cd92f1/
```

`manifest.json` records the payload identity, every file SHA-256 and byte
length, the pinned `8cd093…` worker binary, the source revision, two independent
hash implementations, `test_csv_read=false`, the replayed 3-win control
verdict, and source retention through 2026-09-06. The temporary originals were
not deleted. Reverify with:

```text
python3 scripts/dlgp_budget_campaign.py verify-round2 \
  --bundle artifacts/dlgp-budget-first/round2/62d250dd6d9600b526ec9513eaa44bfd01b007b060feaf6622d02daec4cd92f1
```

## Frozen experiment ladder

1. Reproduce the complete round-2 evidence and the 1,000-evaluation control
   prefixes for `471f11cc`, `4f68d277`, and `b5beff4a`, seeds 101 and 202.
2. Run the unchanged control on the three current winners and seven frontier
   losses. Every run checkpoints at 1k, 2k, and 5k. Continue to 10k only when
   at least four CV7 folds improve RMSE by 0.1%, or top-20 archive Jaccard is
   below 0.8 while mean CV7 RMSE is no more than 0.5% worse. Total search work
   is capped at 200,000 evaluations.
3. Only after the checkpoint and launch gates pass, run the protected Best5
   screen on `4f68d277` and `b5beff4a`: six arms, two seeds, exactly 24 jobs at
   the existing 1,100-operation ledger. At most one non-control mechanism may
   advance, and only after the control matches the reproduction envelope.
4. A Best5 survivor is compared against control on the other five frontier
   losses. It must win at least 7/10 seed-dataset pairs, improve median
   log-RMSE by 1%, and retain every existing robust win. There is no 240-job
   expansion without this gate.
5. Heterogeneous residual screens use `c7d81571`, `c32fa09b`, `24713c7f`, and
   `b5beff4a`, 2,000 GP evaluations, and two seeds. Production-compatible and
   research-only tracks are evaluated separately. Each external base is
   recreated inside every outer-training partition; sparse weights and eta
   are learned there. Full-train predictions are forbidden before selection
   freezes.
6. Surviving tracks expand over all seven frontier losses through the same
   5k/10k ladder. A final track must keep the three existing wins and reach at
   least 7/22 robust wins.
7. Freeze at most one protocol per license track, then run it uniformly on all
   20 confirmation datasets with fresh seeds 303 and 404. Every dataset
   selection freezes before its test bytes are opened. There is no
   post-confirmation retuning under the same campaign ID.

Robust win means that both frozen seeds are strictly below the hash-frozen best
non-ASPFM RMSE. `bd49e740` remains in the development denominator and is
unbeatable by construction because its comparator RMSE is zero.

## Contracts and isolation

`BudgetLadderBundleV1` binds populations, selections, receipts, prediction
roots, CV7 values, archive semantics, accounting, and every nested prefix.
Resume, retry, cold-cache, and warm-cache artifacts must carry the same prefix
roots. `BudgetStageInventoryV1` enforces the exact 10×2 Stage-1 grid and the
200,000-evaluation cap.

`FrozenPredictorV1` can bind a GP pipeline or hash-pinned external components,
including inference protocol, upstream revision, license text identity, and
declared license track. TabFM, TabPFN, TabICL, and TabDPT are research-only;
CatBoost, LightGBM, XGBoost, and GP control are production-compatible. Artifact
roots are disjoint, and a research component is rejected from a production
predictor.

Search-side dataset bindings contain dataset, training, and metadata hashes,
but no test path or hash. Only `SealedTestInventoryV1` may carry test bindings.
Nested OOF receipts reject row overlap and duplicate-feature atomic groups
that cross an outer fold. Full-train external predictions require a prior
selection-freeze hash.

## Launch and storage gates

The hard disk floor is exactly 2% (`20_000` micros). Below 5% is a warning,
not a blocker. Rendered SCQ specs carry the expected disk floor and project
weight. Submission must call the project-weight guard immediately beforehand;
drift aborts and is never silently repaired.

Launch remains blocked until the exact proposal checkpoint, fresh evaluator
and supply evidence, reviewed acknowledgement, a real `scq` executable, and a
matching observed project weight are all present. All rendered specs must pass
`scq check`; rendering itself has no submission path.

`scripts/dlgp_safe_cleanup.py` accepts only exact rebuildable target paths,
checks for active build/evaluator/SCQ processes, verifies the round-2 bundle
before and after, and writes a pre/post cleanup manifest. Old logs require a
typed explicit allow-list and age. `/home/ubuntu/TO_DELETE`, Codex history, and
old `.jain-scq` backups remain quarantined and are never automatic targets.
Model/checkpoint caches, SOTArena data, Golden42 evidence, sources, `.git`,
referenced release artifacts, and round-2 scratch evidence are protected.
