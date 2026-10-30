# H09 CPU experiment — no lift established

The DSL-preserving conditional residual has a worse point estimate than its fixed tree
baseline on the development validation packet. The paired central 97.5% interval crosses
zero; no improvement or independently confirmed harm is established. No further arms,
regularization search, or validation reruns followed the terminal evaluation.

| DL-Val result (lower selected constrained fitness is better) | Value |
| --- | ---: |
| Tree baseline | -0.789990439 |
| Tree + conditional residual | 1.289807976 |
| Lift, baseline minus residual | -2.079798415 |
| Paired whole-family 97.5% CI | [-5.329693987, 0.119209401] |
| Validation contexts / connected family components | 91 / 19 |

The predeclared DL-Train partition used 729 tree-fit, 220 residual-calibration and 270
regularization-selection contexts. Alpha=100 was selected solely on training families.
The archived complete candidate pools were ranked without measured-fitness or gate
filtering. Full 412-column DSL main effects were retained, with an additional 16×8
conditional interaction block. Context summaries came only from delivered train-only
tables. This tests a non-neural H09 residual; it does not claim a DOPE representation
advantage or reconstruct an unavailable historical implementation.

This is **DEVELOPMENT_ONLY** evidence: the validation families were used by earlier waves.
It performs zero new native GP fits and supplies no new outer-test performance. The
positive-CI criterion for HWV-01 Golden45 scoring was not met. Regression Elo is
unavailable; binary and multiclass were not evaluated. No synthetic scientific scores,
proxy Elo, serving claim, or model promotion is made.

Reproduce the implementation checks with `pytest research/h-followups/h09 -q`.
[PROTOCOL.md](PROTOCOL.md) fixes the comparison before fits; [RESULTS.json](RESULTS.json)
records source, plan, models, inputs, software versions, per-family effects and receipt
hashes. Scientific source: `1dfa5043459d43f7e01520421867aa37a7aaeefe`. The already executed plan
and one terminal result are under `~/jope-data/hwv-04/run-v1/`; logs and CPU receipts are
under `~/jope_plan/evidence/HWV-04/local/`. Keep those immutable, including every
context's selected program and constrained fitness, and the serialized fitted scorers.
A replay of the same terminal data is not a new independent evaluation.

Run commands (original output directory must not be reused):

```sh
# Set CPU/thread/resource safeguards described in PROTOCOL.md first.
python3 research/h-followups/h09/run.py plan --output ~/jope-data/hwv-04/new-run
python3 research/h-followups/h09/run.py train --output ~/jope-data/hwv-04/new-run
python3 research/h-followups/h09/run.py score --output ~/jope-data/hwv-04/new-run
```

The source and protocol hashes, catalog/metadata joins, model digest and DOPE checkpoint
identity are checked between phases. Receipts are exclusive-create; a partial or failed
scientific attempt cannot silently restart or overwrite a terminal result. All phases
ran with CUDA hidden, nice 10, eight math threads and a 30-minute ceiling; actual train
and score times were 37.05s and 6.01s, with sampled host CPU peaks
22.6% and 17.4%. The copied checkpoint and actual DOPE composite
`c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814`
matched before and after. There were no neural updates/forwards, holdout/sealed/BeyondArena
payload reads, GPU jobs, live-loop edits or remote storage writes.
