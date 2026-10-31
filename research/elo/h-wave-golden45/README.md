# H-wave Golden45 evaluation input loader

This is the HWV-01 input-validation pre-stage. It does not fit, train, infer,
or score a candidate. The current run is blocked until frozen candidate output
interfaces exist for both H38 `F2_SMART_LM80_STD_SVD32` and GOV-03 h40b.

`loader.py --dry-run` validates only the 91 GOV-03 allowlisted DL-Val
development contexts, their family split, their hash-pinned `nq=0` training
tables, and the selected r782 checkpoint hash. It refuses any other cohort.
The loader for candidate prediction rows accepts only an explicitly declared
method/config and context IDs in that allowlist. It does not invent missing
predictions or turn absent rows into losses.

Example CPU preflight:

```sh
nice -n 10 python3 research/elo/h-wave-golden45/loader.py --dry-run \
  --selected /home/ubuntu/jope-data/gov-03/custody/selected.json \
  --training-tables /home/ubuntu/jope-data/gov-03/custody/tables \
  --checkpoint /home/ubuntu/jope-data/gov-03/custody/round-782.pt
```

The eventual Golden45 evaluation still requires both frozen inference paths,
the benchmark-valid public-development roster and its 45 protocol exclusions,
and a qualification-lock GPU run. This pre-stage's 91-context smoke is not a
Golden45 result and is not independent candidate qualification. No sealed,
holdout, or meta-test input is accepted or opened here.

Synthetic row-loader checks:

```sh
nice -n 10 python3 research/elo/h-wave-golden45/loader.py --self-test
```
