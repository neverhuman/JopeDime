# H152 — zero-update checkpoint soup of saved flywheel adapters

**Status: preregistered; not yet run.** It runs only after a follow-up review PASS and the merge. See
[PREREG.md](PREREG.md).

- **Arms.**
  - MID = 0.5·R782 + 0.5·R882 (primary).
  - SOUP4 = the mean of R782/R785/R859/R882 (secondary).
  - A025 and A075 (exploratory).

  Only the 29,211 trainable adapter parameters are averaged; everything else comes from R882.
- **Primary.** 91-context DL-Val equal-family gain against R882, with three 19-family bootstrap seeds. Holm over
  MID and SOUP4, and ties reject.
- **Also reported.** Elo against R882 = 1500. TRAIN panel and family-disjoint TRAIN folds, both descriptive and
  TRAIN-exposed.
- **Code.**
  - [runner.py](runner.py) imports the merged H139 evaluator byte-pinned. It holds Torch at 1 thread, binds the
    executed bytes to the reviewed head and the merged source, and stops on disk, load, scratch or nice.
  - [monitor.py](monitor.py) is the supervisor. It stops on process-tree threads > 4, RSS ≥ 6 GB, scratch ≥ 1 GB,
    disk ≥ 95% used or < 40 GB free, or load > 85.
  - [test_h152.py](test_h152.py) has 13 data-free tests, including negative controls for every guard and for the
    binding.
  - [preflight.json](preflight.json) is the supervised smoke: soups built, R882 only scored, exact 91/91
    regression.
