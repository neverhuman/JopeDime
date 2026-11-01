# H152 — zero-update checkpoint soup of saved flywheel adapters

**Status: preregistered; not yet run.** See [PREREG.md](PREREG.md).

- **Arms.**
  - MID = 0.5·R782 + 0.5·R882 (primary).
  - SOUP4 = the mean of R782/R785/R859/R882 (secondary).
  - A025 and A075 (exploratory).

  Only the 29,211 trainable adapter parameters are averaged; everything else comes from R882.
- **Primary.** 91-context DL-Val equal-family gain against R882, with three 19-family bootstrap seeds. Holm over
  MID and SOUP4, and ties reject.
- **Also reported.** Elo against R882 = 1500 and a descriptive TRAIN panel.
- **Code.** [runner.py](runner.py) imports the merged H139 evaluator byte-pinned. [test_h152.py](test_h152.py) has
  6 data-free controls. [preflight.json](preflight.json) is the zero-eval smoke.
