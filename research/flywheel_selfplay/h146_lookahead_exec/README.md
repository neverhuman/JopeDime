# H146 — Lookahead optimizer on the r882 adapter continuation

**Status: preregistered; not yet run.** See [PREREG.md](PREREG.md).

- **Arms.** CTRL is the live H140/H144 recipe: AdamW at learning rate 1e-4 and weight decay 1e-4, 40 steps, batch
  16, slate 192, gradient clip 1. LA wraps the same AdamW in Lookahead with k = 5 and α = 0.5. LA_K10 uses k = 10
  and α = 0.5, and was declared before any outcome.
- **Seeds and folds.** Fresh seeds 146101–146103 and family fold seed 146140. Fold 0 is descriptive only.
- **Training scope.** Only the 29,211-parameter adapter trains. Every cell starts from a SHA-pinned r882 copy.
- **Primary.** 91-context flywheel DL-Val, with equal-family gains against r882 and against CTRL. The point estimate
  is the median over 3 seeds, with a 19-family bootstrap 95% CI.
- **Gates.** Holm over LA and LA_K10, the margin, the CI, at least 2 of 3 seeds and family concentration.
- **Also reported.** Elo against r882 = 1500 and invalid (−2) picks.
- **Code.** The executor is copied from merged H144 and the Lookahead state from H93/H122; neither study's files
  were edited. [test_h146.py](test_h146.py) has 23 data-free controls.
