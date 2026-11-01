# H163 — Accepted adapter prediction ensemble

H163 tested prediction-level cost ensembling of read-only accepted checkpoints R782, R785, R859, and R882 against the family-weighted out-of-fold utility of R882. The frozen protocol is in [PREREG.md](PREREG.md); compact results are in [metrics.json](metrics.json).

## Outcome

**Complete — NO_ADVANCE.** The exact 91-context R882 DL-Val gate passed first: 91/91 matches, zero maximum context error, and mean `1.1377941898164525`. On the 1,219-context / 205-family TRAIN cohort, none of ENS2, ENS4, ENS4_W, or RANKAVG4 passed the strict selector on any of the 15 outer-fold/seed decisions. The selected policies therefore matched R882 for all assessed contexts and seeds: median equal-family gain `0.0000`, shared-family median-seed bootstrap 95% CI `[0.0000, 0.0000]`, Elo `1500.0`, and zero changed choices. Each alternative is `NO_ADVANCE`.

Every selected DL-Val policy also had mean utility `1.1377941898164523` (R882 reference `1.1377941898164525`), Elo `1500.0`, and zero changed choices. The 17 TRAIN `-2` outcomes and zero DL-Val `-2` outcomes were unchanged from R882.

The three seeds randomize selector/bootstrap procedures only. The prediction panels are deterministic and repeated across seeds, so the three seed results are not independent prediction samples. The selector’s zero admissions explain why the final selected policies reproduce R882; they do not establish that every unconstrained ensemble score vector is identical to R882.

## Execution and audit

One CPU-only xbabe2 attempt ran at nice 19. It recorded 5,240 model forward calls and zero optimizer updates. The persisted audit receipt lists 1,335 opened paths, zero refused paths, and zero protected reads; it binds source commit `221fa753ada0094300b8b9587cba20345cc47407`, the manifest, pinned checkpoints, DOPE composite, frozen tensors, and review status. The monitor completed with peak RSS about 5.51 GB, three process-tree threads, peak load 79.25 on 102 visible cores, and all terminal resource guards satisfied.

Raw prediction panels and per-fold selection locks remain outside Git at `/home/ubuntu/jope-data/h163/attempt1`; attempt custody and resource evidence are at `/home/ubuntu/jope_plan/evidence/H163/attempt1`.

## Interpretation

R882 historically saw these 1,219 TRAIN contexts. H163 is conditional continuation evidence, not unseen-family generalization. No serving change, checkpoint promotion, or flywheel promotion follows from this result.
