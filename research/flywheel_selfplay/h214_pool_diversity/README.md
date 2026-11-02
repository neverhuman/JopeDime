# H214 — pool diversity at fixed width16

Preregistered design: NATIVE16 vs FPS16-of64, exactly16 scored programs including
the common native retrieved TRAIN parent. TEMP16 is NOT_RUN by the final
coordinator ruling; Holm m=1. Frozen r882, three seeds, DL-Val91/19 families,
zero updates. See [PREREG.md](PREREG.md) and [manifest.json](manifest.json).

The executable adapter reuses merged H212 smoke/evaluator/predictor code.
A separate supervisor enforces the xbabe1 CPU limits and fixed2h deadline.
One real TRAIN smoke at the exact merged head must pass before attempt-001.
The regression tests use synthetic inputs and fake native callbacks only.

No scientific attempt or H214 utility readout has occurred at registration.
The two unchanged-main hygiene failures are explicitly nonblocking under
hw_t0230_res08. Final utility, bootstrap CI and Elo remain unmeasured.


## Prelaunch source-pin correction (2026-11-02 11:30Z)

The first supervisor refused before creating a worker, reading model/cohort data,
or consuming a scientific attempt: the registered H212 runner bytes had changed
on main. Bind H212 commit `c39a96589f340e3d8faa833840e8ddf224aed2a5` (#1788),
whose only runner change removes duplicate scratch-directory creation in its
unused `run` entry point. The reused smoke/evaluator/predictor helpers and analysis
are unchanged. Verify the committed and working runner hashes, pin analysis before
import, and read the original input manifest from that immutable Git commit.
Later H212 S3 receipt metadata does not change H214's registered input closure.
A positive data-free regression now loads the real upstream helper and manifest;
negative controls require refusal on runner/analysis drift. No estimand, seed,
width, arm, model, evaluator or input identity changes.

The conservative original campaign clock remains 11:08:24.326111Z to
13:08:24.326111Z on 2026-11-02, including this repair. No budget reset.
Scientific attempts consumed and utility readouts remain zero before real smoke.
