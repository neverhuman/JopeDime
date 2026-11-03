# H239 — parameter-subset continuation

**PREREGISTERED. No H239 outcomes or training yet.**

Hypothesis: restricting adapter updates to biases/norms or the final scoring head, or decaying LR by depth, reduces harmful drift and improves DL-Val91 over an r882 copy.

[Preregistered protocol](PREREG.md), [tensor census](TENSORS.md), [exact membership](trainable_tensors.json), [manifest](manifest.json), [metrics schema](metrics.schema.json).

CONTROL: 47 tensors/29,211 parameters; BIAS_ONLY primary: 23/755; HEAD_ONLY: 2/25; LLRD: 47/29,211. Three seeds, 15 cells, 2,400 logical updates. Native91 first, TRAIN smoke, then full wave. DOPE frozen. A1 cooperative memory guard and at most two bounded resumes apply.

Outcome, metrics and verdict: NOT_RUN. Next: merge this preregistration, pass admission, execute the fixed study, and close by 08:00Z Nov 3.
