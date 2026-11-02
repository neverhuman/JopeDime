# H208 — harm-constrained nested blend

**NOT RUN; preregistration only.** Owner: rd_astra / [ASTRA].

[PREREG.md](PREREG.md) reacts to H200's NO_ADVANCE result: add a5% FIT-family harm
constraint to nested lambda selection, and explicitly disable early stopping.
A repaired unconstrained control shares the same fresh60 fixed-iteration fits.
H203's per-context gating study remains separate; H208 drops the rank arm.

[manifest.json](manifest.json) pins the parent evidence, inputs, seeds and
[fold_manifest.json](fold_manifest.json). Native r882 reference remains
1.1377941898164525; no H208 gains, harm, Elo or verdict exist yet.

Plan: one4-hour xbabe3 CPU attempt after merged preregistration, reviewed executor
and fresh resource admission, including disk<93.6%. Executor is not implemented;
this PR launches nothing. If CPU admission fails, queue per owner direction;
no GPU is claimed and existing H204–H207 allocations/requests stay intact.
