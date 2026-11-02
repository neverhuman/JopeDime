# H225 — multi-step candidate generation and continuous regret

Status: **PREREGISTERED / NOT_RUN**. Owner: sol_v1_dsc01.

The hypothesis is that larger adapter proposals move beyond the r882 top-1
plateau and TRAIN inner-family continuous regret selects them more reliably
than discrete utility. MSCR is primary; MSTOP1 isolates the selection score;
MSCR_SHUF is the within-family selection-label placebo; r882 is the reference.

The fixed eight-proposal grid spans 8/32 steps and 0.5/1/2 times the r882 learning
rate plus two independent minibatch replicas. Three seeds and five fresh outer
family folds give 120 outer proposals, 240 inner fits, 360 total fits and 8,280
updates, plus a two-update TRAIN smoke. Details, strict selection fallback,
input pins and family assignments are in [PREREG.md](PREREG.md),
[manifest.json](manifest.json) and [fold_manifest.json](fold_manifest.json).

Primary: pooled-A native gain LCB95>0, all three seed gains positive and
per-seed harm<=10%. Co-primary: DL-Val91 context-mean gain>+0.002 with positive
family-bootstrap lower bound. No candidate outcome, CI, harm, Elo, proposal
diversity count or verdict has been measured. r882's Elo reference is1500.

Next step: merge the locally tested preregistration, bind its merged source,
archive the prereg metadata to S3, pass the bounded smoke, then launch the single
CPU attempt. The supervisor pauses the owned worker on load5>60; compute stops
at19:00Z and closeout is due19:30Z on2026-11-02. Run data stays in
`~/jope-data/h225`; the live checkpoint and other studies stay outside its writes.
