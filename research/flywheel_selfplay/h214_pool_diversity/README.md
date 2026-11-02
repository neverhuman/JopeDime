# H214 — pool diversity at fixed width16

**CLOSED INCOMPLETE.** The single TRAIN smoke failed during input setup before
model instantiation, inference, proposal generation or candidate evaluation.
No H214 efficacy verdict is available. Median gain, bootstrap CI, Elo, pool
diversity and oracle-best utility are unmeasured (`null` in [metrics.json](metrics.json)).

The hypothesis was that FPS16 selected from64 native proposals improves frozen
r882 selection over NATIVE16 at equal evaluated cost: exactly16 programs including
the common family-disjoint TRAIN parent. Preregistered seeds214101–214103,
DL-Val91/19 families, equal-family means and median over seeds,10,000 shared
family bootstrap draws, strict LCB95>0.001546, Holm m=1 and ties REJECT.
TEMP16 remains NOT_RUN by the final coordinator ruling. See [PREREG.md](PREREG.md)
and [manifest.json](manifest.json) for the frozen protocol and input identities.

## Execution and failure

Preregistration PR#1808 merged as543aa33a. Its first supervisor refused before
worker creation because H212 runner bytes had advanced on main. PR#1832 bound
reviewed H212 commitc39a9658 and the original input manifest without changing
scientific definitions. Source4cbb66807a08124e89c5350361f68e8cc71fb699 passed30
data-free tests, Ruff/format/path/diff and all five required records gates.
Hosted CI was explicitly waived, not passed; the two known unchanged-main
hygiene failures remain disclosed in the PRs. These checks did not qualify the
full runtime setup path.

On xbabe1 at11:34:22Z, supervisor543018 launched smoke worker543049, CPU only,
nice19 and four-core affinity with single-thread inner math. Setup failed at
`h139.roster`: H214 omitted H212's `h139.ARM_DIR = REFERENCE_DIR` relocation,
so it sought absent `~/jope-data/h139/inputs/best_dev_checkpoint.json`.
The approved H195 reference already exists and matches SHA30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a.
This is an adapter input-path failure, with no measured comparison between arms.

There was one preworker refusal and one smoke-worker launch; zero primary
attempts, zero H214 utility readouts and zero optimizer updates. `attempt-001`
was never created. Both processes stopped. No smoke retry or primary launch
followed under the single-smoke instruction. The conservative original campaign
clock11:08:24.326111Z–13:08:24.326111Z on2026-11-02 was never reset.
The detached launcher's process exit status was not retained; the raw worker
INCOMPLETE receipt and traceback are retained. No missing result counts as PASS.

The frozen r882 copy and DOPE checkpoint file still match their registered hashes.
A post-run DOPE tensor-composite check was NOT_RUN because no model was instantiated;
file identity is not claimed as completed inference/freeze qualification.
No protected rows, outer rows, GPU, live loop or H210 processes were used.

## Evidence and next step

Raw failure and launch receipts remain at
`xbabe1:~/jope-data/h214/preflight-4cbb66807a08/`.
Controller custody: `/home/ubuntu/jope_plan/evidence/H214/res08-t0200/SMOKE-CLOSEOUT.json`
and its `smoke-failure-4cbb6680/` raw copies; [metrics.json](metrics.json) binds hashes.
The frozen source, prior pin-refusal receipt and both launch controls are retained.

The next step is a separately authorized prospective reference-path fix and a
source/input-only setup preflight before registering any new execution. This
closeout makes no scientific ADVANCE or NO_ADVANCE claim and authorizes no retry.

## Versioned terminal evidence custody

The allowlisted terminal evidence is published at
`s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H214/terminal-67e25d74535c-105e24b035cd94b0/H214-terminal-manifest.json`,
VersionId `ZgyLPLG.9ot3uGweg46CvUavdAgy3qOJ`, SHA-256
`dcb079443c89701c5114d1b74c033d71f716558a841665abec145bff2d0cebc1`. The manifest binds the archive's exact
VersionId and all 48 member names, sizes and SHA-256s; [manifest.json](manifest.json)
and [metrics.json](metrics.json) retain both versioned object pointers.
Both objects were uploaded with create-only `If-None-Match: *`, downloaded using
the returned VersionIds, and checked byte-for-byte; every archive member hash matched.

This publication contains only explicitly allowlisted launch/error/closeout
metadata and pinned H214 source blobs. It is not a full raw-root or REL-07 lane
archive. Original raw roots remain retained. Missing periodic resource samples,
the second detached process exit status and post-model tensor-composite observation
remain unavailable. CLOSED_INCOMPLETE, null outcomes, both startup failures and
the original campaign clock are unchanged. No scientific retry or model/data/native
evaluation was performed for this publication.
