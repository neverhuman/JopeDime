# H187 operational relaunch: attempt-002

Authority: `~/nudges/hwave_t0836_h187_relaunch.md` (owner t0836).
Attempt-001/source e112c6e2a ended INCOMPLETE after a transient host-load sample
138.596 >108.8 although host CPU was36.5%. It completed376 optimizer calls, with
377 started (last call in flight),377 completed backward calls, two B cells and
zero candidate A/DL-Val cells. Baseline DL-Val91 had passed. No candidate outcome
was inspected for adapting this relaunch. Keep its failure/resource/partial-cell
artifacts intact and disclose them in the closeout; do not mix them with attempt002.

The owner explicitly authorizes a fresh attempt002 with the SAME preregistration,
seeds, arms, objectives, initializations, samplers, folds, timing rule and decisions.
The original PREREG.md, design.py and objectives.py stay byte-identical. This patch
only changes permitted hosts, resource supervision and the exclusive attempt claim.

Preferred host xbabe1: CPU nice19/math1, <=4 aggregate threads, RSS<6GB, CPU<=85%,
MemAvailable>=16GiB, free space>=80GB and total study output<1GB. The prior xbabe2
fractional disk threshold is replaced by the owner-specified80GB floor on xbabe1.
No flywheel PID2297929, run-20261030 or other lane/source is modified. Use own
`.agent/worktrees/h187-tokcodex` with `~/lane_work/h187-codex` as an alias.

If xbabe1 inputs cannot port within10min, use xbabe2 with its40GB/93.8% disk gates.
On xbabe2 the monitor requires THREE consecutive high-load samples (0.5s sampling)
to stop on loadavg>85% of online cores. CPU>85% still stops on a single sample.
A below-threshold load resets the streak. The worker delegates load supervision
to that monitor so its old single-sample check cannot prematurely abort the job.
Other memory/RSS/thread/disk/time limits remain. xbabe1 retains the old load cap
at its own online-core count. The attempt-002 exclusive claim has a separate name.

Only operational source hashes/registrations change; keep the existing independent
scientific prereg review5728731368, supplement it with local resource-regression
tests and exact-source operational review, and merge before launch. Fresh native91
utility/candidate-cost parity still gates all TRAIN access and optimizer updates.
