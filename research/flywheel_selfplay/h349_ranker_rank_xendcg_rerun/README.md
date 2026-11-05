# H349 — rank_xendcg rerun

**INCOMPLETE_RESOURCE_GUARD — scientific outcome UNKNOWN.** The guard stopped only
H349 at 2026-11-05T05:14:48.151377+00:00 when host CPU reached 85.183339% (>85%).
30/36 tasks were fully locked and archive-verified; 6 tasks were partial.
No outer scoring ran. No gain, harm, Elo, per-root paired CI, or advance decision
was measured. Missing outcomes are unknown; this is not XENDCG_NOT_ADVANCED.
No restart, replay, reduced design, extra capacity probe, or partial scoring occurred.

Hypothesis: a fixed listwise surrogate may improve TRAIN robustness. BASE is
unchanged archived H303 NESTED/H308 P0/P1/P2 x roots30001–30003 x4folds,
1219 TRAIN contexts/205 families. Treatment changes only ranker objective to
rank_xendcg, plus objective_seed copied from each fit's original random_state;
sigmoid remains BASE1.0. Restoring the objective and removing only its explicit
seed restores both pinned H287 full function ASTs. All other options, labels,
streams, 18 configurations, tie rules, and the invalid sentinel and veto remain unchanged.

Preregistration `d9886e2ecf5fda82727b8b6ecc9af723d7326757` was published 2026-11-05T04:07:47.620783+00:00,
before first fit 2026-11-05T04:09:46.483913+00:00. See [PREREG.md](PREREG.md) and
[metrics.json](metrics.json). Frozen BASE regression passed all 9 panels and 36 fold/root
pick and lock hashes, plus 3 exact medians; 972 native BASE rankers were lambdarank/sigmoid1.
TRAIN identities matched before and after the attempt. 38 data-free controls and
7 adversarial custody controls passed; these do not qualify a complete study.

Audit: 35,911 completed native fit records and 35,916 constructor records.
Four rankers and one veto lack completed fit records; their outcomes are unknown.
Every parseable completed native record matches rank_xendcg/original-state
objective_seed/sigmoid1, or the unchanged binary veto. Final hashes for
90 completed veto models match BASE.
Completed-task guards cover 30 tasks; final guards for the 6 partial tasks
are unavailable. All partial artifacts/tails remain preserved as bytes in custody.
The 36-lock barrier prevented every outer prediction; all per-root means,
medians and paired confidence intervals therefore remain UNKNOWN.

Elo label: reference = r882 slot0 (DL-Train outer); r3175 not measured on TRAIN.
No speed verdict. The locked advance gates were not applied.

Resources: xbabe2 CPU only, admitted cores [15, 17, 18, 19, 20, 21],
nice 19, one numeric thread, MemoryMax=6G/MemorySwapMax=0. One five-second
probe admitted 6 cores at host CPU 16.164370%, below the 70% admission limit. Initial and first-task
forecasts passed; fixed compute 06:15Z/close 06:30Z Nov5 were unchanged. Unit stopped
by SIGTERM 15 at Thu 2026-11-05 05:14:48 UTC (unit Result=success is
systemd's stop disposition, not study completion). Actual charged CPU
23,075.723250 CPU-seconds, peak memory 3.318 GiB,
zero swap, and observed study scratch peak 1,551,588,067 bytes.

Custody: create-only s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H349/jrb01-20261105T040000Z/closeout/closeout-manifest.json, VersionId
`rMnIEhgHp8DWqE.NGkWsOdVkY96RNkCI`, SHA256 `aa6d7cdbf4255e436f40e42b04bb70461f00db8cd006e8e9d85083db05b9e63b`. All 4,375 files in
27 parts passed exact-version GET, whole SHA and every-member verification
before removing only H349 scratch. Complete and partial task artifacts retain their original states. Byte-exact executed source/vendor and receipts remain in
`/home/ubuntu/jope_plan/evidence/H349/jrb01-20261105-t0400`. No models in Git. DOPE c51a56d0 was not loaded or changed; this was a CPU selection-layer study.

Next: no additional fit or dev91 is authorized. Any future rerun requires a new
owner allocation/preregistration. This attempt remains incomplete.

Data-free acceptance command after reconstructing the archived source/vendor layout in approved scratch: `CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 19 taskset -c 15 python3 -B -m unittest discover -s /home/ubuntu/jope-data/h349-jrb01/source -p "test_*.py" -v`. These controls perform no model fits.
