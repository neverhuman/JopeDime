# Prospective H-wave preregistration / launch admission template

Authority: **Jepson via Grok Bot, card answer 8:22 PM MT**.
Adopted 2026-11-02 for H204/H205/H206 and later explicitly authorized launches.
Fill every placeholder, commit this prereg and open its PR before scored inference.

Wave / owner / run ID / host / source commit / PR: REQUIRED.
Hypothesis, fixed treatment/controls/seeds, immutable input and family-fold manifest:
REQUIRED. Freeze the model/optimizer, budgets, selector and stopping rules before
outcome access. Sealed D4 confirmation families never enter training/search.

Numeric admission rule: **<=4 numeric threads (OMP/MKL/OpenBLAS/torch intra-op);
CUDA runtime/driver threads exempt**. Total process-tree OS threads are recorded
as diagnostics. They are not the numeric thread limit and do not identify which
threads are driver threads. This supersedes the old aggregate-threads rule in
prospective launch templates, including the rule copied by H196's monitor.
Historical preregs, source pins, launches and resource observations stay historical;
this template does not authorize a repeat of H196.

Before importing numeric libraries, export OMP_NUM_THREADS, MKL_NUM_THREADS,
OPENBLAS_NUM_THREADS (normally all 1), NUMEXPR_NUM_THREADS=1, RAYON_NUM_THREADS=1,
OMP_MAX_ACTIVE_LEVELS=1 and OMP_NESTED=FALSE. Set torch intra-op explicitly.
No application-created concurrent numeric tasks or nested parallelism. If multiple
numeric workers are admitted, their conservative per-worker bounds sum to <=4.
Include exact torch and threadpoolctl versions in the launch package inventory.

Use hwave_numeric_threads.capture_numeric_threads(torch) INSIDE each numerical
worker after library setup and at stage boundaries; persist the complete report
in launch.json and running resource receipts. It inventories runtime native pools
via threadpoolctl, so a loaded BLAS/OpenMP pool exceeding its environment cap is
still rejected. Supervisors pass all live numeric worker reports to
hwave_numeric_threads.admit_numeric_threads(reports, os_threads=tree_thread_count).
Missing/unknown reports or excess numeric limits refuse admission. A supervisor's
own environment cannot attest to a child's runtime limits. A changed process tree,
library inventory or thread setting requires fresh admission before more work.

Keep every other resource/science guard: nice19, host CPU<=85%, disk<93.8%,
wave-specific cache/RSS/memory/wall and CPU/GPU-hour budgets; one claimed job per
GPU; protected live flywheel PID2297929 untouched. No GPU work without explicit
wave authorization and resource admission. H206 has the stricter queued
pre-reservation disk gate (<=~88%, 40GB cache, ~50GB headroom); prereg/source
preparation does not release that queue.

Artifact retention (Jepson t16u, 2026-11-02): **save the gene, not the model**.
Fit objects (`.pt`, `.pth`, `.ckpt`, `.safetensors`, pickled/joblib estimators and
optimizer state) are NOT persisted by default. A wave's durable record is its
gene: source commit, config/prereg sha256, seeds, input/fold manifest sha256,
parent checkpoint id (e.g. r882 sha256), and per-fit scores/predictions. Persist
those as compact JSON/JSONL or `.npz` panels, zstd-compress, upload to the
wave's S3 prefix with VersionId+sha256 in the evidence manifest, then delete the
local copy once the run is done. A model object may be kept only when the
prereg names it (a promotion candidate, or a scored-inference input that cannot
be regenerated bit-exactly); keep at most one per arm, record its sha256 and S3
VersionId, and say in the prereg how to rebuild it (`runner.py --seed ...`).
Fit caches and per-fold scratch go under the wave run dir's `cache/`, or another
declared directory, and are deleted at wave close. Build trees use the shared
off-root `CARGO_TARGET_DIR`, never a per-worktree `target/`. The wave closes
only after a retention receipt has listed every deleted path, its bytes and
sha256, and the S3 key or regeneration command that replaces it. See
`xbabe2:~/ops/disk_steward/RETENTION_POLICY.md`.

Native r882 91/91 exact and mean1.1377941898164525 precede candidate forwards.
Use the declared 205-family pooled cross-fit LCB95>0 and harm<=10% gate; freeze
selection before A and defer the final read to sanctioned sealed confirmation.
Evidence manifest binds S3 URI, exact VersionId and SHA256. Outcome/verdict fields
remain unknown until actual execution. Central merger only; no author merge.
