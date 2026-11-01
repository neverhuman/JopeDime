# H133 A6 attempt 013 closeout

A6 merged in PR #1245 at `74b0e915917658effd6f32268d70cc1e7439f376`. The single authorized attempt ran on CPU at nice 19 on xbabe2. The supervised launcher used the 466,723,755-byte train-only input closure and recorded zero pause events.

The unchanged-r882 regression was the first worker action. Prediction, utility, CONTROL fit-loss, and fit-gradient paired differences were all exactly zero, but the rounded family utility was `1.1255361`, below the required `1.1377942`. The worker stopped before any treatment cell: zero optimizer updates and zero cells. The baseline regression read 245 fold-0 context payloads and one comparator utility; no treatment payload or arm outcome metric was read. The result is incomplete and has no H133 verdict. Medians, family-cluster confidence intervals, Holm, harm, and Elo were not computed.

The launch and regression receipts, worker log, resource snapshots, and pause log remain in `/home/ubuntu/jope-rescue-20260921/h133-optim-geometry/attempt_A6_013/`. The consolidated closeout receipt is `/home/ubuntu/jope-rescue-20260921/h133-optim-geometry/a6_attempt_013_closeout.json` (SHA-256 `f7afe8fad12b1ee9bad080250c2b723fa5a01d222644076dab55cf88a8fc9303`).

Attempt 013 is the one authorized A6 launch. No retry or changed regression expectation is authorized by this record. A further attempt requires a new owner ruling that resolves the baseline discrepancy.

The S3 identity check (`aws sts get-caller-identity`) returned `NoCredentials`, so the planned upload to `s3://veox-jopedime-learning-use2/experiments/rd_breakthrough_20261030/H133/H133-A6-attempt013-20261101T002911Z/` could not be performed. The per-run manifest is preserved locally at `/home/ubuntu/jope-rescue-20260921/h133-optim-geometry/attempt_A6_013/manifest.json` (SHA-256 `37aee7ef8fcf8a9b876e37ad304f91fc5ff45d25f37e4bc0bddf9e7141ddad9f`). H50's source manifest contains no original S3 object keys; none are fabricated. Receipts and diagnostics remain in local custody, and no bulk artifacts were deleted.
