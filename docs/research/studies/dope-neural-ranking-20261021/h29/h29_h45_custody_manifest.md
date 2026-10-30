# HWV-02 evidence custody manifest

Task: v1/HWV-02. Digest authority for the recovered bundle is `~/jope_plan/ns/receipts/H00_H41_MASTER_INDEX.md`. All 13 H38–H41 indexed files were copied from the read-only `/home/ubuntu/JopeDime/.agent/worktrees/flywheel-selfplay-20261023/docs/research/studies/dope-neural-ranking-20261021/` tree and verified by SHA-256; all are under 5 MB.

| Wave | Artifact | SHA-256 | Disposition |
|---|---|---|---|
| H38 | `h38/results.json` | `f0d4c413f0c4def25ae0ffbe40f19b089fa0177e4be079f044c3eea5c7849e9b` | FOUND (SHA ok) |
| H38 | `h38/results_v2_reconciled_noncompute.json` | `0afee6439cd69f63edc1a827621b337c77120c9c03c092635b8663433cc4c545` | FOUND (SHA ok) |
| H38 | `h38/h38_bounded_additive_budget_ledger.json` | `8f48b6185eaa9c4a33129f8a58d86785ee0f4451657c30d522241cc12a4b2656` | FOUND (SHA ok) |
| H39 | `h39/results.json` | `686bb13b8b75af8ca64c731744fb666da67bc7b3d148b44cf1d214a99b9ce34b` | FOUND (SHA ok) |
| H39 | `h39/h39_bounded_additive_budget_ledger.json` | `a09ed0e844caae9ef4a6a2dc6ef0b87399b997907e8c504ce785e4bebdc16115` | FOUND (SHA ok) |
| H39 | `h39/h39_reconciled_execution_vs_denied_reservation_ledger.json` | `7d3f9109c38f6a84086574ca173c18e8d8a226cd76d46d5d58b430817086d6a2` | FOUND (SHA ok) |
| H40 | `h40/results.json` | `5d7e0d4034b12b1c7e284db7eed27c3984365f15ff263963c5dd5d356daf70c5` | FOUND (SHA ok) |
| H40 | `h40/h40_bounded_additive_budget_ledger.json` | `9b61e8f51f9d050cf3ab3abf74fcfe1ec2717264baa82856867f2622b06242be` | FOUND (SHA ok) |
| H40 | `h40/h40_timing_decomposition_and_resource_estimate_v2_closeout.json` | `682b74756f697b69aca67d42bc16321f1ebb24a2a04cb51df75ad98638f223e5` | FOUND (SHA ok) |
| H40 | `h40/h40_timing_decomposition_and_resource_estimate_v3_closeout.json` | `a058e36eaff6ac7da8049d1e252de2e4dcf36d616757744da26cfefe621dc60e` | FOUND (SHA ok) |
| H41 | `h41/results.json` | `57df980cef8423029f9e808a66d7d8155c22be4658e802673d0bcf349323439b` | FOUND (SHA ok) |
| H41 | `h41/results_v2_closeout_reconciled.json` | `4a02ed926ffffb6c61529b24f81f25979c8a14f51053556603c59d72f302b257` | FOUND (SHA ok) |
| H41 | `h41/results_v3_closeout_reconciled.json` | `d91bbede45c158e9c4e9b32c9b6079bfec9a1e4eb893d2d6d8ccc102fa7f1e20` | FOUND (SHA ok) |

## H29–H34 result evidence

**Corrected disposition (super_merge_claude, owner ruling `merge_0605.md`: "a custody record must be correct"): FOUND for H29, H30, H31, H32, H33 and H34.** This supersedes the earlier `EVIDENCE_MISSING` classification. That search ran on `Mac-Studio.local`, which is not xbabe2. On xbabe2 (Linux), the archive path `/mnt/jope-archive/jopedime-research/dope-neural-ranking-20261021/` exists and holds `h29`–`h34`.

**Source.** The published public-research documents for these waves were committed to git branches that never reached `main`, and those branches are preserved as git bundles in the archive (read-only):

- `h34/public-git-custody/h34-public-research.bundle` (sha256 `2b2bff6078776dea2c55ed3a9d3a12ebea66226090b470b700521b5ff76cd61d`), head `1b94942a9b667d6b10bee9039a4e364ce655a3be` (`codex/research-h34-results-20261023`). It is cumulative: it contains every H29–H34 file listed below.
- `h33/public-git-custody/h33-verified-public-research.bundle` (sha256 `87bcf987c8b961f8c986d0505ccd58846d8886934d5deaf0effae2b6616447c9`), head `5378e1f7fb681719d39a1b21ae9bcaff3188ed5d`, an ancestor of the h34 head.
- All six wave bundles pass `git bundle verify` against this repository.

**Digest authority.** `H00_H41_MASTER_INDEX.md` lists no H29–H34 artifacts. Each file is therefore verified against the SHA-256 recorded in its wave's public-git-custody receipt instead:

- H29–H32 against `h32/public-git-custody/RECEIPT.json` (sha256 `ab210a51c66d289aded55b176426571aaf0eed4b3325983c840d8df8b15d6767`);
- H33 and H34 against each wave's `PUBLIC_BLOB_MANIFEST.json` and `SECOND_HOST_RECEIPT.json`.

**Result: 41/41 files match and 0 mismatch.** 40 are committed below, with a `sha256sum -c` manifest in `h29/h29_h34_custody.sha256`. One is a pointer. Every committed file is under 5 MB (290 KB in total).

| Wave | Artifact | SHA-256 | Disposition |
|---|---|---|---|
| H29 | `h29/README.md` | `3f4479b792954069b4dfea3eb548ebb80d311f39fee4557bbb6ec73b003ef125` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H29 | `h29/metadata-summary.json` | `e98087536e553afaf90700ff69ad8d03c33e80f4922a0510f926c1263c892b97` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/README.md` | `269105b547ca6245f53c1fa8aff69f7dfddf121cfdbfac73d0a0e4798dcc8190` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/STORAGE.md` | `4c57358763fea9c26ed9cb3507e78e53858a2641a6fd15e5ef61bfbe97756b47` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/critical-report.md` | `9960a4ae9b9e65dccb13760a8d59984fe2c799828840f3200c43800953f98fe0` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/failure-cost-ledger.json` | `55f63fdce167d132e86139c044b4124882b6dce41fe64b1383358bf80275339d` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/final-results.PENDING.json` | `9c1ac24ffe52458670514d28bdf8c9941ede584b41df31f3b9ee3b01517c29a5` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/final-results.json` | `74d791c04926993de280d257e946df9236ee3eee3c924495c32486c34ee313de` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/monitor-prefix-curves.json` | `4f8f22067523bcde1949b2f3bf46c44b76ca286416a95dfe930c2be3b4b70f27` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/protocol.json` | `aa7cda2532ac3e692403f74bb89a27ca8dc9329d3c255037157837d310617cb3` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/review-entry.md` | `303c4723f6e56791b666e7b31b864687dfd95817d1ba8d411d3598dd92837341` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/source-receipts.json` | `64263f56d0ae119df21dd84e88eeea7e899925f5327ff2804c527cbac91c54be` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/storage-custody.json` | `962d556ec50c40270841fca4efdadfca5cf32372ead0f555c9e5db312a109391` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H30 | `h30/train-curves.json` | `9e173be3ae4b35e4b174684e747708c82d4d9a32642796cc04d82fc10a9ba8be` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H31 | `h31/README.md` | `11225d3bbdbf96237007b7f867492f1ed6b34adbed802844ad36051e57fd3fd8` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H31 | `h31/costs-and-failures.json` | `06176da99c9dc8904de5f05b92e808f3ed60de733de38a89ef2bdf614693f073` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H31 | `h31/critical-report.md` | `1293710526d5d7381deed35442ccb57e089ba0fe9af5bcfa349afdb81ed3ab19` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H31 | `h31/posthoc-gate-penalty-arithmetic.json` | `f255e1294cdc632bdac0fe9c5166472870fce28be1d1ba92f16cd29945146452` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H31 | `h31/posthoc-gate-penalty-arithmetic.md` | `fb338bbdd3dd979a283506923198fdc28b75e41bfa5c8d9209a452e76f2e0c28` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H31 | `h31/protocol.json` | `ba4cc9ad04a3491da5f398bc0ab673cfe1c3da53c05a0bb4bda31fe85078dbe1` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H31 | `h31/results.json` | `a04f66b3a7d73f7a2f3a31fa4afa7d623a754237f3c6062bbd56873adcde629e` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H31 | `h31/review-guide.md` | `3b27623654876bfa56652b5e5acfa872e7b5c9e5ad5f5dad846ef4a8c037b4f9` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H31 | `h31/source-receipts.json` | `08215c48ec5083ac77e5a1cb1a7fc799767d5de34c2831c86aa6616bfc533cf7` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H31 | `h31/train-curves.json` | `bdd88ce53304925ba9398944c0ea646997477207ed2dfe1ff0564b4c6aaa4764` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H32 | `h32/MECHANISM_AND_NEXT.md` | `7d30c1028a6148d9e9453200a1b005b5b2593adce6326f3177543f4da5f68590` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H32 | `h32/RESULTS.md` | `f35caf9221be238f3783e55d285a37dd7c04b0276c15a56115d8ba43e5d7eb34` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H32 | `h32/mechanism-source-pins.json` | `db9559850b5fbf84268b742c5b8a3658672365cfa231c68e38c749796c1bf046` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H32 | `h32/results.json` | `8e970def17fd2a4fdd2efa06f08a4ad4a07457b406cf2226c5e7658a9fd3e12d` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H32 | `h32/synthetic-callback-curve.csv` | `58a2fabbc5c32d37c3040194a1726a44fe78d9d9c0ed8d2ecb3ee2e9c7fd3b13` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`) |
| H33 | `h33/README.md` | `e108a8219c9032e2b4de06370b795230cbf60c1994213b6811e9b546bb728f65` | FOUND (receipt SHA ok: `h33/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H33 | `h33/RESULTS.json` | `4c6816035c05f112b8a6b1cf2de6789cf452cc5df2847ac1ad9f8c6ad69e0485` | FOUND (receipt SHA ok: `h33/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H33 | `h33/TRAIN_MONITOR_CURVES.csv` | `597770b630b930a9503e2c9b19968428b0f63093b6903b20e4d53a2d977123b2` | FOUND (receipt SHA ok: `h33/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H33 | `h33/protocol.json` | `56fc2e8abdc566fadf8bd639338331d2f966c7da3e3f40f1269b1b650bd8f2e1` | FOUND (receipt SHA ok: `h33/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H33 | `h33/source_receipts.json` | `aec230542eafac0fd037314fd1825b6a2d4fece0103f33e8cb6b358b527111d1` | FOUND (receipt SHA ok: `h33/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H34 | `h34/README.md` | `4ae5b5fddefa9b836e3b3e57bcb21a07bafd67bf2ae014b7c833a035cb63f146` | FOUND (receipt SHA ok: `h34/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H34 | `h34/curves.json` | `198da327c9d771f09c641346c8eb321e90f7ee3bdf2f2dc107bc583987e03f30` | FOUND (receipt SHA ok: `h34/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H34 | `h34/failure-cost-ledger.json` | `a94fe1273daea0a1eac05531d4bf0c31ba57f0a9dca82937a61575367ed184b3` | FOUND (receipt SHA ok: `h34/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H34 | `h34/protocol.json` | `94365aba0b4b05cf314efc03ad439a5397ea009daf5f4eb9142b96bef61c67d6` | FOUND (receipt SHA ok: `h34/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H34 | `h34/results.json` | `521f0bc100c2ab49454a7b6a2d861a643a7bce0f57fa379266a5ddc648ff0505` | FOUND (receipt SHA ok: `h34/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H34 | `h34/source-receipts.json` | `a7a266557bf5fed8827aa6ffd8a547f1fcdb11464825fab1e4f0ff47e024156d` | FOUND (receipt SHA ok: `h34/public-git-custody/PUBLIC_BLOB_MANIFEST.json` + `SECOND_HOST_RECEIPT.json`) |
| H32 | `h32/derive_exact_gains.py` (4306 B) | `0870772fca6dcefedf0b0599e5ce01641e7aeec5a44d8046b9444a9981f13ac5` | FOUND (receipt SHA ok: `h32/public-git-custody/RECEIPT.json`), **pointer only**: not committed, so tracked Python, the README count and PYTHON_EXCEPTIONS stay unchanged (outside Owns). Recover it from the h34 bundle below. |

The wave dispositions are: H29 `FOUND`; H30 `FOUND`; H31 `FOUND`; H32 `FOUND`; H33 `FOUND`; H34 `FOUND`. The raw execution trees (containers, qualification and readback directories) stay in the archive at `/mnt/jope-archive/jopedime-research/dope-neural-ranking-20261021/h29..h34/` as recovery leads. They are not copied. H42–H45 have no directories in this archive either (0 entries); this amendment does not change their status.

## Safety and verification notes

- No sealed meta-test, holdout, roster, prediction array, checkpoint, or model-weight payload was copied or opened. S3 inspection was object-list metadata only.
- No DOPE dataset content was accessed or changed; frozen checksum `c51a56d0` was not recomputed because that would require opening the corpus.
- No large binary was copied. No new metrics are claimed or recomputed.
- The recovered H29–H34 files are the already-published public-research documents from the bundles above, extracted with `git checkout <bundle head> -- <paths>`. No archive file was modified.
