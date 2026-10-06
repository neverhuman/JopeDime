"""UPSTREAM-SCALE-01 counterfactual check (Track S): do the Codex encoders' training-side inputs depend on outer-test rows?

Finding UPSTREAM-SCALE-01: research_2k data objects are min-max scaled upstream over rows that include the outer-test rows,
so the JTT1/Codex train bytes embed test-dependent per-column constants. Owner red-team ask (relayed with the DL-D5 round):
for a sample of research datasets, perturb or remove the TEST rows, redo the source min-max scaling, run the Codex train-only
re-normalization, and assert that the training-side encoder inputs are bit-identical, or document the exact residual; if it
does not cancel, re-derive the scaling from training rows only (lane-local). The result gates GPU-training admission.
Track S spec A (component fidelity) and spec B (lineage, prescan, sample design, variants, comparison) are the contract.
numpy only.

Never: torch, a GPU, encoder weights (models/* are not opened), encoder inference, S3, or an import of a Codex file (they
import torch/cuda at module level; they are read and sha256-pinned instead). Writes go only to --out (create-only files)
and to the optional create-only --receipt (dltrain.common.write_receipt).

DATA CHAIN (spec B LINEAGE; every hop is sha256- and byte-checked and a mismatch is a STOP)
epoch-000 included row -> prepared.json (prepared_ref; its context equals the row on context_sha256, input_raw_sha256,
input_ref, outer_fold, dataset_hash and sample_identity_sha256; outer_repeat 0) -> research manifest (frozen: FROZEN_ROOT
+ key; incoming: content-addressed INTAKE/verified-objects/<sha>; items/<item>/materialization.json is rewritten
continuously and is only a cross-check) -> data object (gzip CSV, NO header, target = last column) and membership.
A = np.loadtxt(gunzip(data), delimiter=',', dtype='float64', ndmin=2) is the Codex parse (worker.py:43). T = the sorted
test indices of (fold, repeat 0); R = the ascending complement (inventory.py:22-24). The 45 feature-sanitized views carry
derived identities; their manifest, data object and Codex identity are the parent's. Proofs per context: P1 the JTT1 input
rebuilt from A equals input_ref.raw_sha256/raw_bytes; P2 a cached JTT1 decodes (jtt1.train_view, train range only) to the
rebuilt labels, A[R][:, kept] and A[R, F]; P3 exactly one Codex packet fold with train_indices == R and equal hashes;
P4 every Codex index 81af92a8 output of the (identity, fold) has train_indices == R; P5 local outcomes (worker.py:10-11 job
ids, worker.py:70 shard owner) bind the same split and kept set and their recorded side values replicate bit for bit.

CODEX TRAIN-SIDE PATH (spec A; replicated dtype for dtype; line numbers in the pinned Codex files)
records (worker.py:43-54): raw = A[R]; train-only target-leak exclusion in float64; X = raw[:, kept].astype(float32),
y = raw[:, -1].astype(float32). hyperion (gpu_encoders.py:74-83): lo/hi = np.percentile(X, 1/99, axis=0).astype(float32)
under numpy 1.26.4 (a float64 lerp, emulated in _lerp_np126; numpy 2's scalar-q path lerps in float32), span =
where(hi - lo < 1e-9, 1, hi - lo), u = clip((X - lo)/span*255, 0, 255).astype(uint8); tiles, permutations and the
CUDA-side xt depend only on (h, w) and u. foundation (:19-22, :92-96): sampling(n) seed 20260707; v = ((X - X.min(0))/
span).astype(float32), span floor 1e-12; inputs v[ctx], v[qry] and yz = (y[ctx] - mu)/(std + 1e-8) with mean and std
emulating numpy 1.26.4 (_methods.py; numpy 2 divides the variance mean in float64). target (:103-104): v = ((y - lo)/span)
.clip(0, 1) with span = np.float32 - float computed in float64 as numpy 1.26.4 does; inputs v[classical], v[neural]
(blake2b stable_seed index sets), lengths [n], task [1]. tabpfn35 (:107-110): raw X[ctx], y[ctx], X[qry] with no Codex
normalization. context2048 (worker.py:113-118): no row input; identical iff hyperion, foundation and target are.
Side arrays (feature_q01/q99, feature_min/span, context_y_mean/std, train_target_min/span) carry upstream units by design:
they are reported, never part of the gate, and must never become training features.

VARIANTS (spec B; phi_S(x) = (x - min_S)/(max_S - min_S) in IEEE float64, subtract then divide, divisor 1 when max ==
min, applied to all F+1 columns; each variant's outer-train matrix is handed to records() exactly as worker.records
receives the parsed rows, so the target-leak exclusion is re-run per variant)
A   as materialized: A[R].
A1  control: phi over all rows, no row change (bitwise A on exactly-[0,1] columns).
B   test rows removed, rescaled on outer-train rows only: phi_R(A[R]).
C   per column the smallest-index test row at the test minimum is set to lo_all - 0.25r and the one at the test maximum to
    hi_all + 0.5r (r = hi_all - lo_all, or 1); if both are one row the maximum moves to the next test row (wrapping; the
    count of such columns is reported). C = phi over all rows of the perturbed matrix, restricted to R.
C2  the test rows D holding a test-only extreme of a considered column (Codex-kept features of A plus the target) are
    dropped; S = R + (T - D); C2 = phi_S(A[R]). D empty gives C2 == A1. Single pass; residual test-only extremes counted.
FIX F_V = phi over V's own outer-train rows, for V in A1, C, C2 (the owner's lane-local train-only rescaling); F_A is B.
B32 diagnostic (spec A P4, not gated): phi_R over fl32(A[R]) widened to float64, which separates Codex float32 arithmetic
    (B32 vs A) from the 9-digit print and float32 lattice term (B vs B32).

COMPARISON (per context: X vs A for X in A1, B, C, C2; B, C, C2 vs A1; F_V vs B; B32 vs A and B vs B32)
Bitwise equality on uint32/uint8 views after shape and dtype checks; float32 ULP histogram (0, 1, 2, 3-4, 5-16, 17-256,
>256) with ULP distance |k_a - k_v|, k = i if i >= 0 else -(i & 0x7FFFFFFF) over the int32 view (signed-zero flips counted
separately); |delta| in units of 2^-24; uint8 delta histogram with |delta| = 1 and >= 2, span-floor-column and saturation
flips counted separately; differing elements per data-object column. Up to --positions differing elements per array are
listed as [data-object row, data-object column or 'target', A bits, V bits, ulp, |delta|*2^24] (hyperion: [row, col, u_A,
u_V, delta, t_A bits, t_V bits]); positions name outer-train rows only, outer-test rows appear only as counts.
Class per component: IDENTICAL; STRUCTURAL (binding change, any uint8 |delta| >= 2 or any flip in a span-floor column,
|delta| > 2^-12 on a [0,1] array, > 2^-12*max(1, |yz_A|) for yz, any tabpfn35 raw-input difference); NUMERICAL (any other
difference). tabpfn35 also carries the informational spec A P9 refinement (constant and identical-code categorical columns
are inert, y_ctx is numerical, a fingerprint change is structural); the gate uses the spec B rule.
GATE: PASS only if every fed input is IDENTICAL for every sampled context under B, C and C2. FIX-ROUTE GATE: PASS only if
F_A1, F_C and F_C2 are IDENTICAL to B for every sampled context. Fix-route differences are attributed to near-zero columns
(0 < B < 1e-10, spec A P13: a variant's float64 offset wipes such values), to construction-risk columns (2^-27 times the
variant's offset from the train minimum, in train ranges, exceeds the smallest positive B: the simulated upstream cannot
keep float32 resolution, e.g. the S-FLOOR severities), to rounding columns (every bitwise float32 cast difference is a
signed-zero flip or an adjacent-float32 straddle of a rounding midpoint with a float64 gap under 1/8 float32 ULP), or
left unexplained. The tool reports both gates and admits nothing. STOP (exit 2, the run is not valid evidence): a lineage or sha256 mismatch, a replication mismatch on a
local context, equal_in_order false under the pinned pre-review registry, torch or boto3 imported.

SAMPLE (spec B, deterministic): U = included, unsealed in the pinned registry, all five Codex components complete in the
local outcomes, not the ANCHOR. rank = sha256('jope.dl-scale-leak-sample/v1|' + context_sha256). ANCHOR first, then per
stratum (frozen f0-f4, incoming f0-f4) F-severe, H-floor, U-control (strict: phi_R(A[R]) bitwise A[R] on every considered
column), R-flagged; then T-only frozen, T-only incoming, SAN, OUT01, SUB-inside, SMALL (each the first eligible context by
rank not already selected); then R-any by rank. At most one context per (cohort, fold, dataset) and two per dataset.
Under the pre-review registry this reproduces spec B's 48 contexts exactly, in order (SPEC_B_SAMPLE_48).
S-FLOOR supplement: the unsealed contexts where the foundation 1e-12 span floor is reachable, analysed with positions and
reported outside the gate denominators. CENSUS: every unsealed included context x every comparison, counts only, plus the
replication gate on every local context. A0: the lane's metafeatures vector of A (from_raw on the rebuilt JTT1) against B
with the inner labels held fixed; whether recomputed inner labels survive B is reported separately.

CLI
  python -m dltrain.scale_check run --sample N --out RUN/tmp/scale-check-<label>/ [--receipt
      RUN/receipts/findings/UPSTREAM-SCALE-01.check-<utc>.json] [--registry R] [--holdout H] [--no-census]
      [--census-limit K] [--positions P] [--no-a0] [--max-pause-s S]
  Exit 0 gate PASS, 1 gate FAIL (residual documented), 2 STOP (a pin, lineage, replication or processing failure: the run
  is not valid evidence), 3 ERROR.
"""
import argparse
import collections
import gzip
import hashlib
import io
import json
import math
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np

from dltrain import common, cpuguard, jtt1

SCHEMA_RESULT = "jope.dl-scale-check/v2"
SCHEMA_RECEIPT = "jope.dl-finding-check/v1"
FINDING_ID = "UPSTREAM-SCALE-01"

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
EPOCH_PATH = RUN / "epochs" / "epoch-000.json"
EPOCH_SHA256 = "bcfb5230bc0f5f32f6b42edbfa3a6bf76b89e0593b593b9042290e8c077a87ba"
EPOCH_BYTES = 16633383
REGISTRY_PRE = RUN / "families" / "registry.pre-review.json"
REGISTRY_PRE_SHA256 = "224ce71d89f692c7d13ae81abbcf63cef797f6498086dc6a07f330acc2d5a4ef"
REGISTRY_FINAL = RUN / "families" / "registry.json"
HOLDOUT_DEFAULT = RUN / "holdout" / "holdout.json"
JTT1_CACHE = RUN / "inputs" / "jtt1"
FINDINGS_DIR = RUN / "receipts" / "findings"
TMP_DIR = RUN / "tmp"
CODEX_INDEX_SHA256 = "81af92a86988b01f1f42475e28390659cfc09d34aa4dd72f5d526f71a1da5db8"
CODEX_INDEX_PATH = RUN / "encodings" / "codex-index" / f"{CODEX_INDEX_SHA256}.json"
CODEX_INDEX_BYTES = 159490137
FINDING_PINS = {
    "UPSTREAM-SCALE-01.json": "a112c834bd724218f402520e934aaacd88e0ee62db657188e7ec4aecff25079d",
    "UPSTREAM-SCALE-01.refs.json": "189f73a33d9d3e4a808fe22050bbe143b428da6bde4628c7696798785eb33b63",
}
FROZEN_ROOT = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-20261001")
FROZEN_KEY_PREFIX = "research/regression-2k/v1/"
FROZEN_CONTEXTS = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-per-fold-hwm-20261001/gp/contexts")
INTAKE_ROOT = Path("/home/ubuntu/scratch/jopedime-regression-research-2k-incoming-intake-v1")
INTAKE_OBJECTS = INTAKE_ROOT / "verified-objects"
INCOMING_ITEMS = INTAKE_ROOT / "items"
CODEX_ROOT = Path("/home/ubuntu/scratch/jopedime-regression-neural-encoding-v1")
# Read-only peer source, never imported. The replica below was written against exactly these bytes.
CODEX_PINS = {
    "gpu_encoders.py": "ff41938421be0ab2312db8bbfb6497c6dc7d0eb7164eb1cfea4af04cfd0965b5",
    "worker.py": "880bd9acd9c9fe5fe82e46f0cd14d079d448c0bb5cf96fbfbcd7cda8cf7e02eb",
    "publish.py": "d029f86636e943ed91fe0738209fea62764b589f3c9993b8bcd04874db0a8b90",
    "inventory.py": "218aa087f71c12f1b5cf3a17a9c943bbe8ab0732f72e0f73ab17f489d68f5a95",
    "custody.py": "a8280327c69410a44d308d446ed81fa11f9b53730a1ff82c09133f2443ef422f",
    "packet.json": "6f204f4461e0bfedf280937f169fc73502cc41e8634b002b120beced57e6291e",
    "vendor/target_maker/data.py": "e20c00a8c2ce5894e2ad7dab9b48e05c353a4736aba471e6d481b7c0931bf0fd",
    "vendor/target_maker/encode.py": "ce4161f5d7d10dd4e1bd442050b6c03af0432e792037279ffbf08375204c89ca",
    "vendor/target_graph.py": "df01e4cc5873f41b61c231f6e8a13724e80569ffa6e70c07d51cdcbf304c3ecd",
    "vendor/hyperion_reference.py": "e860f52266fe9d9e50a52b41e5d26484d77ad3f74c310b3993b79a355e1422ce",
    "vendor/tab35/encoder.py": "eb3262e272ba58c681a5deceb024785e57c0acd5d6de00c98bf0807a4dbf7ab2",
}
CODEX_NODES = ("xbabe1", "xbabe2", "xbabe3")   # worker.py:8
LOCAL_NODE = "xbabe2"                          # the shard whose outcomes are on this host
ANCHOR_CONTEXT = "6ebcb49c66f39192af38ffa7393829365b8d7c5e62fd12fd431476dc37785299"
ANCHOR_VIEW = "014b93f43dbaa1a4__r0__f0"
RANK_DOMAIN = "jope.dl-scale-leak-sample/v1|"
# The 48 contexts spec B selected (pre-review registry), in selection order. A run reports whether its selection
# reproduces this list; under the pinned pre-review registry a difference is a STOP, under another registry it is expected.
SPEC_B_SAMPLE_48 = (
    "6ebcb49c66f39192af38ffa7393829365b8d7c5e62fd12fd431476dc37785299", "256287f8bef845f6c34be10471db0bb5144b101e4117b5ac18b80834b4ab2d4d",
    "2e0bc6eb25daed446ff2b102b2b3edf0ea267c44390af0149b1340ad2b340eb5", "42cce7c473564732ead5d284775ca1df2cbdf92e8d27da16a9192deabf07f543",
    "7ea31858afa6969b20a42e6045423adfbd22cf106551d58fe55b52db6599dc0c", "80605ba3498b5dac29de671b6f0426e55c1ecf0a2b2828a78b081f34735d6011",
    "1227013ac65d86280c6cdf08f6ed80502889a5c7582891415f2b43810109bf3f", "334230c68bb827bec9a024ff1ce01e04ac2ab9c0fcf07afb7f66c16915da1d88",
    "01a043913cf4127a1ef6414b8b5328312c1ce8a0cf198f596719d11810602d7e", "d5f631f1f364bff338f53084c06142a60ecfb12118f3a2b21224031bafaabf59",
    "94773adae3c13ac0b7316093b26fe9b761409e9d058b8ca6fa4879e559543f24", "32e7e34709e6213e9a623385250156ae44fbff8233c5ca4a74023054be7fd79c",
    "c345141f8e537abb2d4c85cfe648e03c8646ff489eb6d6cefd1d9b9f9b04cad9", "64539b91ca5c9ed0cbd906c9a8bad345602b243b8a3f57b66ca82d084dd1e1e1",
    "331fc27268ba37e3e48fb21b9986ce6f4325fd9c6b6229f2782d70f9cf1212c1", "7ec5b6b7f6fecce54b469600696513ca71b0e36a64c5f135e56322caec355354",
    "af2ea621225276987c9c5ea371fc3c1fe267d444ae71e7d9478a4d5a4041640b", "e2d6e07a7eb5fb0170ff75a828c032522ae320335c75338ce50eb538e0c66fd1",
    "294bb592b3c78eaa03714d632accf6bfb3820d4746fe806ea7aa2dbf5a1d07e9", "532db87b9b6f422c443d44ac5911faccb2f9f224073898b30cff583e42decc4e",
    "6948c2522ac83fae31d35f04b9b615f5c89c8d7d8cc19ff676ce19770d549f40", "7b139d315bafa0b30e3bc0990bfa5b9d0fceb9c3f01a2ceb7b44bd61ff625264",
    "bc0a64c2ce5b59546e91ebc5852c8d2dbac8107ddd8f0340ee01b9681f3117f1", "118a78a0f13feca4e09fda6fa16c05d5d8abbd4f5660080190d9055c520759ce",
    "e1dc2da419c31deb75accbe3760f1b3648ec5cd2f099a824d3bf1e6f6d24fdf4", "efa7fde7817e1610aa99f9a8febacc157c211702e502a086105d1795ab021f1d",
    "a0c62647fe30f27900166b8e02301698ddb64aa7614dfb0bbb7e243cf34c167f", "0f9f4e1b38992d7988be2a46187dbf3fb22f349ea3dd31d80b346bac85d695fa",
    "e205ca61a303c98854491e96ad8ee617365d77f2a8e3da8803b91d2964345a5c", "498d9882a9e92b9c91d5fa6c2330e98406cafd8b693e9582d2750953f3d2a8a0",
    "2c3758fba4d9a573d6e11bbe2adb2e571ab4a0954b6dae956fd838266e2c5794", "3df2c1d69945c7f5d66a7f563cd66f936ec307ba88d45e67bde580cd88282765",
    "774f663b06e6dd1a7b93ce7818a14f6471a1feb56003e28d0dcc68c136f2f229", "09ca1450587882b34a6da8baa5df6546675df05f07169038f3d58dce39c6dbc3",
    "562b22ca8121ded4cb014bfa0084a9becb1d2e638e74d3d47abcb829c5941468", "56c31b370dd53361baa6cee0747ae7964e26fcae17a2c35ec7e10ac4d0d7ebc0",
    "cb70d28c718099829697861ce5b4a863aa725a1a012b3f23a0da5b8474676152", "d263cfb0388cd75acedc8437134df254bf7c905d0828686b7ad7850d8182c6e9",
    "0cd1b59c1d0052a7ce6f42848d261b90c907cd596e0d6eaad3df2fffab039bd3", "734d56a9918afd11fc8404bd2781493775249a74e97cd3a54b1753e39837cd1b",
    "4378c16f1e50e5e8108639b45609f121021bbc7ed20e8158144e0cbec0bd2d70", "fc11716adc918c728c3c93ca2af39f9c1d923ee373618040816a5a234a9e3b35",
    "3ae406cd82cecab34dba1444815b5207342ae50f2cbe9d942d2786b8bf8f0372", "8dcea0d1d95697dd9d8f4cd27f2b0672ef139bc27822548df5098e99cd163e80",
    "8f23e8090df0fbc77f97a5a672e84dcdef01206ca2f28ad7a28cee292e820d86", "767f19a8986a348b8a4747c862320ed3dd8ce41be5bfbcd03e892108c9ba207e",
    "a675d39e8ff300add6dbd99ef6660affc82d45a55c52925e0211a97ee207ceaa", "be7c719f514964b27011dabcc93eb8259f89bd25a0643327288b998770a149a9",
)
# sha256(common.canonical(list of the 48 context_sha256 in selection order)) and the U pins, from spec B.
SPEC_B_SAMPLE_SHA256 = "4ec7a440f9c91208de901ef26d1c24d940f3ca740221cdd142f5b2ea9378a598"
SPEC_B_U = {"size": 1558, "flagged": 1046,
            "sha256": "135e34012158b1241fc7086a1370e2595aa8ef823540e8960b21cdd60d46cd79"}
# Spec B SUPPLEMENT S-FLOOR: the only unsealed contexts where the foundation 1e-12 span floor is reachable (all non-local).
S_FLOOR_SPEC = ("d3fab2adaa931253__r0__f2__test_adaptive_8a9da1d1fb7d",
                "5f774c42d440516c__r0__f1__test_adaptive_1340e8eb9a20",
                "272b350c633761db__r0__f1__test_adaptive_94a8853abab8")
# Spec B LINEAGE pins and PRE-SCAN totals: an independent cross-check, reported (a mismatch is flagged, not a STOP).
SPEC_B_EPOCH = {"rows": 5017, "included": 4873, "frozen": 2834, "incoming": 2039, "datasets": 1006,
                "frozen_by_fold": [570, 570, 570, 568, 556], "incoming_by_fold": [404, 417, 412, 407, 399]}
SPEC_B_REGISTRY = {"families": 888, "sealed_families": 33, "sealed_included_contexts": 186, "sealed_frozen": 168,
                   "sealed_incoming": 18}
SPEC_B_INDEX = {"outputs": 19341, "complete": 19334, "skipped_null_identity": 7, "keys": 3868}
SPEC_B_LINEAGE = {"index_covered_non_sanitized": 3136, "index_absent_non_sanitized": 1692,
                  "local_complete_included": 1624, "local_complete_sanitized": 22, "local_complete_unsealed": 1558}
_PER_CTX_ALL = {"0": 1594, "1": 1237, "2": 715, "3": 379, "4": 259, "5": 167, "6": 106, "7": 79, "8": 43, "9": 44, "10+": 250}
_PER_CTX_UNSEALED = {"0": 1498, "1": 1189, "2": 690, "3": 370, "4": 252, "5": 166, "6": 106, "7": 79, "8": 43, "9": 44,
                     "10+": 250}
SPEC_B_PRESCAN = {
    "all_included": {
        "contexts": 4873, "flagged": 3279, "flagged_via_feature": 2791, "flagged_via_target": 1304, "flagged_target_only": 488,
        "considered_columns": 100837, "flagged_columns": 12940, "flagged_feature_columns": 11636, "test_only_min": 5573,
        "test_only_max": 8153, "test_only_both": 786, "flagged_columns_per_context": _PER_CTX_ALL,
        "constant_on_R_columns": 88, "constant_on_R_contexts": 60, "b_nonidentity_contexts": 3728,
        "b_nonidentity_columns": 26182, "u_control_eligible": 1143, "exact01_contexts": 2994, "out01_contexts": 185,
        "subsampled_contexts": 1958, "subsampled_flagged": 1438, "floor_contexts": 654, "floor_columns": 3151,
        "floor_varying_contexts": 601, "floor_remapped_contexts": 229, "floor_flagged_contexts": 169, "D_rows": 63373,
        "D_rows_max": 1775, "c2_residual_contexts": 1183, "foundation_floor_risk_contexts": 3,
        "target_floor_risk_contexts": 0, "san_contexts": 45, "san_flagged": 34, "k_gt_128_contexts": 63},
    "unsealed": {
        "contexts": 4687, "flagged": 3189, "flagged_via_feature": 2728, "flagged_via_target": 1251, "flagged_target_only": 461,
        "considered_columns": 99472, "flagged_columns": 12782, "flagged_feature_columns": 11531, "test_only_min": 5504,
        "test_only_max": 8048, "test_only_both": 770, "flagged_columns_per_context": _PER_CTX_UNSEALED,
        "b_nonidentity_contexts": 3632, "u_control_eligible": 1053, "exact01_contexts": 2821, "out01_contexts": 185,
        "subsampled_contexts": 1940, "subsampled_flagged": 1428, "floor_remapped_contexts": 229,
        "floor_varying_contexts": 596, "D_rows": 60543, "c2_residual_contexts": 1154, "constant_on_R_contexts": 60,
        "constant_on_R_local_contexts": 18, "foundation_floor_risk_contexts": 3, "foundation_floor_risk_local_contexts": 0,
        "k_gt_128_contexts": 63, "k_gt_128_local_contexts": 27, "severity_contexts": 3185, "severity_gt_100": 9,
        "severity_gt_100_local": 2},
    "sealed": {"contexts": 186, "flagged": 90, "flagged_columns": 158, "considered_columns": 1365,
               "flagged_target_only": 27, "subsampled_contexts": 18},
}
# Severity quantiles over the unsealed flagged contexts with a varying flagged column, as printed in spec B (compared at
# the printed significant digits).
SPEC_B_SEVERITY_UNSEALED = {"p10": "0.0018", "p25": "0.0172", "p50": "0.0983", "p75": "0.357", "p90": "1.0", "p99": "16.1",
                            "max": "1.72e13"}
# Per stratum, unsealed: [flagged, contexts, target-only, floor_remapped, U-control eligible].
SPEC_B_STRATA_UNSEALED = {
    "frozen f0": [344, 536, 59, 22, 175], "frozen f1": [335, 536, 54, 23, 184], "frozen f2": [329, 536, 57, 27, 186],
    "frozen f3": [330, 534, 61, 21, 188], "frozen f4": [342, 524, 60, 16, 166], "incoming f0": [316, 401, 39, 25, 27],
    "incoming f1": [295, 413, 28, 26, 32], "incoming f2": [310, 408, 36, 24, 28], "incoming f3": [291, 404, 35, 23, 37],
    "incoming f4": [297, 395, 32, 22, 30],
}
SPEC_B_ANCHOR = {"cohort": "frozen", "sealed": False, "owner": "xbabe3", "local_complete": False, "n_train": 1600,
                 "n_test": 399, "F": 20, "k": 20, "flagged_cols": [6], "test_only_min": 1, "test_only_max": 0, "D_rows": 1,
                 "exact01_all_columns": True, "floor_columns": 2, "floor_remapped_by_B": 0}

COMPONENTS = ("hyperion", "foundation", "target", "tabpfn35", "context2048")
VARIANTS = ("A1", "B", "C", "C2")
DIAG_VARIANTS = ("B32",)
GATE_VARIANTS = ("B", "C", "C2")
A1_BASELINE_VARIANTS = ("B", "C", "C2")
FIX_VARIANTS = ("A1", "C", "C2")
GATE_KEYS = tuple(f"{v}_vs_A" for v in GATE_VARIANTS)
FIX_GATE_KEYS = tuple(f"fix_{v}_vs_B" for v in FIX_VARIANTS)
DIAG_KEYS = ("B32_vs_A", "B_vs_B32")
CLASSES = ("IDENTICAL", "NUMERICAL", "STRUCTURAL")
STRUCT_ABS = 2.0 ** -12
TWO24 = 2.0 ** 24
NEAR_ZERO_REL = 1e-10          # spec A P13: train-normalized values in (0, 1e-10) are wiped by a variant's float64 offset
ULP_BINS = (("0", 0, 0), ("1", 1, 1), ("2", 2, 2), ("3-4", 3, 4), ("5-16", 5, 16), ("17-256", 17, 256), (">256", 257, None))
HYPERION_SCALES = (4, 7, 11, 15)
TABPFN_CAT_MIN_SAMPLES = 100   # MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE: inference needs n_ctx > 100
TABPFN_NUM_MIN_UNIQUE = 4      # MIN_UNIQUE_FOR_NUMERICAL_FEATURES: inferred categorical iff fewer distinct values
MOD_CONSTANT, MOD_CATEGORICAL, MOD_NUMERICAL = 0, 1, 2
STRATA = tuple((cohort, fold) for cohort in ("frozen", "incoming") for fold in range(5))
OUT_NAME_RE = re.compile(r"^scale-check-[A-Za-z0-9][A-Za-z0-9._-]*$")
RECEIPT_NAME_RE = re.compile(r"^UPSTREAM-SCALE-01\.check-[0-9]{8}T[0-9]{6}Z\.json$")
POSITION_FIELDS_F32 = ["row", "col", "a_bits", "v_bits", "ulp", "abs_x2^24"]
POSITION_FIELDS_U8 = ["row", "col", "u_a", "u_v", "delta", "t_a_bits", "t_v_bits"]


class ScaleCheckError(RuntimeError):
    """A pin, lineage, replication or processing failure: the run is not valid evidence (exit 2)."""


# The bytes that ran: hashed at import, so a later edit of the file cannot change a run's recorded identity.
MODULE_SHA256 = common.sha256_file(Path(__file__))


# --------------------------------------------------------------------------------------------- Codex replica (numpy)
def codex_canonical(x) -> bytes:
    """custody.py:10, copied (ensure_ascii stays at its default)."""
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def codex_job_id(dataset: dict, fold: dict, component: str, packet: dict) -> str:
    """worker.py:10-11, copied."""
    return hashlib.sha256(codex_canonical(dict(
        schema="neural-encoding-job/v1", identity=dataset["identity"], manifest_ref=dataset["manifest_ref"],
        train_indices_sha256=fold["train_indices_sha256"], train_row_ids_sha256=fold["train_row_ids_sha256"],
        fold=fold["fold"], component=component, model_refs=packet["model_refs"],
        encoder_source_sha256=packet["source_pins"]["gpu_encoders.py"]["sha256"]))).hexdigest()


def codex_shard_owner(identity, fold: int) -> str:
    """worker.py:70: NODES[int(sha(canonical([identity, fold]))[:16], 16) % 3]."""
    return CODEX_NODES[int(hashlib.sha256(codex_canonical([list(identity), int(fold)])).hexdigest()[:16], 16) % 3]


def indices_hash(v) -> str:
    """inventory.py:21: sha256 over 8-byte little-endian indices (non-negative)."""
    a = np.asarray(v, dtype=np.int64)
    if a.ndim != 1 or (a.size and int(a.min()) < 0):
        raise ScaleCheckError("indices must be a 1-D list of non-negative integers")
    return hashlib.sha256(a.astype("<i8").tobytes()).hexdigest()


def records(raw):
    """worker.py:43-54 from `raw = a[t]` on: finiteness, the train-only target-leak exclusion in float64 (absolute
    thresholds 1e-12 and 1e-10, BLAS dot products), kept columns in original order, then the float32 casts."""
    raw = np.asarray(raw)
    if raw.dtype != np.float64 or raw.ndim != 2 or raw.shape[1] < 2:
        raise ScaleCheckError(f"records needs a float64 (n, F+1) matrix, got {raw.dtype} {raw.shape}")
    if not np.isfinite(raw).all():
        raise ScaleCheckError("non-finite outer-train rows (worker.py:43 assert)")
    X, y = raw[:, :-1], raw[:, -1]
    removed = []
    if np.ptp(y) > 1e-12:
        centered = y - y.mean()
        den = float(centered @ centered)
        for i in range(X.shape[1]):
            v = X[:, i]
            slope = float(centered @ (v - v.mean())) / den
            if abs(slope) > 1e-12 and np.max(abs(v - (slope * y + (v.mean() - slope * y.mean())))) <= \
                    1e-10 * max(1, float(np.max(abs(v)))):
                removed.append(i)
    kept = [i for i in range(X.shape[1]) if i not in removed]
    if not kept:
        raise ScaleCheckError("No non-target training features (worker.py:52)")
    X = X[:, kept].astype("float32")
    y = y.astype("float32")
    if not (np.isfinite(X).all() and np.isfinite(y).all()):
        raise ScaleCheckError("non-finite float32 inputs (worker.py:53 assert)")
    return X, y, {"kept": kept, "removed": removed, "n_train": int(raw.shape[0])}


def _lerp_np126(s, q):
    """numpy 1.26.4 np.percentile(X, q, axis=0), method 'linear', scalar q, on s = the float32 columns sorted ascending:
    q = true_divide(q, 100) (function_base.py:4279); vi = (n-1)*q (:109-111); previous = floor(vi), next = previous+1,
    both -1 when vi >= n-1 and 0 when vi < 0 (:4730-4762); gamma = vi - previous stays a float64 (1,) array
    (:4839-4841), so _lerp (:4641-4662) runs in float64: a + d*g when g < 0.5, else b - d*(1-g), with d = fl32(b - a).
    Returns float64; Codex casts with .astype('float32'). numpy 2.x passes a weak scalar gamma and lerps in float32."""
    n = s.shape[0]
    qf = np.true_divide(q, 100)
    vi = (n - 1) * qf
    if vi >= n - 1:
        prev = nxt = -1
    elif vi < 0:
        prev = nxt = 0
    else:
        prev = int(np.floor(vi))
        nxt = prev + 1
    g = np.float64(vi - prev)
    a, b = s[prev], s[nxt]
    d = (b - a).astype(np.float64)
    if g < 0.5:
        return a.astype(np.float64) + d * g
    return b.astype(np.float64) - d * (np.float64(1) - g)


def percentile_np126(X32, q):
    """Column percentiles of a float32 matrix exactly as Codex computed them (numpy 1.26.4), as float32."""
    X32 = np.asarray(X32)
    if X32.dtype != np.float32 or X32.ndim != 2 or X32.shape[0] < 1:
        raise ScaleCheckError("percentile_np126 needs a non-empty float32 matrix")
    return _lerp_np126(np.sort(X32, axis=0), q).astype("float32")


def hyperion_inputs(X32):
    """gpu_encoders.py:74. u is the only data-dependent model input; t = (X - lo)/span*255 before clip/truncation."""
    s = np.sort(X32, axis=0)
    lo = _lerp_np126(s, 1).astype("float32")
    hi = _lerp_np126(s, 99).astype("float32")
    span = np.where(hi - lo < 1e-9, 1, hi - lo)
    t = (X32 - lo) / span * 255
    u = np.clip(t, 0, 255).astype("uint8")
    if span.dtype != np.float32 or t.dtype != np.float32:
        raise ScaleCheckError(f"hyperion dtype drift: span {span.dtype}, t {t.dtype}")
    return {"u": u, "t": t, "lo": lo, "hi": hi, "span": span, "floor": (hi - lo) < 1e-9}


def hyperion_geometry(h: int, w: int):
    """gpu_encoders.py:77-78: sweep 0 is the identity; sweeps 1 and 2 draw the row then the column permutation."""
    perms = []
    for sweep in range(3):
        rng = np.random.default_rng(20260623 + sweep)
        pr = np.arange(h) if sweep == 0 else rng.permutation(h)
        pc = np.arange(w) if sweep == 0 else rng.permutation(w)
        perms.append((pr, pc))
    return perms


def hyperion_tiles(perms, diff_mask):
    """(total, differing) model-input tiles over 3 sweeps x 4 window sizes (gpu_encoders.py:79-84): a tile differs iff it
    holds a differing u cell, because xt is an injective function of the canvas."""
    h, w = diff_mask.shape
    rows, cols = np.nonzero(diff_mask)
    total = differ = 0
    for pr, pc in perms:
        inv_r = np.empty(h, np.int64)
        inv_r[pr] = np.arange(h)
        inv_c = np.empty(w, np.int64)
        inv_c[pc] = np.arange(w)
        for ws in HYPERION_SCALES:
            nr, nc = -(-h // ws), -(-w // ws)
            total += nr * nc
            if rows.size:
                differ += int(np.unique((inv_r[rows] // ws) * nc + inv_c[cols] // ws).size)
    return total, differ


def sampling(n: int):
    """gpu_encoders.py:19-22, copied."""
    if n < 2:
        raise ScaleCheckError("At least two disjoint training rows required")
    p = np.random.default_rng(20260707).permutation(n)
    nc = min(768, max(1, 3 * n // 4), n - 1)
    nq = min(256, n - nc)
    return p[:nc], p[nc:nc + nq]


def mean_np126(a32):
    """numpy 1.26.4 a.mean() of a 1-D float32 array (_methods.py _mean): float32 pairwise sum (umr_sum), then
    ret.dtype.type(ret / rcount) with rcount an np.intp, i.e. a float64 division cast back to float32."""
    a32 = np.asarray(a32)
    if a32.dtype != np.float32 or a32.ndim != 1 or a32.size < 1:
        raise ScaleCheckError("mean_np126 needs a non-empty 1-D float32 array")
    return np.float32(np.float64(np.add.reduce(a32)) / np.float64(a32.shape[0]))


def std_np126(a32):
    """numpy 1.26.4 a.std() (ddof 0) of a 1-D float32 array (_methods.py _var/_std): arrmean = umr_sum(keepdims=True)
    divided in place by an np.intp, which value-based casting makes a float32 division; x = (a - arrmean)**2 in float32;
    float32 pairwise sum; ret.dtype.type(ret / rcount) in float64; float32 sqrt. numpy 2 (NEP 50) does the first division
    in float64 and casts, so the two can differ on a double-rounding tie (spec A C2.4)."""
    a32 = np.asarray(a32)
    if a32.dtype != np.float32 or a32.ndim != 1 or a32.size < 1:
        raise ScaleCheckError("std_np126 needs a non-empty 1-D float32 array")
    n = a32.shape[0]
    m = np.true_divide(np.add.reduce(a32, keepdims=True), np.float32(n), dtype=np.float32)
    x = a32 - m
    x = np.multiply(x, x)
    v = np.float32(np.float64(np.add.reduce(x)) / np.float64(n))
    return np.float32(np.sqrt(v))


def foundation_inputs(X32, y32):
    """gpu_encoders.py:92-96: model inputs x = v[ctx] then v[qry] chunks, and yz."""
    ctx, qry = sampling(len(X32))
    lo = X32.min(0)
    span = X32.max(0) - lo
    span = np.where(span > 1e-12, span, 1)
    v = ((X32 - lo) / span).astype("float32")
    cy = y32[ctx]
    std32 = std_np126(cy)
    mu = float(mean_np126(cy))
    sd = float(std32) + 1e-8
    yz = (cy - mu) / sd
    if span.dtype != np.float32 or yz.dtype != np.float32:
        raise ScaleCheckError(f"foundation dtype drift: span {span.dtype}, yz {yz.dtype}")
    return {"ctx": ctx, "qry": qry, "x_ctx": v[ctx], "x_qry": v[qry], "yz": yz, "lo": lo, "span": span, "mu": mu,
            "sd": sd, "chunk": 32 if X32.shape[1] > 128 else 256,
            "std_np2_native_equal": bool(np.float32(cy.std()).view(np.uint32) == std32.view(np.uint32)),
            "mean_np2_native_equal": bool(float(cy.mean()) == mu)}


def stable_seed(*parts) -> int:
    """vendor/target_maker/data.py:67-74, copied."""
    digest = hashlib.blake2b("|".join(map(str, parts)).encode("utf-8"), digest_size=8).hexdigest()
    return int(digest, 16) & 0x7FFF_FFFF_FFFF_FFFF


def sample_indices(length: int, window_size: int, rng):
    """vendor/target_maker/data.py:324-330 (the index array, i.e. sample_indices(...)[0])."""
    if length <= 0:
        raise ScaleCheckError("Cannot sample from an empty target")
    if length >= window_size:
        return rng.choice(length, size=window_size, replace=False).astype(np.int64)
    return rng.choice(length, size=window_size, replace=True).astype(np.int64)


def window_indices(length: int, window_size: int, stride: int, max_windows, seed: int):
    """vendor/target_maker/encode.py:34-49 (_window_indices), copied."""
    if length <= 0:
        raise ScaleCheckError("Cannot encode an empty target")
    rng = np.random.default_rng(stable_seed("encode", seed, length, window_size, stride))
    if length < window_size:
        return [rng.choice(length, size=window_size, replace=True).astype(np.int64)]
    shuffled = rng.permutation(length)
    windows = []
    stride = max(1, stride)
    for start in range(0, length - window_size + 1, stride):
        windows.append(shuffled[start:start + window_size].astype(np.int64))
        if max_windows is not None and len(windows) >= max_windows:
            break
    if not windows:
        windows.append(shuffled[:window_size].astype(np.int64))
    return windows


def target_inputs(y32):
    """gpu_encoders.py:103-104. train_target_span emulates numpy 1.26.4, where np.float32 - float is a float64; the
    division then casts that Python float to float32, which gives the same v bits in both versions."""
    n = len(y32)
    lo = float(y32.min())
    span = float(np.float64(y32.max()) - lo)
    v = ((y32 - lo) / (span if span > 1e-12 else 1)).clip(0, 1).astype("float32")
    c = np.asarray([sample_indices(n, 32, np.random.default_rng(stable_seed("fusion-signature-inference", 17, 1, n, i, 32)))
                    for i in range(2)])
    q = window_indices(n, 32, 32, 2, 17)
    q = np.asarray((q + q)[:2])
    if v.dtype != np.float32:
        raise ScaleCheckError(f"target dtype drift: {v.dtype}")
    return {"v": v, "c": c, "q": q, "classical": v[c][None], "neural": v[q][None], "lo": lo, "span": span,
            "lengths": np.asarray([n], np.float32), "task": np.asarray([1], np.int64)}


def tabpfn_inputs(X32, y32):
    """gpu_encoders.py:109-110: raw upstream-unit arrays handed to TabPFN fit and get_embeddings; no Codex scaling."""
    ctx, qry = sampling(len(X32))
    return {"ctx": ctx, "qry": qry, "X_ctx": X32[ctx], "y_ctx": y32[ctx], "X_qry": X32[qry]}


def codex_inputs(raw):
    """Every component's training-side input for one variant's outer-train matrix (rows in R order)."""
    X32, y32, binding = records(raw)
    return {"binding": binding, "n": int(X32.shape[0]), "k": int(X32.shape[1]),
            "hyperion": hyperion_inputs(X32), "foundation": foundation_inputs(X32, y32),
            "target": target_inputs(y32), "tabpfn35": tabpfn_inputs(X32, y32)}


# ------------------------------------------------------------------------------------------------------- variants
def phi(M, lo, hi):
    """(x - lo)/(hi - lo) in IEEE float64, subtract then divide (no reciprocal, no FMA); divisor 1 where hi == lo."""
    r = hi - lo
    r = np.where(r == 0, 1.0, r)
    return (M - lo) / r


def column_extremes(A, R):
    AR = A[np.asarray(R, dtype=np.int64)]
    return AR.min(0), AR.max(0), A.min(0), A.max(0)


def test_only_extreme_rows(A, T, cols, lo_R, hi_R, lo_all, hi_all):
    """D: the test rows holding a full-row extreme of a listed column that the outer-train rows do not reach."""
    D = set()
    Tn = np.asarray(T, dtype=np.int64)
    for j in cols:
        col = A[Tn, j]
        if lo_R[j] > lo_all[j]:
            D.update(int(t) for t in Tn[col == lo_all[j]])
        if hi_R[j] < hi_all[j]:
            D.update(int(t) for t in Tn[col == hi_all[j]])
    return sorted(D)


def c2_fit(A, R, T, considered, lo_R, hi_R, lo_all, hi_all):
    """Variant C2's fit population S = R + (T - D), single pass. Returns (D, lo_S, hi_S, residual): residual counts the
    considered columns that still have a test-only extreme over S (iterating would converge to B)."""
    Tn = np.asarray(T, dtype=np.int64)
    D = test_only_extreme_rows(A, Tn, considered, lo_R, hi_R, lo_all, hi_all)
    if D:
        S = np.concatenate([np.asarray(R, dtype=np.int64), np.setdiff1d(Tn, np.asarray(D, dtype=np.int64))])
        lo_S, hi_S = A[S].min(0), A[S].max(0)
    else:
        lo_S, hi_S = lo_all, hi_all
    cons = np.asarray(considered, dtype=np.int64)
    residual = int(np.count_nonzero((lo_R[cons] > lo_S[cons]) | (hi_R[cons] < hi_S[cons])))
    return D, lo_S, hi_S, residual


def c_perturbation(A, T):
    """Variant C's perturbed matrix A' (train rows untouched) and the count of columns whose test minimum and maximum sit
    in one row (a column constant on T); for those the maximum moves to the next test row, wrapping to T[0]."""
    Tn = np.asarray(T, dtype=np.int64)
    if Tn.size == 0:
        raise ScaleCheckError("variant C needs at least one outer-test row")
    lo_all, hi_all = A.min(0), A.max(0)
    r = hi_all - lo_all
    r = np.where(r == 0, 1.0, r)
    At = A.copy()
    single = 0
    for j in range(A.shape[1]):
        col = A[Tn, j]
        t_lo = int(Tn[int(np.argmax(col == col.min()))])
        t_hi = int(Tn[int(np.argmax(col == col.max()))])
        if t_lo == t_hi:
            single += 1
            t_hi = int(Tn[(int(np.searchsorted(Tn, t_lo)) + 1) % len(Tn)])
        At[t_lo, j] = lo_all[j] - 0.25 * r[j]
        At[t_hi, j] = hi_all[j] + 0.5 * r[j]
    return At, single


def build_variants(A, R, T, considered, return_fits=False):
    """The spec B variant outer-train matrices (n_train x (F+1), float64, R order) plus the B32 diagnostic, and their
    test-row metadata (counts only). A, B and B32 never read a test row. With return_fits, also the (lo, hi) each
    perturbation variant was fitted on, in A units (internal: test-derived constants, never written)."""
    R = np.asarray(R, dtype=np.int64)
    Tn = np.asarray(T, dtype=np.int64)
    AR = A[R]
    lo_R, hi_R, lo_all, hi_all = column_extremes(A, R)
    out = {"A": AR, "A1": phi(AR, lo_all, hi_all), "B": phi(AR, lo_R, hi_R)}
    A32 = AR.astype(np.float32).astype(np.float64)
    out["B32"] = phi(A32, A32.min(0), A32.max(0))
    At, single = c_perturbation(A, Tn)
    lo_C, hi_C = At.min(0), At.max(0)
    out["C"] = phi(At[R], lo_C, hi_C)
    D, lo_S, hi_S, residual = c2_fit(A, R, Tn, considered, lo_R, hi_R, lo_all, hi_all)
    out["C2"] = phi(AR, lo_S, hi_S) if D else out["A1"].copy()
    meta = {"D_rows": len(D), "C_single_row_columns": single, "C2_residual_test_only_extreme_columns": residual}
    if return_fits:
        return out, meta, {"A1": (lo_all, hi_all), "C": (lo_C, hi_C), "C2": (lo_S, hi_S)}
    return out, meta


def near_zero_columns(A, R, considered, F):
    """Considered columns whose train-normalized values B = phi_R(A[R]) include one in (0, NEAR_ZERO_REL): a variant's
    float64 offset can wipe such values, so a fix-route difference there is a variant-construction artifact (spec A P13)."""
    AR = A[np.asarray(R, dtype=np.int64)]
    B = phi(AR, AR.min(0), AR.max(0))
    out = []
    for j in considered:
        b = B[:, j]
        if bool(np.any((b > 0) & (b < NEAR_ZERO_REL))):
            out.append("target" if j == F else int(j))
    return out


def construction_risk_columns(A, R, considered, F, fits):
    """Per fix variant V (A1, C, C2): the considered columns where V's float64 construction cannot keep the float32
    resolution of the train-normalized values. V = (s - lo_V)/(hi_V - lo_V) rounds s - lo_V to 2^-53 relative; in
    train-normalized units that is about 2^-53 * |lo_R - lo_V| / (hi_R - lo_R) absolute, which exceeds half a float32 ULP
    of the smallest positive B value when 2^-27 * |lo_R - lo_V| / (hi_R - lo_R) > min(B > 0). This generalizes spec A
    P13's near-zero rule (B < 1e-10 under C's -0.25r offset) to severity-amplified offsets (the S-FLOOR contexts)."""
    AR = A[np.asarray(R, dtype=np.int64)]
    lo_R, hi_R = AR.min(0), AR.max(0)
    B = phi(AR, lo_R, hi_R)
    out = {}
    for name, (lo_V, _hi_V) in fits.items():
        cols = []
        for j in considered:
            span = float(hi_R[j] - lo_R[j])
            if span == 0.0:
                continue                      # constant on R: B and F_V are both exactly 0
            pos = B[:, j][B[:, j] > 0]
            fmin = float(pos.min()) if pos.size else 1.0
            if abs(float(lo_R[j]) - float(lo_V[j])) / span * 2.0 ** -27 > fmin:
                cols.append("target" if j == F else int(j))
        out[name] = cols
    return out


# ----------------------------------------------------------------------------------------------------- comparison
def _ordered_f32(a):
    i = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    return np.where(i >= 0, i, -(i & 0x7FFFFFFF))


def ulp_distance_f32(a, b):
    """|k_a - k_b|, k = i if i >= 0 else -(i & 0x7FFFFFFF) over the int32 view; +0.0 and -0.0 are 0 apart."""
    return np.abs(_ordered_f32(a) - _ordered_f32(b))


def ulp_distance_f64(a: float, b: float) -> int:
    """The float64 analogue for scalars (side values): |k_a - k_b| over the int64 view."""
    def key(x):
        i = int(np.asarray(x, dtype=np.float64).view(np.int64))
        return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
    return abs(key(a) - key(b))


def _bits(x) -> str:
    return "0x%08x" % int(np.asarray(x, dtype=np.float32).view(np.uint32))


def _ulp_hist(ulp):
    return {name: int(np.count_nonzero((ulp >= lo) & (ulp <= hi))) if hi is not None else int(np.count_nonzero(ulp >= lo))
            for name, lo, hi in ULP_BINS}


def _col_label(cols, j):
    return "target" if cols is None or j is None else int(cols[j])


def _differ_by_column(neq, cols):
    """{data-object column (or 'target'): differing elements}, nonzero entries only."""
    if neq.ndim == 2 and cols is not None:
        per = np.count_nonzero(neq, axis=0)
        return {str(int(cols[j])): int(per[j]) for j in np.flatnonzero(per)}
    nd = int(np.count_nonzero(neq))
    return {"target": nd} if nd else {}


def cmp_f32(a, v, rows, cols, cap: int, *, scale=None):
    """Element comparison of a reference and a variant float32 array of one shape. rows maps the first axis to
    data-object rows, cols the second axis to data-object columns (None: a 1-D target array). `scale` multiplies the
    structural threshold 2^-12 per element (yz: max(1, |yz_A|))."""
    a, v = np.asarray(a), np.asarray(v)
    if a.dtype != np.float32 or v.dtype != np.float32:
        raise ScaleCheckError(f"cmp_f32 needs float32 arrays, got {a.dtype} and {v.dtype}")
    a, v = np.ascontiguousarray(a), np.ascontiguousarray(v)
    if a.shape != v.shape:
        return {"kind": "f32", "shape_a": list(a.shape), "shape_v": list(v.shape), "shape_mismatch": True,
                "elements": int(a.size), "differ": int(a.size), "bit_identical": 0, "exceed_2^-12": int(a.size)}
    neq = a.view(np.uint32) != v.view(np.uint32)
    ulp = ulp_distance_f32(a, v)
    absd = np.abs(a.astype(np.float64) - v.astype(np.float64))
    thr = STRUCT_ABS if scale is None else STRUCT_ABS * scale
    exceed = absd > thr
    nd = int(np.count_nonzero(neq))
    out = {"kind": "f32", "shape": list(a.shape), "elements": int(a.size), "bit_identical": int(a.size) - nd,
           "bit_identical_fraction": (int(a.size) - nd) / a.size if a.size else 1.0,
           "differ": nd, "signed_zero_flips": int(np.count_nonzero(neq & (ulp == 0))),
           "ulp_hist": _ulp_hist(ulp), "max_ulp": int(ulp.max()) if a.size else 0,
           "max_abs_x2^24": float(absd.max() * TWO24) if a.size else 0.0, "exceed_2^-12": int(np.count_nonzero(exceed))}
    if nd:
        out["differ_by_column"] = _differ_by_column(neq, cols)
    if cap > 0 and nd:
        flat = np.flatnonzero(neq.ravel())
        pos = []
        for e in flat[:cap]:
            idx = np.unravel_index(int(e), a.shape)
            i = int(idx[0])
            j = int(idx[1]) if a.ndim == 2 else None
            pos.append([int(rows[i]), _col_label(cols, j), _bits(a.flat[e]), _bits(v.flat[e]), int(ulp.flat[e]),
                        float(absd.flat[e] * TWO24)])
        out["positions"] = pos
        out["positions_truncated"] = nd > cap
    return out


def cmp_u8(ua, uv, ta, tv, rows, cols, floor_cols, cap: int):
    """uint8 comparison of hyperion u: delta histogram, |delta| = 1 versus >= 2, flips in span-floor columns (floor in
    either variant), and flips where the clip was active in either variant (saturation)."""
    if ua.dtype != np.uint8 or uv.dtype != np.uint8:
        raise ScaleCheckError(f"cmp_u8 needs uint8 arrays, got {ua.dtype} and {uv.dtype}")
    if ua.shape != uv.shape:
        return {"kind": "u8", "shape_a": list(ua.shape), "shape_v": list(uv.shape), "shape_mismatch": True,
                "elements": int(ua.size), "flips": int(ua.size), "bit_identical": 0}
    d = uv.astype(np.int16) - ua.astype(np.int16)
    neq = d != 0
    nf = int(np.count_nonzero(neq))
    floor2d = np.broadcast_to(np.asarray(floor_cols, dtype=bool)[None, :], ua.shape)
    sat = (ta < 0) | (ta > 255) | (tv < 0) | (tv > 255)
    hist = collections.Counter(int(x) for x in d[neq])
    out = {"kind": "u8", "shape": list(ua.shape), "elements": int(ua.size), "bit_identical": int(ua.size) - nf,
           "bit_identical_fraction": (int(ua.size) - nf) / ua.size if ua.size else 1.0,
           "flips": nf, "flips_abs1": int(np.count_nonzero(np.abs(d) == 1)),
           "flips_ge2": int(np.count_nonzero(np.abs(d) >= 2)),
           "floor_column_flips": int(np.count_nonzero(neq & floor2d)),
           "saturation_flips": int(np.count_nonzero(neq & sat)),
           "nonfloor_abs1_flips": int(np.count_nonzero((np.abs(d) == 1) & ~floor2d)),
           "max_abs_delta": int(np.abs(d).max()) if ua.size else 0,
           "delta_hist": {str(k): hist[k] for k in sorted(hist)}}
    if nf:
        out["differ_by_column"] = _differ_by_column(neq, cols)
    if cap > 0 and nf:
        flat = np.flatnonzero(neq.ravel())
        pos = []
        for e in flat[:cap]:
            i, j = np.unravel_index(int(e), ua.shape)
            pos.append([int(rows[int(i)]), int(cols[int(j)]), int(ua.flat[e]), int(uv.flat[e]), int(d.flat[e]),
                        _bits(ta.flat[e]), _bits(tv.flat[e])])
        out["positions"] = pos
        out["positions_truncated"] = nf > cap
    return out


def _side_f32(a, v):
    a = np.ascontiguousarray(a, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    if a.shape != v.shape:
        return {"identical": False, "shape_mismatch": True}
    neq = a.view(np.uint32) != v.view(np.uint32)
    return {"identical": not bool(neq.any()), "differ": int(np.count_nonzero(neq)),
            "max_ulp": int(ulp_distance_f32(a, v).max()) if a.size else 0}


def _side_scalar(a: float, v: float):
    """Identical or differing with the float64 ULP distance; the values themselves are not written."""
    ulp = ulp_distance_f64(a, v)
    return {"identical": ulp == 0 and bool(np.signbit(a)) == bool(np.signbit(v)), "ulp64": int(ulp)}


def _worst(classes):
    return max(classes, key=CLASSES.index) if classes else "IDENTICAL"


def _klass(differ: bool, structural: bool):
    return "STRUCTURAL" if structural else "NUMERICAL" if differ else "IDENTICAL"


def tabpfn_modalities(Xctx):
    """modality_detection.py (pinned in spec A): per column of X[ctx], CONSTANT if at most one distinct value, CATEGORICAL
    if n_ctx > 100 and fewer than 4 distinct values, else NUMERICAL (-0.0 equals 0.0, as in nunique)."""
    n, k = Xctx.shape
    if n == 0:
        return np.zeros(k, dtype=np.int64)
    s = np.sort(Xctx, axis=0)
    nunique = 1 + np.count_nonzero(s[1:] != s[:-1], axis=0)
    cat = (n > TABPFN_CAT_MIN_SAMPLES) & (nunique < TABPFN_NUM_MIN_UNIQUE)
    return np.where(nunique <= 1, MOD_CONSTANT, np.where(cat, MOD_CATEGORICAL, MOD_NUMERICAL))


def _tabpfn_codes(ctx_col, qry_col):
    """Ordinal codes over the sorted distinct context values; a query value outside them is -1."""
    cats = np.unique(ctx_col)
    idx = np.searchsorted(cats, qry_col)
    known = (idx < cats.size) & (cats[np.minimum(idx, cats.size - 1)] == qry_col)
    return np.searchsorted(cats, ctx_col), np.where(known, idx, -1)


def tabpfn_refined(pa, pv, kept, same_cols):
    """Spec A P9 refinement of the tabpfn35 class (informational; the gate keeps the spec B rule 'any difference is
    STRUCTURAL'). Inert: a column constant on X[ctx] in both (removed before use), or inferred-categorical in both with
    identical ordinal codes for every context and query row. Numerical: y_ctx (float32 z-score, target transform None)
    and numerical-column changes that leave every row's fingerprint input np.around(float64(row), 12) unchanged.
    Structural: a modality change, a categorical code change, or a fingerprint change (the fingerprint hashes absolute
    values). Classes: IDENTICAL, INERT_ONLY, NUMERICAL, STRUCTURAL."""
    y_differ = bool(np.count_nonzero(pa["y_ctx"].view(np.uint32) != pv["y_ctx"].view(np.uint32)))
    if not same_cols:
        return {"class": "STRUCTURAL", "reason": "kept set, removed set or n differs", "y_ctx_differs": y_differ}
    xa_c, xv_c, xa_q, xv_q = pa["X_ctx"], pv["X_ctx"], pa["X_qry"], pv["X_qry"]
    diff = ((xa_c.view(np.uint32) != xv_c.view(np.uint32)).any(axis=0)
            | (xa_q.view(np.uint32) != xv_q.view(np.uint32)).any(axis=0))
    ma, mv = tabpfn_modalities(xa_c), tabpfn_modalities(xv_c)
    kinds = collections.Counter()
    structural, numerical = [], []
    for j in np.flatnonzero(diff):
        if ma[j] != mv[j]:
            kind = "modality_changed"
        elif ma[j] == MOD_CONSTANT:
            kind = "constant_inert"
        elif ma[j] == MOD_CATEGORICAL:
            ca, cv = _tabpfn_codes(xa_c[:, j], xa_q[:, j]), _tabpfn_codes(xv_c[:, j], xv_q[:, j])
            same = np.array_equal(ca[0], cv[0]) and np.array_equal(ca[1], cv[1])
            kind = "categorical_inert" if same else "categorical_codes_changed"
        else:
            kind = "numerical_changed"
            numerical.append(int(j))
        kinds[kind] += 1
        if kind in ("modality_changed", "categorical_codes_changed"):
            structural.append(int(kept[j]))
    fp_rows = 0
    if numerical:
        nc = np.asarray(numerical, dtype=np.int64)
        for xa, xv in ((xa_c, xv_c), (xa_q, xv_q)):
            fa = np.around(xa[:, nc].astype(np.float64), 12)
            fv = np.around(xv[:, nc].astype(np.float64), 12)
            fp_rows += int(np.count_nonzero((fa != fv).any(axis=1)))
    klass = ("STRUCTURAL" if structural or fp_rows else "NUMERICAL" if numerical or y_differ
             else "INERT_ONLY" if kinds else "IDENTICAL")
    return {"class": klass, "column_kinds": dict(sorted(kinds.items())), "fingerprint_rows_changed": fp_rows,
            "structural_columns": structural[:64], "structural_columns_total": len(structural),
            "y_ctx_differs": y_differ}


def compare_inputs(ref, var, R, cap: int = 0):
    """Compare every fed input (and the side arrays) of one variant against a reference; see the module docstring."""
    R = np.asarray(R, dtype=np.int64)
    out = {"components": {}, "side": {}}
    ba, bv = ref["binding"], var["binding"]
    kept_a = list(ba["kept"])
    same_cols = ba["kept"] == bv["kept"] and ba["removed"] == bv["removed"] and ref["n"] == var["n"]
    out["binding"] = {"kept_equal": ba["kept"] == bv["kept"], "removed_equal": ba["removed"] == bv["removed"],
                      "removed_a": list(ba["removed"]), "removed_v": list(bv["removed"]), "n_equal": ref["n"] == var["n"],
                      "class": "IDENTICAL" if same_cols else "STRUCTURAL"}
    comps = out["components"]

    # hyperion: u (the geometry is a function of (h, w) only)
    if same_cols:
        ha, hv = ref["hyperion"], var["hyperion"]
        floor = ha["floor"] | hv["floor"]
        u = cmp_u8(ha["u"], hv["u"], ha["t"], hv["t"], R, kept_a, floor, cap)
        tiles_total, tiles_differ = hyperion_tiles(hyperion_geometry(ref["n"], ref["k"]), ha["u"] != hv["u"])
        structural = u["flips_ge2"] > 0 or u["floor_column_flips"] > 0
        comps["hyperion"] = {"u": u, "tiles_total": tiles_total, "tiles_differ": tiles_differ,
                             "floor_columns_a": int(np.count_nonzero(ha["floor"])),
                             "floor_columns_v": int(np.count_nonzero(hv["floor"])),
                             "floor_status_flips": int(np.count_nonzero(ha["floor"] != hv["floor"])),
                             "class": _klass(u["flips"] > 0, structural)}
        out["side"]["hyperion"] = {"feature_q01": _side_f32(ha["lo"], hv["lo"]), "feature_q99": _side_f32(ha["hi"], hv["hi"])}
    else:
        comps["hyperion"] = {"class": "STRUCTURAL", "reason": "kept set, removed set or n differs"}

    # foundation: x = v[ctx] | v[qry] chunks, and yz
    fa, fv = ref["foundation"], var["foundation"]
    yz = cmp_f32(fa["yz"], fv["yz"], R[fa["ctx"]], None, cap, scale=np.maximum(1.0, np.abs(fa["yz"].astype(np.float64))))
    f = {"yz": yz}
    side_f = {"context_y_mean": _side_scalar(fa["mu"], fv["mu"]), "context_y_std": _side_scalar(fa["sd"], fv["sd"])}
    if same_cols:
        f["x_ctx"] = cmp_f32(fa["x_ctx"], fv["x_ctx"], R[fa["ctx"]], kept_a, cap)
        f["x_qry"] = cmp_f32(fa["x_qry"], fv["x_qry"], R[fa["qry"]], kept_a, cap)
        n_chunks = -(-len(fa["qry"]) // fa["chunk"])
        if f["x_ctx"]["differ"] or yz["differ"]:
            chunks_differ = n_chunks
        else:
            qd = np.flatnonzero((fa["x_qry"].view(np.uint32) != fv["x_qry"].view(np.uint32)).any(axis=1))
            chunks_differ = int(np.unique(qd // fa["chunk"]).size)
        f["chunks_total"], f["chunks_differ"] = n_chunks, chunks_differ
        differ = any(f[x]["differ"] for x in ("x_ctx", "x_qry", "yz"))
        structural = any(f[x]["exceed_2^-12"] for x in ("x_ctx", "x_qry", "yz"))
        side_f.update(feature_min=_side_f32(fa["lo"], fv["lo"]), feature_span=_side_f32(fa["span"], fv["span"]))
        f["class"] = _klass(differ, structural)
    else:
        f["class"] = "STRUCTURAL"
    comps["foundation"] = f
    out["side"]["foundation"] = side_f

    # target: v[classical], v[neural] (lengths and task depend on n only); the full v is a reported superset
    ta, tv = ref["target"], var["target"]
    t = {"classical": cmp_f32(ta["classical"].reshape(-1), tv["classical"].reshape(-1), R[ta["c"].reshape(-1)], None, cap),
         "neural": cmp_f32(ta["neural"].reshape(-1), tv["neural"].reshape(-1), R[ta["q"].reshape(-1)], None, cap),
         "lengths_task_equal": bool(np.array_equal(ta["lengths"], tv["lengths"]) and np.array_equal(ta["task"], tv["task"])),
         "superset_v": cmp_f32(ta["v"], tv["v"], R, None, 0)}
    differ = t["classical"]["differ"] or t["neural"]["differ"] or not t["lengths_task_equal"]
    structural = t["classical"]["exceed_2^-12"] or t["neural"]["exceed_2^-12"] or not t["lengths_task_equal"]
    t["class"] = _klass(bool(differ), bool(structural))
    comps["target"] = t
    out["side"]["target"] = {"train_target_min": _side_scalar(ta["lo"], tv["lo"]),
                             "train_target_span": _side_scalar(ta["span"], tv["span"])}

    # tabpfn35: raw arrays; any difference is structural under spec B (the fingerprint feature hashes raw values)
    pa, pv = ref["tabpfn35"], var["tabpfn35"]
    p = {"y_ctx": cmp_f32(pa["y_ctx"], pv["y_ctx"], R[pa["ctx"]], None, cap)}
    if same_cols:
        p["X_ctx"] = cmp_f32(pa["X_ctx"], pv["X_ctx"], R[pa["ctx"]], kept_a, cap)
        p["X_qry"] = cmp_f32(pa["X_qry"], pv["X_qry"], R[pa["qry"]], kept_a, cap)
        p["ctx_rows_differ"] = int(np.count_nonzero((pa["X_ctx"].view(np.uint32) != pv["X_ctx"].view(np.uint32)).any(axis=1)
                                                    | (pa["y_ctx"].view(np.uint32) != pv["y_ctx"].view(np.uint32))))
        any_diff = any(p[x]["differ"] for x in ("X_ctx", "y_ctx", "X_qry"))
        p["class"] = _klass(any_diff, any_diff)
    else:
        p["class"] = "STRUCTURAL"
    p["refined_p9"] = tabpfn_refined(pa, pv, kept_a, same_cols)
    comps["tabpfn35"] = p

    comps["context2048"] = {"class": _worst([comps[c]["class"] for c in ("hyperion", "foundation", "target")]),
                            "rule": "identical iff hyperion, foundation and target are identical; inherits their residual"}
    out["class"] = _worst([out["binding"]["class"]] + [comps[c]["class"] for c in COMPONENTS])
    return out


def _fed_arrays(cmp):
    """(component, array name, record) for every fed-input comparison record (the target superset is not fed)."""
    for comp, rec in cmp["components"].items():
        for name, sub in rec.items():
            if isinstance(sub, dict) and "kind" in sub and not name.startswith("superset"):
                yield comp, name, sub


def _col_order(c):
    return (c == "target", int(c) if c != "target" else 0)


def rounding_columns(MB, MV, considered, F, max_ulp32_fraction: float = 0.125):
    """Considered columns whose float32 casts (worker.py:53) differ bitwise between B and a fix-route matrix only by
    rounding, as {column: kind}: 'signed_zero' (both casts are zero with different signs: IEEE negative zeros in the
    data object survive phi_R's subtraction but not the simulated upstream's), 'straddle' (the casts are adjacent
    float32 values and the two float64 values are closer than max_ulp32_fraction of a float32 ULP: two float64
    roundings of one exact value on either side of a float32 midpoint, spec A P13's 'float32 midpoint ties after two
    float64 rescales'), or 'mixed'. Every fed array element is a function of its own column's float32 values (and n),
    so a column whose casts agree bitwise has no fed difference."""
    out = {}
    for j in considered:
        b, v = MB[:, j], MV[:, j]
        b32, v32 = b.astype(np.float32), v.astype(np.float32)
        diff = b32.view(np.uint32) != v32.view(np.uint32)
        if not bool(diff.any()):
            continue
        bd, vd, b32d, v32d = b[diff], v[diff], b32[diff], v32[diff]
        zero = (b32d == 0) & (v32d == 0)
        close = np.abs(bd - vd) <= max_ulp32_fraction * np.spacing(np.abs(b32d)).astype(np.float64)
        straddle = ~zero & (ulp_distance_f32(b32d, v32d) == 1) & close
        if bool((zero | straddle).all()):
            out["target" if j == F else int(j)] = ("signed_zero" if bool(zero.all()) else
                                                   "straddle" if bool(straddle.all()) else "mixed")
    return out


def fix_attribution(cmp, near_zero, risk=(), rounding=None):
    """Attribute a fix-route comparison's differing fed elements to near-zero columns (spec A P13: 0 < B < 1e-10) and
    construction-risk columns (construction_risk_columns), both artifacts of simulating the upstream in float64, and to
    rounding columns (rounding_columns: signed-zero flips and sub-ULP float32 rounding-boundary straddles); the rest is
    unexplained."""
    nz = {str(c) for c in near_zero}
    rk = {str(c) for c in risk}
    rounding = rounding or {}
    tc = {str(c) for c in rounding}
    per_col = collections.Counter()
    for _comp, _name, sub in _fed_arrays(cmp):
        for col, n in (sub.get("differ_by_column") or {}).items():
            per_col[col] += n
    total = sum(per_col.values())
    in_nz = sum(n for c, n in per_col.items() if c in nz)
    in_rk = sum(n for c, n in per_col.items() if c in rk)
    in_tc = sum(n for c, n in per_col.items() if c in tc)
    artifact = sum(n for c, n in per_col.items() if c in nz or c in rk)
    explained = sum(n for c, n in per_col.items() if c in nz or c in rk or c in tc)
    same_binding = cmp["binding"]["class"] == "IDENTICAL"
    return {"class": cmp["class"], "binding": cmp["binding"]["class"], "differing_elements": total,
            "elements_in_near_zero_columns": in_nz, "elements_elsewhere": total - in_nz,
            "elements_in_construction_risk_columns": in_rk, "elements_in_rounding_columns": in_tc,
            "elements_unexplained": total - explained,
            "differing_columns": sorted(per_col, key=_col_order), "near_zero_columns": sorted(nz, key=_col_order),
            "construction_risk_columns": sorted(rk, key=_col_order)[:64], "construction_risk_columns_total": len(rk),
            "rounding_columns": {str(c): rounding[c] for c in sorted(rounding, key=lambda c: _col_order(str(c)))[:64]},
            "rounding_kinds": dict(sorted(collections.Counter(rounding.values()).items())),
            "only_near_zero": total > 0 and in_nz == total and same_binding,
            "only_artifact": total > 0 and artifact == total and same_binding,
            "only_explained": total > 0 and explained == total and same_binding}


# ------------------------------------------------------------------------------------------------ JTT1 rebuild, A0
def inner_labels(M, outer_train):
    """research_2k prepare_fold0.inner_assignments / frozen_transfer.inner_assignments (labels only), copied: exact
    feature components by sha256 of the float64 row bytes, target-mean deciles, greedy balance; same-bin, same-size
    components are ordered by that digest, so the labels hash raw feature values."""
    components = {}
    for s in outer_train:
        components.setdefault(hashlib.sha256(np.asarray(M[s, :-1], dtype="<f8").tobytes()).hexdigest(), []).append(int(s))
    if len(components) < 2:
        raise ScaleCheckError("outer training has fewer than two exact-feature components")
    grouped = [(digest, indices, float(np.mean(M[indices, -1]))) for digest, indices in components.items()]
    means = np.asarray([g[2] for g in grouped], dtype=np.float64)
    order = np.argsort(means, kind="mergesort")
    bins = np.empty(len(grouped), dtype=np.int64)
    for rank, gi in enumerate(order):
        bins[gi] = min(9, rank * 10 // len(grouped))
    loads = [0, 0]
    bin_loads = np.zeros((2, 10), dtype=np.int64)
    assignment = {}
    for gi, (digest, indices, _mean) in sorted(enumerate(grouped),
                                               key=lambda item: (int(bins[item[0]]), -len(item[1][1]), item[1][0])):
        tb = int(bins[gi])
        target = min(range(2), key=lambda c: (bin_loads[c, tb], loads[c], c))
        assignment[digest] = target
        loads[target] += len(indices)
        bin_loads[target, tb] += len(indices)
    label = {s: assignment[d] for d, indices, _m in grouped for s in indices}
    return np.asarray([label[int(s)] for s in outer_train], dtype=np.uint8)


def jtt1_rebuild(A, R, T, kept_jtt1, labels_from_parent: bool):
    """(raw JTT1 bytes, inner labels): b'JTT1' + <IIIB(n_train, n_test, p, 2) + labels + A[R+T][:, kept+[F]] float64.
    v6 clean views take their labels from the unsanitized parent matrix (prepare_leakage_remediation.py:74)."""
    F = A.shape[1] - 1
    M = np.ascontiguousarray(A[:, list(kept_jtt1) + [F]], dtype="<f8")
    labels = inner_labels(A if labels_from_parent else M, R)
    ordered = np.asarray(list(R) + list(T), dtype=np.int64)
    raw = (b"JTT1" + struct.pack("<IIIB", len(R), len(T), len(kept_jtt1), 2) + labels.tobytes()
           + np.ascontiguousarray(M[ordered], dtype="<f8").tobytes(order="C"))
    return raw, labels


def jtt1_cache_check(ci: dict, raw_rebuilt: bytes, labels) -> dict | None:
    """Spec B P2 for a context whose JTT1 is in the local cache (no fetch, no S3 import): the cached gz and raw bytes
    verify against input_ref, equal the rebuild, and jtt1.train_view (train range only) yields the rebuilt labels,
    A[R][:, kept] and A[R, F] bit for bit."""
    ref = (ci.get("row") or {}).get("input_ref")
    if ref is None:
        return None
    path = JTT1_CACHE / f"{ref['raw_sha256']}.jtt1.gz"
    if not path.exists():
        return None
    gz = path.read_bytes()
    ok = common.sha256_bytes(gz) == ref["sha256"] and len(gz) == int(ref["bytes"])
    raw = gzip.decompress(gz) if ok else b""
    ok = ok and common.sha256_bytes(raw) == ref["raw_sha256"] and len(raw) == int(ref["raw_bytes"])
    out = {"jtt1_cache_verified": ok, "jtt1_cache_equals_rebuild": ok and raw == raw_rebuilt,
           "jtt1_cache_train_view_equal": False}
    if ok:
        _h, lab, X, y = jtt1.train_view(raw)
        AR = ci["A"][np.asarray(ci["R"], dtype=np.int64)]
        XR = np.ascontiguousarray(AR[:, list(ci["kept_jtt1"])])
        yR = np.ascontiguousarray(AR[:, ci["F"]])
        out["jtt1_cache_train_view_equal"] = bool(
            np.array_equal(lab, labels) and X.shape == XR.shape and y.shape == yR.shape
            and np.array_equal(np.ascontiguousarray(X).view(np.uint64), XR.view(np.uint64))
            and np.array_equal(np.ascontiguousarray(y).view(np.uint64), yR.view(np.uint64)))
    return out


def a0_compare(raw, A, R, kept_jtt1, labels, labels_from_parent: bool):
    """The lane's A0 vector (dltrain.metafeatures) of A, through from_raw on the rebuilt JTT1 (train range only), against
    B = phi_R(A[R]) with the inner labels held fixed. Also reports whether recomputed inner labels survive B."""
    try:
        from dltrain import metafeatures
    except Exception as exc:  # read-only import of a concurrently edited module; the check proceeds without A0
        return {"status": "skipped", "note": f"dltrain.metafeatures not importable: {type(exc).__name__}: {exc}"}
    R = np.asarray(R, dtype=np.int64)
    F = A.shape[1] - 1
    AR = A[R]
    VB = phi(AR, AR.min(0), AR.max(0))
    try:
        out = {"status": "computed", "module": metafeatures.MODULE, "version": metafeatures.VERSION,
               "source_tree_sha256": metafeatures.source_tree_sha256(),
               "feature_list_sha256": metafeatures.FEATURE_LIST_SHA256,
               "labels": "held fixed at the rebuilt JTT1 inner labels"}
        a_bytes, _info = metafeatures.from_raw(raw)
        direct = metafeatures.to_f32le(metafeatures.features(AR[:, list(kept_jtt1)], AR[:, F], labels))
        out["from_raw_equals_direct"] = a_bytes == direct
        b_bytes = metafeatures.to_f32le(metafeatures.features(VB[:, list(kept_jtt1)], VB[:, F], labels))
    except Exception as exc:
        name = type(exc).__name__
        return {"status": "refused" if name == "A0Refused" else "error", "error": f"{name}: {exc}"}
    a = np.frombuffer(a_bytes, dtype="<f4")
    b = np.frombuffer(b_bytes, dtype="<f4")
    neq = a.view(np.uint32) != b.view(np.uint32)
    ulp = ulp_distance_f32(a, b)
    absd = np.abs(a.astype(np.float64) - b.astype(np.float64))
    out.update(bit_identical=int(a.size - np.count_nonzero(neq)), features=int(a.size), differ=int(np.count_nonzero(neq)),
               ulp_hist=_ulp_hist(ulp), max_ulp=int(ulp.max()), max_abs=float(absd.max()),
               a0_a_sha256=common.sha256_bytes(a_bytes), a0_b_sha256=common.sha256_bytes(b_bytes),
               differing_features=[[metafeatures.FEATURES[i], int(ulp[i]), float(absd[i])] for i in np.flatnonzero(neq)])
    # A train-only rebuild would also recompute the inner labels; they order same-size components by a sha256 of the
    # raw row bytes, so they need not survive B. Report how many change and what that does to A0.
    lb_matrix = VB if labels_from_parent else np.concatenate([VB[:, list(kept_jtt1)], VB[:, [F]]], axis=1)
    try:
        labels_b = inner_labels(lb_matrix, np.arange(len(R)))
        out["inner_labels_recomputed_on_B_differ"] = int(np.count_nonzero(labels_b != labels))
        rb = np.frombuffer(metafeatures.to_f32le(metafeatures.features(VB[:, list(kept_jtt1)], VB[:, F], labels_b)),
                           dtype="<f4")
        rneq = a.view(np.uint32) != rb.view(np.uint32)
        out["relabelled_B"] = {"differ": int(np.count_nonzero(rneq)),
                               "max_abs": float(np.abs(a.astype(np.float64) - rb.astype(np.float64)).max()),
                               "differing_features": [metafeatures.FEATURES[i] for i in np.flatnonzero(rneq)]}
    except Exception as exc:
        out["inner_labels_recomputed_on_B_differ"] = f"error: {type(exc).__name__}: {exc}"
    return out


# ------------------------------------------------------------------------------------------------- lineage (local)
def _read_verified(path, sha256: str, size=None) -> bytes:
    data = Path(path).read_bytes()
    if common.sha256_bytes(data) != sha256 or (size is not None and len(data) != int(size)):
        raise ScaleCheckError(f"sha256/bytes mismatch: {path}")
    return data


def load_codex_index(path=CODEX_INDEX_PATH, sha256=CODEX_INDEX_SHA256, size=CODEX_INDEX_BYTES):
    """Spec B P4 input: the Codex output index, reduced to {(identity, fold): {component: (actual train_indices hash,
    recorded train_indices_sha256)}}; outputs without an identity ('absent_or_retired_before_dispatch') are skipped."""
    doc = json.loads(_read_verified(path, sha256, size))
    keys, stats = {}, collections.Counter()
    for o in doc["outputs"]:
        stats["outputs"] += 1
        stats[f"state={o.get('state')}"] += 1
        if o.get("identity") is None:
            stats["skipped_null_identity"] += 1
            continue
        f = o["fold"]
        rec = keys.setdefault((tuple(o["identity"]), int(f["fold"])), {})
        rec[o["component"]] = (indices_hash(f["train_indices"]), f.get("train_indices_sha256"))
    ref = {"path": str(path), "sha256": sha256, "bytes": int(size), "schema": doc.get("schema"), "utc_us": doc.get("utc_us"),
           "packet_ref_sha256": (doc.get("packet_ref") or {}).get("sha256"), "outputs": stats["outputs"],
           "complete": stats["state=complete"], "skipped_null_identity": stats["skipped_null_identity"], "keys": len(keys)}
    return keys, ref


class Sources:
    """Pinned local inputs: the epoch, the family registry, the optional holdout, the Codex packet, index and outcomes."""

    def __init__(self, epoch_path=EPOCH_PATH, epoch_sha256=EPOCH_SHA256, epoch_bytes=EPOCH_BYTES, registry_path=None,
                 holdout_path=None, codex_root=CODEX_ROOT, verify_pins=True, load_index=True):
        self.codex_root = Path(codex_root)
        self.pins = {}
        if verify_pins:
            for rel, want in CODEX_PINS.items():
                got = common.sha256_file(self.codex_root / rel)
                if got != want:
                    raise ScaleCheckError(f"Codex source drift: {rel} {got} != {want}")
                self.pins[rel] = got
            for name, want in FINDING_PINS.items():
                if common.sha256_file(FINDINGS_DIR / name) != want:
                    raise ScaleCheckError(f"finding receipt drift: {name}")
        raw = _read_verified(epoch_path, epoch_sha256, epoch_bytes)
        self.epoch_ref = {"path": str(epoch_path), "sha256": epoch_sha256, "bytes": len(raw)}
        self.epoch = json.loads(raw)
        del raw
        # R2: the registry is pinned; the pre-review registry is the default and is sha256-checked.
        registry_path = Path(registry_path) if registry_path is not None else REGISTRY_PRE
        reg_raw = Path(registry_path).read_bytes()
        reg_sha = common.sha256_bytes(reg_raw)
        if registry_path.resolve() == REGISTRY_PRE.resolve() and reg_sha != REGISTRY_PRE_SHA256:
            raise ScaleCheckError(f"pre-review registry drift: {reg_sha} != {REGISTRY_PRE_SHA256}")
        reg = json.loads(reg_raw)
        self.registry_is_pre_review = reg_sha == REGISTRY_PRE_SHA256
        self.registry_ref = {"path": str(registry_path), "sha256": reg_sha, "bytes": len(reg_raw),
                             "phase": reg.get("phase"), "schema": reg.get("schema"), "families": len(reg["families"]),
                             "pre_review_pinned": self.registry_is_pre_review,
                             "registry_json_present": REGISTRY_FINAL.exists()}
        self.family_of, self.sealed_ctx, self.seed_exposed_ctx = {}, set(), set()
        for fid, fam in reg["families"].items():
            for c in fam.get("contexts", []):
                self.family_of[c] = fid
                if fam.get("sealed"):
                    self.sealed_ctx.add(c)
                if fam.get("seed_selection_exposed"):
                    self.seed_exposed_ctx.add(c)
        self.sealed_families = sum(1 for f in reg["families"].values() if f.get("sealed"))
        del reg, reg_raw
        self.holdout_ref, self.holdout_assignment, self.diag_ctx = None, {}, set()
        if holdout_path is None and HOLDOUT_DEFAULT.exists():
            holdout_path = HOLDOUT_DEFAULT
        if holdout_path is not None:
            h_raw = Path(holdout_path).read_bytes()
            h = json.loads(h_raw)
            self.holdout_ref = {"path": str(holdout_path), "sha256": common.sha256_bytes(h_raw), "bytes": len(h_raw)}
            self.holdout_assignment = dict(h.get("assignment") or {})
            self.diag_ctx = set(h.get("diag") or [])
        packet_path = self.codex_root / "packet.json"
        self.packet = json.loads(_read_verified(packet_path, CODEX_PINS["packet.json"]) if verify_pins
                                 else packet_path.read_bytes())
        self.packet_by_identity = collections.defaultdict(list)
        for d in self.packet["datasets"]:
            for f in d.get("folds", []):
                f["train_indices"] = np.asarray(f["train_indices"], dtype=np.int64)   # memory; compared with np.array_equal
            if not d.get("retired"):
                self.packet_by_identity[tuple(d["identity"])].append(d)
        self.packet_ref = {"path": str(packet_path), "sha256": CODEX_PINS["packet.json"], "datasets": len(self.packet["datasets"]),
                           "retired": sum(1 for d in self.packet["datasets"] if d.get("retired"))}
        self.index, self.index_ref = ({}, None)
        if load_index:
            self.index, self.index_ref = load_codex_index()
        self.outcomes_dir = self.codex_root / "outcomes"
        self.arrays_dir = self.codex_root / "pending-arrays"

    def included(self):
        return [c for c in self.epoch["contexts"] if c.get("status") == "included"]

    def holdout_flag(self, context_sha256: str):
        fid = self.family_of.get(context_sha256)
        a = self.holdout_assignment.get(fid) if fid else None
        return {"assignment": a, "diag": context_sha256 in self.diag_ctx} if self.holdout_ref else None

    def packet_fold(self, identity, fold: int):
        """Spec B P3: exactly one packet dataset with this identity and exactly one (fold, repeat 0), else a STOP."""
        ds = self.packet_by_identity.get(tuple(identity), [])
        if len(ds) != 1:
            raise ScaleCheckError(f"Codex packet holds {len(ds)} datasets for identity {identity[0]} (expected exactly 1)")
        f = [x for x in ds[0]["folds"] if int(x["fold"]) == int(fold) and int(x.get("repeat", 0)) == 0]
        if len(f) != 1:
            raise ScaleCheckError(f"Codex packet holds {len(f)} (fold {fold}, repeat 0) records for {identity[0]}")
        return ds[0], f[0]

    def index_check(self, identity, fold: int, r_hash: str):
        """Spec B P4: None when the (identity, fold) is absent from the index, else whether every component's
        fold.train_indices (and its recorded hash) equals R."""
        rec = self.index.get((tuple(identity), int(fold)))
        if rec is None:
            return None
        return all(h == r_hash and recorded == r_hash for h, recorded in rec.values())

    def outcome_paths(self, d, f):
        return {c: codex_job_id(d, f, c, self.packet) for c in COMPONENTS}

    def local_outcomes(self, d, f):
        """{component: outcome record} when all five are local and complete, else None. A record whose job_id differs
        from its file name is returned as is; replicate() fails it."""
        out = {}
        for comp, jid in self.outcome_paths(d, f).items():
            p = self.outcomes_dir / jid
            if not p.exists():
                return None
            rec = json.loads(p.read_bytes())
            if rec.get("state") != "complete":
                return None
            rec["_expected_job_id"] = jid
            out[comp] = rec
        return out


class DataCache:
    """Parsed data objects and memberships by sha256, so the five folds of a dataset parse once."""

    def __init__(self, max_items=8):
        self.items = collections.OrderedDict()
        self.max_items = max_items

    def get(self, key, loader):
        if key in self.items:
            self.items.move_to_end(key)
            return self.items[key]
        val = loader()
        self.items[key] = val
        while len(self.items) > self.max_items:
            self.items.popitem(last=False)
        return val


def _materialization_crosscheck(item_id: str, manifest_sha: str, dref: dict, mref: dict):
    """R1: items/<item>/materialization.json is rewritten continuously, so it is only a cross-check of the content-addressed
    route: True (it names the same objects), False (it differs) or None (absent or unreadable)."""
    try:
        mat = json.loads((INCOMING_ITEMS / item_id / "materialization.json").read_bytes())
        lo = mat["local_objects"]
        return (Path(mat["manifest"]) == INTAKE_OBJECTS / manifest_sha and lo["data"]["sha256"] == dref["sha256"]
                and lo["membership"]["sha256"] == mref["sha256"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def resolve_context(row: dict, sources: Sources, cache: DataCache) -> dict:
    """The spec B LINEAGE chain for one included epoch context; every hop sha256- and byte-checked, any mismatch raises."""
    view = row["view_id"]
    pref = row["prepared_ref"]
    prep = json.loads(_read_verified(pref["path"], pref["sha256"], pref.get("bytes")))
    c = prep["context"]
    checks = {"context_sha256": c.get("context_sha256") == row["context_sha256"],
              "input_raw_sha256": c.get("input_raw_sha256") == row["input_ref"]["raw_sha256"],
              "input_ref": c.get("input_ref") == row["input_ref"],
              "outer_fold": int(c.get("outer_fold", -1)) == int(row["fold"]),
              "dataset_hash": c.get("dataset_hash") == row["dataset_hash"],
              "sample_identity_sha256": c.get("sample_identity_sha256") == row["sample_identity_sha256"],
              "outer_repeat": int(c.get("outer_repeat", -1)) == 0,
              "view_id": prep.get("view_id", view) == view}
    bad = sorted(k for k, ok in checks.items() if not ok)
    if bad:
        raise ScaleCheckError(f"{view}: prepared context differs from the epoch row on {bad}")
    cohort = row["cohort"]
    sm = prep["source_manifest"]
    if cohort == "frozen":
        if Path(pref["path"]) != FROZEN_CONTEXTS / view / "prepared.json":
            raise ScaleCheckError(f"{view}: frozen prepared.json outside {FROZEN_CONTEXTS}/<view_id>/")
        if not sm["key"].startswith(FROZEN_KEY_PREFIX):
            raise ScaleCheckError(f"{view}: frozen manifest key outside {FROZEN_KEY_PREFIX}")
        manifest_path = FROZEN_ROOT / sm["key"][len(FROZEN_KEY_PREFIX):]
    elif cohort == "incoming":
        item_id = row.get("item_id")
        if not item_id or Path(pref["path"]) != INCOMING_ITEMS / item_id / "run" / "gp" / "contexts" / view / "prepared.json":
            raise ScaleCheckError(f"{view}: incoming prepared.json outside items/<row.item_id>/run/gp/contexts/<view_id>/")
        manifest_path = INTAKE_OBJECTS / sm["sha256"]                       # R1: content addressed
    else:
        raise ScaleCheckError(f"{view}: unknown cohort {cohort!r}")
    manifest = json.loads(_read_verified(manifest_path, sm["sha256"], sm.get("bytes")))
    if manifest.get("dataset_hash") != row["dataset_hash"]:
        raise ScaleCheckError(f"{view}: manifest dataset_hash differs from the epoch row")
    if (manifest.get("target") or {}).get("index") != -1:
        raise ScaleCheckError(f"{view}: manifest target.index is not -1")
    san = c.get("feature_sanitization")
    want_sid = san["parent_sample_identity_sha256"] if san else row["sample_identity_sha256"]
    if manifest.get("sample_identity_sha256") != want_sid:
        raise ScaleCheckError(f"{view}: manifest sample_identity_sha256 differs (sanitized views use the parent's)")
    dref, mref = manifest["objects"]["data"], manifest["objects"]["membership"]
    if cohort == "frozen":
        data_path = FROZEN_ROOT / "objects" / "data" / "sha256" / f"{dref['sha256']}.csv.gz"
        memb_path = FROZEN_ROOT / "objects" / "memberships" / "sha256" / f"{mref['sha256']}.json.gz"
        mat_check = None
    else:
        data_path, memb_path = INTAKE_OBJECTS / dref["sha256"], INTAKE_OBJECTS / mref["sha256"]
        mat_check = _materialization_crosscheck(row["item_id"], sm["sha256"], dref, mref)
    shape = manifest["shape"]

    def load_data():
        gz = _read_verified(data_path, dref["sha256"], dref.get("bytes"))
        body = gzip.decompress(gz)
        a = np.loadtxt(io.BytesIO(body), delimiter=",", dtype="float64", ndmin=2)
        return a, common.sha256_bytes(body)

    A, body_sha = cache.get(("data", dref["sha256"]), load_data)
    n = int(A.shape[0])
    if A.shape != (int(shape["sample_rows"]), int(shape["columns"])) or not np.isfinite(A).all():
        raise ScaleCheckError(f"{view}: data shape {A.shape} != manifest {shape} or non-finite values")
    memb = cache.get(("memb", mref["sha256"]),
                     lambda: json.loads(gzip.decompress(_read_verified(memb_path, mref["sha256"], mref.get("bytes")))))
    if memb.get("sample_csv_sha256") != body_sha or int(memb.get("sample_rows", -1)) != n:
        raise ScaleCheckError(f"{view}: membership sample_csv_sha256/sample_rows differ from the data object")
    fold = int(row["fold"])
    fr = [f for f in memb["folds"] if int(f["fold"]) == fold and int(f.get("repeat", 0)) == 0]
    if len(fr) != 1:
        raise ScaleCheckError(f"{view}: membership has no unique (fold {fold}, repeat 0)")
    T = [int(x) for x in fr[0]["test_indices"]]
    if T != sorted(set(T)) or (T and (T[0] < 0 or T[-1] >= n)):
        raise ScaleCheckError(f"{view}: test indices are not sorted, unique and in range")
    if indices_hash(T) != fr[0]["test_indices_sha256"]:
        raise ScaleCheckError(f"{view}: test_indices_sha256 mismatch")
    tset = set(T)
    R = [i for i in range(n) if i not in tset]
    if (len(R) != int(c["train_rows"]) or len(T) != int(c["test_rows"])
            or len(R) != int(fr[0]["train_rows"]) or len(T) != int(fr[0]["test_rows"])):
        raise ScaleCheckError(f"{view}: train/test rows differ from the prepared context or the membership fold")
    F = A.shape[1] - 1
    removed_jtt1 = sorted(int(i) for i in (san or {}).get("removed_zero_based_feature_indices", []))
    kept_jtt1 = [i for i in range(F) if i not in set(removed_jtt1)]
    if san and kept_jtt1 != [int(i) for i in san["retained_zero_based_feature_indices"]]:
        raise ScaleCheckError(f"{view}: sanitization retained list differs")
    if len(kept_jtt1) != int(c["encoded_features"]):
        raise ScaleCheckError(f"{view}: encoded_features differs from the JTT1-kept columns")
    identity = [manifest["dataset_hash"], manifest["canonical_sha256"], manifest["sample_identity_sha256"]]
    d, pf = sources.packet_fold(identity, fold)
    r_hash = indices_hash(R)
    ids = memb["ordered_source_row_ids"]
    row_ids_sha = common.sha256_bytes(b"".join(bytes.fromhex(ids[i]) for i in R))
    packet_ok = (np.array_equal(pf["train_indices"], np.asarray(R, dtype=np.int64)) and pf["train_indices_sha256"] == r_hash
                 and pf["test_indices_sha256"] == fr[0]["test_indices_sha256"]
                 and d["objects"]["data"]["sha256"] == dref["sha256"])
    if not packet_ok:
        raise ScaleCheckError(f"{view}: Codex packet fold differs from the membership split")
    return {"row": row, "context": c, "cohort": cohort, "fold": fold, "A": A, "R": R, "T": T, "F": F,
            "kept_jtt1": kept_jtt1, "labels_from_parent": bool(san) and str(c.get("schema", "")).endswith("/v6"),
            "sanitized": bool(san), "shape": shape, "subsampled": int(shape["sample_rows"]) < int(shape["source_rows"]),
            "identity": identity, "packet_dataset": d, "packet_fold": pf, "r_hash": r_hash,
            "train_row_ids_ok": pf["train_row_ids_sha256"] == row_ids_sha,
            "index_check": sources.index_check(identity, fold, r_hash) if sources.index_ref else None,
            "index_loaded": sources.index_ref is not None, "materialization_crosscheck": mat_check,
            "owner": codex_shard_owner(identity, fold),
            "refs": {"prepared_sha256": pref["sha256"], "manifest_sha256": sm["sha256"], "data_sha256": dref["sha256"],
                     "membership_sha256": mref["sha256"], "input_raw_sha256": row["input_ref"]["raw_sha256"]}}


def lineage_checks(ci: dict, ref_inputs: dict):
    """Per-context proofs (spec B P1-P4) on top of resolve_context's hard checks: the JTT1 rebuild, the cached JTT1 train
    view, the packet row ids, the Codex index folds, and Codex-kept == JTT1-kept. Returns (booleans, raw, labels)."""
    raw, labels = jtt1_rebuild(ci["A"], ci["R"], ci["T"], ci["kept_jtt1"], ci["labels_from_parent"])
    ref = (ci.get("row") or {}).get("input_ref") or {"raw_sha256": ci["refs"]["input_raw_sha256"]}
    out = {"jtt1_rebuild_matches_input_ref": (common.sha256_bytes(raw) == ref["raw_sha256"]
                                              and ("raw_bytes" not in ref or len(raw) == int(ref["raw_bytes"]))),
           "codex_kept_equals_jtt1_kept": list(ref_inputs["binding"]["kept"]) == list(ci["kept_jtt1"])}
    if ci.get("packet_fold") is not None:      # resolve_context raised on any packet split mismatch
        out["packet_split_matches"] = True
        out["packet_train_row_ids_match"] = bool(ci["train_row_ids_ok"])
    if ci.get("index_check") is not None:
        out["codex_index_train_indices_match"] = bool(ci["index_check"])
    cache = jtt1_cache_check(ci, raw, labels)
    if cache is not None:
        out.update(cache)
    return out, raw, labels


# --------------------------------------------------------------------------------------------------------- prescan
def rank_of(context_sha256: str) -> str:
    return hashlib.sha256((RANK_DOMAIN + context_sha256).encode("utf-8")).hexdigest()


def prescan(ci: dict, ref_inputs: dict) -> dict:
    """Counts-only flags of one context (spec B PRE-SCAN); no test-row value is emitted."""
    A, F = ci["A"], ci["F"]
    Rn = np.asarray(ci["R"], dtype=np.int64)
    kept = list(ref_inputs["binding"]["kept"])
    considered = kept + [F]
    cons = np.zeros(F + 1, dtype=bool)
    cons[considered] = True
    feats = np.zeros(F + 1, dtype=bool)
    feats[kept] = True
    AR = A[Rn]
    lo_R, hi_R, lo_all, hi_all = AR.min(0), AR.max(0), A.min(0), A.max(0)
    tmin, tmax = (lo_R > lo_all) & cons, (hi_R < hi_all) & cons
    flag = tmin | tmax
    varies = hi_R != lo_R
    b_ident = (~varies) | ((lo_R == 0.0) & (hi_R == 1.0))
    B = phi(AR, lo_R, hi_R)
    b_strict = (B.view(np.uint64) == AR.view(np.uint64)).all(axis=0)
    floor = np.zeros(F + 1, dtype=bool)
    floor[kept] = ref_inputs["hyperion"]["floor"]
    X32 = AR[:, kept].astype(np.float32)
    f_risk = int(np.count_nonzero(((X32.max(0) - X32.min(0)) <= 1e-12) & varies[kept]))
    y32 = AR[:, F].astype(np.float32)
    t_risk = bool(varies[F]) and float(np.float64(y32.max()) - float(y32.min())) <= 1e-12
    sev_cols = np.flatnonzero(flag & varies)
    severity = max(float((hi_all[j] - lo_all[j]) / (hi_R[j] - lo_R[j]) - 1.0) for j in sev_cols) if sev_cols.size else None
    D, _lo_S, _hi_S, c2_resid = c2_fit(A, Rn, ci["T"], considered, lo_R, hi_R, lo_all, hi_all)
    exact01 = ((lo_all == 0.0) & (hi_all == 1.0)) | (hi_all == lo_all)
    out01 = (lo_all < 0.0) | (hi_all > 1.0)
    near_zero = np.zeros(F + 1, dtype=bool)
    for j in considered:
        b = B[:, j]
        near_zero[j] = bool(np.any((b > 0) & (b < NEAR_ZERO_REL)))
    return {
        "flagged": bool(flag.any()), "target_flagged": bool(flag[F]), "features_flagged": int((flag & feats).sum()),
        "flagged_columns": int(flag.sum()), "considered_columns": len(considered),
        "flagged_cols": [("target" if j == F else int(j)) for j in np.flatnonzero(flag)][:64],
        "test_only_min": int(tmin.sum()), "test_only_max": int(tmax.sum()), "test_only_both": int((tmin & tmax).sum()),
        "constant_on_R_columns": int((cons & ~varies & (hi_all != lo_all)).sum()),
        "b_nonidentity_columns": int((cons & ~b_ident).sum()), "b_identity_all": bool(b_ident[considered].all()),
        "b_identity_strict": bool(b_strict[considered].all()),
        "exact01_all_columns": bool(exact01.all()), "exact01_considered": bool(exact01[considered].all()),
        "out01": bool(out01.any()), "out01_considered": bool(out01[considered].any()),
        "floor_columns": int(floor.sum()), "floor_varying_columns": int((floor & varies).sum()),
        "floor_remapped_by_B": int((floor & varies & ~b_ident).sum()), "floor_flagged": int((floor & flag).sum()),
        "foundation_floor_risk": f_risk, "target_floor_risk": int(t_risk), "severity": severity,
        "D_rows": len(D), "c2_residual_columns": c2_resid, "near_zero_columns": int(near_zero.sum()),
        "san": bool(ref_inputs["binding"]["removed"]), "removed": list(ref_inputs["binding"]["removed"]),
        "subsampled": ci["subsampled"], "n_train": len(ci["R"]), "n_test": len(ci["T"]), "k": int(ref_inputs["k"]), "F": F,
    }


def _quantiles(values):
    if not values:
        return None
    a = np.sort(np.asarray(values, dtype=np.float64))
    out = {f"p{int(q * 100)}": float(np.quantile(a, q)) for q in (0.10, 0.25, 0.50, 0.75, 0.90, 0.99)}
    out["max"] = float(a[-1])
    out["count"] = int(a.size)
    return out


def prescan_summary(records: list) -> dict:
    """Spec B PRE-SCAN totals (counts only) over a list of {prescan, local_complete} records."""
    s = collections.Counter()
    hist = collections.Counter()
    severities = []
    for r in records:
        p, local = r["prescan"], bool(r.get("local_complete"))
        s["contexts"] += 1
        s["local_contexts"] += int(local)
        s["flagged"] += int(p["flagged"])
        s["flagged_via_feature"] += int(p["features_flagged"] > 0)
        s["flagged_via_target"] += int(p["target_flagged"])
        s["flagged_target_only"] += int(p["target_flagged"] and p["features_flagged"] == 0)
        s["considered_columns"] += p["considered_columns"]
        s["flagged_columns"] += p["flagged_columns"]
        s["flagged_feature_columns"] += p["features_flagged"]
        s["test_only_min"] += p["test_only_min"]
        s["test_only_max"] += p["test_only_max"]
        s["test_only_both"] += p["test_only_both"]
        s["constant_on_R_columns"] += p["constant_on_R_columns"]
        s["constant_on_R_contexts"] += int(p["constant_on_R_columns"] > 0)
        s["constant_on_R_local_contexts"] += int(p["constant_on_R_columns"] > 0 and local)
        s["b_nonidentity_columns"] += p["b_nonidentity_columns"]
        s["b_nonidentity_contexts"] += int(p["b_nonidentity_columns"] > 0)
        s["u_control_eligible"] += int(not p["flagged"] and p["b_identity_all"])
        s["u_control_eligible_strict"] += int(not p["flagged"] and p["b_identity_strict"])
        s["exact01_contexts"] += int(p["exact01_all_columns"])
        s["exact01_considered_contexts"] += int(p["exact01_considered"])
        s["out01_contexts"] += int(p["out01"])
        s["subsampled_contexts"] += int(p["subsampled"])
        s["subsampled_flagged"] += int(p["subsampled"] and p["flagged"])
        s["floor_columns"] += p["floor_columns"]
        s["floor_contexts"] += int(p["floor_columns"] > 0)
        s["floor_varying_columns"] += p["floor_varying_columns"]
        s["floor_varying_contexts"] += int(p["floor_varying_columns"] > 0)
        s["floor_remapped_contexts"] += int(p["floor_remapped_by_B"] > 0)
        s["floor_remapped_columns"] += p["floor_remapped_by_B"]
        s["floor_flagged_contexts"] += int(p["floor_flagged"] > 0)
        s["D_rows"] += p["D_rows"]
        s["D_rows_max"] = max(s["D_rows_max"], p["D_rows"])
        s["c2_residual_contexts"] += int(p["c2_residual_columns"] > 0)
        s["foundation_floor_risk_columns"] += p["foundation_floor_risk"]
        s["foundation_floor_risk_contexts"] += int(p["foundation_floor_risk"] > 0)
        s["foundation_floor_risk_local_contexts"] += int(p["foundation_floor_risk"] > 0 and local)
        s["target_floor_risk_contexts"] += int(p["target_floor_risk"] > 0)
        s["san_contexts"] += int(p["san"])
        s["san_flagged"] += int(p["san"] and p["flagged"])
        s["k_gt_128_contexts"] += int(p["k"] > 128)
        s["k_gt_128_local_contexts"] += int(p["k"] > 128 and local)
        s["near_zero_contexts"] += int(p.get("near_zero_columns", 0) > 0)
        if p["severity"] is not None:
            severities.append(p["severity"])
            s["severity_contexts"] += 1
            s["severity_gt_100"] += int(p["severity"] > 100)
            s["severity_gt_100_local"] += int(p["severity"] > 100 and local)
        hist[min(p["flagged_columns"], 10)] += 1
    out = dict(sorted(s.items()))
    out["flagged_columns_per_context"] = {("10+" if k == 10 else str(k)): v for k, v in sorted(hist.items())}
    out["severity"] = _quantiles(severities)
    return out


def strata_summary(records: list) -> dict:
    """Spec B per-stratum table: [flagged, contexts, target-only, floor_remapped, U-control eligible]."""
    out = {f"{c} f{f}": [0, 0, 0, 0, 0] for c, f in STRATA}
    for r in records:
        p = r["prescan"]
        row = out.setdefault(f"{r['cohort']} f{r['fold']}", [0, 0, 0, 0, 0])
        row[0] += int(p["flagged"])
        row[1] += 1
        row[2] += int(p["target_flagged"] and p["features_flagged"] == 0)
        row[3] += int(p["floor_remapped_by_B"] > 0)
        row[4] += int(not p["flagged"] and p["b_identity_all"])
    return out


def _sig_match(observed, printed: str) -> bool:
    """Whether an observed value rounds to a spec value printed with a given number of significant digits."""
    if observed is None:
        return False
    digits = printed.lower().split("e")[0].replace("-", "").replace(".", "").lstrip("0") or "0"
    return float(f"{float(observed):.{len(digits)}g}") == float(printed)


def crosscheck(observed: dict, expected: dict) -> dict:
    mismatches = {k: {"expected": want, "observed": observed.get(k)} for k, want in expected.items() if observed.get(k) != want}
    return {"checked": len(expected), "all_match": not mismatches, "mismatches": mismatches}


def prescan_crosscheck(all_sum, unsealed_sum, sealed_sum, strata) -> dict:
    """The spec B PRE-SCAN totals against this run's (counts only)."""
    sev = unsealed_sum.get("severity") or {}
    sev_check = {k: {"expected": v, "observed": sev.get(k), "match": _sig_match(sev.get(k), v)}
                 for k, v in SPEC_B_SEVERITY_UNSEALED.items()}
    out = {"all_included": crosscheck(all_sum, SPEC_B_PRESCAN["all_included"]),
           "unsealed": crosscheck(unsealed_sum, SPEC_B_PRESCAN["unsealed"]),
           "sealed": crosscheck(sealed_sum, SPEC_B_PRESCAN["sealed"]),
           "strata_unsealed": crosscheck(strata, SPEC_B_STRATA_UNSEALED),
           "severity_unsealed": {"all_match": all(v["match"] for v in sev_check.values()), "values": sev_check}}
    out["all_match"] = all(v["all_match"] for v in out.values())
    return out


# ------------------------------------------------------------------------------------------------ replication gate
def replicate(ci: dict, ref_inputs: dict, outcomes: dict, arrays_dir: Path) -> dict:
    """Spec B P5: the A-side replica against the local Codex outcome records (job ids, identities, folds, bindings,
    recipes) and the side arrays in pending-arrays, bit for bit."""
    checks = {}
    R = list(ci["R"])
    r_hash = indices_hash(R)
    b = ref_inputs["binding"]
    for comp in COMPONENTS:
        checks[f"{comp}.job_id"] = outcomes[comp].get("job_id") == outcomes[comp].get("_expected_job_id")
    for comp in ("hyperion", "foundation", "target", "tabpfn35"):
        rec = outcomes[comp]
        ib = rec.get("input_binding") or {}
        checks[f"{comp}.binding"] = (ib.get("kept_feature_indices") == b["kept"]
                                     and ib.get("removed_target_feature_indices") == b["removed"]
                                     and ib.get("selection_fit") == "outer_train_only"
                                     and ib.get("outer_train_indices_sha256") == r_hash)
        checks[f"{comp}.fold"] = rec["fold"].get("train_indices") == R and int(rec["fold"]["train_rows"]) == len(R)
        checks[f"{comp}.identity"] = rec.get("identity") == ci["packet_dataset"]["identity"]
    h, f, t, p = (ref_inputs[c] for c in ("hyperion", "foundation", "target", "tabpfn35"))
    perms = hyperion_geometry(ref_inputs["n"], ref_inputs["k"])
    hr = outcomes["hyperion"]["recipe"]
    checks["hyperion.permutations"] = hr.get("permutations") == [{"rows": pr.tolist(), "columns": pc.tolist()} for pr, pc in perms]
    checks["hyperion.scales"] = hr.get("scales") == list(HYPERION_SCALES) and hr.get("sweeps") == 3
    fr = outcomes["foundation"]["recipe"]
    checks["foundation.indices"] = (fr.get("context_local_indices") == f["ctx"].tolist()
                                    and fr.get("query_local_indices") == f["qry"].tolist())
    checks["foundation.context_y"] = fr.get("context_y_mean") == f["mu"] and fr.get("context_y_std") == f["sd"]
    checks["foundation.chunk_seed"] = fr.get("query_chunk") == f["chunk"] and fr.get("seed") == 20260707
    tr = outcomes["target"]["recipe"]
    checks["target.indices"] = (tr.get("classical_local_indices") == t["c"].tolist()
                                and tr.get("neural_local_indices") == t["q"].tolist())
    checks["target.min_span"] = tr.get("train_target_min") == t["lo"] and tr.get("train_target_span") == t["span"]
    checks["target.seed"] = tr.get("seed") == 17 and tr.get("structural_zero_dimensions") == 376
    pr_ = outcomes["tabpfn35"]["recipe"]
    checks["tabpfn35.indices"] = (pr_.get("context_local_indices") == p["ctx"].tolist()
                                  and pr_.get("query_local_indices") == p["qry"].tolist() and pr_.get("seed") == 1729)
    # worker.py:118 records every parent completed before it (tabpfn35 too); the three model parents must be the local ones
    parents = (outcomes["context2048"].get("recipe") or {}).get("parent_refs") or {}
    checks["context2048.parents"] = all(k in parents and parents[k] == outcomes[k].get("outcome_ref")
                                        for k in ("hyperion", "foundation", "target"))
    for comp, names, mine in (("hyperion", ("feature_q01", "feature_q99"), (h["lo"], h["hi"])),
                              ("foundation", ("feature_min", "feature_span"), (f["lo"], f["span"]))):
        ref = outcomes[comp]["array_ref"]
        path = arrays_dir / outcomes[comp]["job_id"]
        if not path.exists():
            checks[f"{comp}.arrays"] = None
            continue
        if common.sha256_file(path) != ref["sha256"]:
            checks[f"{comp}.arrays"] = False
            continue
        with np.load(path, allow_pickle=False) as z:
            got = [np.asarray(z[nm]) for nm in names]
        checks[f"{comp}.arrays"] = all(g.dtype == np.float32 and g.shape == m.shape
                                       and bool((g.view(np.uint32) == m.view(np.uint32)).all()) for g, m in zip(got, mine))
    failed = sorted(k for k, v in checks.items() if v is False)
    missing = sorted(k for k, v in checks.items() if v is None)
    return {"pass": not failed and not missing, "failed": failed, "missing": missing, "checks": len(checks),
            "std_np2_native_equal": bool(f["std_np2_native_equal"])}


# ------------------------------------------------------------------------------------------------ per-context run
def analyse(ci: dict, sources, *, positions: int, want_a0: bool, want_replication: bool):
    """Everything for one context: A-side inputs, prescan, variants, comparisons, lineage proofs, replication, A0."""
    A, R, T, F = ci["A"], ci["R"], ci["T"], ci["F"]
    Rn = np.asarray(R, dtype=np.int64)
    inputs = {"A": codex_inputs(A[Rn])}
    ps = prescan(ci, inputs["A"])
    considered = list(inputs["A"]["binding"]["kept"]) + [F]
    mats, vmeta, fits = build_variants(A, R, T, considered, return_fits=True)
    for name in VARIANTS + DIAG_VARIANTS:
        inputs[name] = codex_inputs(mats[name])
    cmp = {f"{name}_vs_A": compare_inputs(inputs["A"], inputs[name], Rn, positions) for name in VARIANTS}
    for name in A1_BASELINE_VARIANTS:
        cmp[f"{name}_vs_A1"] = compare_inputs(inputs["A1"], inputs[name], Rn, 0)
    # spec A P4 diagnostic: Codex float32 arithmetic (B32 vs A) versus the print and lattice term (B vs B32)
    cmp["B32_vs_A"] = compare_inputs(inputs["A"], inputs["B32"], Rn, positions)
    cmp["B_vs_B32"] = compare_inputs(inputs["B32"], inputs["B"], Rn, positions)
    # FIX ROUTE: the lane re-derives the scaling on the outer-train rows of whatever data object the upstream produced.
    # F_V = phi_R(V[R]) for each counterfactual upstream V; F_A is B. Bit-identity with B means the fix cancels V.
    near_zero = near_zero_columns(A, R, considered, F)
    risk = construction_risk_columns(A, R, considered, F, fits)
    fix_attr = {}
    for name in FIX_VARIANTS:
        M = mats[name]
        FM = phi(M, M.min(0), M.max(0))
        key = f"fix_{name}_vs_B"
        cmp[key] = compare_inputs(inputs["B"], codex_inputs(FM), Rn, positions)
        fix_attr[key] = fix_attribution(cmp[key], near_zero, risk[name], rounding_columns(mats["B"], FM, considered, F))
    lineage, raw, labels = lineage_checks(ci, inputs["A"])
    rep, local = None, None
    if want_replication and sources is not None:
        outs = sources.local_outcomes(ci["packet_dataset"], ci["packet_fold"])
        local = outs is not None
        rep = replicate(ci, inputs["A"], outs, sources.arrays_dir) if outs else {"pass": None, "note": "no local outcomes"}
    a0 = a0_compare(raw, A, R, ci["kept_jtt1"], labels, ci["labels_from_parent"]) if want_a0 else None
    return {"prescan": ps, "variant_meta": vmeta, "comparisons": cmp, "fix_attribution": fix_attr,
            "near_zero_columns": near_zero, "construction_risk_columns": {k: v[:64] for k, v in risk.items()},
            "lineage": lineage, "replication": rep, "local": local,
            "owner": ci.get("owner"), "a0": a0,
            "std_np2_native_equal": bool(inputs["A"]["foundation"]["std_np2_native_equal"])}


def context_identity(ci: dict, sources: Sources) -> dict:
    row = ci["row"]
    return {"view_id": row["view_id"], "context_sha256": row["context_sha256"], "dataset_hash": row["dataset_hash"],
            "cohort": ci["cohort"], "fold": ci["fold"], "family_id": sources.family_of.get(row["context_sha256"]),
            "seed_selection_exposed": row["context_sha256"] in sources.seed_exposed_ctx,
            "holdout": sources.holdout_flag(row["context_sha256"]), "refs": ci["refs"], "owner": ci["owner"],
            "n_train": len(ci["R"]), "n_test": len(ci["T"]), "F": ci["F"], "rank": rank_of(row["context_sha256"])}


# ------------------------------------------------------------------------------------------------- sample design
def select_sample(records: list, n_target: int, anchor=None, u_control: str = "b_identity_strict"):
    """Spec B deterministic sample over prescan records of U (each {context_sha256, dataset_hash, cohort, fold, rank,
    prescan}); anchor is the ANCHOR record (outside U) or None. u_control names the prescan flag of the U-control
    definition (strict by default, spec B R4). Returns [(category, record)] in selection order."""
    chosen, picked = [], set()
    per_ds = collections.Counter()
    per_stratum = set()
    by_rank = sorted(records, key=lambda r: r["rank"])

    def ok(r):
        return (r["context_sha256"] not in picked and per_ds[r["dataset_hash"]] < 2
                and (r["cohort"], r["fold"], r["dataset_hash"]) not in per_stratum)

    def take(cat, r):
        chosen.append((cat, r))
        picked.add(r["context_sha256"])
        per_ds[r["dataset_hash"]] += 1
        per_stratum.add((r["cohort"], r["fold"], r["dataset_hash"]))

    def first(pred, pool=by_rank):
        for r in pool:
            if ok(r) and pred(r):
                return r
        return None

    if anchor is not None:
        take("ANCHOR", anchor)
    for cohort, fold in STRATA:
        pool = [r for r in by_rank if r["cohort"] == cohort and r["fold"] == fold]
        sev = sorted((r for r in pool if r["prescan"]["flagged"] and r["prescan"]["severity"] is not None),
                     key=lambda r: (-r["prescan"]["severity"], r["rank"]))
        r = first(lambda _r: True, sev)
        if r is not None:
            take("F-severe", r)
        for cat, pred in (("H-floor", lambda r: r["prescan"]["flagged"] and r["prescan"]["floor_remapped_by_B"] > 0),
                          ("U-control", lambda r: not r["prescan"]["flagged"] and r["prescan"][u_control]),
                          ("R-flagged", lambda r: r["prescan"]["flagged"])):
            r = first(pred, pool)
            if r is not None:
                take(cat, r)
    specials = (
        ("T-only frozen", lambda r: r["cohort"] == "frozen" and r["prescan"]["target_flagged"]
         and r["prescan"]["features_flagged"] == 0),
        ("T-only incoming", lambda r: r["cohort"] == "incoming" and r["prescan"]["target_flagged"]
         and r["prescan"]["features_flagged"] == 0),
        ("SAN", lambda r: r["prescan"]["san"]),
        ("OUT01", lambda r: r["prescan"]["out01"]),
        ("SUB-inside", lambda r: r["prescan"]["subsampled"] and r["prescan"]["b_nonidentity_columns"] > 0
         and not r["prescan"]["flagged"]),
    )
    # each special is the first eligible context in U by rank that is not already selected (a new context is added)
    for cat, pred in specials:
        r = first(pred)
        if r is not None:
            take(cat, r)
    if by_rank:
        nmin = min(r["prescan"]["n_train"] for r in by_rank)
        r = first(lambda r: r["prescan"]["n_train"] == nmin)
        if r is not None:
            take("SMALL", r)
    for r in by_rank:
        if len(chosen) >= n_target:
            break
        if ok(r):
            take("R-any", r)
    return chosen[:n_target]


# ------------------------------------------------------------------------------------------------------ aggregation
class Census:
    """Counts and histograms over contexts x comparisons; no positions, no identifiers of any kind."""

    def __init__(self):
        self.contexts = 0
        self.meta = collections.Counter()
        self.by = collections.defaultdict(lambda: {"contexts": 0, "class": collections.Counter(),
                                                   "component_class": collections.defaultdict(collections.Counter),
                                                   "tabpfn_refined": collections.Counter(),
                                                   "arrays": collections.defaultdict(collections.Counter),
                                                   "max": collections.defaultdict(float),
                                                   "delta_hist": collections.Counter(),
                                                   "fix": collections.Counter()})

    def add(self, res: dict):
        self.contexts += 1
        vm = res.get("variant_meta") or {}
        self.meta["D_rows"] += vm.get("D_rows", 0)
        self.meta["C_single_row_columns"] += vm.get("C_single_row_columns", 0)
        self.meta["C2_residual_contexts"] += int(vm.get("C2_residual_test_only_extreme_columns", 0) > 0)
        self.meta["near_zero_contexts"] += int(bool(res.get("near_zero_columns")))
        self.meta["std_np2_native_differs"] += int(res.get("std_np2_native_equal") is False)
        for key, cmp in res["comparisons"].items():
            b = self.by[key]
            b["contexts"] += 1
            b["class"][cmp["class"]] += 1
            b["component_class"]["binding"][cmp["binding"]["class"]] += 1
            refined = (cmp["components"]["tabpfn35"].get("refined_p9") or {}).get("class")
            if refined:
                b["tabpfn_refined"][refined] += 1
            for comp, rec in cmp["components"].items():
                b["component_class"][comp][rec["class"]] += 1
                for arr, sub in rec.items():
                    if not isinstance(sub, dict) or "kind" not in sub:
                        continue
                    name = f"{comp}.{arr}"
                    a = b["arrays"][name]
                    a["elements"] += sub.get("elements", 0)
                    a["contexts_with_difference"] += int(bool(sub.get("differ", sub.get("flips", 0))))
                    if sub["kind"] == "f32":
                        a["differ"] += sub.get("differ", 0)
                        a["exceed_2^-12"] += sub.get("exceed_2^-12", 0)
                        a["signed_zero_flips"] += sub.get("signed_zero_flips", 0)
                        for k, v in (sub.get("ulp_hist") or {}).items():
                            a["ulp_" + k] += v
                        b["max"][name + ".max_ulp"] = max(b["max"][name + ".max_ulp"], sub.get("max_ulp", 0))
                        b["max"][name + ".max_abs_x2^24"] = max(b["max"][name + ".max_abs_x2^24"], sub.get("max_abs_x2^24", 0.0))
                    else:
                        for k in ("flips", "flips_abs1", "flips_ge2", "floor_column_flips", "saturation_flips",
                                  "nonfloor_abs1_flips"):
                            a[k] += sub.get(k, 0)
                        b["max"][name + ".max_abs_delta"] = max(b["max"][name + ".max_abs_delta"], sub.get("max_abs_delta", 0))
                        b["delta_hist"].update({k: v for k, v in (sub.get("delta_hist") or {}).items()})
                if comp == "hyperion" and "tiles_total" in rec:
                    b["arrays"]["hyperion.tiles"]["total"] += rec["tiles_total"]
                    b["arrays"]["hyperion.tiles"]["differ"] += rec["tiles_differ"]
                    b["arrays"]["hyperion.tiles"]["floor_status_flips"] += rec.get("floor_status_flips", 0)
            fa = (res.get("fix_attribution") or {}).get(key)
            if fa:
                b["fix"]["contexts_differing"] += int(fa["differing_elements"] > 0 or fa["binding"] != "IDENTICAL")
                b["fix"]["contexts_only_near_zero"] += int(fa["only_near_zero"])
                b["fix"]["contexts_only_artifact"] += int(fa["only_artifact"])
                b["fix"]["contexts_only_explained"] += int(fa["only_explained"])
                b["fix"]["contexts_with_unexplained"] += int(fa["elements_unexplained"] > 0)
                b["fix"]["contexts_with_construction_risk_columns"] += int(fa["construction_risk_columns_total"] > 0)
                b["fix"]["contexts_with_rounding_columns"] += int(bool(fa["rounding_columns"]))
                for kind, cnt in (fa.get("rounding_kinds") or {}).items():
                    b["fix"][f"rounding_columns_{kind}"] += cnt
                b["fix"]["elements_in_near_zero_columns"] += fa["elements_in_near_zero_columns"]
                b["fix"]["elements_in_construction_risk_columns"] += fa["elements_in_construction_risk_columns"]
                b["fix"]["elements_in_rounding_columns"] += fa["elements_in_rounding_columns"]
                b["fix"]["elements_elsewhere"] += fa["elements_elsewhere"]
                b["fix"]["elements_unexplained"] += fa["elements_unexplained"]

    @staticmethod
    def _array_doc(v):
        d = dict(sorted(v.items()))
        if d.get("elements"):
            d["bit_identical_fraction"] = 1.0 - d.get("differ", d.get("flips", 0)) / d["elements"]
        return d

    def doc(self):
        out = {"contexts": self.contexts, "variant_meta": dict(sorted(self.meta.items())), "comparisons": {}}
        for key in sorted(self.by):
            b = self.by[key]
            out["comparisons"][key] = {
                "contexts": b["contexts"], "class": dict(sorted(b["class"].items())),
                "component_class": {c: dict(sorted(v.items())) for c, v in sorted(b["component_class"].items())},
                "tabpfn_refined_p9": dict(sorted(b["tabpfn_refined"].items())),
                "arrays": {n: self._array_doc(v) for n, v in sorted(b["arrays"].items())},
                "max": {k: (float(v) if isinstance(v, float) else int(v)) for k, v in sorted(b["max"].items())},
                "hyperion_delta_hist": dict(sorted(b["delta_hist"].items(), key=lambda kv: int(kv[0])))}
            if b["fix"]:
                out["comparisons"][key]["fix_attribution"] = dict(sorted(b["fix"].items()))
        return out


def census_compact(doc: dict) -> dict:
    """The census without per-array histograms and maxima (for the receipt): classes, component classes, the P9
    refinement, fix-route attribution and variant metadata."""
    return {"contexts": doc["contexts"], "variant_meta": doc["variant_meta"],
            "comparisons": {k: {x: v[x] for x in ("class", "component_class", "tabpfn_refined_p9", "fix_attribution")
                                if x in v} for k, v in doc["comparisons"].items()}}


def headline(gate, fix_gate, census_doc, s_floor_doc, rep_counts, a0_sum) -> dict:
    """One-screen result for the family-review evidence (counts only)."""
    def classes(doc, keys):
        return {k: (doc.get("comparisons") or {}).get(k, {}).get("class", {}) for k in keys}
    fixes = {k: (census_doc.get("comparisons") or {}).get(k, {}).get("fix_attribution", {}) for k in FIX_GATE_KEYS}
    return {"sample_gate": {"result": gate["result"], "contexts": gate["contexts"], **gate["by_variant"]},
            "sample_fix_route": {"result": fix_gate["result"], **fix_gate["by_variant"]},
            "census_contexts": census_doc.get("contexts", 0),
            "census_gate_classes": classes(census_doc, ("A1_vs_A",) + GATE_KEYS),
            "census_fix_route_classes": classes(census_doc, FIX_GATE_KEYS),
            "census_fix_route_attribution": {k: {x: v.get(x, 0) for x in ("contexts_differing", "contexts_only_artifact",
                                                                           "contexts_only_explained",
                                                                           "contexts_with_unexplained",
                                                                           "elements_unexplained")}
                                             for k, v in fixes.items()},
            "census_diagnostic_classes": classes(census_doc, DIAG_KEYS),
            "s_floor": s_floor_doc.get("per_context", []),
            "replication_local": {k: {x: v.get(x, 0) for x in ("local_complete", "replicated", "replication_pass")}
                                  for k, v in rep_counts.items()},
            "a0_sample": {k: a0_sum.get(k, 0) for k in ("status=computed", "bit_identical_vectors",
                                                        "inner_labels_change_under_B", "relabelled_B_vectors_differ")}}


def gate_verdict(sample_results: list, keys=GATE_KEYS, rule=None) -> dict:
    """PASS only if every fed input is IDENTICAL in every listed comparison for every sampled context."""
    per = {k: collections.Counter() for k in keys}
    comp = {k: {c: collections.Counter() for c in ("binding",) + COMPONENTS} for k in keys}
    refined = {k: collections.Counter() for k in keys}
    for r in sample_results:
        for k in keys:
            cmp = r["comparisons"][k]
            per[k][cmp["class"]] += 1
            comp[k]["binding"][cmp["binding"]["class"]] += 1
            for c in COMPONENTS:
                comp[k][c][cmp["components"][c]["class"]] += 1
            refined[k][(cmp["components"]["tabpfn35"].get("refined_p9") or {}).get("class", "n/a")] += 1
    identical = len(sample_results) > 0 and all(per[k]["IDENTICAL"] == len(sample_results) for k in keys)
    return {"result": "PASS" if identical else "FAIL",
            "rule": rule or "PASS only if every fed input is IDENTICAL for every sampled context under B, C and C2",
            "contexts": len(sample_results),
            "by_variant": {k: dict(sorted(per[k].items())) for k in keys},
            "by_variant_component": {k: {c: dict(sorted(comp[k][c].items())) for c in comp[k]} for k in keys},
            "tabpfn35_refined_p9_informational": {k: dict(sorted(refined[k].items())) for k in keys}}


FIX_RULE = ("FIX ROUTE: PASS only if, after the lane-local train-only rescaling phi_R, every fed input is IDENTICAL to B "
            "for every sampled context whatever the upstream did with the test rows (A1, C, C2)")


def residual_doc(sample_census: dict) -> dict:
    """Attribution of the documented residual to its sources (spec A predictions, spec B RESIDUAL SOURCES)."""
    out = {}
    for v in GATE_KEYS + FIX_GATE_KEYS + DIAG_KEYS:
        c = (sample_census.get("comparisons") or {}).get(v)
        if not c:
            continue
        arr = c["arrays"]
        u = arr.get("hyperion.u", {})
        out[v] = {
            "class": c["class"], "component_class": c["component_class"], "tabpfn35_refined_p9": c["tabpfn_refined_p9"],
            "hyperion_u_flips": {k: u.get(k, 0) for k in ("flips", "flips_abs1", "flips_ge2", "floor_column_flips",
                                                         "saturation_flips", "nonfloor_abs1_flips")},
            "hyperion_tiles": dict(arr.get("hyperion.tiles", {})),
            "foundation": {n: {k: arr.get(f"foundation.{n}", {}).get(k, 0) for k in ("elements", "differ", "exceed_2^-12")}
                           for n in ("x_ctx", "x_qry", "yz")},
            "target": {n: {k: arr.get(f"target.{n}", {}).get(k, 0) for k in ("elements", "differ", "exceed_2^-12")}
                       for n in ("classical", "neural")},
            "tabpfn35": {n: {k: arr.get(f"tabpfn35.{n}", {}).get(k, 0) for k in ("elements", "differ")}
                         for n in ("X_ctx", "y_ctx", "X_qry")},
            "max": c["max"]}
        if "fix_attribution" in c:
            out[v]["fix_attribution"] = c["fix_attribution"]
    out["sources"] = {
        "exact_arithmetic_non_invariance": [
            "hyperion span floor (hi - lo < 1e-9 -> span = 1): u keeps variant units in floor columns",
            "tabpfn35 receives raw upstream-unit X[ctx], y[ctx], X[qry]; its fingerprint feature hashes raw values",
            "foundation yz adds an absolute 1e-8 to std (live in float32 when std < 0.25)",
            "absolute span floors (foundation 1e-12, target 1e-12) and the target-leak thresholds (1e-12, 1e-10)"],
        "rounding_only": [
            "float32 cast of rescaled float64 values (worker.py:53); A's values sit on the float32 lattice of the "
            "upstream all-row scale (spec A P3), which no re-normalization can move",
            "float32 (X - lo)/span in foundation and target: 1-2 ULP; mean/std recomputed in float32 for yz",
            "hyperion uint8 truncation at lattice levels where 255*(x - lo)/(hi - lo) is an exact integer: +-1 flips"],
        "diagnostic_B32": "B32_vs_A isolates Codex float32 arithmetic on a lattice-free rescale; B_vs_B32 isolates the "
                          "9-digit decimal print and float32 lattice term (spec A P4)",
        "fix_route_artifacts": "fix-route differences in near-zero columns (0 < B < 1e-10, spec A P13) or construction-"
                               "risk columns (2^-27 * |lo_R - lo_V| / train range > min positive B: the variant's float64 "
                               "offset cannot keep float32 resolution) come from how the counterfactual upstream was "
                               "simulated in float64, not from the train-only fix; rounding columns hold only signed-"
                               "zero flips (IEEE -0.0 in the data object survives phi_R, not the simulated upstream; "
                               "numerically equal, but TabPFN's fingerprint hashes the bytes) or float32 rounding-"
                               "boundary straddles with a float64 gap under 1/8 float32 ULP; 'elements_unexplained' "
                               "are the rest",
        "fix_route_for_owner": "B is the owner's train-only rescaling and can be applied lane-local to CPU-side inputs; "
                               "Codex embeddings were computed on A, so making them train-only needs re-encoding on B "
                               "(GPU, not admitted) or excluding the non-invariant components and every upstream-unit "
                               "side array; tabpfn35 must be excluded or re-encoded (spec A P15)"}
    return out


def refs_correction(ci: dict) -> dict:
    """Spec B R5: UPSTREAM-SCALE-01 (.json evidence.rows, .refs.json data_object_local.rows) record 1998 rows for the
    anchor data object; the object has 1999 (manifest sample_rows, np.loadtxt). The receipts are immutable, so the
    correction is recorded here for the family-review evidence. The column claims are re-verified."""
    refs = json.loads((FINDINGS_DIR / "UPSTREAM-SCALE-01.refs.json").read_bytes())
    finding = json.loads((FINDINGS_DIR / "UPSTREAM-SCALE-01.json").read_bytes())
    A, Rn = ci["A"], np.asarray(ci["R"], dtype=np.int64)
    lo_all, hi_all, lo_R, hi_R = A.min(0), A.max(0), A[Rn].min(0), A[Rn].max(0)
    n = int(A.shape[0])
    corrections = []
    for receipt, field, recorded in (("UPSTREAM-SCALE-01.refs.json", "data_object_local.rows",
                                      (refs.get("data_object_local") or {}).get("rows")),
                                     ("UPSTREAM-SCALE-01.json", "evidence.rows", (finding.get("evidence") or {}).get("rows"))):
        corrections.append({"receipt": receipt, "field": field, "recorded": recorded, "observed": n,
                            "correct": recorded == n})
    not_both = [int(j) for j in np.flatnonzero((lo_R != 0.0) | (hi_R != 1.0))]
    return {"anchor_view_id": ci["row"]["view_id"], "receipts_immutable": True, "corrections": corrections,
            "evidence": {"np_loadtxt_shape": [n, int(A.shape[1])], "manifest_sample_rows": int(ci["shape"]["sample_rows"]),
                         "fold0_train_rows": len(ci["R"]), "fold0_test_rows": len(ci["T"]),
                         "likely_cause": "a header-assuming read of a header-less CSV drops one row"},
            "column_claims": {"columns": int(A.shape[1]),
                              "columns_exactly_0_1_over_all_rows": int(np.count_nonzero((lo_all == 0.0) & (hi_all == 1.0))),
                              "fold0_train_columns_not_reaching_both_extremes": not_both,
                              "fold0_train_min_of_those_columns": [float(lo_R[j]) for j in not_both]}}


# ------------------------------------------------------------------------------------------------------------ run
def _check_out_dir(out: Path) -> Path:
    out = Path(out).resolve()
    if out.parent != TMP_DIR.resolve() or not OUT_NAME_RE.match(out.name):
        raise ScaleCheckError(f"--out must be {TMP_DIR}/scale-check-<label>/, got {out}")
    if out.exists() and any(out.iterdir()):
        raise ScaleCheckError(f"--out {out} exists and is not empty (create-only)")
    out.mkdir(parents=True, exist_ok=True)
    return out


def _check_receipt_path(path) -> Path:
    path = Path(path).resolve()
    if path.parent != FINDINGS_DIR.resolve() or not RECEIPT_NAME_RE.match(path.name):
        raise ScaleCheckError(f"--receipt must be {FINDINGS_DIR}/UPSTREAM-SCALE-01.check-<YYYYmmddTHHMMSSZ>.json")
    if path.exists():
        raise ScaleCheckError(f"--receipt {path} exists (create-only)")
    return path


def _write_json(out: Path, name: str, doc) -> dict:
    data = common.canonical(doc) + b"\n"
    res = common.immutable_write(out / name, data)
    return {"path": res["path"], "sha256": res["sha256"], "bytes": res["bytes"]}


def _write_jsonl_gz(out: Path, name: str, rows) -> dict:
    body = b"".join(common.canonical(r) + b"\n" for r in rows)
    data = gzip.compress(body, compresslevel=6, mtime=0)
    res = common.immutable_write(out / name, data)
    return {"path": res["path"], "sha256": res["sha256"], "bytes": res["bytes"], "raw_sha256": common.sha256_bytes(body),
            "rows": len(rows)}


def _holdout_evaluation(flag) -> bool:
    """A sampled context whose family is held out or dev, or that is a diag context: kept, but flagged, because evaluation
    inputs need the same assurance."""
    return bool(flag) and (flag.get("assignment") in ("heldout", "dev") or bool(flag.get("diag")))


def _epoch_counts(sources: Sources) -> dict:
    rows = sources.epoch["contexts"]
    inc = [r for r in rows if r.get("status") == "included"]
    by = collections.Counter((r["cohort"], int(r["fold"])) for r in inc)
    sealed = [r for r in inc if r["context_sha256"] in sources.sealed_ctx]
    return {"rows": len(rows), "included": len(inc), "frozen": sum(1 for r in inc if r["cohort"] == "frozen"),
            "incoming": sum(1 for r in inc if r["cohort"] == "incoming"),
            "datasets": len({r["dataset_hash"] for r in inc}),
            "frozen_by_fold": [by[("frozen", f)] for f in range(5)], "incoming_by_fold": [by[("incoming", f)] for f in range(5)],
            "families": sources.registry_ref["families"], "sealed_families": sources.sealed_families,
            "sealed_included_contexts": len(sealed), "sealed_frozen": sum(1 for r in sealed if r["cohort"] == "frozen"),
            "sealed_incoming": sum(1 for r in sealed if r["cohort"] == "incoming")}


def _detail(cat: str, rec: dict, row: dict, sources: Sources, cache: DataCache, args, errors, first_errors):
    """Pass 2 for one selected context: positions, A0, replication."""
    try:
        ci = resolve_context(row, sources, cache)
        res = analyse(ci, sources, positions=args.positions, want_a0=not args.no_a0, want_replication=True)
    except Exception as exc:
        errors[f"detail:{type(exc).__name__}"] += 1
        first_errors.append(f"detail {row['view_id']}: {type(exc).__name__}: {exc}")
        return None
    ident = context_identity(ci, sources)
    ident.update(category=cat, local_complete=rec["local_complete"])
    return res, {"context": ident, **res}


def run(args) -> int:
    t0 = time.time()
    started = common.utc_iso()
    out = _check_out_dir(args.out)
    receipt_path = _check_receipt_path(args.receipt) if args.receipt else None
    if args.only and receipt_path:
        raise ScaleCheckError("--only bypasses the spec B sample; it cannot write a receipt")
    if not 1 <= args.sample <= 4873:
        raise ScaleCheckError("--sample must be in [1, 4873]")
    guard = args.guard if getattr(args, "guard", None) is not None else cpuguard.Guard(
        journal_path=out / "guard.jsonl", max_pause_s=args.max_pause_s)
    guard.wait()
    sources = Sources(registry_path=args.registry, holdout_path=args.holdout)
    cache = DataCache()
    included = sources.included()
    included.sort(key=lambda r: (r["dataset_hash"], r["cohort"], r["view_id"]))
    errors = collections.Counter()
    first_errors = []
    records_all, unsealed_records = [], []
    census, census_flagged = Census(), Census()
    rep_counts = {"unsealed": collections.Counter(), "sealed": collections.Counter()}
    rep_failed_examples = []
    lineage_all, lineage_census = collections.Counter(), collections.Counter()
    xcheck = collections.Counter()
    census_limit = None if args.no_census else (args.census_limit if args.census_limit is not None else -1)
    census_rows = []
    census_done = 0
    anchor_record, corrections = None, None

    # pass 1: prescan, lineage proofs and local replication of every included context; census of the unsealed ones
    for i_row, row in enumerate(included):
        guard.tick()
        if i_row % 250 == 0:
            print(json.dumps({"progress": "pass1", "done": i_row, "of": len(included), "census": census_done,
                              "errors": sum(errors.values()), "elapsed_s": round(time.time() - t0, 1)}),
                  file=sys.stderr, flush=True)
        sealed = row["context_sha256"] in sources.sealed_ctx
        try:
            ci = resolve_context(row, sources, cache)
            xcheck[f"schema {ci['cohort']} {ci['context'].get('schema')}"] += 1
            if ci["cohort"] == "incoming":
                xcheck[f"materialization_crosscheck={ci['materialization_crosscheck']}"] += 1
            xcheck[f"index {'sanitized' if ci['sanitized'] else 'plain'} covered={ci['index_check'] is not None}"] += 1
            do_census = (not sealed) and census_limit is not None and (census_limit < 0 or census_done < census_limit)
            if do_census:
                res = analyse(ci, sources, positions=0, want_a0=False, want_replication=True)
                ps, lin, local, rp = res["prescan"], res["lineage"], bool(res["local"]), res["replication"]
                census.add(res)
                if ps["flagged"]:
                    census_flagged.add(res)
                census_done += 1
                for k, v in lin.items():
                    lineage_census[f"{k}={v}"] += 1
                census_rows.append({
                    "view_id": row["view_id"], "context_sha256": row["context_sha256"], "local": local, "owner": ci["owner"],
                    "classes": {k: c["class"] for k, c in res["comparisons"].items()},
                    "tabpfn35_refined_p9": {k: res["comparisons"][k]["components"]["tabpfn35"]["refined_p9"]["class"]
                                            for k in GATE_KEYS + FIX_GATE_KEYS},
                    "fix_attribution": {k: {x: v[x] for x in ("differing_elements", "elements_in_near_zero_columns",
                                                              "elements_in_construction_risk_columns",
                                                              "elements_in_rounding_columns", "elements_unexplained",
                                                              "only_near_zero", "only_artifact", "only_explained")}
                                        for k, v in res["fix_attribution"].items()},
                    "lineage": lin, "replication": rp, "variant_meta": res["variant_meta"]})
            else:
                ref = codex_inputs(ci["A"][np.asarray(ci["R"], dtype=np.int64)])
                ps = prescan(ci, ref)
                lin, _raw, _labels = lineage_checks(ci, ref)
                outs = sources.local_outcomes(ci["packet_dataset"], ci["packet_fold"])
                local = outs is not None
                rp = replicate(ci, ref, outs, sources.arrays_dir) if local else None
            for k, v in lin.items():
                lineage_all[f"{k}={v}"] += 1
            bucket = rep_counts["sealed" if sealed else "unsealed"]
            bucket["owner_" + ci["owner"]] += 1
            bucket["local_complete"] += int(local)
            bucket["local_complete_sanitized"] += int(local and ci["sanitized"])
            bucket["local_not_owned_by_" + LOCAL_NODE] += int(local and ci["owner"] != LOCAL_NODE)
            bucket[f"owned_by_{LOCAL_NODE}_not_local"] += int(ci["owner"] == LOCAL_NODE and not local)
            if local and rp is not None and rp.get("pass") is not None:
                bucket["replicated"] += 1
                bucket["replication_pass"] += int(rp["pass"])
                bucket["std_np2_native_differs"] += int(rp.get("std_np2_native_equal") is False)
                if not rp["pass"] and len(rep_failed_examples) < 20:
                    rep_failed_examples.append({"view_id": "sealed context (identity withheld)" if sealed else row["view_id"],
                                                "failed": rp["failed"], "missing": rp["missing"]})
            rec = {"context_sha256": row["context_sha256"], "dataset_hash": row["dataset_hash"], "cohort": ci["cohort"],
                   "fold": ci["fold"], "rank": rank_of(row["context_sha256"]), "prescan": ps, "sealed": sealed,
                   "local_complete": bool(local), "owner": ci["owner"]}
            records_all.append(rec)
            if not sealed:
                rec["view_id"] = row["view_id"]
                unsealed_records.append(rec)
                if row["context_sha256"] == ANCHOR_CONTEXT:
                    anchor_record = rec
                    corrections = refs_correction(ci)
        except ScaleCheckError as exc:
            errors["ScaleCheckError"] += 1
            if len(first_errors) < 20:
                first_errors.append("sealed context (identity withheld)" if sealed else f"{row['view_id']}: {exc}")
        except Exception as exc:  # recorded and turned into STOP; the run continues so every failure is counted
            errors[type(exc).__name__] += 1
            if len(first_errors) < 20:
                first_errors.append("sealed context (identity withheld)" if sealed
                                    else f"{row['view_id']}: {type(exc).__name__}: {exc}")

    # sample (spec B) and the S-FLOOR supplement
    U = [r for r in unsealed_records if r["local_complete"] and r["context_sha256"] != ANCHOR_CONTEXT]
    u_sha = common.sha256_bytes(common.canonical(sorted(r["context_sha256"] for r in U)))
    if args.only:
        wanted = set(args.only)
        chosen = [("ONLY", r) for r in unsealed_records if r["view_id"] in wanted]
        if len(chosen) != len(wanted):
            raise ScaleCheckError("--only names a context that is not an unsealed included context")
        loose = chosen
    else:
        chosen = select_sample(U, args.sample, anchor_record, u_control="b_identity_strict")
        loose = select_sample(U, args.sample, anchor_record, u_control="b_identity_all")
    picked = [r["context_sha256"] for _c, r in chosen]
    n_ref = min(len(picked), len(SPEC_B_SAMPLE_48))
    equal_in_order = n_ref > 0 and tuple(picked[:n_ref]) == SPEC_B_SAMPLE_48[:n_ref]
    sample_doc = {
        "algorithm": "spec B DETERMINISTIC SAMPLE (rank = sha256('jope.dl-scale-leak-sample/v1|' + context_sha256))",
        "u_control": "strict: phi_R(A[R]) bitwise equal to A[R] on every considered column (spec B R4)",
        "requested": args.sample, "selected": len(chosen), "universe_U": len(U),
        "U_flagged": sum(1 for r in U if r["prescan"]["flagged"]), "U_sha256": u_sha,
        "U_matches_spec_b": {"size": len(U) == SPEC_B_U["size"],
                             "flagged": sum(1 for r in U if r["prescan"]["flagged"]) == SPEC_B_U["flagged"],
                             "sha256": u_sha == SPEC_B_U["sha256"]},
        "anchor_included": anchor_record is not None and any(c == "ANCHOR" for c, _ in chosen),
        "u_control_forms_select_same": [r["context_sha256"] for _c, r in loose] == picked,
        "contexts": [{"category": c, "view_id": r["view_id"], "context_sha256": r["context_sha256"]} for c, r in chosen]}
    sample_doc["contexts_sha256"] = common.sha256_bytes(common.canonical(picked))
    enforced = sources.registry_is_pre_review and not args.only
    sample_doc["spec_b_reference"] = {
        "equal_in_order": equal_in_order, "compared": n_ref, "common": len(set(picked) & set(SPEC_B_SAMPLE_48)),
        "contexts_sha256_matches_spec_b": len(picked) == 48 and sample_doc["contexts_sha256"] == SPEC_B_SAMPLE_SHA256,
        "registry_is_pre_review": sources.registry_is_pre_review, "enforced": enforced,
        "note": "under the pinned pre-review registry equal_in_order must hold (else STOP); another registry may "
                "legitimately select other contexts"}
    s_floor_recs = sorted((r for r in unsealed_records if r["prescan"]["foundation_floor_risk"] > 0), key=lambda r: r["rank"])
    s_floor_doc = {"definition": "unsealed contexts where a kept column varies on R in float64 but its float32 train span "
                                 "is <= 1e-12 (the foundation span floor is reachable); analysed with positions; reported "
                                 "outside the 48-context gate denominators",
                   "contexts": [{"view_id": r["view_id"], "context_sha256": r["context_sha256"],
                                 "local_complete": r["local_complete"], "severity": r["prescan"]["severity"]}
                                for r in s_floor_recs],
                   "equals_spec_b": sorted(r["view_id"] for r in s_floor_recs) == sorted(S_FLOOR_SPEC)}

    # pass 2: detailed sample and S-FLOOR (positions, A0, replication)
    print(json.dumps({"progress": "pass2", "sample": len(chosen), "s_floor": 0 if args.only else len(s_floor_recs),
                      "census": census_done, "errors": sum(errors.values()), "elapsed_s": round(time.time() - t0, 1)}),
          file=sys.stderr, flush=True)
    rows_by_sha = {r["context_sha256"]: r for r in included}
    sample_results, detail_rows = [], []
    sample_census = Census()
    for cat, rec in chosen:
        guard.tick()
        got = _detail(cat, rec, rows_by_sha[rec["context_sha256"]], sources, cache, args, errors, first_errors)
        if got is not None:
            sample_results.append(got[0])
            sample_census.add(got[0])
            detail_rows.append(got[1])
    s_floor_results, s_floor_rows = [], []
    s_floor_census = Census()
    if not args.only:
        for rec in s_floor_recs:
            guard.tick()
            got = _detail("S-FLOOR", rec, rows_by_sha[rec["context_sha256"]], sources, cache, args, errors, first_errors)
            if got is not None:
                s_floor_results.append(got[0])
                s_floor_census.add(got[0])
                s_floor_rows.append(got[1])

    gate = gate_verdict(sample_results)
    fix_gate = gate_verdict(sample_results, FIX_GATE_KEYS, FIX_RULE)
    diag_gate = gate_verdict(sample_results, DIAG_KEYS, "DIAGNOSTIC (not a gate): B32 vs A and B vs B32 (spec A P4)")
    s_floor_doc["classes"] = gate_verdict(s_floor_results, GATE_KEYS + FIX_GATE_KEYS,
                                          "S-FLOOR supplement (informational; outside the gate denominators)")
    s_floor_doc["per_context"] = [{"view_id": d["context"]["view_id"],
                                   "classes": {k: d["comparisons"][k]["class"] for k in GATE_KEYS + FIX_GATE_KEYS}}
                                  for d in s_floor_rows]
    rep_sample = collections.Counter()
    for res in sample_results + s_floor_results:
        rp = res["replication"]
        if rp and rp.get("pass") is not None:
            rep_sample["local"] += 1
            rep_sample["pass"] += int(rp["pass"])
        else:
            rep_sample["not_local"] += 1
    lineage_sample = collections.Counter(f"{k}={v}" for res in sample_results + s_floor_results
                                         for k, v in res["lineage"].items())
    a0_sum = collections.Counter()
    a0_max_ulp, a0_max_abs, a0_max_abs_relabelled = 0, 0.0, 0.0
    for res in sample_results:
        a0 = res["a0"] or {}
        a0_sum[f"status={a0.get('status')}"] += 1
        if a0.get("status") == "computed":
            a0_sum["bit_identical_vectors"] += int(a0["differ"] == 0)
            a0_sum["features_differ"] += a0["differ"]
            a0_sum["from_raw_equals_direct"] += int(bool(a0.get("from_raw_equals_direct")))
            lab = a0.get("inner_labels_recomputed_on_B_differ")
            a0_sum["inner_labels_change_under_B"] += int(isinstance(lab, int) and lab > 0)
            rel = a0.get("relabelled_B") or {}
            a0_sum["relabelled_B_vectors_differ"] += int(rel.get("differ", 0) > 0)
            a0_max_abs_relabelled = max(a0_max_abs_relabelled, float(rel.get("max_abs", 0.0)))
            a0_max_ulp = max(a0_max_ulp, a0["max_ulp"])
            a0_max_abs = max(a0_max_abs, a0["max_abs"])
    a0_notes = sorted({(res["a0"] or {}).get("note") or (res["a0"] or {}).get("error") for res in sample_results} - {None})

    all_sum, uns_sum = prescan_summary(records_all), prescan_summary(unsealed_records)
    sealed_sum = {k: v for k, v in prescan_summary([r for r in records_all if r["sealed"]]).items()
                  if k in ("contexts", "flagged", "flagged_columns", "considered_columns", "flagged_target_only",
                           "subsampled_contexts", "local_contexts")}
    strata = strata_summary(unsealed_records)
    epoch_counts = _epoch_counts(sources)
    lineage_counts = {
        "index_covered_non_sanitized": xcheck["index plain covered=True"],
        "index_absent_non_sanitized": xcheck["index plain covered=False"],
        "local_complete_included": rep_counts["unsealed"]["local_complete"] + rep_counts["sealed"]["local_complete"],
        "local_complete_sanitized": (rep_counts["unsealed"]["local_complete_sanitized"]
                                     + rep_counts["sealed"]["local_complete_sanitized"]),
        "local_complete_unsealed": rep_counts["unsealed"]["local_complete"]}
    anchor_facts = None
    if anchor_record is not None:
        p = anchor_record["prescan"]
        anchor_facts = {"cohort": anchor_record["cohort"], "sealed": anchor_record["sealed"], "owner": anchor_record["owner"],
                        "local_complete": anchor_record["local_complete"], "n_train": p["n_train"], "n_test": p["n_test"],
                        "F": p["F"], "k": p["k"], "flagged_cols": p["flagged_cols"], "test_only_min": p["test_only_min"],
                        "test_only_max": p["test_only_max"], "D_rows": p["D_rows"],
                        "exact01_all_columns": p["exact01_all_columns"], "floor_columns": p["floor_columns"],
                        "floor_remapped_by_B": p["floor_remapped_by_B"], "severity": p["severity"]}
    spec_crosscheck = {
        "epoch_registry": crosscheck(epoch_counts, {**SPEC_B_EPOCH, **SPEC_B_REGISTRY}),
        "index": crosscheck(sources.index_ref or {}, SPEC_B_INDEX),
        "lineage": crosscheck(lineage_counts, SPEC_B_LINEAGE),
        "prescan": prescan_crosscheck(all_sum, uns_sum, sealed_sum, strata),
        "anchor": crosscheck(anchor_facts or {}, SPEC_B_ANCHOR) | {
            "severity_is_1/58": anchor_facts is not None and anchor_facts["severity"] is not None
            and math.isclose(anchor_facts["severity"], 1 / 58, rel_tol=1e-6)},
        "sample": {"U": sample_doc["U_matches_spec_b"], "equal_in_order": equal_in_order,
                   "contexts_sha256": sample_doc["spec_b_reference"]["contexts_sha256_matches_spec_b"],
                   "s_floor": s_floor_doc["equals_spec_b"],
                   "all_match": (all(sample_doc["U_matches_spec_b"].values()) and equal_in_order and s_floor_doc["equals_spec_b"]
                                 and sample_doc["spec_b_reference"]["contexts_sha256_matches_spec_b"])},
        "note": "spec B numbers are an independent cross-check; a mismatch is flagged for review, it is not a STOP"}
    spec_crosscheck["all_match"] = all(
        [spec_crosscheck[k]["all_match"] for k in ("epoch_registry", "index", "lineage", "prescan", "anchor")]
        + [spec_crosscheck["anchor"]["severity_is_1/58"]] + list(sample_doc["U_matches_spec_b"].values())
        + [equal_in_order, sample_doc["spec_b_reference"]["contexts_sha256_matches_spec_b"], s_floor_doc["equals_spec_b"]])

    stop_reasons = []
    if sum(errors.values()):
        stop_reasons.append(f"processing/lineage errors: {dict(errors)}")
    for label, bucket in rep_counts.items():
        if bucket["replicated"] != bucket["replication_pass"]:
            stop_reasons.append(f"replication gate failed on a local {label} context")
        if bucket["local_complete"] != bucket["replicated"]:
            stop_reasons.append(f"a local {label} context was not replicated")
        if bucket["local_not_owned_by_" + LOCAL_NODE]:
            stop_reasons.append(f"a local {label} context is not owned by shard {LOCAL_NODE} (worker.py:70)")
    if rep_sample["local"] != rep_sample["pass"]:
        stop_reasons.append("replication gate failed on a local sample or S-FLOOR context")
    for counter, label in ((lineage_sample, "sample"), (lineage_census, "census"), (lineage_all, "all included")):
        bad = {k: v for k, v in counter.items() if k.endswith("=False")}
        if bad:
            stop_reasons.append(f"{label} lineage mismatch: {bad}")
    if sources.index_ref and sources.index_ref.get("packet_ref_sha256") != CODEX_PINS["packet.json"]:
        stop_reasons.append("Codex index packet_ref differs from the pinned packet")
    if enforced and not equal_in_order:
        stop_reasons.append("equal_in_order = false under the pinned pre-review registry")
    if len(sample_results) != len(chosen) or len(s_floor_results) != (0 if args.only else len(s_floor_recs)):
        stop_reasons.append("a selected context could not be analysed")
    if "torch" in sys.modules or "boto3" in sys.modules:
        stop_reasons.append("torch or boto3 was imported")

    files = {}
    files["prescan.jsonl.gz"] = _write_jsonl_gz(out, "prescan.jsonl.gz", [
        {k: r[k] for k in ("view_id", "context_sha256", "dataset_hash", "cohort", "fold", "rank", "local_complete", "owner",
                           "prescan")} for r in unsealed_records])
    files["sample.json"] = _write_json(out, "sample.json", sample_doc)
    files["sample-detail.jsonl.gz"] = _write_jsonl_gz(out, "sample-detail.jsonl.gz", detail_rows)
    if s_floor_rows:
        files["s-floor-detail.jsonl.gz"] = _write_jsonl_gz(out, "s-floor-detail.jsonl.gz", s_floor_rows)
    if census_rows:
        files["census-contexts.jsonl.gz"] = _write_jsonl_gz(out, "census-contexts.jsonl.gz", census_rows)
    sample_summary = {k: v for k, v in sample_doc.items() if k != "contexts"} | {
        "categories": dict(collections.Counter(c for c, _ in chosen)),
        "flagged": sum(1 for _c, r in chosen if r["prescan"]["flagged"]),
        "target_flagged": sum(1 for _c, r in chosen if r["prescan"]["target_flagged"]),
        "floor_remapped": sum(1 for _c, r in chosen if r["prescan"]["floor_remapped_by_B"] > 0),
        "subsampled": sum(1 for _c, r in chosen if r["prescan"]["subsampled"]),
        "out01": sum(1 for _c, r in chosen if r["prescan"]["out01"]),
        "k_gt_128": sum(1 for _c, r in chosen if r["prescan"]["k"] > 128),
        "local_complete": sum(1 for _c, r in chosen if r["local_complete"]),
        "datasets": len({r["dataset_hash"] for _c, r in chosen}),
        "families": len({sources.family_of.get(r["context_sha256"]) for _c, r in chosen}),
        "seed_selection_exposed": sum(1 for _c, r in chosen if r["context_sha256"] in sources.seed_exposed_ctx),
        "holdout_flagged": (None if not sources.holdout_ref else
                            sum(1 for _c, r in chosen if _holdout_evaluation(sources.holdout_flag(r["context_sha256"]))))}
    verdict = {"gate": gate["result"], "fix_route_gate": fix_gate["result"], "stop": bool(stop_reasons),
               "stop_reasons": stop_reasons, "valid_evidence": not stop_reasons,
               "codex_inputs_test_row_invariant_as_is": gate["result"] == "PASS" and not stop_reasons,
               "train_only_fix_cancels_perturbations": fix_gate["result"] == "PASS" and not stop_reasons,
               "spec_b_crosscheck_all_match": spec_crosscheck["all_match"],
               "admission_rule": "GPU training needs this check to pass: bit-identical encoder inputs as is (gate), or the "
                                 "scaling fixed to train-only (fix route). Adopting the fix route is an owner/orchestrator "
                                 "decision; Codex embeddings computed on A would need re-encoding on B or exclusion. This "
                                 "tool admits nothing."}
    census_doc = census.doc()
    head = headline(gate, fix_gate, census_doc, s_floor_doc, rep_counts, a0_sum)
    summary = {
        "schema": SCHEMA_RESULT, "finding": FINDING_ID, "started_utc": started, "finished_utc": common.utc_iso(),
        "elapsed_s": round(time.time() - t0, 3), "verdict": verdict, "headline": head,
        "gate": gate, "fix_route_gate": fix_gate, "diagnostics_b32": diag_gate, "residual": residual_doc(sample_census.doc()),
        "position_fields": {"f32": POSITION_FIELDS_F32, "u8": POSITION_FIELDS_U8,
                            "row": "data-object row of an outer-train row", "col": "data-object column, or 'target'"},
        "sample": sample_summary, "sample_comparisons": sample_census.doc(),
        "s_floor": s_floor_doc | {"comparisons": s_floor_census.doc()},
        "replication": {"sample_and_s_floor": dict(rep_sample),
                        "all_local": {k: dict(sorted(v.items())) for k, v in rep_counts.items()},
                        "failed_examples": rep_failed_examples},
        "lineage": {"sample": dict(sorted(lineage_sample.items())), "census": dict(sorted(lineage_census.items())),
                    "all_included": dict(sorted(lineage_all.items())), "crosschecks": dict(sorted(xcheck.items())),
                    "counts": lineage_counts, "epoch_registry": epoch_counts},
        "a0": dict(a0_sum) | {"max_ulp": a0_max_ulp, "max_abs": a0_max_abs, "max_abs_relabelled_B": a0_max_abs_relabelled,
                              "labels": "held fixed at the rebuilt JTT1 inner labels; relabelled_B recomputes them on B",
                              "notes": a0_notes},
        "census": census_doc | {"limit": census_limit, "unsealed_contexts": len(unsealed_records)},
        "census_flagged": census_flagged.doc(),
        "prescan": {"all_included": all_sum, "unsealed": uns_sum, "sealed": sealed_sum, "strata_unsealed": strata,
                    "anchor": anchor_facts, "sealed_contexts": sum(1 for r in records_all if r["sealed"]),
                    "sealed_families": sources.sealed_families},
        "spec_b_crosscheck": spec_crosscheck, "corrections_R5": corrections,
        "errors": {"counts": dict(errors), "first": first_errors},
        "inputs": {"epoch": sources.epoch_ref, "registry": sources.registry_ref, "holdout": sources.holdout_ref,
                   "codex_pins": sources.pins, "codex_packet": sources.packet_ref, "codex_index": sources.index_ref,
                   "finding_pins": FINDING_PINS, "scale_check_sha256": MODULE_SHA256, "numpy": np.__version__,
                   "python": sys.version.split()[0]},
        "policy": {"s3_calls": 0, "jtt1_gets": 0, "gpu": False, "torch_imported": "torch" in sys.modules,
                   "boto3_imported": "boto3" in sys.modules, "encoder_weights_opened": False,
                   "codex_files_imported": False, "test_rows": "counts only; positions name outer-train rows",
                   "sealed_identities_written": False},
        "guard": guard.summary() if hasattr(guard, "summary") else None,
        "argv": common.redact_argv(sys.argv),
    }
    files["summary.json"] = _write_json(out, "summary.json", summary)
    if receipt_path is not None:
        receipt = {"schema": SCHEMA_RECEIPT, "finding": FINDING_ID, "utc_us": common.utc_us(),
                   "check": "test-row counterfactual invariance of the Codex encoders' training-side inputs "
                            "(A/A1/B/C/C2, fix route, B32 diagnostic)",
                   "owner_ask": "perturb or remove the TEST rows, redo the source min-max scaling, run the Codex train-only "
                                "re-normalization, and assert the training-side encoder inputs are bit-identical (or "
                                "document the exact residual); if it does not cancel, re-derive the scaling from training "
                                "rows only (lane-local); gates GPU-training admission",
                   "verdict": verdict, "headline": head, "gate": gate, "fix_route_gate": fix_gate,
                   "diagnostics_b32": diag_gate, "residual": summary["residual"], "sample": sample_summary,
                   "sample_contexts": sample_doc["contexts"], "s_floor": s_floor_doc,
                   "census": census_compact(census_doc) | {"limit": census_limit,
                                                           "unsealed_contexts": len(unsealed_records)},
                   "census_flagged": census_compact(summary["census_flagged"]),
                   "replication": summary["replication"], "lineage": summary["lineage"], "a0": summary["a0"],
                   "prescan": summary["prescan"], "spec_b_crosscheck": spec_crosscheck, "corrections_R5": corrections,
                   "errors": summary["errors"], "inputs": summary["inputs"], "policy": summary["policy"], "outputs": files,
                   "elapsed_s": summary["elapsed_s"], "argv": summary["argv"]}
        common.write_receipt(receipt_path, receipt)
    print(json.dumps({"out": str(out), "gate": gate["result"], "fix_route_gate": fix_gate["result"],
                      "fix_by_variant": fix_gate["by_variant"], "stop": bool(stop_reasons), "stop_reasons": stop_reasons,
                      "sample": len(sample_results), "s_floor": len(s_floor_results), "census": census.contexts,
                      "spec_b_crosscheck_all_match": spec_crosscheck["all_match"],
                      "receipt": str(receipt_path) if receipt_path else None, "by_variant": gate["by_variant"]},
                     sort_keys=True))
    if stop_reasons:
        return 2
    return 0 if gate["result"] == "PASS" else 1


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="dltrain.scale_check", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="prescan, census, spec B sample, S-FLOOR, both gates; exit 0 PASS, 1 FAIL, 2 STOP, 3 ERROR")
    r.add_argument("--sample", type=int, default=48, help="spec B sample size (default 48; a smaller N is a prefix)")
    r.add_argument("--out", required=True, type=Path, help=f"{TMP_DIR}/scale-check-<label>/ (new or empty; create-only)")
    r.add_argument("--receipt", type=Path,
                   help=f"create-only {FINDINGS_DIR}/UPSTREAM-SCALE-01.check-<YYYYmmddTHHMMSSZ>.json")
    r.add_argument("--registry", type=Path,
                   help=f"family registry (default: the pinned pre-review registry {REGISTRY_PRE.name}, sha256 "
                        f"{REGISTRY_PRE_SHA256[:12]}...; another registry is recorded and not held to spec B's 48)")
    r.add_argument("--holdout", type=Path, help="holdout.json for flagging held/dev/diag contexts (default RUN/holdout)")
    r.add_argument("--no-census", action="store_true", help="prescan only outside the sample (no census)")
    r.add_argument("--census-limit", type=int, help="census at most K unsealed contexts (smoke runs)")
    r.add_argument("--positions", type=int, default=256, help="differing positions listed per array (default 256)")
    r.add_argument("--no-a0", action="store_true", help="skip the A0 comparison")
    r.add_argument("--max-pause-s", type=float, default=3600.0, help="CPU-admission pause budget before GuardTimeout")
    r.add_argument("--only", action="append", help="debug: analyse these view_ids instead of the spec B sample (no receipt)")
    args = ap.parse_args(argv)
    try:
        return run(args)
    except ScaleCheckError as exc:
        print(json.dumps({"stop": True, "error": str(exc)}), file=sys.stderr)
        return 2
    except cpuguard.GuardTimeout as exc:
        print(json.dumps({"error": f"GuardTimeout: {exc}"}), file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(_main())
