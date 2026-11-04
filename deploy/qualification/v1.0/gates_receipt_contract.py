from __future__ import annotations

import re

RELEASE_GATE_IDS = "M1 M2 M3 M4 M5 S1 S2 S3 S4 R1 R2 R3 R4 A1 A2 A3 A4 A5 E1 E2 E3 U1 U2 T1 T2 T3 D1 D2".split()
STATUSES = {"PASS", "FAIL", "NOT_RUN", "BLOCKED"}
HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HEX_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
M3_WAIVER_RULING_SHA256 = (
    "5c45cf3b6fea741d753de5be0eaec3ce9443602da9bf2258a96393705f9b8ddf"
)
M3_WAIVER_PROPOSER_SHA256 = (
    "48ebbfc081451baf0470ae25e1ae04621823ac4de3dfb0d8e93576206c8dbe39"
)
