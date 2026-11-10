"""Data-free H281 controls: the exactness classifier (Outcome A needs 91/91 utilities, 91/91 picks and the mean to
1e-12), the VRAM admission rule (free >= need + 3 GiB), the H278 source pins, the export-manifest pin and the
writer-path constants. CUDA is never initialised."""

import hashlib
import json
from pathlib import Path

import gpu_rescore as g

HERE = Path(__file__).resolve().parent
H278 = HERE.parent / "h278_rerank8_veto_frozen_export_transfer"
GIB = 2**30


def test_classify_exactness():
    s = 1.1389572403748425
    assert g.classify(91, 91, s, s) == "A_EXACT"
    assert g.classify(91, 91, s + 5e-13, s) == "A_EXACT"
    assert g.classify(90, 91, s, s) == "B_NOT_EXACT"
    assert g.classify(91, 90, s, s) == "B_NOT_EXACT"
    assert g.classify(91, 91, s + 2e-12, s) == "B_NOT_EXACT"
    assert (
        g.classify(90, 90, 1.1366829009431036, s) == "B_NOT_EXACT"
    )  # the H278 CPU result


def test_vram_admission():
    assert g.NEED_BYTES == int(1.5 * GIB) and g.HEADROOM_BYTES == 3 * GIB
    assert g.admit(int(4.5 * GIB)) and not g.admit(int(4.5 * GIB) - 1)
    assert g.ALLOCATOR_CAP_BYTES == 2 * GIB


def test_writer_path_constants():
    assert g.WRITER_PID == 1402163
    assert g.CKPT_SHA256.startswith("1d6ea900") and g.DOPE.startswith("c51a56d0")
    assert g.SOURCE_SHA256.startswith("b9f1468a")
    for name, prefix in (
        ("device_pool_cache.py", "bb4426ead00f9243"),
        ("dope_freeze_guard.py", "38cbed30f8361a5e"),
    ):
        assert hashlib.sha256((HERE / name).read_bytes()).hexdigest().startswith(prefix)


def test_h278_pins_and_export():
    m = json.loads((HERE / "manifest.json").read_text())
    for name, digest in m["h278_sources"].items():
        assert hashlib.sha256((H278 / name).read_bytes()).hexdigest() == digest
    assert m["export_manifest_sha256"].startswith("2fce784a959179e0")
    assert m["total_updates"] == 0 and m["wave"] == "H281"
