"""Assemble collected GPU outputs (tools/v2_gpu_encode.py OUT/<view_id>/) into a publishable tree PUB/shard-<host>/<view_id>/,
re-verifying every file against its context.json and adding the owner-ruled per-component `label_exposure` block and the
P3 `holdout_assignment` (sealed_excluded / dev / heldout / train_*), so trainers can refuse what the protocol excludes.

Exposure source: the pretraining-overlap receipt (5f16216d), looked up by view_id, else by the epoch-0 view_id a rebuilt
context replaces, else the most conservative status over the dataset's epoch-0 contexts (exposure is dataset-level).
Rulings: TARGET-ENC and HYPERION-ARMB-LABELS (owner-decisions-20261005T220413Z.json); hyperion_v6 inherits armB and its
fine-tune saw the features (never targets) of every dataset in the v6 pack. Nothing is uploaded here; the tree is only written.

Usage: v2_gpu_assemble.py --out OUT [--out OUT2 ...] --pub PUB
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
from pathlib import Path

RUN = Path("/home/ubuntu/scratch/jopedime-dl-training-v1")
OVERLAP = RUN / "receipts/trainonly-v2/pretraining-overlap-20261005T203956Z.jsonl"
OVERLAP_SHA = "5f16216d2350ef30c06c570c0f33cbf8957771d1023f11f6f11ba0523405d24a"
TARGET_COPY = RUN / "tmp/target-copy-datasets.json"
V6_PACK = RUN / "build-v2/HYPERION-V6/pack/pack-manifest.jsonl"
DEC = "owner-decisions-20261005T220413Z.json#sha256=e32a400338854f9373eea481ed7f296e4efaeb553c9b210c6e6ca3533acbc369"
V6_RECEIPT = "v2-trainonly/models/hyperion-v6/4d656b867d97/receipt.json (sha256 affcdaf78f5757e01d8d61adf812b008936dc44e5e038f05bd4b88fc9be58ee9)"
H5, TG = "hyperion_v5_arm_b@88b9e834", "target_fused_v1@be96adc6"
RANK = {"Exposed": 2, "Unknown": 1, "Unseen": 0}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_refs():
    ob = OVERLAP.read_bytes()
    if sha(ob) != OVERLAP_SHA:
        raise SystemExit("overlap receipt sha256 mismatch")
    by_view, by_ds = {}, {}
    for line in ob.decode().splitlines():
        x = json.loads(line)
        by_view[x["view_id"]] = x["encoders"]
        cur = by_ds.setdefault(x["dataset_hash"], {})
        for k in (H5, TG):
            s = x["encoders"][k]["status"]
            if RANK.get(s, 1) >= RANK.get(cur.get(k, "Unseen"), 0):
                cur[k] = s
    tc = set(json.loads(TARGET_COPY.read_text()))
    v6 = {json.loads(l)["dataset_hash"] for l in V6_PACK.read_text().splitlines() if l.strip()}
    hold = json.loads((RUN / "holdout/holdout.json").read_text())["assignment"]
    reg = json.loads((RUN / "families/g2/registry.json").read_text())
    ds2fam = {m["dataset_hash"]: fid for fid, f in reg["families"].items() for m in (f.get("members") or []) if m.get("dataset_hash")}
    return by_view, by_ds, tc, v6, hold, ds2fam


def statuses(ctx, by_view, by_ds):
    for v in (ctx["view_id"], (ctx.get("epoch0") or {}).get("view_id")):
        if v and v in by_view:
            return by_view[v][H5]["status"], by_view[v][TG]["status"], "view_id" if v == ctx["view_id"] else "epoch0_view_id"
    d = by_ds.get(ctx["dataset_hash"]) or {}
    return d.get(H5, "Unknown"), d.get(TG, "Unknown"), "dataset_level"


CTX2048_CONTROL = {"status": "UNVERIFIED", "note": "pilot must-fail control did not cover context2048 (CTX2048-CONTROL)"}


def tags(ctx, refs):
    by_view, by_ds, tc, v6, hold, ds2fam = refs
    hs, ts, basis = statuses(ctx, by_view, by_ds)
    isc = ctx["dataset_hash"] in tc
    fam = ds2fam.get(ctx["dataset_hash"])
    lift = "excluded (target-copy dataset)" if isc else f"per-stratum: {hs}"
    t = {
        "hyperion": {"ruling": "HYPERION-ARMB-LABELS", "label_exposure": "armB landmark auxiliary (dataset-level label statistics) and target-copy features in pretraining",
                     "pretraining_overlap_status": hs, "status_basis": basis, "target_copy_dataset": isc, "lift": lift,
                     "headline_stratum": hs == "Unseen" and not isc, "decision": DEC},
        "target": {"ruling": "TARGET-ENC", "label_exposed": True, "research_only": True, "production_feature": False, "lift": "excluded",
                   "pretraining_overlap_status": ts, "status_basis": basis, "secondary_stratum_candidate": ts == "Unseen", "decision": DEC},
        "context2048": {"ruling": "TARGET-ENC (inherits target and hyperion)", "label_exposed": True, "research_only": True, "production_feature": False,
                        "lift": "excluded", "secondary_stratum_candidate": ts == "Unseen" and hs == "Unseen" and not isc, "decision": DEC,
                        "leak_control": CTX2048_CONTROL},
        "foundation": {"pretraining_overlap_status": "Unknown", "note": "Starlight fine-tune data not enumerated; no label-exposure ruling"},
        "tabpfn35": {"research_only": True, "ruling": "L3 evaluation-only", "pretraining_overlap_status": "Unknown"},
        "hyperion_v6": {"ruling": "HYPERION-ARMB-LABELS + HYPERION-V6", "inherits": "v5 armB (label exposure as for hyperion)",
                        "v6_finetune": "label-free; features of all rows of each dataset in the v6 pack, never targets",
                        "dataset_in_v6_pack": ctx["dataset_hash"] in v6, "pretraining_overlap_status_armB": hs, "status_basis": basis,
                        "target_copy_dataset": isc, "lift": lift, "headline_stratum": hs == "Unseen" and not isc,
                        "v6_receipt": V6_RECEIPT, "decision": DEC},
    }
    return {k: v for k, v in t.items() if f"{k}.npy" in ctx["files"]}, {"family_id": fam, "assignment": hold.get(fam, "no_family")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", action="append", required=True)
    ap.add_argument("--pub", required=True)
    ap.add_argument("--ctx2048-control", default=None, help="PATH#SHA256 of a pilot receipt whose context2048 must-fail control passed")
    a = ap.parse_args()
    global CTX2048_CONTROL
    if a.ctx2048_control:
        path, want = a.ctx2048_control.split("#", 1)
        rb = Path(path).read_bytes()
        if sha(rb) != want:
            raise SystemExit("ctx2048 control receipt sha256 mismatch")
        res = json.loads(rb)["pilot_results"]
        ok = res and all(r["outputs_bit_identical"].get("context2048") and r["control_differs"].get("context2048")
                         and all(r.get("context2048_single_input_control_differs", {}).values()) for r in res)
        if not ok:
            raise SystemExit("ctx2048 control receipt does not show a passing context2048 control")
        CTX2048_CONTROL = {"status": "VERIFIED", "pilot_receipt": f"{path}#sha256={want}", "contexts": len(res),
                           "checks": "clean path bit-identical; composition from leaky inputs differs; each single leaky input differs"}
    refs = load_refs()
    counts, bad = {}, 0
    for out in a.out:
        for p in sorted(glob.glob(f"{out}/*/context.json")):
            d = os.path.dirname(p)
            if not os.path.exists(f"{d}/.done"):
                counts["not_done"] = counts.get("not_done", 0) + 1
                continue
            ctx = json.loads(Path(p).read_text())
            for f, m in ctx["files"].items():
                b = Path(d, f).read_bytes()
                if sha(b) != m["sha256"] or len(b) != m["bytes"]:
                    bad += 1
            ctx["label_exposure"], ctx["holdout_assignment"] = tags(ctx, refs)
            host = ctx["gpu"]["components"][next(iter(ctx["gpu"]["components"]))]["host"]
            dst = Path(a.pub) / f"shard-{host}" / ctx["view_id"]
            if not (dst / ".done").exists():
                dst.mkdir(parents=True, exist_ok=True)
                for f in ctx["files"]:
                    shutil.copy2(Path(d, f), dst / f)
                (dst / "context.json").write_text(json.dumps(ctx, sort_keys=True) + "\n")
                (dst / ".done").write_text("ok\n")
            for k in ("assembled", f"assignment_{ctx['holdout_assignment']['assignment']}",
                      f"hyperion_{(ctx['label_exposure'].get('hyperion') or ctx['label_exposure'].get('hyperion_v6') or {}).get('pretraining_overlap_status', (ctx['label_exposure'].get('hyperion_v6') or {}).get('pretraining_overlap_status_armB'))}"):
                counts[k] = counts.get(k, 0) + 1
    print(json.dumps({"counts": counts, "file_mismatches": bad}, sort_keys=True))
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
