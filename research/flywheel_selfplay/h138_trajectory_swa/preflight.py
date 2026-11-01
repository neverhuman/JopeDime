"""Manual input-hash and unscored constructor proof; no inference or updates."""

from pathlib import Path
import datetime
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import sys
import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--custody", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    C = args.custody.resolve()
    HERE = Path(__file__).resolve().parent
    W = HERE.parents[2]

    def sha(p):
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()

    m = json.loads((C / "portable-inputs.json").read_text())
    assets = json.loads((C / "asset-pins.json").read_text())
    pins = json.loads((HERE / "manifest.json").read_text())
    assert sha(C / "portable-inputs.json") == pins["portable_input_manifest_sha256"]
    assert sha(C / "asset-pins.json") == pins["asset_pins_sha256"]
    assert json.loads((C / "checkpoint-copies.json").read_text()) == pins["checkpoints"]
    assert sha(C / "inputs/reference.json") == pins["reference_sha256"]
    for name, digest in pins["executor_sha256"].items():
        assert sha(HERE / name) == digest, name
    for name, digest in pins["source_sha256"].items():
        assert sha(W / name) == digest, name
    assert sys.version == assets["python"]
    for r in m["files"].values():
        p = C / "inputs/data" / r["relative"]
        assert p.stat().st_size == r["bytes"] and sha(p) == r["sha256"], p
    for p, h in assets["assets"].items():
        assert sha(p) == h, p
    for p, v in assets["packages"].items():
        assert importlib.metadata.version(p) == v, p
    opened = []

    def audit(event, args):
        if event != "open" or not isinstance(args[0], (str, bytes)):
            return
        p = str(Path(os.fsdecode(args[0])).resolve())
        if any(
            x in p
            for x in [
                "/pools/",
                "context_features.npz",
                "pool_priors.npz",
                "selected.json",
                "reference.json",
                "/jope-data/",
                "source_manifest",
                "/sealed",
                "/holdout",
                "meta-test",
                "meta_test",
            ]
        ):
            raise RuntimeError(
                "preflight forbids study/protected metadata or payload: " + p
            )
        if str(C / "inputs") in p:
            opened.append(p)

    sys.addaudithook(audit)
    sys.path.insert(0, str(HERE))
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    import runner

    h = runner.load(
        "h138_preflight_h124",
        W / "research/flywheel_selfplay/h124_gbm_shrink_blend/runner.py",
    )
    h.INPUT = C / "inputs/data/h50"
    h.R882 = C / "inputs/round-882.pt"
    manifest = pins
    net, guard, names, frozen, r = runner.build(C, h, manifest, "R882")
    r.update(
        status="PASS",
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        normalizer_keys=list(guard.DOPE_NORMALIZER_BUFFERS),
        actual_threads=len(list(Path("/proc/self/task").iterdir())),
        peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        constructor_study_payload_reads=0,
        constructor_study_metadata_reads=0,
        preconstruction_input_hash_reads=len(m["files"]),
        preconstruction_input_hash_bytes=m["bytes"],
        opened=sorted(set(opened)),
        model_constructor_only=True,
        source_sha256=sha(Path(__file__)),
        source_commit=os.environ.get("H138_SOURCE_COMMIT"),
    )
    assert r["actual_threads"] <= 4 and r["peak_rss"] < 6_000_000_000
    args.output.write_text(json.dumps(r, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in r.items() if k not in ["adapter_names", "opened"]})
    )


if __name__ == "__main__":
    main()
