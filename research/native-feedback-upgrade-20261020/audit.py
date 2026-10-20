"""Post-run verification: exact acquisition replay, target consumption, inference.

Does not alter frozen experiment code, preregistration, checkpoints or results.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import torch

from feedback import canonical, digest, choose, ensemble_predict, ValueAdapter
from predict import load_adapters, select


def main(root):
    torch.set_num_threads(2)
    plan = json.loads((root / "preregistration.json").read_text())
    features = np.load(root / "features.npz")
    out = root / "experiment"
    archive = tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["zstd", "-dc", str(root / "inputs.tar.zst")])))
    rows = {r["context_id"]: r for r in (json.loads(line) for line in archive.extractfile("inputs/pools.jsonl"))}
    result = {"schema": "jope.native-feedback-post-run-audit/v1", "checks": [], "all_passed": False}
    for arm in ["continuation", "uniform_feedback", "targeted_feedback"]:
        previous_targets = json.loads((out / "initial-targets.json").read_text())
        initial_count = len(previous_targets)
        for wave in range(1, 4):
            stem = f"{arm}-wave{wave}"
            checkpoint = out / ("initial" if wave == 1 else f"{arm}-wave{wave-1}")
            models, model_manifest = load_adapters(checkpoint)
            events = json.loads((out / (stem + "-events.json")).read_text())
            current = json.loads((out / (stem + "-targets.json")).read_text())
            target_index = {t["outcome_sha256"]: t for t in current}
            if arm == "continuation":
                assert len(current) == initial_count and current == previous_targets
            else:
                acquired = {(t["context_id"], t["member"]) for t in previous_targets}
                expected = []
                for cid in sorted(c for c, r in plan["split"].items() if r["partition"] == "train"):
                    mean, std = ensemble_predict(models, torch.tensor(features[cid]))
                    available = [i for i in range(len(rows[cid]["members"])) if (cid, i) not in acquired]
                    rng = np.random.default_rng(int(digest(["acquire", wave, cid])[:16], 16))
                    mode = "uniform" if arm == "uniform_feedback" else "targeted"
                    for _ in range(min(2, len(available))):
                        member, propensity = choose(available, mean, std, mode=mode, rng=rng)
                        expected.append((cid, rows[cid]["members"][member]["program_sha256"], propensity))
                        available.remove(member)
                actual = [(e["context_id"], e["program_sha256"], e["propensity"]) for e in events if e["wave"] == wave]
                assert expected == actual, (arm, wave, "acquisition not reproduced")
            consumption = json.loads(gzip.decompress((out / (stem + "-consumption.json.gz")).read_bytes()))
            assert len(consumption) == 3
            for member in consumption:
                assert member["parameter_before"] != member["parameter_after"]
                assert member["optimizer_updates"] == plan["updates_per_wave"]
                for step in member["steps"]:
                    assert all(plan["split"][cid]["partition"] == "train" for cid in step["contexts"])
                    for identity in step["target_hashes"]:
                        target = target_index[identity]
                        assert target["wave"] <= wave
                        assert plan["split"][target["context_id"]]["partition"] == "train"
            current_models, manifest = load_adapters(out / stem)
            assert manifest["target_snapshot_sha256"] == digest(current)
            assert manifest["parameters"] == [r["parameter_after"] for r in consumption]
            result["checks"].append({"arm": arm, "wave": wave, "acquisition_exactly_reproduced": True,
                                     "child_parameter_changed": True, "consumed_targets_validated": True,
                                     "guide_parameters": model_manifest["parameters"],
                                     "child_parameters": manifest["parameters"], "targets": len(current)})
            previous_targets = current
    # End-to-end research inference, deliberately stripping every measured q.
    cid = sorted(c for c, r in plan["split"].items() if r["partition"] == "assessment")[0]
    meta = json.loads((root / "meta.json").read_text())
    encoding = np.load(root / "blocks_kumo_tabular_s.npy", mmap_mode="r")[meta["cids"].index(cid)]
    members = [{"tokens": m["tokens"], "program_sha256": m["program_sha256"]} for m in rows[cid]["members"]]
    proposal = select(root, out / "uniform_feedback-wave3", encoding, members, 4)
    predictions = json.loads((out / "assessment-predictions.json").read_text())[cid]["uniform_feedback"]
    expected = sorted(range(len(members)), key=lambda i: (-predictions[i], members[i]["program_sha256"]))[:4]
    assert [m["program_sha256"] for m in proposal["selected"]] == [members[i]["program_sha256"] for i in expected]
    for got, i in zip(proposal["selected"], expected):
        assert abs(got["predicted_value_score"] - predictions[i]) < 1e-6
    result["inference"] = {"context_id": cid, "checkpoint_reload_predictions_match": True,
                            "measured_outcomes_absent_from_inputs": True, "proposal": proposal}
    result["all_passed"] = True
    destination = out / "post-run-audit.json"
    with destination.open("xb") as f:
        f.write(canonical(result))
    print(json.dumps({"all_passed": True, "wave_checks": len(result["checks"]),
                      "next_wave_choices_reproduced": True, "inference_reload_passed": True}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
