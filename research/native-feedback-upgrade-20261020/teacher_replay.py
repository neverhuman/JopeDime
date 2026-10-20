"""Reproduce the original 263-context teacher-CE KPI without training.

The value adapter leaves both native backbones unchanged; ranking improvements
must not be represented as an improvement in this different teacher-loss KPI.
"""
import argparse
import io
import json
from pathlib import Path
import subprocess
import tarfile

import numpy as np
import torch

from dltrain import model as M, dsl_tokenizer as T, automaton as A
from feedback import canonical
from prepare import PINS, PARAMS, checked, load_native, forward


def main(root):
    M.configure_determinism(2)
    for name in ["meta.json", "hwm_programs.json", "preprocessing.json", "grammar.json", "blocks_kumo_tabular_s.npy"]:
        checked(root / name, PINS[name])
    meta = json.loads((root / "meta.json").read_text())
    preprocessing = json.loads((root / "preprocessing.json").read_text())
    train = set(preprocessing["fit_context_ids"])
    validation = sorted(set(meta["cids"]) - train)
    assert len(validation) == 263
    lookup = {cid: i for i, cid in enumerate(meta["cids"])}
    programs = {p["cid"]: p for p in json.loads((root / "hwm_programs.json").read_text())["programs"]}
    vocab = T.Vocab.from_names({})
    doc = json.loads((root / "grammar.json").read_text())
    grammar = A.Grammar(doc["grammar"], doc["signatures"], vocab)
    sequences = {cid: T.tokenize(programs[cid]["dsl"], vocab) for cid in validation}
    mean, scale = np.array(preprocessing["mean"]), np.array(preprocessing["scale"])
    raw = np.load(root / "blocks_kumo_tabular_s.npy", mmap_mode="r")
    checked(root / "prior-comparison.tar.zst", PINS["prior-comparison.tar.zst"])
    archive = tarfile.open(fileobj=io.BytesIO(subprocess.check_output(["zstd", "-dc", str(root / "prior-comparison.tar.zst")])))
    reference = json.loads(archive.extractfile("P01_SAME_INPUT_PARENT_REPLAY_QUALIFICATION_20261019.json").read())
    expected = {"native17773": reference["arms"]["v1-rolling-000017773"]["full_263_validation"]["per_context_mean_ce"],
                "native429": reference["arms"]["cold-epoch1-000429-20261019T1303Z"]["full_263_validation"]["per_context_mean_ce"]}
    result = {"schema": "jope.native-teacher-kpi-replay/v1", "contexts": 263,
              "population": "original reused validation, never adapter training", "models": {},
              "adapter_changes_teacher_loss": False, "sealed_confirmation": False}
    with torch.inference_mode():
        for name in PARAMS:
            net = load_native(root, name)
            losses = {}
            for cid in validation:
                seq = sequences[cid]
                z = np.clip((raw[lookup[cid]].astype(np.float64) - mean)/scale, -10, 10).astype(np.float32)
                out = forward(net, [seq], z)
                table, ids = grammar.compile_masks([seq], A.Mode.SYNTAX)
                legal = torch.tensor(table[ids[0][1:]])
                logits = out.logits[0].masked_fill(~legal, -float("inf"))
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor(seq[1:])).item()
                losses[cid] = loss
            actual = float(np.mean(list(losses.values())))
            result["models"][name] = {"teacher_ce": actual, "previous": expected[name],
                                      "absolute_difference": abs(actual-expected[name]),
                                      "matched_within_1e-5": abs(actual-expected[name]) < 1e-5,
                                      "parameter_sha256": M.parameter_sha256(net), "per_context": losses}
    result["all_matched"] = all(v["matched_within_1e-5"] for v in result["models"].values())
    result["correction"] = "Earlier teacher-kpi-replay.json compared CE against composite-loss references. Original failure retained; this receipt uses the archive's explicit per_context_mean_ce field."
    with (root / "experiment/teacher-ce-replay-corrected.json").open("xb") as f:
        f.write(canonical(result))
    print(json.dumps({"all_matched": result["all_matched"], "models": {k: {x: y for x, y in v.items() if x != "per_context"} for k, v in result["models"].items()}}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    main(p.parse_args().root)
