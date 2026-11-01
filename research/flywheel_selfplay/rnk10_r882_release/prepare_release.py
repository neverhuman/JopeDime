#!/usr/bin/env python3
"""Prepare the owner-pinned r882 adapter for the existing release signer.

This is an explicit research export. Production never opens a PyTorch file;
the resulting bytes still need the established release signature and loader.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess

import torch
from safetensors import safe_open

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PARENT = HERE.parent / "export_fused_bundle.py"
CHECKPOINT_SHA = "0084479aa0703667a0f3ccfe2beda60741c204795ec57b7c08542401ec6cdb64"
POINTER_SHA = "30853cad78d860870338316dfb5a4e869e46dc5c22d42fe4ab9d5f24f31c373a"
EVALUATION_IDENTITY = "e79a7715c1531ccb8c4055ab96971101cd4034ec3a71d83bca6d25b0e604ca83"
CATALOG_SHA = "d5a4b21e6167dbd637e940acd3a11d6357c147bb021ea2a8929f06c71951e34f"
ENVELOPE_SHA = "c39e5e05407969e322ced4aac0001bf88240969a4dc79483fe38a213287129d4"
CARRIER_SOURCE = "de5719a566fdd37926d4b5b9786ddd6f25343949"
DEFAULT_CHECKPOINT = Path(
    "/home/ubuntu/jope-data/RNK10-t2115-pin-20261101/round-882.pt"
)
BASE = Path("/home/ubuntu/jope-data/rnk12-s1-r782/verified-assets")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pinned(path: Path, expected: str, size: int | None = None) -> bytes:
    if any(p.is_symlink() for p in (path, *path.parents)) or not path.is_file():
        raise ValueError(f"regular custody input required: {path}")
    length = path.stat().st_size
    if length > 64 * 1024 * 1024 or (size is not None and length != size):
        raise ValueError(f"custody length mismatch: {path}")
    data = path.read_bytes()
    if digest(data) != expected or (size is not None and len(data) != size):
        raise ValueError(f"custody hash/length mismatch: {path}")
    return data


def exporter():
    spec = importlib.util.spec_from_file_location("rnk10_frozen_exporter", PARENT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(checkpoint: Path, output: Path, packer: Path) -> dict:
    if checkpoint != DEFAULT_CHECKPOINT:
        raise ValueError("only the D2 custody checkpoint path is authorized")
    if output.parent != Path("/home/ubuntu/jope-data") or not output.name.startswith(
        "RNK10-t0140-"
    ):
        raise ValueError("output must be a new RNK10-t0140 custody directory")
    if any(p.is_symlink() for p in output.parents):
        raise ValueError("output parents must not be symlinks")
    if output.exists() or output.is_symlink():
        raise ValueError("release output must be new")
    checkpoint_data = pinned(checkpoint, CHECKPOINT_SHA, 7_493_476)
    pointer_data = pinned(checkpoint.with_name("best_dev_checkpoint.json"), POINTER_SHA)
    pointer = json.loads(pointer_data)
    if (
        pointer.get("best_round") != 882
        or pointer.get("evaluation_identity") != EVALUATION_IDENTITY
    ):
        raise ValueError("r882 pointer identity mismatch")
    fused = exporter()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        state.get("round") != 882
        or state.get("dope_sha256") != fused.FROZEN_DOPE_COMPOSITE
    ):
        raise ValueError("r882 checkpoint lineage mismatch")
    state = state["state_dict"]
    fused.validate_state(state)
    catalog_data = pinned(BASE / "release-catalog.json", CATALOG_SHA, 240_292)
    base_catalog = json.loads(catalog_data)
    envelope = json.loads(pinned(BASE / "appliance-manifest.json", ENVELOPE_SHA))
    entries = {row["id"]: row for row in envelope["entries"]}

    def base_asset(name):
        row = entries[base_catalog["assets"][name]["signed_asset_id"]]
        return pinned(BASE / name, row["sha256"], row["bytes"])

    # The independent, committed exporter inventory supplies the tensor schema.
    # Only its 47 non-anchor adapter tensors may differ from frozen r782.
    trainable = {
        "adapter_10tok." + name
        for name in fused.ADAPTER_SHAPES
        if name not in {"prog_mean", "prog_std"}
    }
    unchanged = 0
    for filename in ("dope.safetensors", "jope.safetensors"):
        base_asset(filename)
        with safe_open(BASE / filename, framework="pt") as source:
            for name in source.keys():
                if name not in trainable:
                    if not torch.equal(source.get_tensor(name), state[name]):
                        raise ValueError(f"frozen non-adapter state changed: {name}")
                    unchanged += 1
    if len(trainable) != 47 or sum(state[name].numel() for name in trainable) != 29_211:
        raise ValueError("r882 adapter inventory differs")

    payloads = {
        "dope.safetensors": fused.safetensors_bytes(
            {k: v for k, v in state.items() if k.startswith("dope_model.")}
        ),
        "jope.safetensors": fused.safetensors_bytes(
            {k: v for k, v in state.items() if not k.startswith("dope_model.")}
        ),
        "featurizer.json": base_asset("featurizer.json"),
        "bank.jsonl.zst": base_asset("bank.jsonl.zst"),
    }
    if digest(payloads["dope.safetensors"]) != base_catalog["dope_weights_sha256"]:
        raise ValueError("frozen DOPE serialized identity changed")
    manifest = json.loads(base_asset("manifest.json"))
    manifest.pop("bundle_digest")
    manifest["jope_weights_sha256"] = digest(payloads["jope.safetensors"])
    manifest["qualified"] = []
    manifest["asset_policy"] = (
        "internal-research-only; R1 explicit opt-in; not quality-qualified"
    )
    sources = manifest["reference_source_sha256"]
    for name in (
        "checkpoint.pt",
        "round-782.pt",
        "custody/CHECKPOINT.json",
        "custody/MANIFEST.json",
    ):
        sources.pop(name, None)
    sources.update({"round-882.pt": CHECKPOINT_SHA, "r882-pointer.json": POINTER_SHA})
    manifest["bundle_digest"] = fused.compute_bundle_digest(manifest, payloads)
    payloads["manifest.json"] = fused.canonical_json(manifest) + b"\n"

    # Retain only captured TRAIN inputs. Old r782 predicted outputs would be
    # misleading goldens for the changed adapter, so they are not selected.
    names = sorted(
        name
        for name in base_catalog["assets"]
        if name.startswith("golden/contexts/") and name.endswith(".safetensors")
    )
    if len(names) != 50:
        raise ValueError("expected fifty pinned TRAIN context inputs")
    payloads.update({name: base_asset(name) for name in names})
    assets = {
        name: {
            "signed_asset_id": "model/r882/" + name,
            "tensors": base_catalog["assets"][name].get("tensors"),
        }
        for name in sorted(payloads)
    }
    catalog = dict(
        schema="jope.r882-release-catalog/v1",
        bundle_digest=manifest["bundle_digest"],
        dope_logical_composite=fused.FROZEN_DOPE_COMPOSITE,
        dope_weights_sha256=manifest["dope_weights_sha256"],
        assets=assets,
        source_checkpoint_sha256=CHECKPOINT_SHA,
        provenance="R1 t0140: frozen architecture inventory; exact r882 custody adapter; TRAIN inputs only; no quality declaration",
    )
    payloads["release-catalog.json"] = fused.canonical_json(catalog) + b"\n"
    output.mkdir(parents=True)
    model = output / "models/bundles/r882"
    model.mkdir(parents=True)
    for name, data in payloads.items():
        path = model / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    context = output / "pack-context"
    profiles = context / "deploy/appliance/profiles"
    profiles.mkdir(parents=True)
    (context / "descriptors").mkdir()
    descriptor = {
        "schema": "jope.model-bundle/v2",
        "bundle_id": "r882",
        "files": [
            dict(
                path=name,
                sha256=digest(data),
                bytes=len(data),
                kind="weights" if name.endswith(".safetensors") else "metadata",
                license=dict(
                    spdx_expression="LicenseRef-Jain-Internal-Model",
                    redistribution="internal",
                ),
            )
            for name, data in sorted(payloads.items())
        ],
    }
    (context / "descriptors/r882.json").write_bytes(
        fused.canonical_json(descriptor) + b"\n"
    )
    profile = "\n".join(
        [
            'schema = "jope.appliance-profile/v1"',
            'name = "r882-authenticated-internal"',
            'description = "R1 internal model authentication; no release/default quality qualification"',
            'scope = "internal"',
            'redistribution = ["internal"]',
            "methods = []",
            'bundles = "descriptors"',
            "max_payload_mib = 64",
            "[codec]",
            'name = "shuffle+zstd"',
            "level = 3",
            "frame_mib = 16",
            "[[component]]",
            'id = "r882"',
            'bundle = "r882"',
            'role = "model"',
            'backends = ["cpu"]',
            "files = " + json.dumps(sorted(payloads)),
            "",
        ]
    )
    (profiles / "r882-authenticated-internal.toml").write_text(profile)
    for name in ("LICENSE", "Cargo.lock"):
        data = subprocess.check_output(
            ["git", "show", CARRIER_SOURCE + ":" + name], cwd=REPO
        )
        (context / name).write_bytes(data)
    # This carrier and its notices are the already built S1 internal package
    # inputs. They do not claim to contain the new learned engine code.
    shutil.copyfile(BASE.parent / "code", output / "code")
    shutil.copyfile(BASE.parent / "THIRD_PARTY.json", output / "THIRD_PARTY.json")
    commands = [
        [str(packer), "--repo", str(context), "lock"],
        [
            str(packer),
            "--repo",
            str(context),
            "pack",
            "--profile",
            str(profiles / "r882-authenticated-internal.toml"),
            "--code",
            str(output / "code"),
            "--models",
            str(output / "models"),
            "--notices",
            str(output / "THIRD_PARTY.json"),
            "--source-sha",
            CARRIER_SOURCE,
            "--stage",
            str(output / "stage"),
            "--receipt",
            str(output / "pack-receipt.json"),
        ],
    ]
    for command in commands:
        subprocess.run(command, check=True, cwd=REPO)
    receipt = dict(
        schema="jope.rnk10.r882-export/v1",
        checkpoint_sha256=digest(checkpoint_data),
        pointer_sha256=POINTER_SHA,
        frozen_dope_composite=fused.FROZEN_DOPE_COMPOSITE,
        frozen_nonadapter_tensors=unchanged,
        adapter_tensors=47,
        adapter_parameters=29_211,
        bundle_digest=manifest["bundle_digest"],
        catalog_sha256=digest(payloads["release-catalog.json"]),
        catalog_bytes=len(payloads["release-catalog.json"]),
        jope_weights_sha256=manifest["jope_weights_sha256"],
        signing_message_sha256=digest(
            (output / "stage/signing-message.bin").read_bytes()
        ),
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        carrier_source_commit=CARRIER_SOURCE,
        carrier_is_new_engine=False,
        packer_sha256=digest(packer.read_bytes()),
        commands=commands,
        quality_qualified=False,
        production_loader_requires_release_signature=True,
    )
    (output / "export-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-import", action="store_true", required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--packer", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.checkpoint, args.output, args.packer), indent=2))


if __name__ == "__main__":
    main()
