#!/usr/bin/env python3
"""Read the pinned round-882 packet on xbabe1 and stream DL-Train files only."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import threading
from pathlib import Path, PurePosixPath

SELECTED_SHA256 = "981960ad628f5029f8c54552c7b1cb1d440072e7e9dcffecdea52c6a3493a0f1"
SOURCE_MANIFEST_SHA256 = "99a53569fcaf59f9309e69329c830ae1684adff382da3ae22e0b2645c4ebc4d4"
CPU_FUSED_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
SIGNED_BUNDLE = Path("/home/ubuntu/jope-data/RNK10-t0140-t0250-rnk11/verified-r882")
SIGNED_PACKAGE = Path("/home/ubuntu/jope-data/RNK10-t0140-t0250-rnk11/r882.jopedime")
SIGNED_RECEIPT = Path("/home/ubuntu/jope_plan/signing/receipts/release-r882-20261101T094716Z.json")
RECEIPT_SHA256 = "be1e4a4584d89d5b09b26287c8a58a43391943d03f12616280bdbb87b217f5f0"
SOURCE_ROOT = "flywheel-full-20261030/input"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path.home() / "jope-data/RNK10-M5-lift-t0815/input",
    )
    parser.add_argument("--source-host", default="xbabe1")
    args = parser.parse_args()
    destination = args.destination.resolve()
    if destination.exists() and any(destination.iterdir()):
        raise SystemExit(f"destination must be empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    selected_bytes = subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", args.source_host,
         f"cat /home/ubuntu/jope-data/{SOURCE_ROOT}/selected.json"],
        text=False,
    )
    if hashlib.sha256(selected_bytes).hexdigest() != SELECTED_SHA256:
        raise SystemExit("selected roster pin mismatch")
    selected = json.loads(selected_bytes)
    rows = selected.get("train")
    if not isinstance(rows, list) or len(rows) != 1219:
        raise SystemExit("expected exactly 1,219 DL-Train rows")
    if len({row.get("context_sha256") for row in rows}) != 1219:
        raise SystemExit("duplicate DL-Train context identity")
    families = set()
    members = [
        f"{SOURCE_ROOT}/source_manifest.json",
        f"{SOURCE_ROOT}/cpu_exact_fused.py",
    ]
    train_rows = []
    for row in rows:
        cid = row.get("context_sha256")
        if (
            not isinstance(cid, str)
            or len(cid) != 64
            or any(ch not in "0123456789abcdef" for ch in cid)
            or row.get("dl_split") != "DL-Train"
            or row.get("task") != "regression"
            or row.get("source_split") not in ("train", "inner")
        ):
            raise SystemExit("invalid or non-training roster row")
        if not row.get("family_id"):
            raise SystemExit("missing DL-Train family identity")
        families.add(row["family_id"])
        train_rows.append(row)
        members.extend(
            [f"{SOURCE_ROOT}/pools/{cid}.npz", f"{SOURCE_ROOT}/tables/{cid}.jtt1"]
        )
    if len(families) != 205:
        raise SystemExit(f"expected 205 train families, got {len(families)}")

    # Paths are generated only from fixed roots and validated hexadecimal IDs.
    member_paths = [PurePosixPath(member) for member in members]
    if any(path.is_absolute() or ".." in path.parts for path in member_paths):
        raise SystemExit("unsafe archive member path")
    file_list = ("\n".join(str(path) for path in member_paths) + "\n").encode()
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        args.source_host,
        "tar",
        "--create",
        "--file=-",
        "--directory=/home/ubuntu/jope-data",
        "--files-from=-",
    ]
    source = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert source.stdin is not None and source.stdout is not None
    write_error: list[BaseException] = []

    def send_file_list() -> None:
        try:
            assert source.stdin is not None
            source.stdin.write(file_list)
            source.stdin.close()
        except BaseException as error:
            write_error.append(error)

    sender = threading.Thread(target=send_file_list, daemon=True)
    sender.start()
    expected = {str(path) for path in member_paths}
    with tarfile.open(fileobj=source.stdout, mode="r|") as archive:
        for member in archive:
            relative = PurePosixPath(member.name)
            if member.name not in expected or not member.isfile():
                source.kill()
                raise SystemExit(f"unexpected packet member: {member.name}")
            if relative.parts[:2] != PurePosixPath(SOURCE_ROOT).parts:
                source.kill()
                raise SystemExit(f"train packet member escaped pinned source root: {member.name}")
            target = destination / Path(*relative.parts[2:])
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = archive.extractfile(member)
            if stream is None:
                raise SystemExit(f"missing packet contents: {member.name}")
            with target.open("wb") as output:
                while block := stream.read(1 << 20):
                    output.write(block)
    source.stdout.close()
    sender.join()
    if write_error:
        raise SystemExit(f"could not send train-only file list: {write_error[0]}")
    if source.wait() != 0:
        raise SystemExit("xbabe1 train-only packet stream failed")

    required_hashes = {
        "source_manifest.json": SOURCE_MANIFEST_SHA256,
        "cpu_exact_fused.py": CPU_FUSED_SHA256,
    }
    for name, expected_hash in required_hashes.items():
        if sha256(destination / name) != expected_hash:
            raise SystemExit(f"packet hash mismatch: {name}")

    signed_hashes = {
        "r882.jopedime": "7523af9700e21325a69f9f71b8b60eb4897d46f3948546e4024010b1ce129a60",
        "jope.safetensors": "ff3f9ea5d76fa6c13cd74d40a9effbf7e9c891d98746dc542d851726b9e3138c",
        "dope.safetensors": "0f5e98fa29cf9c45bfaac5a3cf001b13a0de37d7b6e154c582d7ec689b21a84d",
        "manifest.json": "9a83b2225f57ee451b4813790dad3cb4c41249d4f5a14f7c80e2c0e41540891f",
        "release-catalog.json": "3af56a817bfae39b70f0b2f60668fe88fae178650ee596e1ba7a180acba1e0c6",
        "appliance-manifest.json": "4ae903b2179f94753a0efb5924b580e8e1e8cb197eee04c71a391a3d7ff3e029",
        "appliance-signature.bin": "2fa70bfd9c83b06c24d3c96cea14c2684ca068346e0a5a5217dc92a08402d823",
    }
    signed_dir = destination / "signed-r882"
    signed_dir.mkdir()
    for name, expected_hash in signed_hashes.items():
        source_path = SIGNED_PACKAGE if name == "r882.jopedime" else SIGNED_BUNDLE / name
        if sha256(source_path) != expected_hash:
            raise SystemExit(f"signed R882 input mismatch: {name}")
        shutil.copyfile(source_path, signed_dir / name)
    receipt = json.loads(SIGNED_RECEIPT.read_text())
    if (
        receipt.get("kind") != "release"
        or receipt.get("label") != "r882"
        or receipt.get("verify") != "PASS"
        or receipt.get("message_sha256") != "45e21d576cec91b5171bb8e279ef24fef761456fa916e4de1ed8ed2b8bd6eba5"
    ):
        raise SystemExit("watcher-signed R882 receipt identity mismatch")
    if sha256(SIGNED_RECEIPT) != RECEIPT_SHA256:
        raise SystemExit("watcher-signed R882 receipt hash mismatch")
    shutil.copyfile(SIGNED_RECEIPT, destination / "signed-r882-receipt.json")
    signature_material = Path("/home/ubuntu/jope-data/RNK10-t0140-t0200-rnk11/stage")
    for name, expected_hash in {
        "signing-message.bin": "45e21d576cec91b5171bb8e279ef24fef761456fa916e4de1ed8ed2b8bd6eba5",
        "manifest.sig": "d35ac4dcdeba1aae42d9ad4469f4681c83ef57df969939ad0101215524a1a780",
    }.items():
        source_path = signature_material / name
        if sha256(source_path) != expected_hash:
            raise SystemExit(f"signed R882 signature material mismatch: {name}")
        shutil.copyfile(source_path, signed_dir / name)
    manifest = {
        "schema": "jope.rnk10-m5-train-packet/v1",
        "source_selected_sha256": SELECTED_SHA256,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "train_contexts": len(train_rows),
        "train_families": len(families),
        "task": "regression",
        "split": "DL-Train",
        "payload_scope": "1219 train pools and train-only JTT1 tables plus pinned signed R882 bundle; no DL-Val/sealed/outer payloads",
        "rows": train_rows,
    }
    (destination / "selected_train.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(json.dumps({"staged": len(train_rows), "families": len(families), "destination": str(destination)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
