"""Recover the required study from original user messages, without funding work."""

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path

from scientific_metadata import canonical, immutable

REPOSITORY = "/home/ubuntu/jope-dime"
ORIGINALS = {
    "seed-selection": {
        "session_id": "01a08c89-86c6-7143-a069-191334939097",
        "timestamp": "2026-09-10T18:17:00.191Z",
        "message_sha256": "2ccb7c530a3514bb551361de823fdddbe61ed0ddcea532d0c38044c26b9c7aea",
    },
    "condition-selection": {
        "session_id": "01a08d64-ccfb-7113-b235-14d0fa30056f",
        "timestamp": "2026-09-10T22:16:33.831Z",
        "message_sha256": "a5637d74a18e52018bd6b554834dad32489c0b7f8a9d0bfc84ad801323e0d6e4",
    },
    "governing-constraints": {
        "session_id": "01a091e5-f87b-7880-877b-dc0bafa4f3cf",
        "timestamp": "2026-09-11T19:16:05.116Z",
        "message_sha256": "60b50a7106d2a5af7d4471cf78d72045be650198b52c2a9e4b949827e5de43bd",
    },
}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def message_text(record):
    require(record.get("type") == "response_item", "original response item required")
    payload = record.get("payload", {})
    require(
        payload.get("type") == "message" and payload.get("role") == "user",
        "only an original user message can select this study",
    )
    content = payload.get("content")
    require(isinstance(content, list) and content, "original user content unavailable")
    require(
        all(
            isinstance(item, dict) and item.get("type") == "input_text" and isinstance(item.get("text"), str)
            for item in content
        ),
        "unsupported original user message content",
    )
    return "\n".join(item["text"] for item in content)


def signature(value):
    return {
        key: getattr(value, key)
        for key in ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    }


def custody_json(path, limit):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "custody symlink rejected")
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode), "regular custody metadata required")
        data = stream.read(limit + 1)
        require(
            len(data) <= limit and signature(before) == signature(os.fstat(stream.fileno())),
            "custody metadata changed or exceeds bound",
        )

    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, "duplicate custody metadata key")
            result[key] = value
        return result

    value = json.loads(data, object_pairs_hook=unique)
    canonical(value)
    return value, hashlib.sha256(data).hexdigest()


def publish_bytes(path, data):
    path = Path(path)
    require(not any(p.is_symlink() for p in (path, *path.parents)), "custody symlink rejected")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
    except FileExistsError:
        require(path.is_file() and path.read_bytes() == data, "original custody conflict")
        return
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def capture_user_message(source, output, expected_message_sha256, line=9):
    """Copy only the specified original user record and session header.

    Other records are streamed into a source digest and are never decoded,
    retained or reported. Capture proves byte custody, not a budget balance.
    """
    source, output = Path(source), Path(output)
    require(type(line) is int and 1 < line <= 1000, "bounded original line required")
    require(not any(p.is_symlink() for p in (source, *source.parents)), "session symlink rejected")
    fd = os.open(source, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        before = os.fstat(stream.fileno())
        require(
            stat.S_ISREG(before.st_mode) and before.st_size <= 64 * 1024 * 1024,
            "bounded regular original session required",
        )
        hasher, bytes_read = hashlib.sha256(), 0
        header_bytes = record_bytes = None
        for index in range(1, line + 1):
            raw = stream.readline(1024 * 1024 + 1)
            require(
                raw and len(raw) <= 1024 * 1024 and raw.endswith(b"\n"),
                "original line missing or exceeds custody bound",
            )
            bytes_read += len(raw)
            require(bytes_read <= 64 * 1024 * 1024, "session grew beyond custody bound")
            hasher.update(raw)
            if index == 1:
                header_bytes = raw
            if index == line:
                record_bytes = raw
        while data := stream.read(1024 * 1024):
            bytes_read += len(data)
            require(bytes_read <= 64 * 1024 * 1024, "session grew beyond custody bound")
            hasher.update(data)
        require(
            signature(before) == signature(os.fstat(stream.fileno())) == signature(source.stat()),
            "original session changed during capture",
        )
    header, record = json.loads(header_bytes), json.loads(record_bytes)
    require(
        header.get("type") == "session_meta"
        and header["payload"].get("cwd") == REPOSITORY
        and header["payload"].get("source") == "cli",
        "unrelated or delegated session rejected",
    )
    message = message_text(record).encode()
    require(
        hashlib.sha256(message).hexdigest() == expected_message_sha256,
        "original user message differs from inspected source",
    )
    copies = {
        "session-header.json": header_bytes,
        "user-record.json": record_bytes,
        "user-message.txt": message,
    }
    for name, data in copies.items():
        publish_bytes(output / name, data)
    receipt = dict(
        schema="parallel.original_user_message_custody.v1",
        source_path=str(source),
        source_line=line,
        source_sha256=hasher.hexdigest(),
        source_stat=signature(before),
        session_id=header["payload"]["id"],
        original_timestamp=record["timestamp"],
        message_sha256=expected_message_sha256,
        files={
            name: dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            for name, data in copies.items()
        },
        scope="one original repository user message; no other session content copied",
        funding_balance_verified=False,
        numerical_work_authorized=False,
    )
    immutable(output / "provenance.json", receipt)
    return receipt


def verify_original(directory, expected):
    """Reopen copied original bytes; a plausible caller-written receipt is insufficient."""
    directory = Path(directory)
    receipt, receipt_sha = custody_json(directory / "provenance.json", 128 * 1024)
    require(
        receipt.get("schema") == "parallel.original_user_message_custody.v1",
        "original message custody unavailable",
    )
    header, header_sha = custody_json(directory / "session-header.json", 128 * 1024)
    record, record_sha = custody_json(directory / "user-record.json", 1024 * 1024)
    require(
        header.get("type") == "session_meta"
        and header["payload"].get("cwd") == REPOSITORY
        and header["payload"].get("source") == "cli"
        and header["payload"].get("id") == expected["session_id"],
        "original session identity changed",
    )
    text = message_text(record)
    require(
        record.get("timestamp") == expected["timestamp"]
        and hashlib.sha256(text.encode()).hexdigest() == expected["message_sha256"],
        "original selected instruction changed",
    )
    require(
        receipt["session_id"] == expected["session_id"]
        and receipt["original_timestamp"] == expected["timestamp"]
        and receipt["message_sha256"] == expected["message_sha256"]
        and receipt["files"]["session-header.json"]["sha256"] == header_sha
        and receipt["files"]["user-record.json"]["sha256"] == record_sha,
        "original custody binding changed",
    )
    return text, dict(
        path=str(directory / "provenance.json"), sha256=receipt_sha, user_record_sha256=record_sha, **expected
    )


def recover_selection(custody_root):
    originals = {
        name: verify_original(Path(custody_root) / name, expected) for name, expected in ORIGINALS.items()
    }
    seed_text = originals["seed-selection"][0]
    seed_line = next((line for line in seed_text.splitlines() if line.startswith("- Seeds `")), "")
    seeds = [int(value) for value in re.findall(r"`(\d+)`", seed_line)]
    require(seeds == [20260909, 20260910, 20260911], "original three training seeds unavailable")
    condition_text = originals["condition-selection"][0]
    table = {}
    for line in condition_text.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) == 5 and cells[1] in ("A", "B", "C", "D"):
            require(cells[1] not in table, "ambiguous original condition table")
            table[cells[1]] = cells[2:4]
    require(
        table
        == {
            "A": ["Strictly admitted narrow support", "Family-balanced broad"],
            "B": ["Strictly admitted narrow support", "50/50 fresh/broad"],
            "C": ["All eligible broader support at freeze", "Family-balanced broad"],
            "D": ["All eligible broader support at freeze", "50/50 fresh/broad"],
        },
        "original support/replay conditions unavailable",
    )
    latest = originals["governing-constraints"][0]
    require(
        all(
            value in latest
            for value in (
                "1,000 new committed steps",
                "300-second task caps",
                "Do not add a reset ablation",
                "No eligible new support means an unavailable support condition",
            )
        ),
        "governing original study constraints changed",
    )
    return dict(
        schema="parallel.required_learning_selection.v1",
        original_selection_recovered=True,
        seeds=seeds,
        conditions=[
            dict(
                identity=name,
                support="narrow" if name in ("A", "B") else "broader",
                replay="broad" if name in ("A", "C") else "fresh_balanced",
                original_support=table[name][0],
                original_replay=table[name][1],
            )
            for name in ("A", "B", "C", "D")
        ],
        constraints=dict(
            global_batch=2048,
            new_committed_steps=1000,
            training_ceiling_seconds=300,
            optimizer="carry",
            optimizer_reset_ablation_allowed=False,
            independent_branches=True,
            unavailable_if_no_eligible_new_support=True,
        ),
        original_instructions={name: value[1] for name, value in originals.items()},
        support_membership_verified=False,
        continuation_qualified=False,
        funding_balance_verified=False,
        launch_allowed=False,
        scope="recovered original selection; no support inventory, numerical qualification or grant created",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--source", type=Path, required=True)
    capture.add_argument("--output", type=Path, required=True)
    capture.add_argument("--message-sha256", required=True)
    capture.add_argument("--line", type=int, default=9)
    selection = sub.add_parser("selection")
    selection.add_argument("--custody-root", type=Path, required=True)
    selection.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        value = capture_user_message(args.source, args.output, args.message_sha256, args.line)
    else:
        value = recover_selection(args.custody_root)
        immutable(args.output, value)
    print(canonical(value).decode())


if __name__ == "__main__":
    main()
