"""Verify the shard catalogue and expose compact contiguous group ranges."""

import hashlib
import json
import struct

from resume_expired_cycle import read, require, sha

SHARD_FIELDS = (
    "context",
    "path",
    "sha256",
    "records",
    "max_length",
    "records_sha256",
    "labels_sha256",
    "legal_sets",
)


def load(root, corpus, progress=None):
    root, corpus = root.resolve(), corpus.resolve()
    require(corpus.is_relative_to(root), "corpus outside workspace")
    index = read(corpus / "index.json")
    require(
        index["schema"] == "parallel.teaching_shards.v1" and index["record_bytes"] == 532,
        "unknown shard layout",
    )
    require(
        read(corpus / "provenance.json")["index_sha256"] == sha(corpus / "index.json"), "corpus index changed"
    )
    require(sha(corpus / "vocabulary.json") == index["vocabulary_sha256"], "corpus vocabulary changed")
    require(sha(corpus / "groups.bin") == index["groups_sha256"], "group offsets changed")
    shards = [{key: s[key] for key in SHARD_FIELDS} for s in index["shards"]]
    require(
        hashlib.sha256(json.dumps(shards, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
        == index["records_sha256"],
        "shard catalogue identity changed",
    )
    offsets = (corpus / "groups.bin").read_bytes()
    require(len(offsets) == 16 * len(shards), "group count changed")
    count, previous, groups, segments = 0, -1, [], []
    for number, shard in enumerate(shards):
        context = shard["context"]
        size = shard["records"]
        require(
            type(context) is int and previous < context < len(index["contexts"]) and size >= 8,
            "invalid shard context or size",
        )
        previous = context
        directory = (root / shard["path"]).resolve()
        require(directory.is_relative_to(root), "shard outside workspace")
        require(sha(directory / "shard.json") == shard["sha256"], "shard manifest changed")
        require(read(directory / "shard.json") == dict(shard, sha256=""), "shard binding changed")
        require(
            sha(directory / "records.bin") == shard["records_sha256"]
            and sha(directory / "labels.jsonl") == shard["labels_sha256"],
            "shard payload changed",
        )
        require(
            struct.unpack_from("<QQ", offsets, number * 16) == (count, size)
            and (directory / "records.bin").stat().st_size == size * 532,
            "shard dimensions changed",
        )
        row = index["contexts"][context]
        require(row["partition"] in ("meta_train", "meta_validation"), "sealed supervision")
        groups.append(
            dict(
                context=context,
                family=row["family"],
                partition=row["partition"],
                records=range(count, count + size),
                max_length=shard["max_length"],
            )
        )
        segments.append(dict(directory=directory, offset=count, records=size, group=number))
        count += size
        if progress is not None:
            progress(number + 1, count)
    require(count == index["records"], "manifest record count changed")
    index["groups"] = groups
    return index, segments


def workspace(corpus):
    return next(parent.parent for parent in corpus.resolve().parents if parent.name == "runs")
