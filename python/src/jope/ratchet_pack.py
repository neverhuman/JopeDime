"""Zero-copy reader for Rust-authored ratchet training schedule packs.

Sampling, target supersession, and game ordering are already committed by Rust.
This module only validates the immutable artifact and exposes NumPy views for
fast pinned-memory collation; it never selects labels or computes performance.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np


PACK_MAGIC = b"JRPACK\x00\x01"
HEADER_BYTES = 256
GROUP_BYTES = 128
ROW_BYTES = 64

GROUP_DTYPE = np.dtype(
    {
        "names": [
            "dataset_hash_bytes",
            "catalog_index",
            "row_start",
            "row_count",
            "max_genome_tokens",
            "encoding_dimensions",
            "encoding_byte_offset",
            "target_revision",
            "conditional_draw_probability",
            "source",
            "block_sha256_bytes",
            "source_family_sha256_bytes",
            "microbatch_index",
            "group_position",
        ],
        "formats": [
            "V8",
            "<u4",
            "<u4",
            "<u4",
            "<u4",
            "<u4",
            "<u8",
            "<u8",
            "<f8",
            "u1",
            "V32",
            "V32",
            "<u2",
            "<u2",
        ],
        "offsets": [0, 8, 12, 16, 20, 24, 28, 36, 44, 52, 56, 88, 120, 122],
        "itemsize": GROUP_BYTES,
    }
)

ROW_DTYPE = np.dtype(
    {
        "names": [
            "group_index",
            "label_kind",
            "flags",
            "genome_offset",
            "genome_length",
            "semantic_sha256_bytes",
            "tier1_robust_cv",
            "tier2_test_rmse",
            "auxiliary_cv_target",
            "auxiliary_test_target",
        ],
        "formats": [
            "<u2",
            "u1",
            "u1",
            "<u8",
            "<u4",
            "V32",
            "<f4",
            "<f4",
            "<f4",
            "<f4",
        ],
        "offsets": [0, 2, 3, 4, 12, 16, 48, 52, 56, 60],
        "itemsize": ROW_BYTES,
    }
)


@dataclass(frozen=True)
class RatchetSchedulePack:
    manifest: dict
    binary: np.memmap
    groups: np.ndarray
    rows: np.ndarray
    genome_blob: memoryview

    def microbatches(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield committed game groups and their row views without copying."""

        microbatch_ids = self.groups["microbatch_index"]
        for microbatch_index in sorted(int(value) for value in np.unique(microbatch_ids)):
            group_indices = np.flatnonzero(microbatch_ids == microbatch_index)
            first_group = int(group_indices[0])
            last_group = int(group_indices[-1]) + 1
            if not np.array_equal(group_indices, np.arange(first_group, last_group)):
                raise ValueError("schedule-pack microbatch groups are not contiguous")
            group_view = self.groups[first_group:last_group]
            if not np.array_equal(
                group_view["group_position"], np.arange(len(group_view), dtype=np.uint16)
            ):
                raise ValueError("schedule-pack microbatch group order is not canonical")
            first_row = int(group_view[0]["row_start"])
            last_row = int(group_view[-1]["row_start"] + group_view[-1]["row_count"])
            if sum(int(group["row_count"]) for group in group_view) != last_row - first_row:
                raise ValueError("schedule-pack microbatch rows are not contiguous")
            yield group_view, self.rows[first_row:last_row]

    def genome(self, row_index: int) -> str:
        row = self.rows[row_index]
        start = int(row["genome_offset"])
        end = start + int(row["genome_length"])
        if start < 0 or end > len(self.genome_blob):
            raise ValueError("schedule-pack genome slice is out of bounds")
        return bytes(self.genome_blob[start:end]).decode("utf-8")


@dataclass(frozen=True)
class RatchetEncodingCatalog:
    catalog_sha256: str
    data_sha256: str
    rows: np.memmap
    dataset_dimensions: int
    target_dimensions: int


@dataclass(frozen=True)
class RatchetTrainingMicrobatch:
    groups: np.ndarray
    labels: np.ndarray
    dataset_encodings: np.ndarray
    target_encodings: np.ndarray


@dataclass(frozen=True)
class RatchetPinnedHostMicrobatch:
    """Views into one persistent pinned-host ring slot."""

    groups: np.ndarray
    labels: np.ndarray
    contexts: Any
    dataset_encodings: Any
    target_encodings: Any
    group_indices: Any
    label_kinds: Any
    flags: Any
    targets: Any
    slot: int
    generation: int


@dataclass(frozen=True)
class RatchetDeviceMicrobatch:
    dataset_encodings: Any
    target_encodings: Any
    group_indices: Any
    label_kinds: Any
    flags: Any
    targets: Any


class PinnedTrainingBatchRing:
    """Reusable double-buffered collation for genuinely asynchronous H2D.

    The permanent catalog remains mmap-backed. Selected context/label columns
    are copied into persistent ``torch.empty(..., pin_memory=True)`` slots;
    no per-step pinned allocation occurs. A CUDA event protects each slot until
    its nonblocking transfer has consumed the host bytes.
    """

    def __init__(
        self,
        pack: RatchetSchedulePack,
        catalog: RatchetEncodingCatalog,
        *,
        depth: int = 2,
        require_pinned: bool = True,
    ) -> None:
        import torch

        if depth < 2:
            raise ValueError("pinned training ring requires at least two slots")
        if pack.manifest["encoding_catalog_sha256"] != catalog.catalog_sha256:
            raise ValueError("schedule pack and encoding catalog authorities differ")
        if require_pinned and not torch.cuda.is_available():
            raise RuntimeError("production pinned collation requires CUDA")
        batches = list(pack.microbatches())
        max_groups = max(len(groups) for groups, _ in batches)
        max_rows = max(len(labels) for _, labels in batches)
        dimensions = catalog.dataset_dimensions + catalog.target_dimensions
        self.pack = pack
        self.catalog = catalog
        self.depth = depth
        self._torch = torch
        self._cursor = 0
        self._events: list[Any | None] = [None] * depth
        self._states = ["free"] * depth
        self._generations = [0] * depth
        self._copy_stream: Any | None = None
        self._copy_device: str | None = None
        pin = bool(require_pinned)
        self._contexts = [
            torch.empty((max_groups, dimensions), dtype=torch.float32, pin_memory=pin)
            for _ in range(depth)
        ]
        self._group_indices = [
            torch.empty(max_rows, dtype=torch.int64, pin_memory=pin)
            for _ in range(depth)
        ]
        self._label_kinds = [
            torch.empty(max_rows, dtype=torch.uint8, pin_memory=pin)
            for _ in range(depth)
        ]
        self._flags = [
            torch.empty(max_rows, dtype=torch.uint8, pin_memory=pin)
            for _ in range(depth)
        ]
        self._targets = [
            torch.empty((max_rows, 4), dtype=torch.float32, pin_memory=pin)
            for _ in range(depth)
        ]

    def stage(
        self, groups: np.ndarray, labels: np.ndarray
    ) -> RatchetPinnedHostMicrobatch:
        slot = self._cursor
        self._cursor = (self._cursor + 1) % self.depth
        if self._states[slot] == "staged":
            raise RuntimeError(
                "pinned ring slot was staged but never transferred/released; "
                "consume batches as an iterator instead of retaining host views"
            )
        event = self._events[slot] if self._states[slot] == "in_flight" else None
        if event is not None:
            event.synchronize()
            self._events[slot] = None
            self._states[slot] = "free"
        if self._states[slot] != "free":
            raise RuntimeError("pinned ring slot has an invalid lifecycle state")
        self._generations[slot] += 1
        generation = self._generations[slot]
        indices = groups["catalog_index"].astype(np.int64, copy=False)
        dimensions = self.catalog.dataset_dimensions + self.catalog.target_dimensions
        if np.any(indices >= len(self.catalog.rows)):
            raise ValueError("schedule pack references an unknown encoding catalog row")
        if np.any(groups["encoding_dimensions"] != dimensions):
            raise ValueError("schedule pack encoding dimensions differ from catalog")
        expected_offsets = indices.astype(np.uint64) * np.uint64(dimensions * 4)
        if np.any(groups["encoding_byte_offset"] != expected_offsets):
            raise ValueError("schedule pack encoding offsets differ from fixed catalog layout")
        group_count = len(groups)
        row_count = len(labels)
        contexts = self._contexts[slot][:group_count]
        # Write directly from the mmap into the persistent pinned destination;
        # fancy indexing would allocate an intermediate pageable NumPy array.
        np.take(self.catalog.rows, indices, axis=0, out=contexts.numpy())
        group_indices = self._group_indices[slot][:row_count]
        label_kinds = self._label_kinds[slot][:row_count]
        flags = self._flags[slot][:row_count]
        targets = self._targets[slot][:row_count]
        np.copyto(group_indices.numpy(), labels["group_index"], casting="safe")
        np.copyto(label_kinds.numpy(), labels["label_kind"], casting="no")
        np.copyto(flags.numpy(), labels["flags"], casting="no")
        target_view = targets.numpy()
        for column, name in enumerate(
            (
                "tier1_robust_cv",
                "tier2_test_rmse",
                "auxiliary_cv_target",
                "auxiliary_test_target",
            )
        ):
            np.copyto(target_view[:, column], labels[name], casting="no")
        self._states[slot] = "staged"
        return RatchetPinnedHostMicrobatch(
            groups=groups,
            labels=labels,
            contexts=contexts,
            dataset_encodings=contexts[:, : self.catalog.dataset_dimensions],
            target_encodings=contexts[:, self.catalog.dataset_dimensions :],
            group_indices=group_indices,
            label_kinds=label_kinds,
            flags=flags,
            targets=targets,
            slot=slot,
            generation=generation,
        )

    def release_untransferred(self, batch: RatchetPinnedHostMicrobatch) -> None:
        """Release a staged host slot after validation/testing without H2D."""

        slot = batch.slot
        if (
            slot < 0
            or slot >= self.depth
            or batch.generation != self._generations[slot]
            or self._states[slot] != "staged"
        ):
            raise ValueError("pinned host batch is stale or was already consumed")
        self._states[slot] = "free"

    def transfer(
        self,
        batch: RatchetPinnedHostMicrobatch,
        *,
        device: str = "cuda",
        context_dtype: Any | None = None,
    ) -> RatchetDeviceMicrobatch:
        torch = self._torch
        if batch.slot < 0 or batch.slot >= self.depth:
            raise ValueError("pinned microbatch names an unknown ring slot")
        if (
            batch.generation != self._generations[batch.slot]
            or self._states[batch.slot] != "staged"
        ):
            raise ValueError("pinned microbatch is stale or was already transferred")
        if not str(device).startswith("cuda"):
            raise ValueError("nonblocking ratchet transfer requires a CUDA device")
        if self._copy_stream is None:
            self._copy_stream = torch.cuda.Stream(device=device)
            self._copy_device = str(device)
        elif self._copy_device != str(device):
            raise ValueError("pinned training ring is already bound to another CUDA device")
        dtype = context_dtype or torch.bfloat16
        event = torch.cuda.Event()
        with torch.cuda.stream(self._copy_stream):
            # Transfer the full 2,048-d context once, then split on device.
            device_contexts = batch.contexts.to(
                device=device, dtype=dtype, non_blocking=True
            )
            output = RatchetDeviceMicrobatch(
                dataset_encodings=device_contexts[:, : self.catalog.dataset_dimensions],
                target_encodings=device_contexts[:, self.catalog.dataset_dimensions :],
                group_indices=batch.group_indices.to(device=device, non_blocking=True),
                label_kinds=batch.label_kinds.to(device=device, non_blocking=True),
                flags=batch.flags.to(device=device, non_blocking=True),
                targets=batch.targets.to(device=device, non_blocking=True),
            )
            event.record(self._copy_stream)
        torch.cuda.current_stream(device=device).wait_event(event)
        self._events[batch.slot] = event
        self._states[batch.slot] = "in_flight"
        return output

    def microbatches(self) -> Iterator[RatchetPinnedHostMicrobatch]:
        """Yield host batches that must be transferred or released immediately."""

        for groups, labels in self.pack.microbatches():
            yield self.stage(groups, labels)

    def device_microbatches(
        self,
        *,
        device: str = "cuda",
        context_dtype: Any | None = None,
    ) -> Iterator[RatchetDeviceMicrobatch]:
        """Prefetch committed batches on a dedicated H2D stream."""

        for groups, labels in self.pack.microbatches():
            yield self.transfer(
                self.stage(groups, labels),
                device=device,
                context_dtype=context_dtype,
            )


def open_encoding_catalog(
    path: str | Path,
    *,
    catalog_sha256: str,
    data_sha256: str,
    row_count: int = 2322,
    dataset_dimensions: int = 1536,
    target_dimensions: int = 512,
) -> RatchetEncodingCatalog:
    data_path = Path(path).resolve()
    dimensions = dataset_dimensions + target_dimensions
    expected_bytes = row_count * dimensions * 4
    if data_path.stat().st_size != expected_bytes:
        raise ValueError("encoding catalog has the wrong fixed f32 byte length")
    digest = hashlib.sha256()
    with data_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != data_sha256:
        raise ValueError("encoding catalog data differs from its immutable manifest")
    rows = np.memmap(
        data_path, dtype="<f4", mode="r", shape=(row_count, dimensions), order="C"
    )
    return RatchetEncodingCatalog(
        catalog_sha256, data_sha256, rows, dataset_dimensions, target_dimensions
    )


def materialize_training_microbatches(
    pack: RatchetSchedulePack, catalog: RatchetEncodingCatalog
) -> Iterator[RatchetTrainingMicrobatch]:
    """Copy only the tiny selected context rows into contiguous H2D-ready arrays.

    Labels and genomes remain zero-copy views of the pack. The permanent 18 MiB
    catalog stays mmap-backed; each yielded context copy is only the selected
    games for one committed gradient-accumulation microbatch.
    """

    if pack.manifest["encoding_catalog_sha256"] != catalog.catalog_sha256:
        raise ValueError("schedule pack and encoding catalog authorities differ")
    dimensions = catalog.dataset_dimensions + catalog.target_dimensions
    for groups, labels in pack.microbatches():
        indices = groups["catalog_index"].astype(np.int64, copy=False)
        if np.any(indices >= len(catalog.rows)):
            raise ValueError("schedule pack references an unknown encoding catalog row")
        if np.any(groups["encoding_dimensions"] != dimensions):
            raise ValueError("schedule pack encoding dimensions differ from catalog")
        expected_offsets = indices.astype(np.uint64) * np.uint64(dimensions * 4)
        if np.any(groups["encoding_byte_offset"] != expected_offsets):
            raise ValueError("schedule pack encoding offsets differ from fixed catalog layout")
        contexts = np.ascontiguousarray(catalog.rows[indices], dtype=np.float32)
        yield RatchetTrainingMicrobatch(
            groups=groups,
            labels=labels,
            dataset_encodings=contexts[:, : catalog.dataset_dimensions],
            target_encodings=contexts[:, catalog.dataset_dimensions :],
        )


def open_schedule_pack(path: str | Path) -> RatchetSchedulePack:
    binary_path = Path(path).resolve()
    stem = binary_path.name.removesuffix(".pack.bin")
    manifest_path = binary_path.with_name(f"{stem}.manifest.json")
    completion_path = binary_path.with_name(f"{stem}.complete.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    _validate_metadata(manifest, completion, binary_path)

    mapped = np.memmap(binary_path, dtype=np.uint8, mode="r")
    header = memoryview(mapped)
    if bytes(header[:8]) != PACK_MAGIC:
        raise ValueError("schedule-pack magic is invalid")
    u32 = lambda offset: int.from_bytes(header[offset : offset + 4], "little")
    u64 = lambda offset: int.from_bytes(header[offset : offset + 8], "little")
    version = u32(8)
    header_bytes = u32(12)
    group_count = u32(16)
    row_count = u32(20)
    group_stride = u32(24)
    row_stride = u32(28)
    groups_start = u64(32)
    rows_start = u64(40)
    genomes_start = u64(48)
    genome_bytes = u64(56)
    total_bytes = u64(64)
    if (
        version != 1
        or header_bytes != HEADER_BYTES
        or group_count != 8
        or group_stride != GROUP_BYTES
        or row_stride != ROW_BYTES
        or groups_start != HEADER_BYTES
        or rows_start != HEADER_BYTES + group_count * GROUP_BYTES
        or genomes_start != rows_start + row_count * ROW_BYTES
        or total_bytes != genomes_start + genome_bytes
        or total_bytes != mapped.size
    ):
        raise ValueError("schedule-pack fixed-width header is inconsistent")
    if bytes(header[72:104]).hex() != manifest["schedule_sha256"]:
        raise ValueError("schedule-pack header differs from schedule authority")
    if bytes(header[104:136]).hex() != manifest["packed_plan_sha256"]:
        raise ValueError("schedule-pack header differs from packing-plan authority")
    if bytes(header[136:168]).hex() != manifest["encoding_catalog_sha256"]:
        raise ValueError("schedule-pack header differs from encoding-catalog authority")

    groups = np.ndarray(
        shape=(group_count,), dtype=GROUP_DTYPE, buffer=mapped, offset=groups_start
    )
    rows = np.ndarray(shape=(row_count,), dtype=ROW_DTYPE, buffer=mapped, offset=rows_start)
    if len({bytes(value) for value in groups["dataset_hash_bytes"]}) != 8:
        raise ValueError("schedule pack does not contain eight distinct dataset games")
    if np.any(groups["row_start"] + groups["row_count"] > row_count):
        raise ValueError("schedule-pack group row range is out of bounds")
    if np.any(rows["group_index"] >= group_count):
        raise ValueError("schedule-pack row references an unknown group")
    genome_blob = memoryview(mapped)[genomes_start : genomes_start + genome_bytes]
    return RatchetSchedulePack(manifest, mapped, groups, rows, genome_blob)


def _validate_metadata(manifest: dict, completion: dict, binary_path: Path) -> None:
    if manifest.get("schema") != "RatchetSchedulePack.v1" or not manifest.get("complete"):
        raise ValueError("schedule-pack manifest is incomplete or unsupported")
    digest = hashlib.sha256()
    with binary_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    if digest.hexdigest() != manifest.get("pack_sha256"):
        raise ValueError("schedule-pack binary differs from its content address")
    canonical_manifest_sha256 = hashlib.sha256(
        json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        completion.get("schema") != "RatchetSchedulePackCompletion.v1"
        or completion.get("pack_sha256") != manifest.get("pack_sha256")
    ):
        raise ValueError("schedule-pack completion marker names another pack")
    if completion.get("manifest_sha256") != canonical_manifest_sha256:
        raise ValueError("schedule-pack completion marker differs from its manifest")
