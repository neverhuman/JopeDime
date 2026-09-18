"""`_freeze_corpus` must describe the TOURNAMENT's corpus, not the live pointer.

`training/current-corpus.json` used to have exactly one producer, the promotion
tournament, so reading it back was a harmless way for `freeze_corpus` to report
what it had just frozen. It now has two: `jope_selfplay_corpus_freezer` advances
it at the live `COUNT(*)` precisely so a tournament parked at `phase_index 9`
can no longer starve the DL trainer.

With two producers the read-back is a silent corpus swap, and every downstream
phase consumes `artifacts.freeze_corpus.manifest` verbatim —
`_build_packed_corpus` packs it, `_launch_checkpoint_tournament` and
`jope_selfplay_promotion.rs:1301-1310` render the trainer spec from it. A
tournament pinned at watermark 1,805,437 would have trained its candidate on the
4.69M-row corpus, gated it against that corpus's evidence, and filed the result
under 1,805,437.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest


import test_exact_controller as controller  # noqa: E402
import test_exact_packed_corpus  # noqa: E402
import test_exact_train  # noqa: E402

TOURNAMENT_WATERMARK = 1_805_437
POINTER_WATERMARK = 4_690_000


def write_json(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    return path


def snapshot(root: Path, watermark: int) -> Path:
    """A corpus snapshot in the layout `build_corpus` writes."""
    directory = root / "training" / f"snapshot-{watermark:09d}-full_consolidation-eq32v32c128"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "corpus.train.jsonl").write_text("{}\n")
    (directory / "corpus.val.jsonl").write_text("{}\n")
    if (directory / "corpus-manifest.json").is_file():
        return directory
    manifest = {
        "schema": "JopeTestOnlyTrainingCorpus.v1",
        "source_label_watermark": watermark,
        "source_family_disjoint": True,
        "train": str(directory / "corpus.train.jsonl"),
        "validation": str(directory / "corpus.val.jsonl"),
    }
    write_json(directory / "corpus-manifest.json", manifest)
    return directory


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A root where the standalone freezer has ALREADY run ahead of the
    tournament: the pointer names watermark 4,690,000 while the tournament is
    still pinned at 1,805,437."""
    monkeypatch.setattr(controller, "ROOT", tmp_path)
    tournament = snapshot(tmp_path, TOURNAMENT_WATERMARK)
    ahead = snapshot(tmp_path, POINTER_WATERMARK)
    ahead_manifest = ahead / "corpus-manifest.json"
    write_json(
        tmp_path / "training" / "current-corpus.json",
        {
            "manifest": str(ahead_manifest),
            "manifest_sha256": sha256_of(ahead_manifest),
            "packed_manifest": str(tmp_path / "training" / "packed" / "ahead" / "manifest.json"),
            "packed_manifest_sha256": "0" * 64,
            "watermark": POINTER_WATERMARK,
        },
    )

    def fake_build_corpus(min_rows, *, freeze_watermark=None, corpus_mode="x"):
        # Returns the manifest IN MEMORY, exactly as build_corpus does, without
        # re-writing the snapshot: tests that poison the on-disk manifest are
        # testing the read-back, so the fake must not undo them.
        directory = snapshot(tmp_path, int(freeze_watermark))
        return {
            "schema": "JopeTestOnlyTrainingCorpus.v1",
            "source_label_watermark": int(freeze_watermark),
            "source_family_disjoint": True,
            "train": str(directory / "corpus.train.jsonl"),
            "validation": str(directory / "corpus.val.jsonl"),
        }

    def fake_materialize(manifest_path, destination):
        binding = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
        write_json(
            Path(destination) / binding / "manifest.json",
            {"binding_sha256": binding, "source": str(manifest_path)},
        )
        return {"binding_sha256": binding, "row_count": 1}

    monkeypatch.setattr(test_exact_train, "build_corpus", fake_build_corpus)
    monkeypatch.setattr(test_exact_packed_corpus, "materialize", fake_materialize)
    return tmp_path, tournament


def test_the_artifact_names_the_tournaments_corpus_not_the_pointers(root):
    tmp_path, tournament = root
    result = controller._freeze_corpus(
        {"watermark": TOURNAMENT_WATERMARK, "corpus_mode": "full_consolidation"}
    )
    assert result["manifest"] == str(tournament / "corpus-manifest.json")
    assert result["watermark"] == TOURNAMENT_WATERMARK
    # The pointer is reported, never followed.
    assert result["pointer_watermark"] == POINTER_WATERMARK
    assert result["pointer_manifest"] != result["manifest"]


def test_the_packed_manifest_is_bound_to_the_tournaments_corpus(root):
    tmp_path, tournament = root
    result = controller._freeze_corpus(
        {"watermark": TOURNAMENT_WATERMARK, "corpus_mode": "full_consolidation"}
    )
    packed = json.loads(Path(result["packed_manifest"]).read_text())
    assert packed["source"] == str(tournament / "corpus-manifest.json")
    assert result["packed_manifest_sha256"] == sha256_of(Path(result["packed_manifest"]))


def test_every_hash_in_the_artifact_is_of_the_file_it_names(root):
    result = controller._freeze_corpus(
        {"watermark": TOURNAMENT_WATERMARK, "corpus_mode": "full_consolidation"}
    )
    for path_key, hash_key in (
        ("manifest", "manifest_sha256"),
        ("packed_manifest", "packed_manifest_sha256"),
        ("pointer", "pointer_sha256"),
    ):
        assert result[hash_key] == sha256_of(Path(result[path_key])), path_key


def test_a_manifest_at_the_wrong_watermark_is_refused(root, monkeypatch):
    """The snapshot directory is the authority; if what is on disk there is not
    the tournament's watermark, the phase must fail rather than proceed."""
    tmp_path, tournament = root
    manifest_path = tournament / "corpus-manifest.json"
    poisoned = json.loads(manifest_path.read_text())
    poisoned["source_label_watermark"] = POINTER_WATERMARK
    manifest_path.write_text(json.dumps(poisoned, sort_keys=True))
    with pytest.raises(RuntimeError, match="not the tournament's 1805437"):
        controller._freeze_corpus(
            {"watermark": TOURNAMENT_WATERMARK, "corpus_mode": "full_consolidation"}
        )


def test_a_corrupt_global_pointer_still_fails_closed(root):
    """The pointer is no longer the source of the artifact, but it is still
    verified: a pointer whose manifest hash does not match its file means the
    live corpus the TRAINER reads has been tampered with."""
    tmp_path, _ = root
    pointer_path = tmp_path / "training" / "current-corpus.json"
    pointer = json.loads(pointer_path.read_text())
    pointer["manifest_sha256"] = "f" * 64
    pointer_path.write_text(json.dumps(pointer, sort_keys=True))
    with pytest.raises(RuntimeError, match="pointer failed hash verification"):
        controller._freeze_corpus(
            {"watermark": TOURNAMENT_WATERMARK, "corpus_mode": "full_consolidation"}
        )
