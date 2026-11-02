"""Worker-side corpus reopen must repeat the integrity check."""

from pathlib import Path

import numpy as np


def test_mmap_collator_verifies_corpus_when_opened_in_worker(monkeypatch, tmp_path):
    from jope.dime_score import train

    calls = []
    corpus = object()
    monkeypatch.setattr(
        train,
        "TokenizedCorpus",
        lambda root, *, verify: calls.append((Path(root), verify)) or corpus,
    )
    monkeypatch.setattr(
        train.np,
        "load",
        lambda path, **kwargs: np.zeros((1, 1), dtype=np.float32),
    )
    monkeypatch.setattr(
        train,
        "_collate_cpu",
        lambda opened, indexes, embeddings, lookup: (opened, indexes),
    )

    root = tmp_path / "corpus"
    collator = train._MmapCollator(root, tmp_path / "embeddings.npy", np.zeros((1, 1)))
    assert collator([0]) == (corpus, [0])
    assert calls == [(root, True)]
