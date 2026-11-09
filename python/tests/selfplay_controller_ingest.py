from __future__ import annotations

import hashlib
import json
import struct

import pytest
from migrate_jain_web_cpu import cpu_payload


def test_corrected_ingest_requires_hash_bound_prediction_evidence(tmp_path, monkeypatch):
    import test_exact_ingest as ingest
    from test_exact_common import selfplay_authority, sha256_text

    monkeypatch.setattr(ingest, "CORPUS", tmp_path / "corpus.sqlite")
    output = tmp_path / "receipt"
    predictions = output / "predictions/prediction.f64le"
    predictions.parent.mkdir(parents=True)
    prediction_bytes = struct.pack("<dd", 0.25, 0.75)
    predictions.write_bytes(prediction_bytes)
    genome = "(pipe X (ridge))"
    semantic = sha256_text(genome)
    authority = selfplay_authority(
        task="regression",
        wave="strict-wave-000001",
        frozen_seed_sha256="1" * 64,
        model_checkpoint_sha256="2" * 64,
        hall_of_fame_sha256="3" * 64,
    )
    manifest = {
        "task": "regression",
        "dataset_hash": "dataset",
        "train_sha256": "4" * 64,
        "test_sha256": "5" * 64,
        "candidates": [
            {
                "semantic_sha256": semantic,
                "canonical_genome": genome,
                "source": "accepted_dl_checkpoint",
            }
        ],
        "label_authority": authority,
    }
    (output / ".portable-manifest.json").write_text(json.dumps(manifest))
    (output / "SHARD_OK").write_text("ok\n")
    receipt = {
        "task": "regression",
        "dataset_hash": "dataset",
        "semantic_sha256": semantic,
        "canonical_genome": genome,
        "tier": "test_exact",
        "metric": "r2",
        "score": 0.4,
        "secondary_metric": "rmse",
        "secondary_score": 0.5,
        "valid": True,
        "runtime_seconds": 1.0,
        "train_sha256": "4" * 64,
        "test_sha256": "5" * 64,
        "manifest_sha256": "6" * 64,
        "created_at_ms": 7,
        "predictions_path": "predictions/prediction.f64le",
        "predictions_sha256": hashlib.sha256(prediction_bytes).hexdigest(),
        "prediction_count": 2,
        "balanced_accuracy": None,
        **{
            "test_adaptive": authority["test_adaptive"],
            "benchmark_valid": authority["benchmark_valid"],
            "selection_bias": authority["selection_bias"],
            "generation": authority["generation"],
            "frozen_seed_sha256": authority["frozen_seed_sha256"],
            "model_checkpoint_sha256": authority["model_checkpoint_sha256"],
            "hall_of_fame_sha256": authority["hall_of_fame_sha256"],
        },
    }
    receipt_path = output / "dataset.receipts.jsonl"
    receipt_path.write_text(json.dumps(receipt) + "\n")
    database = ingest.corpus_db()
    predictions.write_bytes(prediction_bytes[:-1])
    with pytest.raises(RuntimeError, match="row alignment drift"):
        ingest.validate_and_ingest("job", output, database)
    predictions.write_bytes(prediction_bytes)
    result = ingest.validate_and_ingest("job", output, database)
    database.commit()
    stored = database.execute(
        "SELECT predictions_sha256,prediction_count,strict_wave FROM observations"
    ).fetchone()
    database.close()
    assert result == {"rows": 1, "valid": 1, "failed": 0}
    assert tuple(stored) == (
        hashlib.sha256(prediction_bytes).hexdigest(),
        2,
        "strict-wave-000001",
    )


def test_jain_cpu_clone_removes_every_gpu_admission_path():
    source = {
        "Config": {
            "Image": "jain",
            "Env": [
                "SAFE=value",
                "NVIDIA_VISIBLE_DEVICES=all",
                "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
            ],
        },
        "Image": "sha256:" + "a" * 64,
        "HostConfig": {
            "DeviceRequests": [{"Capabilities": [["gpu"]]}],
            "Devices": [
                {"PathOnHost": "/dev/nvidia0"},
                {"PathOnHost": "/dev/safe-device"},
            ],
            "Runtime": "nvidia",
        },
    }
    payload = cpu_payload(source)
    assert payload["HostConfig"]["DeviceRequests"] == []
    assert payload["HostConfig"]["Devices"] == [{"PathOnHost": "/dev/safe-device"}]
    assert payload["HostConfig"]["Runtime"] == "runc"
    assert payload["Env"] == [
        "SAFE=value",
        "NVIDIA_VISIBLE_DEVICES=void",
        "JAIN_DISABLE_GPU_EXEC=1",
    ]
