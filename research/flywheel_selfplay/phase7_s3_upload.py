"""Shared S3 upload and byte-readback verification for Phase 7 publishers."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable


def s3_upload_and_verify(
    local_path: Path,
    s3_key: str,
    *,
    bucket: str,
    aws_profile: str,
    results_dir: Path,
    sha256_file: Callable[[Path], str],
) -> dict[str, Any]:
    """Upload one file, then verify the stored bytes through an S3 readback."""
    env = os.environ.copy()
    env["AWS_PROFILE"] = aws_profile
    local_sha = sha256_file(local_path)
    local_size = local_path.stat().st_size

    subprocess.run(
        ["aws", "s3", "cp", str(local_path), f"s3://{bucket}/{s3_key}", "--only-show-errors"],
        env=env,
        check=True,
        timeout=30,
    )
    head_out = subprocess.check_output(
        [
            "aws",
            "s3api",
            "head-object",
            "--bucket",
            bucket,
            "--key",
            s3_key,
            "--output",
            "json",
        ],
        env=env,
        text=True,
        timeout=20,
    )
    head_data = json.loads(head_out)

    verify_tmp = results_dir / f".verify_{local_path.name}"
    subprocess.run(
        ["aws", "s3api", "get-object", "--bucket", bucket, "--key", s3_key, str(verify_tmp)],
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=30,
    )
    remote_sha = sha256_file(verify_tmp)
    verify_tmp.unlink(missing_ok=True)
    if remote_sha != local_sha:
        raise RuntimeError(f"S3 SHA256 mismatch for {s3_key}: {remote_sha} != {local_sha}")

    return {
        "local_path": str(local_path),
        "s3_uri": f"s3://{bucket}/{s3_key}",
        "s3_key": s3_key,
        "local_sha256": local_sha,
        "remote_get_sha256": remote_sha,
        "sha256_verified": remote_sha == local_sha,
        "size_bytes": int(local_size),
        "content_length": int(head_data.get("ContentLength", 0)),
        "version_id": str(head_data.get("VersionId", "")),
        "etag": str(head_data.get("ETag", "")).strip('"'),
        "last_modified": str(head_data.get("LastModified", "")),
    }
