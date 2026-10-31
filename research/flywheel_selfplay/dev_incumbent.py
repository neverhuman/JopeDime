"""Development-only adapter ratchet; preserve rejected proposals and restore optimizer.

The dev cohort is repeatedly observed for selection and cannot establish sealed-test
quality. Keep the comparison identity fixed and never consult the sealed meta-test.
"""

from __future__ import annotations

import copy
import math


def admission_margin(value):
    """Validate an explicitly supplied strict-admission utility margin."""
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("adapter admission margin must be finite and nonnegative")
    return value


def validate_admission_options(strict_enabled, min_gain):
    """Keep the legacy policy unless strict admission and its margin are explicit."""
    if strict_enabled:
        if min_gain is None:
            raise ValueError("--strict-adapter-admission requires --adapter-min-gain")
        return admission_margin(min_gain)
    if min_gain is not None:
        raise ValueError("--adapter-min-gain requires --strict-adapter-admission")
    return None


def add_admission_arguments(parser):
    """Add opt-in flags without changing existing command defaults."""
    parser.add_argument(
        "--strict-adapter-admission",
        action="store_true",
        help="opt in to strict candidate > incumbent + explicit margin admission",
    )
    parser.add_argument(
        "--adapter-min-gain",
        type=float,
        default=None,
        help="explicit nonnegative margin; requires --strict-adapter-admission",
    )
    return parser


def publish_checkpoint_archive(path, s3_prefix, profile="veox-jepson-prod"):
    """Host AWS CLI only; deletion requires a version-pinned independent GET."""
    import hashlib
    import json
    import subprocess
    import tempfile
    from pathlib import Path
    from urllib.parse import urlparse

    path = Path(path)
    location = urlparse(s3_prefix)
    if (
        location.scheme != "s3"
        or location.netloc != "veox-jopedime-learning-use2"
        or not location.path.startswith("/experiments/")
    ):
        raise ValueError("retention requires the approved learning experiments prefix")
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    key = location.path.lstrip("/").rstrip("/") + "/" + path.name
    uri = f"s3://{location.netloc}/{key}"
    subprocess.run(
        [
            "aws",
            "s3",
            "cp",
            str(path),
            uri,
            "--profile",
            profile,
            "--only-show-errors",
            "--metadata",
            "sha256=" + digest,
        ],
        capture_output=True,
        timeout=120,
        check=True,
    )
    head = json.loads(
        subprocess.run(
            [
                "aws",
                "s3api",
                "head-object",
                "--bucket",
                location.netloc,
                "--key",
                key,
                "--profile",
                profile,
            ],
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        ).stdout
    )
    if head["ContentLength"] != path.stat().st_size or not head.get("VersionId"):
        raise ValueError("S3 archive size/version verification failed")
    with tempfile.TemporaryDirectory(prefix="s3-readback-", dir=path.parent) as folder:
        downloaded = Path(folder) / "object"
        subprocess.run(
            [
                "aws",
                "s3api",
                "get-object",
                "--bucket",
                location.netloc,
                "--key",
                key,
                "--version-id",
                head["VersionId"],
                "--profile",
                profile,
                str(downloaded),
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
        with downloaded.open("rb") as stream:
            if (
                downloaded.stat().st_size != path.stat().st_size
                or hashlib.file_digest(stream, "sha256").hexdigest() != digest
            ):
                raise ValueError("S3 archive readback hash mismatch")
    return {
        "uri": uri,
        "version_id": head["VersionId"],
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def retain_checkpoint_payloads(
    output,
    archive_root,
    protected_paths=(),
    keep_last=20,
    guarded_roots=(),
    minimum_free_bytes=50 * 1024**3,
    s3_prefix="s3://veox-jopedime-learning-use2/experiments/flywheel-full-20261030/run-20261030/retention/",
    scored_round=None,
    rotate_bytes=8 * 1024**2,
):
    """Archive/restore-verify old payloads before pruning; callers hold the run lock.

    Root receipts and unknown files stay in place. No model, dataset or optimizer
    access is needed; host S3 custody must succeed. The canonical best pointer and explicit resume/pins are fences.
    """
    import hashlib
    import json
    import os
    import shutil
    import stat
    import subprocess
    import tarfile
    import tempfile
    import time
    import uuid
    from pathlib import Path

    if keep_last < 1:
        raise ValueError("checkpoint retention must keep at least one round")
    root, archive_root = Path(output).resolve(), Path(archive_root).resolve()
    roots = [
        root,
        *(Path(p).resolve() for p in guarded_roots),
        Path(__file__).resolve().parents[2],
    ]
    # Managed worktrees sit below the canonical checkout; exclude that whole tree.
    roots.extend(p for p in Path(__file__).resolve().parents if (p / ".git").is_dir())
    if any(archive_root.is_relative_to(p) for p in roots):
        raise ValueError("archive must be outside guarded roots and the checkout")

    def digest(p):
        with p.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    def durable_json(path, value):
        with path.open("w") as stream:
            json.dump(value, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())

    pointer = root / "best_dev_checkpoint.json"
    pointer_bytes = pointer.read_bytes() if pointer.exists() else None
    declared = list(protected_paths)
    if pointer_bytes:
        best = json.loads(pointer_bytes)["best_checkpoint"]
        if best != "initial_model":
            # Runner pointers can be cwd-relative; explicit pins below are run-relative.
            best = Path(best)
            declared.append(best.resolve() if best.is_file() else root / best)
    protected = set()
    for name in declared:
        if name is None:
            continue
        path = Path(name)
        path = path if path.is_absolute() else root / path
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"protected checkpoint is missing or not regular: {path}")
        protected.add(path.resolve())
    protected_hashes = {str(p): digest(p) for p in protected}
    completed, accepted = set(), set()
    for p in root.glob("round-*.json"):
        number = p.stem.removeprefix("round-")
        if number.isdecimal():
            n = int(number)
            observation = json.loads(p.read_text())
            if p.is_symlink() or observation.get("round") != n:
                raise ValueError("invalid completed-round receipt")
            completed.add(n)
            if observation.get("adapter_admission", {}).get("accepted"):
                accepted.add(n)
    # Legacy fixtures/receipts without admission metadata keep their recent checkpoints.
    recent = set(sorted(accepted or completed)[-keep_last:])
    if completed:
        recent.add(
            max(completed)
        )  # Logical restart frontier, including rejected rounds.
    if (root / "round-882.pt").exists():
        protected.add((root / "round-882.pt").resolve())
        protected_hashes[str(root / "round-882.pt")] = digest(root / "round-882.pt")
    candidates = []
    rejected = root / "rejected"
    if rejected.is_symlink():
        raise ValueError("symlinked rejected directory")
    for p in [*root.glob("round-*.pt"), *rejected.glob("round-*.pt")]:
        number = p.stem.removeprefix("round-")
        if (
            number.isdecimal()
            and int(number) in completed
            and (p.parent == rejected or int(number) not in recent)
            and p.resolve() not in protected
        ):
            candidates.append(p)
    old_directories = []
    for directory in root.glob("rust_eval_round*"):
        number = directory.name.removeprefix("rust_eval_round")
        eligible_eval = completed if scored_round is not None else completed - recent
        if number.isdecimal() and int(number) in eligible_eval:
            if directory.is_symlink() or not directory.is_dir():
                raise ValueError("invalid evaluation directory")
            old_directories.append(directory)
            for p in directory.rglob("*"):
                if p.is_symlink():
                    raise ValueError("symlink in evaluation directory")
                # t533 allows candidate artifacts only after compact score/tree publication.
                eligible_suffix = (
                    (".jtt1", ".json") if scored_round is not None else (".jtt1",)
                )
                if p.is_file() and p.suffix.lower() in eligible_suffix:
                    candidates.append(p)
    rotations = set()
    for pattern in ["summary.jsonl", "*.log"]:
        for p in root.glob(pattern):
            if p.is_file() and not p.is_symlink() and p.stat().st_size >= rotate_bytes:
                candidates.append(p)
                rotations.add(str(p.relative_to(root)))
    rows = {}
    for p in candidates:
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
            raise ValueError("unsafe retention candidate")
        if p.resolve() in protected:
            continue
        s = p.stat()
        rows[str(p.relative_to(root))] = {
            "bytes": s.st_size,
            "mtime_ns": s.st_mtime_ns,
            "mode": stat.S_IMODE(s.st_mode),
            "sha256": digest(p),
        }
    if not rows:
        return {"pruned_files": 0}
    if pointer_bytes is None:
        raise ValueError("best checkpoint pointer required before pruning")
    existing = archive_root
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < minimum_free_bytes + 2 * sum(
        r["bytes"] for r in rows.values()
    ):
        raise ValueError("insufficient archive/restore headroom; nothing pruned")
    archive_root.mkdir(parents=True, exist_ok=True)
    name = f"retention-{max(completed)}-{time.time_ns()}-{uuid.uuid4().hex}"
    archive = archive_root / (name + ".tar.zst")
    manifest = {
        "source_run": str(root),
        "keep_last": keep_last,
        "protected": protected_hashes,
        "archive": str(archive),
        "files": rows,
    }
    manifest_path = archive_root / (name + ".manifest.json")
    durable_json(manifest_path, manifest)
    with subprocess.Popen(
        ["zstd", "-T2", "-1", "--long=27", "-o", str(archive)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    ) as compressor:
        with tarfile.open(fileobj=compressor.stdin, mode="w|") as tar:
            for rel in sorted(rows):
                tar.add(root / rel, arcname=rel, recursive=False)
        compressor.stdin.close()
        error = compressor.stderr.read()
        if compressor.wait() != 0:
            raise RuntimeError(
                f"checkpoint archive failed: {error.decode(errors='replace')}"
            )
    with archive.open("rb") as stream:
        os.fsync(stream.fileno())
    archive_hash = digest(archive)
    # Stream into a fresh independent directory; never extract unchecked tar paths.
    with tempfile.TemporaryDirectory(
        prefix="verify-retention-", dir=archive_root
    ) as folder:
        restored = Path(folder)
        seen = set()
        with subprocess.Popen(
            ["zstd", "-dc", str(archive)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ) as decoder:
            with tarfile.open(fileobj=decoder.stdout, mode="r|") as tar:
                for member in tar:
                    if (
                        not member.isfile()
                        or member.name not in rows
                        or member.name in seen
                    ):
                        raise ValueError("unexpected archive member")
                    expected = rows[member.name]
                    if (
                        member.size != expected["bytes"]
                        or member.mode != expected["mode"]
                    ):
                        raise ValueError("archive metadata mismatch")
                    target = restored / member.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        tar.extractfile(member) as source,
                        target.open("wb") as destination,
                    ):
                        shutil.copyfileobj(source, destination)
                    target.chmod(expected["mode"])
                    os.utime(target, ns=(expected["mtime_ns"], expected["mtime_ns"]))
                    if (
                        target.stat().st_size != expected["bytes"]
                        or digest(target) != expected["sha256"]
                    ):
                        raise ValueError("restored checkpoint hash mismatch")
                    seen.add(member.name)
            if decoder.wait() != 0 or seen != set(rows):
                raise ValueError("incomplete checkpoint archive")
    if digest(archive) != archive_hash or pointer.read_bytes() != pointer_bytes:
        raise ValueError("archive or best pointer changed during retention")
    # Preserve the hash manifest in S3 too, before any source deletion/rotation.
    remote_archive = publish_checkpoint_archive(archive, s3_prefix)
    remote_manifest = publish_checkpoint_archive(manifest_path, s3_prefix)
    for path, expected in protected_hashes.items():
        if digest(Path(path)) != expected:
            raise ValueError("protected checkpoint changed during retention")
    for rel, expected in rows.items():
        p = root / rel
        if (
            p.is_symlink()
            or p.stat().st_mtime_ns != expected["mtime_ns"]
            or digest(p) != expected["sha256"]
        ):
            raise ValueError("retention source changed; nothing pruned")
    # A fresh dependency check is mandatory. Missing lsof fails closed.
    for offset in range(0, len(rows), 256):
        paths = [str(root / rel) for rel in list(rows)[offset : offset + 256]]
        check = subprocess.run(
            ["lsof", "-t", "--", *paths], capture_output=True, text=True, check=False
        )
        if check.returncode != 1 or check.stdout.strip() or check.stderr.strip():
            raise ValueError("active or unknown checkpoint consumer; nothing pruned")
    receipt = {
        "archive": str(archive),
        "archive_sha256": archive_hash,
        "manifest": str(manifest_path),
        "restored_verified_files": len(rows),
        "protected": protected_hashes,
        "pruned_files": 0,
        "s3_archive": remote_archive,
        "s3_manifest": remote_manifest,
    }
    receipt_path = archive_root / (name + ".receipt.json")
    if shutil.disk_usage(archive_root).free < minimum_free_bytes:
        raise ValueError("free-space reserve reached; nothing pruned")
    durable_json(receipt_path, receipt)
    directory_fd = os.open(archive_root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
    for rel in rows:
        if pointer.read_bytes() != pointer_bytes:
            raise ValueError("best pointer changed; remaining payloads retained")
        if rel in rotations:
            with (root / rel).open("wb") as stream:
                stream.flush()
                os.fsync(stream.fileno())
        else:
            (root / rel).unlink()
        receipt["pruned_files"] += 1
    # Preserve nonempty receipt directories; remove only empty eval subdirectories.
    for directory in old_directories:
        if directory.is_dir() and not directory.is_symlink():
            for p in sorted(
                directory.rglob("*"), key=lambda p: len(p.parts), reverse=True
            ):
                if p.is_dir() and not any(p.iterdir()):
                    p.rmdir()
            if not any(directory.iterdir()):
                directory.rmdir()
    durable_json(receipt_path, receipt)
    return receipt


def batch_seed(seed: int, round_idx: int, step: int):
    """A private batch stream advances by logical round/step, never best_round."""
    import hashlib

    payload = f"flywheel-batch/v1:{seed}:{round_idx}:{step}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


class RejectedCandidateError(RuntimeError):
    pass


class CandidateNovelty:
    """Refuse previously rejected parameter bytes before invoking an evaluator."""

    def __init__(self, identity: str, history):
        self.identity = identity
        self.rejected = {}
        for row in history:
            decision = row.get("adapter_admission", row)
            self.record(
                {
                    **decision,
                    "candidate_parameter_sha256": decision.get(
                        "candidate_parameter_sha256",
                        row.get("candidate_parameter_sha256"),
                    ),
                }
            )

    def record(self, decision):
        if (
            decision.get("evaluation_identity") == self.identity
            and decision.get("accepted") is False
            and decision.get("candidate_parameter_sha256")
        ):
            sha = decision["candidate_parameter_sha256"]
            self.rejected[sha] = min(
                self.rejected.get(sha, decision["round"]), decision["round"]
            )

    def evaluate(self, parameter_sha: str, evaluate):
        if parameter_sha in self.rejected:
            raise RejectedCandidateError(
                f"refuse rejected candidate {parameter_sha}; first rejected round "
                f"{self.rejected[parameter_sha]}; evaluator not called"
            )
        return evaluate()


class DevIncumbent:
    def __init__(
        self,
        model,
        optimizer,
        utility: dict,
        identity: str,
        checkpoint: str,
        round_idx: int,
        *,
        strict_admission: bool = False,
        min_gain: float | None = None,
    ):
        self.min_gain = validate_admission_options(strict_admission, min_gain)
        self.strict_admission = strict_admission
        self.identity = identity
        self.best_utility = self._mean(utility)
        self.best_stats = copy.deepcopy(utility)
        self.best_checkpoint = checkpoint
        self.best_round = round_idx
        self.model_state = copy.deepcopy(model.state_dict())
        self.optimizer_state = copy.deepcopy(optimizer.state_dict())

    @staticmethod
    def _mean(utility):
        value = float(utility["mean"])
        if not math.isfinite(value):
            raise ValueError("non-finite development utility")
        return value

    def restore(self, model, optimizer):
        model.load_state_dict(self.model_state, strict=True)
        optimizer.load_state_dict(copy.deepcopy(self.optimizer_state))

    def accepts(self, utility: dict, identity: str):
        if identity != self.identity:
            raise ValueError("development evaluation identity changed")
        value = self._mean(utility)
        if self.strict_admission:
            return value > self.best_utility + self.min_gain
        return value >= self.best_utility

    def decide(
        self,
        model,
        optimizer,
        utility: dict,
        identity: str,
        checkpoint: str,
        round_idx: int,
    ):
        if identity != self.identity:
            raise ValueError("development evaluation identity changed")
        value = self._mean(utility)
        old = self.best_utility
        # The default remains legacy tie acceptance; strict margin mode is opt-in.
        accepted = value > old + self.min_gain if self.strict_admission else value >= old
        if accepted:
            self.best_utility = value
            self.best_stats = copy.deepcopy(utility)
            self.best_checkpoint = checkpoint
            self.best_round = round_idx
            self.model_state = copy.deepcopy(model.state_dict())
            self.optimizer_state = copy.deepcopy(optimizer.state_dict())
        else:
            self.restore(model, optimizer)
        event = {
            "round": round_idx,
            "evaluation_identity": identity,
            "candidate_utility": value,
            "previous_best_utility": old,
            "accepted": accepted,
            "action": (
                "accept_improvement"
                if accepted and value > old
                else "accept_tie"
                if accepted
                else "restore_best_model_and_optimizer"
            ),
            "best_utility": self.best_utility,
            "best_round": self.best_round,
            "best_checkpoint": self.best_checkpoint,
        }
        if self.strict_admission:
            event["admission_policy"] = "strict-margin/v1"
            event["min_gain"] = self.min_gain
        return event


def proposal_plan(
    seed: int, round_idx: int, history: list[dict], max_steps: int = 2, k: int = 3
):
    """Bounded retry policy; retain best-dev admission while changing rejected proposals."""
    import hashlib
    import json

    def signature(row):
        dev = row.get("candidate_dev_utility", row.get("dev_utility", {}))
        return hashlib.sha256(json.dumps(dev, sort_keys=True).encode()).hexdigest()

    streak = 0
    same = 0
    last_signature = None
    latest = None
    for row in sorted(history, key=lambda r: r["round"], reverse=True):
        decision = row.get("adapter_admission")
        if not decision or decision["accepted"]:
            break
        if latest is None:
            latest = row
            last_signature = signature(row)
        streak += 1
        if signature(row) == last_signature and same == streak - 1:
            same += 1
    retry_level = (
        min(3, 1 + int(latest.get("proposal_plan", {}).get("retry_level", 0)))
        if streak
        else 0
    )
    # GP progress is independent of adapter admission. Accepted dev ties must
    # not disable the GP escape path indefinitely. Unknown/missing rounds do not
    # count as completed zero-promotion rounds.
    gp_drought = 0
    gp_policy_rounds = 0
    expected_round = round_idx - 1
    for row in sorted(history, key=lambda r: r["round"], reverse=True):
        if (
            row["round"] != expected_round
            or row.get("promotions") != 0
            or row.get("GP", {}).get("evaluated", 0) <= 0
        ):
            break
        gp_drought += 1
        prior_policy = row.get("proposal_plan", {}).get("gp_drought_policy", {})
        if (
            prior_policy.get("version") == "promotion-drought-burst/v1"
            and prior_policy.get("phase") is not None
        ):
            gp_policy_rounds += 1
        expected_round -= 1
    threshold, burst, cooldown = 18, 3, 3
    # Historical rounds before installation did not execute this protocol;
    # advance the burst/cooldown clock only for its actual completed rounds.
    burst_phase = gp_policy_rounds % (burst + cooldown)
    drought_exploration = gp_drought >= threshold and burst_phase < burst
    rejection_exploration = same >= k or streak >= k
    explore = rejection_exploration or drought_exploration
    return {
        "version": "rejection-escape/v1",
        "round_seed": seed + round_idx * 104729,
        "consecutive_rejections": streak,
        "same_candidate_streak": same,
        "retry_level": retry_level,
        "lr_factor": 0.25**retry_level,
        "steps": min(max_steps, 1) if streak else max_steps,
        "minimum_lr": 1e-6,
        "training_rotation": round_idx,
        "gp_exploration": explore,
        "gp_exploration_reason": "adapter_rejection"
        if rejection_exploration
        else "promotion_drought_burst"
        if drought_exploration
        else "ranked_exploitation",
        "gp_drought_policy": {
            "version": "promotion-drought-burst/v1",
            "threshold_rounds": threshold,
            "burst_rounds": burst,
            "cooldown_rounds": cooldown,
            "completed_zero_promotion_rounds": gp_drought,
            "phase": burst_phase if gp_drought >= threshold else None,
        },
        "gp_generated_per_context": 16 if explore else 8,
        "gp_max_evaluated_per_context": 4,
        "reason": "smaller_lr_one_step_and_new_seed_after_rejection"
        if streak
        else "new_round_seed",
    }


def gp_selection_indices(costs, seed: int, explore: bool):
    """Reserve two of four native evaluations for novel exploration when stalled."""
    import random

    order = sorted(range(len(costs)), key=lambda i: (float(costs[i]), i))
    if not explore or len(order) <= 2:
        return order[:4]
    chosen = order[:2]
    chosen.extend(random.Random(seed).sample(order[2:], min(2, len(order) - 2)))
    return chosen


def project_score_margins(parameters, constraints, max_passes=2, tolerance=1e-6):
    """Project an optimizer step onto label-free incumbent score-margin constraints.

    Each constraint is (name, differentiable margin callback, minimum margin).
    Only the supplied trainable parameters move; the prediction/evaluation rule
    stays raw argmin. This is part of the optimizer step, not a score override.
    """
    import torch

    params = list(parameters)
    if not params or any(not p.requires_grad for p in params):
        raise ValueError("projection accepts only trainable adapter parameters")
    corrections = []
    for pass_idx in range(max_passes):
        repaired = False
        for name, margin_fn, floor in constraints:
            margin = margin_fn()
            value = float(margin.detach())
            if not math.isfinite(value) or not math.isfinite(floor) or floor < 0:
                raise ValueError("non-finite or negative margin constraint")
            if value >= floor:
                continue
            grads = torch.autograd.grad(margin, params, allow_unused=True)
            norm_sq = sum(
                float((g.detach() ** 2).sum()) for g in grads if g is not None
            )
            if not math.isfinite(norm_sq) or norm_sq <= 1e-20:
                raise RuntimeError(
                    "violated score margin has no usable adapter gradient"
                )
            amount = (floor + min(tolerance, max(floor * 0.1, 1e-8)) - value) / norm_sq
            with torch.no_grad():
                for p, g in zip(params, grads):
                    if g is not None:
                        p.add_(g, alpha=amount)
            corrections.append(
                {
                    "constraint": name,
                    "pass": pass_idx,
                    "before": value,
                    "floor": floor,
                    "correction_l2": abs(amount) * math.sqrt(norm_sq),
                }
            )
            repaired = True
        if not repaired:
            break
    with torch.no_grad():
        after = [
            {"constraint": name, "margin": float(fn()), "floor": floor}
            for name, fn, floor in constraints
        ]
    return {
        "corrections": corrections,
        "after": after,
        "satisfied": all(
            math.isfinite(v["margin"]) and v["margin"] >= v["floor"] - tolerance
            for v in after
        ),
    }


def warm_restart_adapter(model, optimizer, seed, round_idx):
    """Perturb only trainable adapter tensors; reset LR independently of restored LR.

    Restart scales use a fixed round-keyed schedule, never development scores.
    The old four-phase scale table produced four update-size clusters despite
    changing batch/noise seeds. Keep its bounds, remove that short cycle, and
    retain the complete incumbent's Adam moments/step. DOPE/anchor stay frozen.
    """
    import hashlib

    import torch

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable adapter")
    generator = torch.Generator(device=params[0].device).manual_seed(seed)
    digest = hashlib.sha256(
        f"adapter-restart-scales/v2:{seed}:{round_idx}".encode()
    ).digest()
    # Independent 53-bit fractions are exactly representable; no global RNG use.
    sigma_fraction = (int.from_bytes(digest[:8], "little") >> 11) / 2**53
    lr_fraction = (int.from_bytes(digest[8:16], "little") >> 11) / 2**53
    sigma = min(
        3e-4, max(1e-5, math.exp(math.log(1e-5) + sigma_fraction * math.log(30.0)))
    )
    lr = min(
        3e-4, max(2.5e-5, math.exp(math.log(2.5e-5) + lr_fraction * math.log(12.0)))
    )
    with torch.no_grad():
        for p in params:
            p.add_(
                torch.randn(
                    p.shape, dtype=p.dtype, device=p.device, generator=generator
                ),
                alpha=sigma,
            )
    for group in optimizer.param_groups:
        group["lr"] = lr
    return {
        "version": "seeded-adapter-warm-restart/v2",
        "scale_policy": "round-hash-loguniform/v1",
        "round": round_idx,
        "lr_bounds": [2.5e-5, 3e-4],
        "sigma_bounds": [1e-5, 3e-4],
        "seed": seed,
        "sigma": sigma,
        "lr": lr,
    }
