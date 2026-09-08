from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn import functional as F

from .data import TASK_BINARY, TASK_IDS, deterministic_shuffle, read_target_input, stable_seed
from .model import TargetEncoder, build_model_from_checkpoint


def normalize_task(task: str | int) -> int:
    if isinstance(task, int):
        if task in (0, 1):
            return task
        raise ValueError(f"Unknown task id: {task}")
    key = task.lower()
    if key not in TASK_IDS:
        raise ValueError(f"Task must be one of {sorted(TASK_IDS)}")
    return TASK_IDS[key]


def load_encoder(checkpoint_path: Path, device: torch.device | str | None = None) -> TargetEncoder:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    model = build_model_from_checkpoint(checkpoint).to(device)
    model.eval()
    return model


def _window_indices(length: int, window_size: int, stride: int, max_windows: int | None, seed: int) -> list[np.ndarray]:
    if length <= 0:
        raise ValueError("Cannot encode an empty target")
    rng = np.random.default_rng(stable_seed("encode", seed, length, window_size, stride))
    if length < window_size:
        return [rng.choice(length, size=window_size, replace=True).astype(np.int64)]
    shuffled = rng.permutation(length)
    windows: list[np.ndarray] = []
    stride = max(1, stride)
    for start in range(0, length - window_size + 1, stride):
        windows.append(shuffled[start : start + window_size].astype(np.int64))
        if max_windows is not None and len(windows) >= max_windows:
            break
    if not windows:
        windows.append(shuffled[:window_size].astype(np.int64))
    return windows


@torch.no_grad()
def encode_values(
    model: TargetEncoder,
    values: Iterable[float] | np.ndarray,
    task: str | int,
    *,
    window_size: int = 96,
    stride: int | None = None,
    max_windows: int | None = None,
    seed: int = 17,
    device: torch.device | str | None = None,
) -> np.ndarray:
    device = torch.device(device or next(model.parameters()).device)
    task_id = normalize_task(task)
    arr = np.asarray(list(values) if not isinstance(values, np.ndarray) else values, dtype=np.float32).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if task_id == TASK_BINARY:
        arr = (arr >= 0.5).astype(np.float32)
    else:
        arr = np.clip(arr, 0.0, 1.0).astype(np.float32, copy=False)
    original_length = int(arr.size)
    stride = stride or window_size
    windows = _window_indices(original_length, window_size, stride, max_windows, seed)
    embeddings: list[torch.Tensor] = []
    for idx in windows:
        sampled = arr[idx].astype(np.float32, copy=False)
        values_t = torch.from_numpy(sampled[None, :]).to(device=device, dtype=torch.float32)
        mask_t = torch.ones_like(values_t)
        task_t = torch.tensor([task_id], device=device, dtype=torch.long)
        length_t = torch.tensor([original_length], device=device, dtype=torch.float32)
        effective_t = torch.tensor([min(original_length, window_size)], device=device, dtype=torch.float32)
        replacement_t = torch.tensor(
            [0.0 if original_length >= window_size else 1.0 - original_length / float(window_size)],
            device=device,
            dtype=torch.float32,
        )
        embedding = model(
            values_t,
            task_t,
            length_t,
            effective_t,
            mask=mask_t,
            replacement_fraction=replacement_t,
            return_aux=False,
        )
        embeddings.append(embedding.detach().float().cpu())
    combined = F.normalize(torch.cat(embeddings, dim=0).mean(dim=0, keepdim=True), dim=1)
    return combined.squeeze(0).numpy().astype(np.float32, copy=False)


def encode_file(
    checkpoint_path: Path,
    input_path: Path,
    task: str | int,
    *,
    output_path: Path | None = None,
    window_size: int = 96,
    stride: int | None = None,
    max_windows: int | None = None,
    seed: int = 17,
    device: str | None = None,
) -> np.ndarray:
    model = load_encoder(checkpoint_path, device=device)
    target = deterministic_shuffle(read_target_input(input_path), str(Path(input_path).resolve()), seed)
    embedding = encode_values(
        model,
        target,
        task,
        window_size=window_size,
        stride=stride,
        max_windows=max_windows,
        seed=seed,
        device=device,
    )
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".json":
            output_path.write_text(json.dumps({"embedding": embedding.tolist()}) + "\n", encoding="utf-8")
        else:
            np.save(output_path, embedding)
    return embedding
