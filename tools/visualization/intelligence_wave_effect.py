"""An editorial brightness wave over pixels from the recorded neural render."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

# 1080p plate coordinates. The table ends at x=611 and the source window starts
# at x=1310. The vertical interval includes the network's floor reflection.
NETWORK_ROI = (620, 225, 1309, 970)


def _smoothstep(value):
    value = np.clip(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


def apply_wave(
    frame: Image.Image,
    progress: float,
    energy: float = 1.0,
    *,
    roi: tuple[int, int, int, int] = NETWORK_ROI,
) -> Image.Image:
    """Brighten existing RGB signal as a left-to-right wave passes through it.

    Progress is editorial and runs from 0 to 1. Energy ranges from 0 to 2.
    Outside the ROI and on black pixels, the output matches the input exactly.
    """
    if frame.mode != "RGB":
        raise ValueError("intelligence wave requires an RGB frame")
    if not math.isfinite(progress) or not 0.0 <= progress <= 1.0:
        raise ValueError("wave progress must be between 0 and 1")
    if not math.isfinite(energy) or not 0.0 <= energy <= 2.0:
        raise ValueError("wave energy must be between 0 and 2")
    left, top, right, bottom = roi
    if not (0 <= left < right <= frame.width and 0 <= top < bottom <= frame.height):
        raise ValueError("network ROI escapes the frame")

    result = frame.copy()
    if energy == 0.0 or progress in (0.0, 1.0):
        return result

    width, height = right - left, bottom - top
    x = np.arange(width, dtype=np.float32) + 0.5
    y = np.arange(height, dtype=np.float32) + 0.5
    depth = x / width
    feather_x = _smoothstep(np.minimum(x, width - x) / 24.0)
    feather_y = _smoothstep(np.minimum(y, height - y) / 36.0)

    center = -0.08 + 1.16 * progress
    front = np.exp(-0.5 * ((depth - center) / 0.075) ** 2)
    tail = 0.26 * np.exp(-0.5 * ((depth - center + 0.10) / 0.12) ** 2)
    fade = _smoothstep(progress / 0.12) * _smoothstep((1.0 - progress) / 0.12)
    envelope = energy * fade * feather_y[:, None] * (front + tail) * feather_x

    pixels = np.asarray(frame.crop(roi), dtype=np.float32)
    peak = pixels.max(axis=2)
    strength = envelope * _smoothstep((peak - 5.0) / 28.0)
    cyan = np.array([0.12, 0.92, 1.0], dtype=np.float32)
    violet = np.array([0.82, 0.27, 1.0], dtype=np.float32)
    tint = cyan + _smoothstep(depth)[:, None] * (violet - cyan)
    lift = strength[:, :, None] * (0.85 * pixels + 0.28 * peak[:, :, None] * tint)
    # The remaining headroom softens the gain on already bright pixels.
    brightened = pixels + (255.0 - pixels) * np.minimum(lift / 255.0, 1.0)
    result.paste(Image.fromarray(np.rint(brightened).astype(np.uint8)), (left, top))
    return result
