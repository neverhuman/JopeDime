"""Read native RGB16 PNG plates and blend native-resolution 8-bit graphics."""

from __future__ import annotations

import hashlib
import struct
import subprocess
import tempfile
from threading import Thread

import numpy as np


UHD_SIZE = (3840, 2160)


def png_frames(paths, *, ffmpeg="ffmpeg", size=UHD_SIZE, expected_sha256=None):
    """Yield RGB16 arrays in path order using one FFmpeg PNG decoder.

    Pass a sequence of expected hashes in the same order as the paths to bind
    the bytes read by FFmpeg to a render receipt. Paths may be nonconsecutive.
    """
    paths = tuple(paths)
    hashes = tuple(expected_sha256) if expected_sha256 is not None else (None,) * len(paths)
    if len(paths) != len(hashes):
        raise ValueError("PNG paths and expected hashes must have the same length")
    if not paths:
        return
    width, height = size
    frame_bytes = width * height * 6
    command = [ffmpeg, "-nostdin", "-v", "error", "-xerror", "-f", "image2pipe", "-c:v", "png",
               "-i", "pipe:0", "-map", "0:v:0", "-fps_mode", "passthrough",
               "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"]
    with tempfile.TemporaryFile() as stderr:
        with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr) as process:
            writer_errors = []

            def feed():
                try:
                    with process.stdin as sink:
                        for path, expected in zip(paths, hashes, strict=True):
                            data = path.read_bytes()
                            if (len(data) < 29 or data[:16] != b"\x89PNG\r\n\x1a\n\0\0\0\rIHDR"
                                    or struct.unpack(">IIBB", data[16:26]) != (width, height, 16, 2)):
                                raise ValueError(f"Expected native {width}×{height} RGB16 PNG at {path}")
                            if expected and hashlib.sha256(data).hexdigest() != expected:
                                raise ValueError(f"PNG differs from its render receipt at {path}")
                            sink.write(data)
                except (OSError, ValueError) as error:
                    writer_errors.append(error)

            writer = Thread(target=feed, name="native PNG input")
            writer.start()
            count = 0
            try:
                while data := process.stdout.read(frame_bytes):
                    if len(data) != frame_bytes:
                        raise ValueError("Incomplete RGB16 frame from the PNG decoder")
                    count += 1
                    yield np.frombuffer(data, dtype="<u2").reshape(height, width, 3)
                writer.join()
                if writer_errors:
                    raise writer_errors[0]
                if process.wait() or count != len(paths):
                    stderr.seek(0)
                    raise RuntimeError(f"PNG decoder emitted {count}/{len(paths)} frames: {stderr.read().decode(errors='replace')}")
            finally:
                if process.poll() is None:
                    process.kill()
                writer.join()


def rgb8_to_rgb16(image):
    """Expand an already native-size RGB8 raster without resampling it."""
    return np.asarray(image, dtype=np.uint8).astype("<u2") * 257


def blend_rgba8(target, overlay):
    """Alpha composite RGBA8 into a writable RGB16 array in place."""
    rgba = np.asarray(overlay, dtype=np.uint8)
    if target.dtype != np.dtype("<u2") or rgba.shape != (*target.shape[:2], 4) or target.shape[2] != 3:
        raise ValueError("RGBA overlay must match the RGB16 destination")
    alpha = rgba[..., 3:4].astype(np.uint32)
    rgb = rgba[..., :3].astype(np.uint32) * 257
    target[...] = (target.astype(np.uint32) * (255 - alpha) + rgb * alpha + 127) // 255


def blend_rgb16(target, reference, amount):
    """Blend two RGB16 regions in place with a scalar amount in [0, 1]."""
    if amount >= 1.:
        target[...] = reference
    elif amount > 0.:
        alpha = round(amount * 65535)
        target[...] = (target.astype(np.uint32) * (65535 - alpha)
                       + reference.astype(np.uint32) * alpha + 32767) // 65535
