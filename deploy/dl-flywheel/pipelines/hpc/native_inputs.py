"""Compatibility surface for the shared native-input implementation."""

from native_inputs_core import (
    CANONICALIZER_SHA,
    MAX_RAW,
    PROTOCOL,
    WORKER_SHA,
    NativeInputs,
    OriginalScoreContext,
    Unsupported,
    bound,
    canonical,
    digest,
    hash64,
    masked_training,
    need,
    sha,
)

# Keep compatibility with callers that persist or inspect these public types.
OriginalScoreContext.__module__ = __name__
Unsupported.__module__ = __name__
NativeInputs.__module__ = __name__

__all__ = [
    "CANONICALIZER_SHA",
    "MAX_RAW",
    "PROTOCOL",
    "WORKER_SHA",
    "NativeInputs",
    "OriginalScoreContext",
    "Unsupported",
    "bound",
    "canonical",
    "digest",
    "hash64",
    "masked_training",
    "need",
    "sha",
]
