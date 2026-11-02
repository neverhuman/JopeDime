"""Compatibility facade for the live canonical batch-stream implementation.

The frozen hpc_verified_v5 package keeps its reviewed copy; live callers share
this implementation through one source module.
"""
from . import batch_stream_core as _core
from .batch_stream_core import *

# Preserve the historic public import path for pickled dataclass instances.
for _type in (_core.Batch, _core.Waiting, _core.BatchStream):
    _type.__module__ = __name__
