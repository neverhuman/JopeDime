"""Compatibility facade for the live canonical snapshot adapter.

The hpc_verified_v5 module remains an immutable reviewed snapshot; live callers
share the implementation in training_snapshot_core.py.
"""
from . import training_snapshot_core as _core

for _name, _value in vars(_core).items():
    if not _name.startswith('_'):
        globals().setdefault(_name, _value)

need = _core._live_need
need.__module__ = __name__
need.__name__ = 'need'
need.__qualname__ = 'need'

byte_sha = _core._live_byte_sha
byte_sha.__module__ = __name__
byte_sha.__name__ = 'byte_sha'
byte_sha.__qualname__ = 'byte_sha'

document = _core._live_document
document.__module__ = __name__
document.__name__ = 'document'
document.__qualname__ = 'document'

normalized_ref = _core._live_normalized_ref
normalized_ref.__module__ = __name__
normalized_ref.__name__ = 'normalized_ref'
normalized_ref.__qualname__ = 'normalized_ref'

_Reader = _core._live_Reader
_Reader.__module__ = __name__
_Reader.__name__ = '_Reader'
_Reader.__qualname__ = '_Reader'

decode_input = _core._live_decode_input
decode_input.__module__ = __name__
decode_input.__name__ = 'decode_input'
decode_input.__qualname__ = 'decode_input'

mean_baselines = _core._live_mean_baselines
mean_baselines.__module__ = __name__
mean_baselines.__name__ = 'mean_baselines'
mean_baselines.__qualname__ = 'mean_baselines'

_consumer = _core._live_consumer
_consumer.__module__ = __name__
_consumer.__name__ = '_consumer'
_consumer.__qualname__ = '_consumer'

_authority = _core._live_authority
_authority.__module__ = __name__
_authority.__name__ = '_authority'
_authority.__qualname__ = '_authority'

_objects_by_suffix = _core._live_objects_by_suffix
_objects_by_suffix.__module__ = __name__
_objects_by_suffix.__name__ = '_objects_by_suffix'
_objects_by_suffix.__qualname__ = '_objects_by_suffix'

_dataset = _core._live_dataset
_dataset.__module__ = __name__
_dataset.__name__ = '_dataset'
_dataset.__qualname__ = '_dataset'

_encodings = _core._live_encodings
_encodings.__module__ = __name__
_encodings.__name__ = '_encodings'
_encodings.__qualname__ = '_encodings'

LoadedSnapshot = _core._live_LoadedSnapshot
LoadedSnapshot.__module__ = __name__
LoadedSnapshot.__name__ = 'LoadedSnapshot'
LoadedSnapshot.__qualname__ = 'LoadedSnapshot'

load_snapshot = _core._live_load_snapshot
load_snapshot.__module__ = __name__
load_snapshot.__name__ = 'load_snapshot'
load_snapshot.__qualname__ = 'load_snapshot'

save_receipt = _core._live_save_receipt
save_receipt.__module__ = __name__
save_receipt.__name__ = 'save_receipt'
save_receipt.__qualname__ = 'save_receipt'
