"""Continuous-runtime compatibility facade for the canonical Kumo snapshot loader.

The campaign and continuous runtimes share one live implementation in
``deploy/dl-flywheel/jope_flywheel/training_snapshot_kumos.py``. This facade
keeps the continuous runtime's historical 256 MiB read ceiling and binds
receipts to this deployed entry point.
"""
from __future__ import annotations

import hashlib
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

_SHARED_PATH = Path(__file__).resolve().parents[4] / 'jope_flywheel' / 'training_snapshot_kumos.py'
_SHARED_NAME = f'{__package__}._canonical_training_snapshot_kumos'
_SPEC = spec_from_file_location(_SHARED_NAME, _SHARED_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f'canonical Kumo snapshot loader is unavailable: {_SHARED_PATH}')
_CANONICAL = module_from_spec(_SPEC)
sys.modules[_SHARED_NAME] = _CANONICAL
_SPEC.loader.exec_module(_CANONICAL)

globals().update({name: value for name, value in vars(_CANONICAL).items()
                  if not name.startswith('__')})

MAX_READ = 256 * 1024**2


class _ContinuousStore:
    """Enforce the continuous runtime's cumulative unique-version read bound."""
    def __init__(self, store):
        self._store = store
        self._bytes_read = 0
        self.nativecanonical_json = getattr(store, 'nativecanonical_json', None)

    def get_version(self, ref):
        size = ref.get('bytes')
        if type(size) is not int or size <= 0 or self._bytes_read + size > MAX_READ:
            raise ValueError('snapshot read budget exceeded')
        result = self._store.get_version(ref)
        self._bytes_read += size
        return result


class _Reader(_CANONICAL._Reader):
    """Keep the private reader import compatible with continuous consumers."""
    def get_version(self, ref):
        key = (ref['bucket'], ref['key'], ref['version_id'])
        if key not in self.cache and self.total + ref['bytes'] > MAX_READ:
            raise ValueError('snapshot read budget exceeded')
        return super().get_version(ref)


def load_snapshot(spec_bytes, expected_sha256, store, *, registry_bytes,
                  authority_bytes, authority_sha256, grammar_bytes, now=None):
    loaded = _CANONICAL.load_snapshot(
        spec_bytes, expected_sha256, _ContinuousStore(store),
        registry_bytes=registry_bytes, authority_bytes=authority_bytes,
        authority_sha256=authority_sha256, grammar_bytes=grammar_bytes, now=now)
    loaded.receipt['loader_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    loaded.receipt['read_budget_bytes'] = MAX_READ
    return loaded
