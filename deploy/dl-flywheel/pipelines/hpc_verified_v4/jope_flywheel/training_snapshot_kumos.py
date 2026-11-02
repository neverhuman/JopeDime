"""Compatibility import for the shared continuous KumoS snapshot loader."""
import importlib.util as _importlib_util
import sys as _sys
from pathlib import Path as _Path


def _shared_implementation():
    name = __package__ + '._shared_training_snapshot_kumos'
    module = _sys.modules.get(name)
    if module is not None:
        return module
    path = (
        _Path(__file__).resolve().parents[3]
        / 'pipelines/continuous/runtime/jope_flywheel/training_snapshot_kumos.py'
    )
    spec = _importlib_util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f'continuous KumoS snapshot source is unavailable: {path}')
    module = _importlib_util.module_from_spec(spec)
    _sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _sys.modules.pop(name, None)
        raise
    return module


_shared = _shared_implementation()
globals().update({
    name: value for name, value in vars(_shared).items()
    if not (name.startswith('__') and name.endswith('__'))
})
del _shared, _shared_implementation, _importlib_util, _sys, _Path
