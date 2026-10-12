"""Portable, fail-closed support for synthetic offline source qualification only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest

sys.dont_write_bytecode = True
FAMILIES = ('incremental_v1', 'capture_reuse_v3', 'budget_preflight_v1', 'adapter_v1')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def wire(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def write_new(path, value):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(wire(value))
        stream.flush()
        os.fsync(stream.fileno())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def prepare_output(output, source, family):
    require(family in FAMILIES or family == 'aggregate', 'known closed family before any import')
    output, source = Path(output).absolute(), Path(source).resolve(strict=True)
    require(not output.is_symlink(), 'output must not be a symlink')
    require(not output.exists(), 'fresh output only; never replace an old receipt or attempt')
    require(output.parent.is_dir(), 'existing output parent required')
    parent = output.parent.resolve(strict=True)
    destination = parent / output.name
    require(destination != source and source not in destination.parents, 'output outside immutable source')
    require(destination not in source.parents, 'output must not contain immutable source')
    output.mkdir(mode=0o700)
    return output


def verify_source(root):
    root = Path(root).resolve(strict=True)
    manifest = root / 'SOURCE_MANIFEST.json'
    require(not manifest.is_symlink(), 'regular source manifest')
    raw = manifest.read_bytes()
    data = json.loads(raw)
    require(data['schema'] == 'jope.standalone-paper-monitor-portable-source/v1', 'source manifest schema')
    rows = data['files']
    expected = {}
    for row in rows:
        relative = row['path']
        require(type(relative) is str and relative and not Path(relative).is_absolute(), 'relative source member')
        require('..' not in Path(relative).parts and relative not in expected, 'unique contained source member')
        expected[relative] = row
    actual = set()
    for path in root.rglob('*'):
        require(not path.is_symlink(), 'no source symlink or linked directory')
        if path.is_file() and path != manifest:
            actual.add(path.relative_to(root).as_posix())
    require(actual == set(expected), 'complete source membership, no unlisted receipt or file')
    for relative, row in expected.items():
        body = (root / relative).read_bytes()
        require(len(body) == row['bytes'] and digest(body) == row['sha256'], 'exact source bytes ' + relative)
    return {'sha256': digest(raw), 'bytes': len(raw), 'files': len(rows)}


def load(name, path, package=False):
    require(name not in sys.modules, 'no preexisting alias ' + name)
    path = Path(path)
    options = {'submodule_search_locations': [str(path.parent)]} if package else {}
    spec = importlib.util.spec_from_file_location(name, path, **options)
    require(spec is not None and spec.loader is not None, 'explicit loader')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def flatten(suite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from flatten(item)
        else:
            yield item


def imported_origins(root):
    root = Path(root).resolve(strict=True)
    result = {}
    for name, module in tuple(sys.modules.items()):
        filename = getattr(module, '__file__', None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if root in path.parents:
            require(path.is_file() and not path.is_symlink(), 'import regular source')
            body = path.read_bytes()
            result[name] = {'path': path.relative_to(root).as_posix(), 'sha256': digest(body), 'bytes': len(body)}
    return result
