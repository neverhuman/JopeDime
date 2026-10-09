"""Strict JSON, identities and bounded reviewed callback bindings."""
import hashlib
import inspect
import json
import math
import sys
import types
from pathlib import Path
import time


class ExportError(ValueError):
    pass


def need(ok, message):
    if not ok:
        raise ExportError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def digest(value):
    return sha(canonical(value))


def hash64(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def document(raw, cap):
    need(type(raw) is bytes and 0 < len(raw) <= cap, 'bounded exact JSON bytes')
    def pairs(values):
        out = {}
        for key, value in values:
            need(key not in out, 'duplicate JSON key')
            out[key] = value
        return out
    def bad(value):
        raise ExportError('nonfinite JSON')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=bad)
    canonical(value)  # Reject nonfinite overflow such as 1e999 as well.
    return value


def reference(value):
    need(type(value) is dict and set(value) == {'bucket', 'key', 'version_id', 'sha256', 'bytes'},
         'closed exact-version reference')
    need(all(type(value[k]) is str and value[k] for k in ('bucket', 'key', 'version_id'))
         and value['version_id'] != 'null' and hash64(value['sha256'])
         and type(value['bytes']) is int and 0 < value['bytes'] <= 32 * 1024**2,
         'valid version/hash/byte reference')
    return value


def safe_file(path, cap):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts and
         all(not x.is_symlink() for x in (path, *path.parents)) and path.is_file(),
         'absolute regular non-symlink source')
    before = path.stat()
    need(before.st_size <= cap, 'file bound')
    raw = path.read_bytes()
    after = path.stat()
    need(len(raw) == before.st_size == after.st_size and
         (before.st_dev, before.st_ino, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_mtime_ns), 'source changed while read')
    return raw


class Budget:
    def __init__(self, seconds):
        need(type(seconds) in (int, float) and math.isfinite(seconds) and 0 < seconds <= 120,
             'bounded local monotonic work window')
        self.deadline = time.monotonic() + seconds

    def tick(self):
        need(time.monotonic() < self.deadline, 'export work deadline exceeded')


class CheckedCallback:
    """A trusted reviewed callable, not a sandbox or permission to execute GP.

    Pin its defining file/qualified name plus all externally reviewed dependency
    files in the request. Production adapters must be source reviewed before use.
    The exporter independently checks returned identity, never a bare success.
    """
    def __init__(self, fn, binding):
        self.fn, self.binding = fn, binding
        self.check()

    def check(self):
        need(type(self.binding) is dict and set(self.binding) == {'path', 'sha256', 'qualname', 'dependencies'},
             'closed callback source binding')
        b = self.binding
        need(inspect.isfunction(self.fn) and self.fn.__qualname__ == b['qualname'] and
             Path(inspect.getsourcefile(self.fn)).resolve() == Path(b['path']), 'callback origin/name')
        raw = safe_file(b['path'], 2 * 1024**2)
        need(hash64(b['sha256']) and sha(raw) == b['sha256'], 'callback source changed')
        need(self.fn.__closure__ is None and self.fn.__defaults__ is None and not self.fn.__kwdefaults__,
             'callback must be a top-level function without runtime default/closure bindings')
        compiled = compile(raw, self.fn.__code__.co_filename, 'exec', dont_inherit=True, optimize=sys.flags.optimize)
        def codes(code):
            for child in code.co_consts:
                if isinstance(child, types.CodeType):
                    yield child
                    yield from codes(child)
        matches = [c for c in codes(compiled) if c.co_qualname == self.fn.__qualname__]
        # Code-object equality is structural; marshal reference/intern flags can
        # change after an otherwise unchanged function executes.
        need(len(matches) == 1 and matches[0] == self.fn.__code__,
             'loaded callback bytecode differs from pinned source')
        need(type(b['dependencies']) is list and len(b['dependencies']) <= 64, 'callback dependency bound')
        for item in b['dependencies']:
            need(set(item) == {'path', 'sha256'} and hash64(item['sha256']) and
                 sha(safe_file(item['path'], 2 * 1024**2)) == item['sha256'], 'callback dependency changed')

    def __call__(self, *args):
        self.check()
        value = self.fn(*args)
        self.check()
        return value
