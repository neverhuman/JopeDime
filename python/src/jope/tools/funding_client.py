"""Protected native funding client bridge; never creates a Python budget ledger.

The native client verifies signatures, service identity, the independently pinned
decision and witness receipts. Python only transports typed requests to that
reviewed executable. Production configuration has no environment/path override.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess


CONFIG = Path('/etc/jope-dime/funding/client.json')
MAX_CONFIG_BYTES = 1_048_576
MAX_REQUEST_BYTES = 8_388_608


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate funding JSON field: ' + key)
        result[key] = value
    return result


def _json(raw):
    def invalid(value):
        raise ValueError('non-finite funding JSON: ' + value)
    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            invalid(value)
        return number
    return json.loads(raw, object_pairs_hook=_unique, parse_constant=invalid, parse_float=finite)


def _protected(path):
    if not path.is_absolute():
        raise RuntimeError('funding installation path must be absolute')
    for current in (path, *path.parents):
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
            raise RuntimeError('funding installation is not independently protected: ' + str(current))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
        os.close(descriptor)
        raise RuntimeError('funding installation file changed')
    return descriptor


class NativeFundingClient:
    def __init__(self):
        try:
            descriptor = _protected(CONFIG)
            with os.fdopen(descriptor, 'rb') as stream:
                raw = stream.read(MAX_CONFIG_BYTES + 1)
            if len(raw) > MAX_CONFIG_BYTES:
                raise RuntimeError('funding client configuration exceeds bound')
            self.config = _json(raw)
            if not isinstance(self.config, dict) or self.config.get('schema') != 'parallel.funding_client.v1':
                raise RuntimeError('unsupported funding client configuration')
            self.executable = Path(self.config['client_executable'])
            self.expected_executable = self.config['client_executable_sha256']
            if (not isinstance(self.expected_executable, str) or len(self.expected_executable) != 64
                    or any(c not in '0123456789abcdef' for c in self.expected_executable)):
                raise RuntimeError('native funding executable digest missing')
            self.domain_id = self.config['trust']['domain_id']
        except (OSError, KeyError, ValueError, TypeError, RuntimeError) as error:
            raise RuntimeError('production budget authorization verifier unavailable: ' + str(error)) from error

    def request(self, operation, **values):
        # Numerical start/participant claims belong to authenticated native
        # supervisor/rank peers. This Python adapter cannot consume their slots.
        if operation not in ('register', 'reserve', 'preflight'):
            raise RuntimeError('Python funding bridge cannot consume numerical execution')
        request = dict(operation=operation, **values)
        raw = json.dumps(request, separators=(',', ':'), allow_nan=False).encode()
        if len(raw) > MAX_REQUEST_BYTES:
            raise RuntimeError('funding request exceeds bound')
        descriptor = _protected(self.executable)
        try:
            before = os.fstat(descriptor)
            digest = hashlib.sha256()
            while True:
                block = os.read(descriptor, 1_048_576)
                if not block:
                    break
                digest.update(block)
            after = os.fstat(descriptor)
            if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                    != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                    or digest.hexdigest() != self.expected_executable):
                raise RuntimeError('reviewed native funding client executable changed')
            # Execute the pinned opened inode with loader-injection variables
            # absent. The native client independently rechecks config and peers.
            result = subprocess.run(['/proc/self/fd/' + str(descriptor), 'funding-request',
                '--config', str(CONFIG)], input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                pass_fds=(descriptor,), timeout=30, check=False,
                env={'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8'})
        finally:
            os.close(descriptor)
        if result.returncode:
            raise RuntimeError('native funding authority rejected request: ' + result.stderr[-4096:].decode(errors='replace'))
        if len(result.stdout) > MAX_REQUEST_BYTES:
            raise RuntimeError('native funding response exceeds bound')
        response = _json(result.stdout)
        if not isinstance(response, dict):
            raise RuntimeError('native funding response is not a typed object')
        return response
