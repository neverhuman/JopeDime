"""Bounded local files and native tools for the owned program bridge."""
import hashlib
import json
import os
from pathlib import Path
import signal
import selectors
import stat
import subprocess
import time
from .protocol import canonical

MAX_FILE = 32 * 1024**2
CAMPAIGN_MONOTONIC_END = None


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def bind_campaign_deadline(deadline):
    global CAMPAIGN_MONOTONIC_END
    end = time.monotonic()+max(0,deadline-time.time())
    CAMPAIGN_MONOTONIC_END = end if CAMPAIGN_MONOTONIC_END is None else min(CAMPAIGN_MONOTONIC_END,end)


def remaining(deadline):
    wall = deadline-time.time()
    return wall if CAMPAIGN_MONOTONIC_END is None else min(wall,CAMPAIGN_MONOTONIC_END-time.monotonic())


def tick(deadline):
    if remaining(deadline) <= 0:
        raise TimeoutError('fixed program bridge deadline')


def read(path, cap=MAX_FILE):
    path = Path(path)
    need(path.is_absolute() and '..' not in path.parts, 'absolute source path')
    fd = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            nxt = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = nxt
        file = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    finally:
        os.close(fd)
    with os.fdopen(file, 'rb') as stream:
        first = os.fstat(stream.fileno())
        need(stat.S_ISREG(first.st_mode) and first.st_size <= cap, 'bounded regular source')
        body = stream.read(cap + 1)
        last = os.fstat(stream.fileno())
    need(len(body) == first.st_size and (first.st_size, first.st_mtime_ns, first.st_ctime_ns) ==
         (last.st_size, last.st_mtime_ns, last.st_ctime_ns), 'source changed during read')
    return body


def bound(ref, cap=MAX_FILE):
    body = read(ref['path'], cap)
    need(sha(body) == ref['sha256'], 'source SHA mismatch: ' + ref['path'])
    need('bytes' not in ref or len(body) == ref['bytes'], 'source byte size mismatch')
    return body


def create(path, body):
    path = Path(path)
    need(path.is_absolute() and all(not p.is_symlink() for p in path.parents), 'owned nonsymlink output')
    with path.open('xb') as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    return {'path': str(path), 'sha256': sha(body), 'bytes': len(body)}


def run(command, directory, name, deadline, timeout):
    tick(deadline)
    log = Path(directory)/(name+'.log')
    started = time.time()
    failure = None
    monotonic_end = time.monotonic()+min(timeout,remaining(deadline))
    total = 0
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
        selector = selectors.DefaultSelector()
        selector.register(child.stdout,selectors.EVENT_READ)
        os.set_blocking(child.stdout.fileno(),False)
        parent_exited = None
        try:
            while selector.get_map():
                now = time.monotonic()
                if remaining(deadline) <= 0 or now >= monotonic_end:
                    failure = 'deadline'; break
                if child.poll() is not None:
                    if parent_exited is None: parent_exited = now
                    if now-parent_exited > 1:
                        failure = 'pipe held after parent exit'; break
                for key,_ in selector.select(min(.1,monotonic_end-now)):
                    block = os.read(key.fileobj.fileno(),65536)
                    if not block: selector.unregister(key.fileobj); continue
                    room = 4*1024**2-total
                    stream.write(block[:room]); total += min(len(block),room)
                    if len(block) > room: failure = 'log byte cap'; break
                if failure: break
            if failure is None:
                try: child.wait(timeout=max(.001,min(remaining(deadline),monotonic_end-time.monotonic())))
                except subprocess.TimeoutExpired: failure = 'deadline'
        finally:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            result = child.wait()
            selector.close(); child.stdout.close()
            stream.flush(); os.fsync(stream.fileno())
    receipt = {'command': command, 'returncode': result, 'failure':failure,
               'started_unix': started, 'finished_unix': time.time(),
               'log_sha256': sha(read(log, 4*1024**2))}
    create(Path(directory)/(name+'-command.json'), canonical(receipt))
    need(result == 0 and failure is None, 'native command failed or timed out: ' + name)
    tick(deadline)
    return receipt


class NativeTools:
    def __init__(self, ref, deadline):
        bound(ref)
        self.ref, self.deadline = ref, deadline

    def call(self, mode, value):
        need(mode in ('--canonical-json', '--canonical-program'), 'native tool mode')
        tick(self.deadline)
        body = canonical(value)
        need(len(body) <= 2*1024**2, 'native input cap')
        bound(self.ref)
        result = subprocess.run([self.ref['path'], mode], input=body, capture_output=True,
                                timeout=min(5, remaining(self.deadline)))
        tick(self.deadline)
        need(result.returncode == 0 and result.stdout.endswith(b'\n') and len(result.stdout) <= 2*1024**2,
             'native canonicalizer rejected input')
        bound(self.ref)
        return result.stdout[:-1]

    def canonical_json(self, value):
        body = self.call('--canonical-json', value)
        need(json.loads(body) == value, 'native canonicalizer changed JSON value')
        return body

    def canonical_program(self, text):
        need(type(text) is str and 0 < len(text.encode()) <= 32768, 'bounded DSL text')
        value = json.loads(self.call('--canonical-program', {'dsl': text}))
        need(sha(value['dsl'].encode()) == value['program_sha256'], 'native program hash')
        return value['dsl']
