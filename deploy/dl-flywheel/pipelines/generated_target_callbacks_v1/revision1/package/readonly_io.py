"""Read-only exact-version transport and bounded native canonicalization.

No HEAD, PUT, evaluation, shell command, queue or database API is exposed.
Production credentials stay inside the explicitly created boto3 GET transport.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import selectors
import resource
import signal
import stat
import subprocess
import time

BUCKET = 'veox-jopedime-learning-use2'
BASE = 'experiments/gp-policy-dl/v2-trainonly/'
PREFIXES = tuple(BASE + x for x in ('datasets/flywheel/v1/', 'receipts/flywheel-datasets/v1/', 'program-results/flywheel/v1/'))

def mark():
    own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
    return {"utc_ns":time.time_ns(),"monotonic_ns":time.monotonic_ns(),"monotonic":time.monotonic(),
            "cpu_user":own.ru_utime,"cpu_system":own.ru_stime,"child_user":child.ru_utime,"child_system":child.ru_stime}

class Failure(ValueError):
    pass

def need(ok, message):
    if not ok:
        raise Failure(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def hash64(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)

def document(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'duplicate JSON key'); result[key] = value
        return result
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda x: (_ for _ in ()).throw(Failure('nonfinite JSON')))
    canonical(value)
    return value

def read_bound(ref, cap=32*1024**2):
    need(type(ref) is dict and set(ref) in ({'path','sha256'}, {'path','sha256','bytes'}) and hash64(ref['sha256']), 'closed local pin')
    p = Path(ref['path'])
    need(p.is_absolute() and '..' not in p.parts and all(not q.is_symlink() for q in (p,*p.parents)), 'absolute nonsymlink local pin')
    fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0 <= before.st_size <= cap, 'bounded regular file')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            raw = stream.read(cap + 1)
        after = os.fstat(fd)
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns) ==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns), 'local file changed during read')
        need(len(raw) == before.st_size and sha(raw) == ref['sha256'] and
             ('bytes' not in ref or ref['bytes'] == len(raw)), 'local exact hash/bytes')
        return raw
    finally:
        os.close(fd)

def reference(ref):
    need(type(ref) is dict and set(ref) == {'bucket','key','version_id','sha256','bytes'}, 'closed exact-version reference')
    need(ref['bucket'] == BUCKET and type(ref['key']) is str and any(ref['key'].startswith(x) for x in PREFIXES)
         and '..' not in ref['key'].split('/') and '\\' not in ref['key'], 'read namespace')
    need(type(ref['version_id']) is str and 0 < len(ref['version_id']) <= 1024 and ref['version_id'] != 'null'
         and hash64(ref['sha256']) and type(ref['bytes']) is int and 0 < ref['bytes'] <= 32*1024**2, 'exact version/hash/bytes')
    return ref

class Window:
    def __init__(self, deadline_unix, max_gets, max_bytes):
        now = time.time()
        need(type(deadline_unix) in (int,float) and math.isfinite(deadline_unix) and 0 < deadline_unix-now <= 900, 'bounded future deadline')
        need(type(max_gets) is int and 0 < max_gets <= 4096 and type(max_bytes) is int and 0 < max_bytes <= 512*1024**2, 'read budget')
        self.deadline_unix = deadline_unix
        self.start = time.monotonic(); self.end = self.start + deadline_unix-now
        self.max_gets,self.max_bytes = max_gets,max_bytes
        self.gets = self.bytes = self.reserved_bytes = 0
        self.events = []
    def remaining(self):
        return min(self.deadline_unix-time.time(),self.end-time.monotonic())
    def tick(self):
        need(self.remaining() > 0, 'read/canonicalization deadline expired')
    def event(self, kind, started, **fields):
        need(len(self.events) < 8192, 'timing event cap')
        ended=mark()
        self.events.append({'kind':kind,'start_utc_ns':started['utc_ns'],'end_utc_ns':ended['utc_ns'],
            'start_monotonic_ns':started['monotonic_ns'],'end_monotonic_ns':ended['monotonic_ns'],
            'elapsed_seconds':ended['monotonic']-started['monotonic'],
            'cpu_user_seconds':ended['cpu_user']-started['cpu_user'],'cpu_system_seconds':ended['cpu_system']-started['cpu_system'],
            'child_user_seconds':ended['child_user']-started['child_user'],'child_system_seconds':ended['child_system']-started['child_system'],
            'cuda_seconds':None,'nested_spans_not_additive':True,**fields})
    def counters(self):
        return {'gets':self.gets,'bytes':self.bytes,'attempted_bytes':self.reserved_bytes,
                'elapsed_seconds':time.monotonic()-self.start,'events':list(self.events)}

class ExactReader:
    def __init__(self, client, window):
        self.__client,self.window = client,window
        self.bindings = {}
    def get_version(self, ref):
        reference(ref); w = self.window; w.tick()
        key = (ref['bucket'],ref['key'],ref['version_id'])
        value = (ref['sha256'],ref['bytes'])
        need(self.bindings.setdefault(key,value) == value, 'conflicting same-version pin')
        need(w.gets < w.max_gets and w.reserved_bytes+ref['bytes'] <= w.max_bytes, 'GET budget exhausted')
        w.gets += 1; w.reserved_bytes += ref['bytes']; started=mark(); response=None; count=0; body_started=None
        try:
            response = self.__client.get_object(Bucket=ref['bucket'],Key=ref['key'],VersionId=ref['version_id'])
            w.event('s3.get_headers',started,key=ref['key'],version_id=ref['version_id']);body_started=mark()
            w.tick()
            need(response.get('VersionId') == ref['version_id'] and response.get('ContentLength') == ref['bytes'], 'GET exact version/length')
            body = bytearray()
            while True:
                w.tick(); block=response['Body'].read(min(65536,ref['bytes']-len(body)+1))
                need(type(block) is bytes,'byte stream required')
                count+=len(block);w.bytes+=len(block)
                need(count<=ref['bytes'] and w.bytes<=w.max_bytes,'bounded GET body')
                if not block: break
                body.extend(block)
            w.tick(); raw=bytes(body)
            need(len(raw)==ref['bytes'] and sha(raw)==ref['sha256'],'GET exact length/SHA')
            return raw,ref['version_id']
        finally:
            if response is not None and response.get('Body') is not None:
                response['Body'].close()
            if body_started is not None:w.event('s3.get_body',body_started,key=ref['key'],bytes_read=count)
            w.event('s3.exact_version_get',started,key=ref['key'],version_id=ref['version_id'],bytes_read=count)

class NativeCanonical:
    def __init__(self, ref, window):
        self.ref = dict(ref);self.window=window
        read_bound(self.ref,8*1024**2)
    def _call(self, mode, value):
        need(mode in ('--canonical-json','--canonical-program'), 'canonicalization modes only')
        self.window.tick(); read_bound(self.ref,8*1024**2)
        data=canonical(value);need(len(data)<=2*1024**2,'native stdin bound')
        started=mark(); p=None; selector=selectors.DefaultSelector(); buffers={'out':bytearray(),'err':bytearray()}
        try:
            p=subprocess.Popen([self.ref['path'],mode],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                               start_new_session=True,close_fds=True)
            for name,stream,events in [('in',p.stdin,selectors.EVENT_WRITE),('out',p.stdout,selectors.EVENT_READ),('err',p.stderr,selectors.EVENT_READ)]:
                os.set_blocking(stream.fileno(),False);selector.register(stream,events,name)
            sent=0;end=time.monotonic()+min(5,self.window.remaining())
            while selector.get_map():
                self.window.tick();need(time.monotonic()<end,'native canonicalization timeout')
                for key,_ in selector.select(min(.05,max(0,end-time.monotonic()))):
                    stream=key.fileobj;name=key.data
                    if name=='in':
                        if sent<len(data):
                            sent+=os.write(stream.fileno(),data[sent:sent+65536])
                        if sent==len(data):selector.unregister(stream);stream.close()
                    else:
                        block=os.read(stream.fileno(),65536)
                        if not block:selector.unregister(stream);stream.close();continue
                        buffers[name].extend(block)
                        need(len(buffers[name]) <= (2*1024**2+1 if name=='out' else 65536),'native output cap')
            need(p.wait(timeout=max(.001,min(1,self.window.remaining())))==0,'canonicalizer nonzero exit')
            self.window.tick();read_bound(self.ref,8*1024**2)
            raw=bytes(buffers['out']);need(raw.endswith(b'\n') and len(raw)<=2*1024**2+1,'canonicalizer framing')
            return raw[:-1]
        finally:
            selector.close()
            if p is not None:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait(timeout=1)
                for stream in (p.stdin,p.stdout,p.stderr):
                    if stream is not None and not stream.closed:stream.close()
            self.window.event('native.canonicalization',started,mode=mode,stdout_bytes=len(buffers['out']),stderr_bytes=len(buffers['err']))
    def canonical_json(self, value):
        raw=self._call('--canonical-json',value);need(document(raw)==value,'native JSON value changed');return raw
    def canonical_program(self, text):
        need(type(text) is str and 0<len(text.encode())<=32768,'bounded program text')
        result=document(self._call('--canonical-program',{'dsl':text}))
        need(type(result) is dict and type(result.get('dsl')) is str and 0<len(result['dsl'].encode())<=65536
             and result.get('program_sha256')==sha(result['dsl'].encode()),'native program identity')
        return result['dsl']
