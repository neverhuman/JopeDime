"""Bounded in-process caches for already verified immutable GP artifacts.

Use CachedS3Library in place of S3Library and CachedNativeTools in place of
NativeTools only in a newly pinned runtime. First reads/canonicalizations retain
the existing verifier. No cache persists to disk or changes lookup/absence rules.
Counters are measured monotonic durations, not reconstructed historical timings.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib,json,threading,time
from jope_flywheel.s3_library import S3Library
from jope_flywheel.program_gate import MAX_BYTES
from jope_flywheel.program_io import NativeTools,bound,tick
from jope_flywheel.protocol import canonical

MIB=1024**2

def need(ok,message):
    if not ok:raise ValueError(message)
def sha(body):return hashlib.sha256(body).hexdigest()
def digest(value):
    need(type(value)is str and len(value)==64 and all(c in '0123456789abcdef'for c in value),'cache SHA256')
    return value

@dataclass(frozen=True)
class Entry:
    body:bytes
    sha256:str
    charge:int

class ByteLRU:
    """Immutable bytes with SHA integrity on every hit and bounded key overhead.

The byte budget includes UTF-8 key storage and bodies; max_entries separately
bounds Python bookkeeping. I/O is outside the lock. Concurrent misses may both
verify a source; neither can insert unverified bytes or exceed the cap.
"""
    def __init__(self,max_bytes,max_entries=4096):
        need(type(max_bytes)is int and 0<max_bytes<=128*MIB,'cache byte budget <=128MiB')
        need(type(max_entries)is int and 0<max_entries<=8192,'cache entry budget')
        self.max_bytes,self.max_entries=max_bytes,max_entries
        self._items=OrderedDict();self.bytes=0;self.lock=threading.RLock()
        self.counters={'hits':0,'misses':0,'evictions':0,'uncached_oversize':0,'integrity_failures':0,'backend_failures':0,'backend_calls':0,'backend_bytes':0,'bytes_served_from_cache':0,'backend_seconds':0.,'integrity_seconds':0.}
    def get(self,key):
        start=time.monotonic()
        with self.lock:
            item=self._items.get(key)
            if item is None:self.counters['misses']+=1;return None
            if type(item.body)is not bytes or sha(item.body)!=item.sha256:
                self._items.pop(key);self.bytes-=item.charge;self.counters['integrity_failures']+=1
                raise ValueError('stored cache bytes corrupted; evicted without reuse')
            self._items.move_to_end(key);self.counters['hits']+=1;self.counters['bytes_served_from_cache']+=len(item.body)
            self.counters['integrity_seconds']+=time.monotonic()-start
            return item.body
    def put(self,key,body):
        need(type(body)is bytes,'cache only immutable verified bytes')
        charge=len(body)+len(repr(key).encode('utf-8'))+128
        with self.lock:
            if charge>self.max_bytes:self.counters['uncached_oversize']+=1;return
            old=self._items.pop(key,None)
            if old is not None:self.bytes-=old.charge
            while self._items and (self.bytes+charge>self.max_bytes or len(self._items)>=self.max_entries):
                _,old=self._items.popitem(last=False);self.bytes-=old.charge;self.counters['evictions']+=1
            self._items[key]=Entry(body,sha(body),charge);self.bytes+=charge
    def backend(self,seconds,size=0,failed=False):
        with self.lock:
            self.counters['backend_calls']+=1;self.counters['backend_seconds']+=seconds
            self.counters['backend_bytes']+=size;self.counters['backend_failures']+=int(failed)
    def snapshot(self):
        with self.lock:return {**self.counters,'entries':len(self._items),'charged_bytes':self.bytes,'max_bytes':self.max_bytes,'max_entries':self.max_entries}

class VerifiedVersionCache:
    """Adapter for get_version(ref)->(body,VersionId), preserving its signature.

Expected ref fields are bucket/key/version_id/sha256/bytes. Extra provenance
fields are allowed but are not cache identity. GET errors and absent objects are
never cached. check() must preserve the wrapped library's operation deadline.
"""
    def __init__(self,get_version,*,check,max_bytes=128*MIB,max_entries=4096):
        need(callable(get_version)and callable(check),'verified getter and deadline check required')
        self.getter,self.check=get_version,check;self.cache=ByteLRU(max_bytes,max_entries)
    @staticmethod
    def identity(ref):
        need(type(ref)is dict,'object reference required')
        for k,limit in [('bucket',255),('key',4096),('version_id',1024)]:
            need(type(ref.get(k))is str and 0<len(ref[k].encode())<=limit,'bounded exact S3 identity:'+k)
        need(ref['version_id']!='null','actual VersionId required');digest(ref['sha256'])
        need(type(ref.get('bytes'))is int and 0<ref['bytes']<=MAX_BYTES,'bounded exact object bytes')
        return tuple(ref[k]for k in ('bucket','key','version_id','sha256','bytes'))
    def get_version(self,ref):
        self.check();key=self.identity(ref);body=self.cache.get(key)
        if body is not None:
            need(len(body)==ref['bytes']and sha(body)==ref['sha256'],'cached object no longer matches exact ref')
            self.check();return body,ref['version_id']
        started=time.monotonic();size=0
        try:
            body,version=self.getter(dict(ref));size=len(body)if type(body)is bytes else 0
            need(type(body)is bytes and version==ref['version_id']and len(body)==ref['bytes']and sha(body)==ref['sha256'],'first exact-Version GET/body verification failed')
            self.check()
        except BaseException:
            self.cache.backend(time.monotonic()-started,size,True);raise
        self.cache.backend(time.monotonic()-started,size);self.cache.put(key,body)
        return body,version
    def snapshot(self):return {'schema':'jope.verified-version-cache/v1',**self.cache.snapshot()}

class CachedS3Library(S3Library):
    """Internal put/lookup/commit calls use this overridden verified read path."""
    def __init__(self,client,*,deadline,nativecanonical_json=None,cache_bytes=128*MIB,cache_entries=4096):
        super().__init__(client,deadline=deadline,nativecanonical_json=nativecanonical_json)
        getter=super().get_version
        self.version_cache=VerifiedVersionCache(getter,check=self.tick,max_bytes=cache_bytes,max_entries=cache_entries)
    def get_version(self,ref):return self.version_cache.get_version(ref)
    def cache_stats(self):return self.version_cache.snapshot()

class CachedNativeTools(NativeTools):
    """Memoize validated outputs keyed by pinned executable SHA and exact input.

Source binding and campaign deadline still apply on a hit. NativeTools performs
all existing subprocess/output validation on a miss. JSON bytes and DSL strings
retain the original return types; no Python substitute canonicalizer is used.
"""
    def __init__(self,ref,deadline,*,cache_bytes=8*MIB,cache_entries=8192):
        super().__init__(ref,deadline);self._pinned_ref=dict(ref);digest(ref['sha256'])
        self.memo=ByteLRU(cache_bytes,cache_entries);self.source_check_seconds=0.
    def _key(self,mode,value):
        tick(self.deadline);need(self.ref==self._pinned_ref,'canonicalizer identity changed')
        body=canonical(value);need(len(body)<=2*MIB,'native input cap')
        start=time.monotonic();bound(self._pinned_ref);self.source_check_seconds+=time.monotonic()-start
        tick(self.deadline);return self._pinned_ref['sha256'],mode,body
    def canonical_json(self,value):
        key=self._key('--canonical-json',value);body=self.memo.get(key)
        if body is not None:
            need(json.loads(body)==value,'cached native JSON changed value');tick(self.deadline);return body
        start=time.monotonic()
        try:
            body=super().canonical_json(value);need(type(body)is bytes,'native JSON bytes');tick(self.deadline)
        except BaseException:self.memo.backend(time.monotonic()-start,failed=True);raise
        self.memo.backend(time.monotonic()-start,len(body));self.memo.put(key,body);return body
    def canonical_program(self,text):
        need(type(text)is str and 0<len(text.encode())<=32768,'bounded DSL text')
        key=self._key('--canonical-program',{'dsl':text});body=self.memo.get(key)
        if body is not None:
            result=body.decode('utf-8');need(0<len(body)<=2*MIB,'cached native DSL');tick(self.deadline);return result
        start=time.monotonic()
        try:
            result=super().canonical_program(text);need(type(result)is str,'native program text');tick(self.deadline)
        except BaseException:self.memo.backend(time.monotonic()-start,failed=True);raise
        body=result.encode();self.memo.backend(time.monotonic()-start,len(body));self.memo.put(key,body);return result
    def cache_stats(self):return {'schema':'jope.native-canonical-cache/v1','canonicalizer_sha256':self._pinned_ref['sha256'],'source_check_seconds':self.source_check_seconds,**self.memo.snapshot()}
