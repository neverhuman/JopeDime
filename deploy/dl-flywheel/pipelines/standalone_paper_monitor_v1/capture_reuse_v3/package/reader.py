"""Local exact-reader source, never remote IO, process control or mutation.

This reader has no production CLI. Future immutable operator configuration must
supply ExternalSeal independently; request JSON cannot select it. All metadata
counters here concern this reader only, not an invented whole-process RSS bound.
"""
import os
import stat
from pathlib import Path
from .vendor.common import BODY_LIMIT, FRESH_LIMIT, TRANSPORT_LIMIT, digest, document, exact_body, file_pin, need
from .reuse import ExternalSeal, ReusedHistory, verify_fresh_manifest

def _safe(path):
    p=Path(path)
    need(not any(x.is_symlink()for x in (p,*p.parents)),'symlink path refused')
    return p

def _identity(s):
    return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)

def _expected(ref):
    return (ref['device'],ref['inode'],ref['bytes'],ref['mtime_ns'])

def _inspect(ref,immutable=False):
    file_pin(ref,immutable=immutable);p=_safe(ref['path']);s=p.lstat()
    need(stat.S_ISREG(s.st_mode)and _identity(s)==_expected(ref),'physical declared identity drift')
    if immutable:
        need(stat.S_IMODE(s.st_mode)==ref['mode']and s.st_uid==ref['uid']and s.st_nlink==ref['nlink'],'external immutable source mode/owner/link')
    return p,s

def _read(ref,remaining,immutable=False):
    need(type(remaining)is int and ref['bytes']<=remaining,'remaining read budget before open')
    p,a=_inspect(ref,immutable);fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW)
    try:
        first=os.fstat(fd);need(stat.S_ISREG(first.st_mode)and _identity(first)==_identity(a),'opened same declared inode')
        if immutable:need((first.st_uid,stat.S_IMODE(first.st_mode),first.st_nlink)==(ref['uid'],ref['mode'],ref['nlink']),'opened immutable authority')
        parts=[];left=ref['bytes']
        while left:
            raw=os.read(fd,min(65536,left));need(raw,'short exact body');parts.append(raw);left-=len(raw)
        last=os.fstat(fd);_,current=_inspect(ref,immutable)
        need(_identity(first)==_identity(last)==_identity(current)and first.st_ctime_ns==last.st_ctime_ns==current.st_ctime_ns,'body changed during same-FD read')
        wire=b''.join(parts);exact_body(wire,ref);return wire
    finally:os.close(fd)

def read_external_seal(external):
    need(type(external)is ExternalSeal,'external operator anchor, not request')
    ref=file_pin(document(external.physical_pin_json),immutable=True)
    wire=_read(ref,1024*1024,True)
    # Exact immutable pin is returned only after actual same-FD local readback.
    return wire,dict(ref),dict(ref)

class ReadSession:
    """One in-process budget ledger, no retries or release of retained charges.

    A qualified future owner must enforce one session per durable external intent.
    This Python object is not a defense against an operator constructing a second
    session. No process/global enforcement or live qualification is claimed here.
    """
    def __init__(self,history):
        need(type(history)is ReusedHistory,'authenticated exact history type before callbacks or attributes')
        history.check();self.history=history
        self.read_charge=history.metadata_bytes_verified
        self.retained_charge=history.metadata_bytes_verified
        self.transport_lower_bound=history.metadata_bytes_verified
        self.fresh_attempted=False;self.historical_attempts=set();self.failed=False
    def _charge(self,n,transport):
        need(not self.failed,'failed session held, no retry')
        need(self.read_charge+n<=FRESH_LIMIT and self.retained_charge+n<=FRESH_LIMIT and
             self.transport_lower_bound+transport<=TRANSPORT_LIMIT,'shared packet budget exhausted before IO')
        self.read_charge+=n;self.retained_charge+=n;self.transport_lower_bound+=transport
    def capture_fresh(self,request_wire):
        need(not self.fresh_attempted and not self.failed,'one fresh capture attempt per session')
        self.fresh_attempted=True
        try:
            plan=verify_fresh_manifest(request_wire,self.history);members=plan['manifest']['members']
            self._charge(len(request_wire),len(request_wire))
            projected=sum(r['bytes']for r in members)
            need(self.read_charge+projected<=FRESH_LIMIT and self.retained_charge+projected<=FRESH_LIMIT,'complete session projection before payload IO')
            for r in members:_inspect(r)
            rows=[]
            for r in members:
                self._charge(r['bytes'],4*((r['bytes']+2)//3))
                wire=_read(r,r['bytes']);rows.append((r['path'],wire))
            return {'state':'FRESH_BYTES_VERIFIED_HISTORY_REUSED_NOT_REREAD',
                'read_reserved_bytes':self.read_charge,'retained_charge_bytes':self.retained_charge,
                'historical_bytes_not_read':self.history.historical_bytes_not_read,
                'historical_members_preserved':self.history.original_member_count,
                'body_wires':tuple(rows),'monitor_or_publication_acceptance':False,
                'RSS_qualified':False,'serialization_qualified':False,'retry_allowed':False}
        except BaseException:
            self.failed=True;raise
    def later_body_use(self,namespace,relative_path):
        key=(namespace,relative_path)
        need(not self.failed and key not in self.historical_attempts,'one exact historical use attempt, no retry')
        self.historical_attempts.add(key)
        try:
            _,graph=self.history.check();matches=[m for n in graph['namespaces']if n['id']==namespace
                for m in n['members']if m['relative_path']==relative_path]
            need(len(matches)==1,'exact named historical body required');r=matches[0]['custody']
            self._charge(r['bytes'],4*((r['bytes']+2)//3))
            return _read(r,r['bytes'],True)
        except BaseException:
            self.failed=True;raise
