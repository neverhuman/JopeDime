"""Bounded fsynced owned-output attempt custody; no remote or database API."""
import json
import os
from pathlib import Path
import time
from .readonly_io import canonical,sha,need

class EventSink:
    def __init__(self,root,config_sha256):
        root=Path(root)
        need(root.is_absolute() and '..'not in root.parts and all(not x.is_symlink()for x in(root,*root.parents)),'owned output nonsymlink path')
        need(root.parent.is_dir() and not root.exists(),'new owned output directory required')
        root.mkdir(mode=0o700)
        parent_fd=os.open(root.parent,os.O_RDONLY)
        try:os.fsync(parent_fd)
        finally:os.close(parent_fd)
        self.root=root;self.config_sha256=config_sha256
        self.fd=os.open(root/'events.jsonl',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_APPEND|os.O_NOFOLLOW,0o600)
        self.seq=0;self.head='0'*64;self.bytes=0;self.closed=False
        directory=os.open(root,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
        self.emit('configuration.started',{'config_sha256':config_sha256,'source_scope':'GET/canonicalization only; no full cutoff/export'})
    def emit(self,kind,payload):
        need(not self.closed and type(kind)is str and 0<len(kind)<=128,'open event sink/type')
        need(self.seq<(8192 if kind in('qualification.failed','qualification.passed')else 8190),'durable event count cap')
        body={'schema':'jope.generated-target-qualification-event/v1','seq':self.seq+1,'previous_sha256':self.head,
              'config_sha256':self.config_sha256,'utc_ns':time.time_ns(),'monotonic_ns':time.monotonic_ns(),'kind':kind,'payload':payload}
        raw=canonical(body);need(len(raw)<=16384 and self.bytes+len(raw)+1<=4*1024**2,'durable event byte cap')
        wire=raw+b'\n';at=0
        while at<len(wire):at+=os.write(self.fd,wire[at:])
        os.fsync(self.fd);self.seq+=1;self.head=sha(raw);self.bytes+=len(wire)
    def snapshot(self):
        return {'events':self.seq,'head_sha256':self.head,'bytes':self.bytes,'config_sha256':self.config_sha256}
    def finish(self,result):
        need(not self.closed,'one final owned receipt')
        result=dict(result);result['durable_event_head']=self.snapshot();result['own_output_writes']=True
        raw=canonical(result);need(len(raw)<=8*1024**2,'final receipt output cap')
        fd=os.open(self.root/'result.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        try:
            at=0
            while at<len(raw):at+=os.write(fd,raw[at:])
            os.fsync(fd)
        finally:os.close(fd)
        directory=os.open(self.root,os.O_RDONLY)
        try:os.fsync(directory)
        finally:os.close(directory)
        os.close(self.fd);self.closed=True
        return result
