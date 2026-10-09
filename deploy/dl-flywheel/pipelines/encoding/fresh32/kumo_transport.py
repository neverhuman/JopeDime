"""Bounded create-only stream of a fixed packet index, never an arbitrary tree."""
import argparse,hashlib,io,json,os,sys,tarfile,time
from pathlib import Path
from campaign import ROOT
CAP=512*1024**2

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(b):return hashlib.sha256(b).hexdigest()
def guard(path):
 p=Path(path);need(p.is_absolute()and all(not q.is_symlink()for q in(p,*p.parents)),'unsafe path');return p

def index(raw,digest):
 need(len(raw)<=1024**2 and sha(raw)==digest,'stage index hash/cap');v=json.loads(raw);root=guard(v['root']);need(root.parent==Path('/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final')and str(root)==ROOT,'owned packet root')
 entries={}
 for ref in v['files']:
  p=guard(ref['path']);rel=p.relative_to(root).as_posix();need(rel not in entries and'..'not in Path(rel).parts and type(ref['bytes'])is int and 0<=ref['bytes']<=32*1024**2,'unique bounded stage file');entries[rel]=ref
 need(len(entries)<=256 and sum(r['bytes']for r in entries.values())==v['bytes']<=CAP,'stage total cap');return root,entries

def send(path,digest,deadline):
 raw=guard(path).read_bytes();root,entries=index(raw,digest);out=sys.stdout.buffer;out.write(str(len(raw)).encode()+b'\n'+raw);out.flush()
 with tarfile.open(fileobj=out,mode='w|')as archive:
  for rel,ref in entries.items():
   need(time.time()<deadline,'stage deadline');p=guard(ref['path']);need(p.is_file()and p.stat().st_size==ref['bytes'],'exact stage source');body=p.read_bytes();need(sha(body)==ref['sha256'],'stage source SHA');info=tarfile.TarInfo(rel);info.size=len(body);info.mode=0o600;archive.addfile(info,io.BytesIO(body))
 need(time.time()<deadline,'stage deadline')

def receive(digest,deadline):
 stream=sys.stdin.buffer;line=stream.readline(32);need(line.endswith(b'\n')and line[:-1].isdigit(),'index frame');n=int(line);need(0<n<=1024**2,'index cap');raw=stream.read(n);root,entries=index(raw,digest);need(not root.exists(),'fresh destination root');root.mkdir(mode=0o700);seen=set()
 with tarfile.open(fileobj=stream,mode='r|')as archive:
  for member in archive:
   need(time.time()<deadline,'stage deadline');need(member.name in entries and member.name not in seen and member.isfile(),'unexpected tar member');ref=entries[member.name];need(member.size==ref['bytes'],'tar size');p=guard(root/member.name);p.parent.mkdir(parents=True,exist_ok=True);source=archive.extractfile(member);h=hashlib.sha256();count=0
   with p.open('xb')as dest:
    while True:
     block=source.read(1024**2)
     if not block:break
     need(time.time()<deadline,'stage deadline');count+=len(block);need(count<=ref['bytes'],'tar grew');h.update(block);dest.write(block)
    dest.flush();os.fsync(dest.fileno())
   need(count==ref['bytes']and h.hexdigest()==ref['sha256'],'copied payload mismatch');seen.add(member.name)
 need(seen==set(entries)and time.time()<deadline,'incomplete stage/deadline')
 with(root/'stage-index.json').open('xb')as f:f.write(raw)
 print(json.dumps({'state':'stage_complete','root':str(root),'files':len(seen),'bytes':sum(r['bytes']for r in entries.values()),'index_sha256':digest}))

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['send','receive']);p.add_argument('--index');p.add_argument('--index-sha',required=True);p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
 if a.action=='send':send(a.index,a.index_sha,a.deadline)
 else:receive(a.index_sha,a.deadline)
if __name__=='__main__':main()
