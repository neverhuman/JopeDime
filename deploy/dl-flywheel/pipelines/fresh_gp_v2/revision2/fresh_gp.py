#!/usr/bin/env python3
"""Pinned cold preparation, broker and bounded fresh128 CPU execution CLI.

All package files are hashed before importing any proposal module. CLI requires
the exact source manifest SHA supplied by the reviewing deployment controller.
"""
import argparse,hashlib,json,os,sys
from pathlib import Path
sys.dont_write_bytecode=True

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-manifest',required=True);p.add_argument('--source-sha',required=True)
    p.add_argument('action',choices=['history-spec','collect','prepare','freeze','probe','broker','relay','install-physical','export-proof','install-proof','preflight','init-registry','run'])
    p.add_argument('--ref',required=True);p.add_argument('--sha',required=True);p.add_argument('--out');p.add_argument('--shard')
    p.add_argument('--preflight-ref');p.add_argument('--preflight-sha');p.add_argument('--body-stdin',action='store_true')
    p.add_argument('--once',action='store_true');p.add_argument('--physical-only',action='store_true')
    a=p.parse_args();root=Path(__file__).absolute().parent
    def checked(path,sha,cap):
        q=Path(path)
        if not q.is_absolute()or any(x.is_symlink()for x in(q,*q.parents))or not q.is_file()or q.stat().st_size>cap:raise ValueError('unsafe source manifest/file')
        raw=q.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('source hash differs')
        return raw
    manifest=json.loads(checked(a.source_manifest,a.source_sha,4*1024**2))
    if manifest['schema']!='jope.fresh128-source/v2':raise ValueError('source manifest schema')
    files=manifest['files'];paths={r['path']for r in files}
    actual={str(q.absolute())for q in root.rglob('*.py')if not q.name.startswith('test_')and'borrowed'not in q.parts}
    if paths!=actual:raise ValueError('full executable package source closure differs')
    for r in files:checked(r['path'],r['sha256'],64*1024**2)
    sys.path[:0]=[str(root),str(root/'runtime')]
    from common import ref,obj,canonical,need
    source={'path':a.source_manifest,'sha256':a.source_sha,'bytes':Path(a.source_manifest).stat().st_size}
    reference={'path':a.ref,'sha256':a.sha,'bytes':Path(a.ref).stat().st_size};request=obj(reference)
    # A pinned entrypoint cannot be paired with another mutable package through
    # a request argument. All operational requests retain this source manifest.
    if a.action in ('prepare',):need(request['source_manifest']==source,'same source package')
    if a.action=='history-spec':
        from make_requests import finalize_history
        result=finalize_history(request['template'],request['coordination'],a.out)
    elif a.action=='collect':
        from history_collect import collect
        result=collect(reference,a.out)
    elif a.action=='prepare':
        from prepare import prepare
        from jope_flywheel.s3_library import client
        class ReadOnly:
            def __init__(self,c):self.c=c
            def get_object(self,**kw):return self.c.get_object(**kw)
        result=prepare(reference,a.out,ReadOnly(client('veox-jepson-prod')))
    elif a.action=='freeze':
        from epoch import freeze
        result=freeze(reference,a.out)
    elif a.action=='probe':
        from transport import probe
        result=probe(reference,hashlib.sha256((root/'transport.py').read_bytes()).hexdigest())
    elif a.action=='broker':
        from broker import serve
        serve(reference);result={'state':'broker_closed'}
    elif a.action=='relay':
        from transport import relay
        result=relay(reference,once=a.once,physical_only=a.physical_only)
    elif a.action in('install-physical','install-proof'):
        need(a.body_stdin,'bounded proof stdin required');raw=sys.stdin.buffer.read(1024*1024+1);need(len(raw)<=1024*1024,'proof stdin cap')
        import transport
        result=getattr(transport,a.action.replace('-','_'))(reference,json.loads(raw))
    elif a.action=='export-proof':
        from transport import export_proof
        result=export_proof(reference)
    elif a.action=='preflight':
        from executor import preflight
        result=preflight(reference,a.shard,a.out)
    elif a.action=='init-registry':
        from executor import initialize_registry
        result=initialize_registry(reference)
    else:
        from executor import run
        need(a.preflight_ref and a.preflight_sha,'fresh preflight ref required')
        result=run(reference,a.shard,{'path':a.preflight_ref,'sha256':a.preflight_sha,'bytes':Path(a.preflight_ref).stat().st_size})
    sys.stdout.buffer.write(canonical(result)+b'\n')

if __name__=='__main__':main()
