"""Exact parent-encoder gzip custody. No general annotated-pin allowance.

The four-field pin is accepted only from the replayed parent's immutable registry
record. Generic source/config/normalizer/other artifacts keep read_bound's strict
2/3-field contract. Both compressed bytes and bounded decompressed payload are
checked; no annotation is silently discarded.
"""
import gzip
import io
from .readonly_io import need,hash64,read_bound,sha

MAX_COMPRESSED = 16*1024**2
MAX_RAW = 16*1024**2  # Exact existing evergreen source() / MAX_RAW bound.


def recheck(local_pins,parent_encoder_pins,parents,replayed_lineages):
    """Recheck strict generic pins and separately derived, role-bound encoders."""
    for ref in local_pins:
        read_bound(ref)
    need(type(parent_encoder_pins)is dict and set(parent_encoder_pins)==set(replayed_lineages),'exact replayed parent encoder roster')
    for lineage,ref in parent_encoder_pins.items():
        need(lineage in parents and parents[lineage]['parent_lineage_sha256']==lineage,'registered parent identity')
        parent=parents[lineage]
        need(ref==parent['encoder'],'parent encoder differs from pinned registry role')
        need(type(ref)is dict and set(ref)=={'path','sha256','bytes','raw_sha256'}
             and hash64(ref['sha256']) and hash64(ref['raw_sha256'])
             and type(ref['bytes'])is int and 0<ref['bytes']<=MAX_COMPRESSED,'closed role-specific parent gzip pin')
        need(ref['raw_sha256']==parent['raw_payload_sha256'],'parent raw hash annotation/registry mismatch')
        zipped=read_bound({k:ref[k]for k in('path','sha256','bytes')},MAX_COMPRESSED)
        with gzip.GzipFile(fileobj=io.BytesIO(zipped))as stream:
            raw=stream.read(MAX_RAW+1)
        need(0<len(raw)<=MAX_RAW,'bounded decompressed parent encoder')
        need(sha(raw)==ref['raw_sha256'],'decompressed parent raw SHA mismatch')
