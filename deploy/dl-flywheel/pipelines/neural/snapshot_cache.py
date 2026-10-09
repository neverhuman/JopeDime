"""Reuse only locally present bytes of already admitted exact S3 versions.
No new trust in a pathname: requested VersionId/SHA/length must be in the exact
previously verified inventory, and every local read rechecks bytes and nofollow.
"""
from pathlib import Path
from jope_flywheel.program_io import bound,need,tick
import flywheel_kumos_snapshot as snapshot

class VerifiedLocalStore(snapshot.Store):
 def __init__(self,client,native,deadline,*,accepted_inventory,local_binary_refs):
  super().__init__(client,native,deadline)
  self.accepted={};self.local={};self.local_bytes_read=0;self.local_hits=0;self.remote_bytes_read=0
  need(len(accepted_inventory)<=4096 and len(local_binary_refs)<=4096,'bounded immutable cache inventories')
  for source in accepted_inventory:
   ref=snapshot.S.normalized_ref(source);key=(ref['bucket'],ref['key'],ref['version_id'])
   need(key not in self.accepted,'duplicate admitted exact version');self.accepted[key]=ref
  for ref in local_binary_refs:
   # Config contains only exact local emitted binaries from the accepted campaign.
   path=Path(ref['path']);need(path.name=='artifact-rust_executable'and path.parent.name.startswith('gp-evaluate-'),'qualified local binary locator')
   pair=(ref['sha256'],ref['bytes']);self.local.setdefault(pair,ref)
 def get_version(self,source):
  tick(self.deadline);ref=snapshot.S.normalized_ref(source);key=(ref['bucket'],ref['key'],ref['version_id'])
  need(self.accepted.get(key)==ref,'cache read outside prior admitted exact-version inventory')
  if key not in self.cache:
   local=self.local.get((ref['sha256'],ref['bytes']))
   if local is not None:
    # Preserve inherited namespace/version/budget checks before inserting cache.
    need(ref['bucket']==snapshot.BUCKET and ref['key']==snapshot.PREFIX+'program-results/flywheel/v1/objects/'+ref['sha256']+'/rust_executable','local cache only emitted executable artifact namespace')
    need(self.bytes_read+ref['bytes']<=snapshot.S.MAX_READ,'total read cap')
    body=bound(local,snapshot.S.MAX_OBJECT);tick(self.deadline)
    need(len(body)==ref['bytes'],'cache exact bytes')
    self.cache[key]=body;self.bytes_read+=len(body);self.local_bytes_read+=len(body);self.local_hits+=1
  before=self.bytes_read;value=super().get_version(ref);self.remote_bytes_read+=self.bytes_read-before
  return value
