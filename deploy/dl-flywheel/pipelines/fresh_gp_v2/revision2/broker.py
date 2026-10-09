"""One live xbabe2 ownership broker, permanent whole-context reservations.

The lock covers cooperating launchers. It does not magically stop foreign code:
an explicit pinned root fencing handoff and fresh physical probes are required.
"""
import fcntl,json,os,sqlite3,time
from pathlib import Path
from common import *
from epoch import BROKER_ROOT,handoff,execution_config
from role_authority import produce

def physical(epoch):
    pointer=json.loads(read(epoch['physical_pointer'],4096));bundle=obj(pointer,4*MIB)
    need(bundle['schema']=='jope.fresh128-physical-bundle/v2'and bundle['epoch_sha256']==epoch['_sha']and
         0<=time.time()-bundle['checked_at_unix']<90,'fresh actual physical fence proof required')
    want={r['sha256']:r for r in epoch['history_inventories']}
    need(len(bundle['probes'])==len(want)and {r['inventory']['sha256']for r in bundle['probes']}==set(want),'all hosts physically probed')
    for p in bundle['probes']:
        inv=obj(want[p['inventory']['sha256']])
        need(p['schema']=='jope.fresh128-physical-probe/v2'and p['inventory']==want[p['inventory']['sha256']]and
             p['host']==inv['host']and p['producer_sha256']==epoch['physical_producer_sha256']and 0<=time.time()-p['checked_at_unix']<90 and
             [v['unit']for v in p['units']]==inv['legacy_units'],'exact fresh declared physical probe')
        for v in p['units']:
            need(v['systemd']['MainPID']=='0'and v['systemd']['ActiveState']in('inactive','failed')and
                 all(g['exists']is False or(g['pids']==[]and g['populated']==0)for g in v['cgroups']),'legacy unit resurrected')
        need(len(p['controllers'])==len(inv['legacy_controllers'])and all(v['closed']is True and all(v[k]==old[k]for k in old)for v,old in zip(p['controllers'],inv['legacy_controllers'])),'controller still active')
    return bundle

def reserve(db,epoch,epoch_sha):
    db.execute('BEGIN IMMEDIATE')
    try:
        need(db.execute('SELECT 1 FROM epochs WHERE sha=?',(epoch_sha,)).fetchone()is None,'one-use epoch already activated; no restart')
        db.execute('INSERT INTO epochs VALUES(?,?,?)',(epoch_sha,canonical(epoch).decode(),time.time()))
        for d in epoch['context_descriptors']:
            db.execute('INSERT INTO contexts VALUES(?,?,?,?,?)',
                       (d['context_id'],d['variant_identity'],epoch_sha,d['descriptor_sha256'],d['partition']))
        db.execute('COMMIT')
    except BaseException:
        if db.in_transaction:db.execute('ROLLBACK')
        raise

def serve(epoch_ref,*,stop=None,hostname=None,broker_root=None,physical_reader=physical,role_reader=produce):
    if hostname is None:host('xbabe2')
    else:need(hostname=='xbabe2','broker on authority host')
    e=obj(epoch_ref);need(e['schema']=='jope.fresh128-epoch/v2','epoch schema');e['_sha']=epoch_ref['sha256']
    source_closure(e['source_manifest']);need(ref(__file__)['sha256']==e['broker_producer_sha256'],'actual broker source');p=obj(e['prepared']);handoff(e['fencing_handoff'],p,e['history_sources'])
    clock=Clock(e['work_deadline_unix'],1500);c=execution_config(epoch_ref)
    home=Path(broker_root or BROKER_ROOT);need(broker_root is not None or str(home)==e['broker_root'],'fixed broker scope')
    need(home.is_dir()and all(not x.is_symlink()for x in(home,*home.parents)),'precreated owned broker root')
    for name in('registry.sqlite','registry.sqlite-wal','registry.sqlite-shm','epoch.lock'):
        need(not(home/name).is_symlink(),'broker source symlink')
    lock=os.open(home/'epoch.lock',os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        db=sqlite3.connect(home/'registry.sqlite',isolation_level=None,timeout=.5)
        db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL')
        db.executescript('CREATE TABLE IF NOT EXISTS epochs(sha TEXT PRIMARY KEY,body TEXT NOT NULL,started REAL NOT NULL);CREATE TABLE IF NOT EXISTS contexts(context_id TEXT PRIMARY KEY,variant TEXT UNIQUE NOT NULL,epoch TEXT NOT NULL,descriptor_sha TEXT NOT NULL,partition TEXT NOT NULL);')
        # Prove closure and current roles before permanent ownership insertion.
        physical_reader(e);role_reader(c);reserve(db,{k:v for k,v in e.items()if k!='_sha'},e['_sha'])
        root=fresh(e['broker_out']);(root/'proofs').mkdir();seq=0;previous='0'*64
        try:
            while clock.remaining()>0:
                clock.tick();phys=physical_reader(e);roles=role_reader(c)
                # The original root fence source/config pins remain live, not a
                # one-time promise silently surviving source replacement.
                fence=obj(e['fencing_handoff'])
                for writer in fence['compatible_native_writers']:
                    bound(writer['dispatch_source']);bound(writer['config'])
                need(db.execute('SELECT count(*)FROM contexts WHERE epoch=?',(e['_sha'],)).fetchone()[0]==128,'permanent ownership complete')
                seq+=1;now=time.time()
                own={'schema':'jope.fresh-gp-context-ownership/v1','passed':True,'host':'xbabe3',
                     'campaign_sha256':e['_sha'],'producer_sha256':e['broker_producer_sha256'],
                     'history_census_sha256':digest(e['history_sources']),'selection_sha256':e['selection_sha256'],
                     'checked_at_unix':now,'lease_expires_unix':min(now+90,e['work_deadline_unix']),
                     'global_deadline_unix':GLOBAL,'legacy_variant_workers_closed':True,'legacy_launch_controllers_closed':True,
                     'active_xbabe2_source_kind':'native_s1_context','active_native_context_overlap':[],
                     'other_variant_context_overlap':[],'all_legacy_units_physically_closed':True,
                     'context_descriptors':e['context_descriptors'],'broker_host':'xbabe2','broker_pid':os.getpid(),
                     'broker_lock':str(home/'epoch.lock'),'physical_bundle_sha256':digest(phys),
                     'fencing_handoff_sha256':e['fencing_handoff']['sha256'],'epoch_sequence':seq}
                body={'schema':'jope.fresh128-broker-proof-bundle/v2','epoch_sha256':e['_sha'],
                      'ownership':own,'roles':roles,'sequence':seq,'previous_sha256':previous}
                previous=digest(body);r=put(root/'proofs',previous+'.json',body);atom(root/'current.json',canonical(r))
                timing(root,'ownership_refreshed',epoch=e['_sha'],sequence=seq,proof_sha256=r['sha256'])
                if stop and stop():break
                time.sleep(min(10,max(0,clock.remaining())))
        finally:
            put(root,'closed.json',{'epoch_sha256':e['_sha'],'last_sequence':seq,'last_sha256':previous,'at':time.time(),
                'context_reservations_retained':True,'automatic_restart':False});db.close()
    finally:os.close(lock)
