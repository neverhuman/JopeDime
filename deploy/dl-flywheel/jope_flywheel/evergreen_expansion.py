"""Persistent bounded expansion controller and command-line entry point."""
import argparse, gzip, json, os, sqlite3, time
from pathlib import Path
from .evergreen_source import (MAX_RAW, SCHEMA, TRANSFORM, byte_sha, canonical, digest,
    need, pinned, safe_path, source, write_new)
from .evergreen_recipes import generate

class Expansion:
    def __init__(self,root,parents,*,target,backlog,config_sha):
        self.root=safe_path(root);self.root.mkdir(parents=True,exist_ok=True)
        for name in ('variants','expansion.sqlite','expansion.sqlite-wal','expansion.sqlite-shm','progress.new','progress.json'):safe_path(self.root/name)
        need(0<target<=100000 and 1<=backlog<=2048 and len(parents)*256>=target,'growth bounds')
        need(len({p['parent_lineage_sha256']for p in parents})==len(parents),'one representative per unique lineage')
        self.parents=sorted(parents,key=lambda p:digest({'order':TRANSFORM,'lineage':p['parent_lineage_sha256']}))
        self.target,self.backlog,self.config_sha=target,backlog,config_sha
        self.db=sqlite3.connect(self.root/'expansion.sqlite',timeout=20,isolation_level=None);self.db.row_factory=sqlite3.Row
        self.db.execute('PRAGMA journal_mode=WAL');self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''CREATE TABLE IF NOT EXISTS config(id INTEGER PRIMARY KEY,sha TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS variants(sequence INTEGER PRIMARY KEY,parent TEXT NOT NULL,ordinal INTEGER NOT NULL,
        state TEXT NOT NULL,identity TEXT UNIQUE,content_sha TEXT UNIQUE,metadata_sha TEXT,error TEXT,created REAL NOT NULL,
        UNIQUE(parent,ordinal));''')
        identity=digest({'config_sha':config_sha,'parents':parents,'target':target,'backlog':backlog})
        self.db.execute('INSERT OR IGNORE INTO config VALUES(1,?)',(identity,))
        if self.db.execute('SELECT sha FROM config WHERE id=1').fetchone()[0]!=identity:
            self.db.close();raise ValueError('immutable controller config mismatch')
    def close(self):self.db.close()
    def snapshot(self):
        states={r[0]:r[1]for r in self.db.execute('SELECT state,count(*) FROM variants GROUP BY state')}
        return {'schema':SCHEMA,'at':time.time(),'target':self.target,'parent_lineages':len(self.parents),
                'states':states,'generated':sum(v for k,v in states.items()if k in ('generated','published','admitted')),
                'published':states.get('published',0)+states.get('admitted',0),'admitted':states.get('admitted',0),
                'backlog_limit':self.backlog,'config_sha256':self.config_sha}
    def reserve(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            rows=self.db.execute("SELECT count(*) FROM variants WHERE state IN ('reserved','generated','published')").fetchone()[0]
            if rows>=self.backlog:self.db.execute('COMMIT');return None
            # Reserved/unknown outcomes remain capacity-occupying until explicit reconciliation.
            seq=self.db.execute('SELECT coalesce(max(sequence)+1,0) FROM variants').fetchone()[0]
            ready=self.db.execute("SELECT count(*) FROM variants WHERE state IN ('reserved','generated','published','admitted')").fetchone()[0]
            if seq>=len(self.parents)*256 or ready>=self.target:self.db.execute('COMMIT');return None
            p=self.parents[seq%len(self.parents)];ordinal=seq//len(self.parents)
            self.db.execute('INSERT INTO variants(sequence,parent,ordinal,state,created) VALUES(?,?,?,?,?)',(seq,p['parent_lineage_sha256'],ordinal,'reserved',time.time()))
            self.db.execute('COMMIT');return seq,p,ordinal
        except BaseException:self.db.execute('ROLLBACK');raise
    def step(self):
        task=self.reserve()
        if task is None:return False
        seq,parent,ordinal=task
        try:
            x,y,f=source(parent);identity,raw,meta=generate(parent,ordinal,x,y,f)
            prior=self.db.execute('SELECT sequence FROM variants WHERE content_sha=?',(byte_sha(raw),)).fetchone()
            need(prior is None,'duplicate dataset content')
            dest=safe_path(self.root/'variants'/identity);dest.mkdir(parents=True,exist_ok=False)
            zipped=gzip.compress(raw,mtime=0);metadata=canonical(meta)
            for name,body in [('encoder.jtt1.gz',zipped),('metadata.json',metadata)]:
                write_new(dest/name,body)
            receipt={'schema':SCHEMA,'sequence':seq,'variant_identity':identity,'raw_sha256':byte_sha(raw),
                     'encoder_sha256':byte_sha(zipped),'metadata_sha256':byte_sha(metadata),'parent_lineage_sha256':parent['parent_lineage_sha256'],
                     'config_sha256':self.config_sha,'at':time.time()}
            write_new(dest/'generation.json',canonical(receipt))
            self.db.execute("UPDATE variants SET state='generated',identity=?,content_sha=?,metadata_sha=? WHERE sequence=? AND state='reserved'",(identity,byte_sha(raw),byte_sha(metadata),seq))
        except BaseException as exc:
            self.db.execute("UPDATE variants SET state='failed',error=? WHERE sequence=? AND state='reserved'",(type(exc).__name__+': '+str(exc)[:400],seq));raise
        return True

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True);ap.add_argument('--config-sha',required=True);ap.add_argument('--once',action='store_true');a=ap.parse_args()
    conf=json.loads(pinned({'path':a.config,'sha256':a.config_sha},32*1024**2));need(conf['schema']==SCHEMA and conf['approved']is True,'authorized config required')
    need(conf['host']=='xbabe2' and os.uname().nodename.split('.')[0]=='xbabe2','dedicated intake host required')
    need(time.time()<conf['deadline_unix'],'controller deadline expired')
    for ref in conf['source_pins']:pinned(ref)
    body=pinned(conf['parent_registry'],32*1024**2);parents=[json.loads(s)for s in body.splitlines()]
    controller=Expansion(conf['out'],parents,target=conf['target'],backlog=conf['backlog_limit'],config_sha=a.config_sha)
    try:
        while time.time()<conf['deadline_unix']:
            for ref in conf['authority_pins']:pinned(ref)
            load=os.getloadavg()[0];mem={l.split(':')[0]:int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.split(':')[0]in('MemAvailable','MemTotal')}
            disk=os.statvfs(conf['out']);free=disk.f_bavail*disk.f_frsize
            held=load>conf['max_host_load'] or mem['MemAvailable']<conf['min_available_memory_bytes'] or free<conf['min_free_disk_bytes']
            advanced=False
            if not held:advanced=controller.step()
            value={**controller.snapshot(),'state':'resource_wait' if held else ('growing' if advanced else 'downstream_backpressure_or_target'),
                   'load1':load,'available_memory':mem['MemAvailable'],'free_disk':free,'deadline_unix':conf['deadline_unix']}
            out=safe_path(conf['out']);tmp=safe_path(out/'progress.new');destination=safe_path(out/'progress.json');write_new(tmp,canonical(value));os.replace(tmp,destination)
            print(json.dumps({k:value[k]for k in ('at','state','generated','published','admitted','target')}),flush=True)
            if a.once:break
            time.sleep(min(1 if advanced else 30,max(0,conf['deadline_unix']-time.time())))
    finally:controller.close()
if __name__=='__main__':main()
