"""Create-once broker custody; never reconstruct missing permanent state.

DB identity and the lock/directory inode are externally pinned by bootstrap.
Each committed reservation also has a create-only file anchor. Missing DB,
replacement, rollback/truncation or incomplete DB/file commit refuses service.
"""
import fcntl,json,os,sqlite3,stat,time
from pathlib import Path
from common import *

ROOT=Path('/home/ubuntu/JopeDime/.agent/proposals/dl-flywheel-fresh-context-broker-20261009-v1')
APP_ID=1246713651
SCHEMA='''CREATE TABLE config(id INTEGER PRIMARY KEY CHECK(id=1),body TEXT NOT NULL,sha TEXT NOT NULL);
CREATE TABLE epochs(sha TEXT PRIMARY KEY,body TEXT NOT NULL,started REAL NOT NULL);
CREATE TABLE contexts(context_id TEXT PRIMARY KEY,variant TEXT UNIQUE NOT NULL,epoch TEXT NOT NULL,descriptor_sha TEXT NOT NULL,partition TEXT NOT NULL);
CREATE TABLE events(seq INTEGER PRIMARY KEY,body TEXT NOT NULL,prev_sha TEXT NOT NULL,sha TEXT NOT NULL);
'''
POLICY={'schema':'jope.permanent-variant-broker-policy/v3','root':str(ROOT),'database':'registry.sqlite',
        'lock':'epoch.lock','whole_context_holds_permanent':True,'automatic_recreation':False,
        'application_id':APP_ID,'user_version':3,'journal_mode':'wal','synchronous':2}

def identity(p):
    p=Path(p);need(all(not q.is_symlink()for q in(p,*p.parents)),'permanent path symlink')
    st=p.stat();return {'device':st.st_dev,'inode':st.st_ino,'mode':stat.S_IFMT(st.st_mode)}
def syncdir(p):
    fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
def schema_rows(db):return [list(r)for r in db.execute("SELECT type,name,tbl_name,sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'ORDER BY name")]
def state(db):return {n:[dict(r)for r in db.execute('SELECT *FROM '+n+' ORDER BY 1')]for n in('config','epochs','contexts')}
def event(db,body):
    row=db.execute('SELECT seq,sha FROM events ORDER BY seq DESC LIMIT 1').fetchone();seq=1 if row is None else row['seq']+1;prev='0'*64 if row is None else row['sha']
    body={**body,'state_sha256':digest(state(db))};head=digest({'seq':seq,'body':body,'previous_sha256':prev})
    db.execute('INSERT INTO events VALUES(?,?,?,?)',(seq,canonical(body).decode(),prev,head))
    return {'seq':seq,'body':body,'previous_sha256':prev,'sha256':head}
def anchor_event(root,e):
    return put(Path(root)/'commits','%08d-%s.json'%(e['seq'],e['sha256']),e)

def bootstrap(request_ref,*,hostname=None,fixture_root=None):
    if hostname is None:host('xbabe2')
    else:need(hostname=='xbabe2','bootstrap authority host')
    r=obj(request_ref);need(r['schema']=='jope.fresh128-broker-bootstrap-request/v3'and r['create_once_authorized']is True and r['broker_root']==str(ROOT),'explicit new permanent bootstrap authorization')
    source_closure(r['source_manifest']);root=Path(fixture_root or ROOT)
    # Existence of ANY earlier directory, including incomplete bootstrap, stops
    # this operation. It cannot create another epoch's empty replacement DB.
    root.mkdir(mode=0o700,parents=False,exist_ok=False);(root/'commits').mkdir(mode=0o700)
    lock=os.open(root/'epoch.lock',os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fd=os.open(root/'registry.sqlite',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.close(fd)
        db=sqlite3.connect((root/'registry.sqlite').as_uri()+'?mode=rw',uri=True,isolation_level=None,timeout=.5);db.row_factory=sqlite3.Row
        try:
            db.execute('PRAGMA journal_mode=WAL');db.execute('PRAGMA synchronous=FULL');db.execute('PRAGMA application_id='+str(APP_ID));db.execute('PRAGMA user_version=3');db.executescript(SCHEMA)
            config={**POLICY,'bootstrap_nonce':os.urandom(32).hex(),'source_manifest':r['source_manifest']}
            db.execute('BEGIN IMMEDIATE');db.execute('INSERT INTO config VALUES(1,?,?)',(canonical(config).decode(),digest(config)));genesis=event(db,{'kind':'bootstrap','config':config,'config_sha256':digest(config)});db.execute('COMMIT')
            gr=anchor_event(root,genesis);syncdir(root/'commits')
            result={'schema':'jope.permanent-variant-broker-bootstrap/v3','root':str(root),'policy':POLICY,'config_sha256':digest(config),
                'schema_sha256':digest(schema_rows(db)),'directory_identity':identity(root),'database_identity':identity(root/'registry.sqlite'),
                'lock_identity':identity(root/'epoch.lock'),'genesis':gr,'created_unix':time.time(),'automatic_recreation':False}
            receipt=put(root,'bootstrap.json',result);syncdir(root);return receipt
        finally:db.close()
    finally:os.close(lock)

class Registry:
    def __init__(self,bootstrap_ref,head_anchor,*,locked=True,readonly=False,fixture_root=None):
        self.bootstrap_ref=bootstrap_ref;self.head_anchor=head_anchor;self.a=obj(bootstrap_ref);self.root=Path(self.a['root']);self.lock=None;self.db=None
        need(self.a['schema']=='jope.permanent-variant-broker-bootstrap/v3'and self.a['policy']==POLICY and
             self.a['automatic_recreation']is False and(self.root==ROOT or self.root==fixture_root),'fixed permanent broker identity')
        need(Path(bootstrap_ref['path'])==self.root/'bootstrap.json','fixed bootstrap receipt location')
        self.check_files()
        try:
            if locked:
                self.lock=os.open(self.root/'epoch.lock',os.O_RDWR|os.O_NOFOLLOW)
                fcntl.flock(self.lock,fcntl.LOCK_EX|fcntl.LOCK_NB);self.check_files()
            mode='ro'if readonly else'rw'
            self.db=sqlite3.connect((self.root/'registry.sqlite').as_uri()+'?mode='+mode,uri=True,isolation_level=None,timeout=.5)
            self.db.row_factory=sqlite3.Row
            if readonly:self.db.execute('PRAGMA query_only=ON')
            self.verify()
        except BaseException:self.close();raise
    def close(self):
        if self.db is not None:self.db.close();self.db=None
        if self.lock is not None:os.close(self.lock);self.lock=None
    def check_files(self):
        bound(self.bootstrap_ref);bound(self.head_anchor,4*MIB)
        for name,key in(('', 'directory_identity'),('registry.sqlite','database_identity'),('epoch.lock','lock_identity')):
            need(identity(self.root/name)==self.a[key],'permanent broker path missing/replaced')
        for suffix in('-wal','-shm','-journal'):
            p=self.root/('registry.sqlite'+suffix)
            need(not p.is_symlink(),'permanent broker sidecar symlink')
            if p.exists():need(stat.S_ISREG(p.stat().st_mode),'permanent broker sidecar not regular')
        need((self.root/'commits').is_dir()and not(self.root/'commits').is_symlink(),'permanent commit anchors missing')
    def verify(self):
        self.check_files();db=self.db;own=not db.in_transaction
        if own:db.execute('BEGIN')
        try:
            need(db.execute('PRAGMA application_id').fetchone()[0]==APP_ID and db.execute('PRAGMA user_version').fetchone()[0]==3 and
                 db.execute('PRAGMA journal_mode').fetchone()[0]=='wal'and db.execute('PRAGMA synchronous').fetchone()[0]==2,'permanent journal/config identity')
            need(digest(schema_rows(db))==self.a['schema_sha256'],'permanent registry schema changed')
            configs=db.execute('SELECT *FROM config').fetchall();need(len(configs)==1 and configs[0]['id']==1 and
                digest(json.loads(configs[0]['body']))==configs[0]['sha']==self.a['config_sha256'],'permanent config binding')
            prev='0'*64;seq=0;last=None;names=[];replayed={'config':[],'epochs':[],'contexts':[]};epoch_keys=set();context_keys=set();variant_keys=set()
            for row in db.execute('SELECT *FROM events ORDER BY seq'):
                seq+=1;need(seq<=4097,'bounded permanent epoch journal');body=json.loads(row['body']);h=digest({'seq':seq,'body':body,'previous_sha256':prev})
                need(row['seq']==seq and row['prev_sha']==prev and row['sha']==h,'permanent event chain')
                e={'seq':seq,'body':body,'previous_sha256':prev,'sha256':h};name='%08d-%s.json'%(seq,h);names.append(name)
                need(read(self.root/'commits'/name,4*MIB)==canonical(e),'permanent external event custody missing/changed')
                if seq==1:
                    need(set(body)=={'kind','config','config_sha256','state_sha256'}and body['kind']=='bootstrap'and
                         digest(body['config'])==body['config_sha256']==self.a['config_sha256'],'bootstrap event authority')
                    replayed['config']=[{'id':1,'body':canonical(body['config']).decode(),'sha':body['config_sha256']}]
                else:
                    need(set(body)=={'kind','epoch','epoch_sha256','started','contexts','state_sha256'}and
                         body['kind']=='epoch_reserved'and body['contexts']==128,'known permanent reservation event')
                    epoch=body['epoch'];key=body['epoch_sha256'];need(digest(epoch)==key and key not in epoch_keys,'unique pinned epoch event')
                    epoch_keys.add(key);ds=epoch['context_descriptors'];need(len(ds)==128,'complete128 event support')
                    replayed['epochs'].append({'sha':key,'body':canonical(epoch).decode(),'started':body['started']})
                    for d in ds:
                        need(d['context_id']not in context_keys and d['variant_identity']not in variant_keys and
                             d['partition']in('train','validation'),'permanent event context uniqueness/partition')
                        context_keys.add(d['context_id']);variant_keys.add(d['variant_identity'])
                        replayed['contexts'].append({'context_id':d['context_id'],'variant':d['variant_identity'],
                            'epoch':key,'descriptor_sha':d['descriptor_sha256'],'partition':d['partition']})
                    replayed['epochs'].sort(key=lambda x:x['sha']);replayed['contexts'].sort(key=lambda x:x['context_id'])
                need(digest(replayed)==body['state_sha256'],'event state cannot be reconstructed from custody')
                prev=h;last=body
            need(seq>=1 and obj(self.head_anchor)==e,'external exact prior head is missing, advanced or rewound')
            need(seq>=1 and last['state_sha256']==digest(state(db)),'permanent state does not match durable journal')
            need(sorted(p.name for p in(self.root/'commits').iterdir())==names,'permanent journal rolled back or extra unknown commit')
            need(obj(self.a['genesis'])==json.loads(read(self.root/'commits'/names[0])),'original bootstrap custody')
            need(replayed==state(db),'materialized ownership differs from event replay')
            self.check_files();return {'events':seq,'head_sha256':prev,'state_sha256':last['state_sha256'],
              'epochs':db.execute('SELECT count(*)FROM epochs').fetchone()[0],'contexts':len(replayed['contexts']),'bootstrap_sha256':self.bootstrap_ref['sha256']}
        finally:
            if own and db.in_transaction:db.execute('ROLLBACK')
    def reserve(self,epoch,epoch_sha,expected_head):
        need(self.lock is not None,'reservation requires permanent owned launch lock')
        need(self.verify()==expected_head,'permanent history advanced since frozen qualification')
        need(digest(epoch)==epoch_sha,'exact frozen epoch bytes')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            self.verify();need(self.db.execute('SELECT 1 FROM epochs WHERE sha=?',(epoch_sha,)).fetchone()is None,'epoch already activated; never restart')
            started=time.time();self.db.execute('INSERT INTO epochs VALUES(?,?,?)',(epoch_sha,canonical(epoch).decode(),started))
            for d in epoch['context_descriptors']:
                self.db.execute('INSERT INTO contexts VALUES(?,?,?,?,?)',(d['context_id'],d['variant_identity'],epoch_sha,d['descriptor_sha256'],d['partition']))
            e=event(self.db,{'kind':'epoch_reserved','epoch':epoch,'epoch_sha256':epoch_sha,'started':started,'contexts':128});self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:self.db.execute('ROLLBACK')
            raise
        # A crash here is an explicit incomplete permanent journal, never an
        # invitation to reconstruct the file or retry any reserved context.
        self.head_anchor=anchor_event(self.root,e);syncdir(self.root/'commits');return self.verify()
