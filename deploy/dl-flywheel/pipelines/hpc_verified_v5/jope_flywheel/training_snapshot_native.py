"""Versioned, lazy native-context KumoS corpus; no optimizer/admission/publication.

load_snapshot(spec_bytes, expected_sha256, store, authority_bytes, now=...) reads
only explicit hash-bound refs. store.read(ref) returns bytes. Large per-context
S1 shards are read on demand by context_loader.get(context_ids, epoch,
batch_index). This is deliberately not an eager multi-million-row TrainData.
NativeBatch.data is None: the support-mask objective must consume real optional
labels directly, never invent y_train to make a winner-only Corpus admissible.

HWM winner DSL provides CE only, no inferred fitness->quality/runtime label.
Native detailed rows gain quality only after a registered exact teacher outer-train
prefix reconstructs the opposite-fold mean baselines from the shard's y_train.
All shard program identities participate in deterministic window rotation;
long/RAW/refused rows remain counted. Only the HWM winner is a CE teacher.
Historical native successor-v1, source shards, encoders and role files stay
unchanged. This source does not establish a fresh unseen holdout: retained
lineage95/5 is the existing flywheel development split.
"""
from __future__ import annotations
from collections import OrderedDict, Counter
from dataclasses import dataclass
import gzip, hashlib, importlib, io, json, math, struct, time
from pathlib import Path
import numpy as np
from .protocol import canonical, digest, partition, sha256

SCHEMA='jope.native-kumos-corpus/v1'
DIM=2051
MAX_META=64*1024**2
MAX_SHARD=16*1024**2
MAX_CACHE=64*1024**2
MAX_BATCH_CONTEXTS=128
MODULES=('dltrain.dsl_tokenizer','dltrain.labels','dltrain.automaton')


def need(ok, message):
    if not ok: raise ValueError(message)


def byte_sha(b): return hashlib.sha256(b).hexdigest()


def document(b):
    def pairs(items):
        out={}
        for k,v in items:
            need(k not in out,'duplicate JSON key');out[k]=v
        return out
    x=json.loads(b,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    canonical(x);return x


def ref_check(r, limit):
    need(type(r)is dict and set(r)=={'path','sha256','bytes'},'exact local custody ref required')
    need(type(r['path'])is str and Path(r['path']).is_absolute() and '..'not in Path(r['path']).parts,'absolute nontraversing ref')
    sha256(r['sha256']);need(type(r['bytes'])is int and 0<r['bytes']<=limit,'ref size cap')


class Reader:
    """No ambient path discovery; exact bytes checked even for a custom store."""
    def __init__(self, store): self.store=store;self.read_bytes=0
    def read(self,r,limit=MAX_META):
        ref_check(r,limit);b=self.store.read(r)
        need(type(b)is bytes and len(b)==r['bytes'] and byte_sha(b)==r['sha256'],'source bytes/SHA drift')
        self.read_bytes+=len(b);return b


class LocalStore:
    def __init__(self,roots): self.roots=tuple(Path(p)for p in roots)
    def read(self,r):
        p=Path(r['path']);need(any(p.is_relative_to(q)for q in self.roots),'path outside configured roots')
        need(all(not q.is_symlink()for q in (p,*p.parents)) and p.is_file(),'symlink/nonregular source')
        need(p.stat().st_size==r['bytes'],'source size drift')
        with p.open('rb')as f:b=f.read(r['bytes']+1)
        return b


@dataclass(frozen=True)
class NativeRecord:
    program_sha256:str
    tokens:tuple[int,...]
    outcome:str
    q:float|None=None
    q_opt:float|None=None
    q_grid:int|None=None
    feasibility:int|None=None
    runtime_seconds:float|None=None
    runtime_censored:bool=False
    ce_eligible:bool=False
    quality_eligible:bool=False
    pairwise_eligible:bool=False
    runtime_eligible:bool=False
    source_kind:str='hwm_winner_ce_only'
    skeleton_sha256:str|None=None


@dataclass(frozen=True)
class NativeBatch:
    contexts:dict
    encodings:dict
    records:dict
    support:dict
    selection_receipt:dict
    data:object=None


class NativeContextLoader:
    def __init__(self,snapshot): self.snapshot=snapshot;self.cache=OrderedDict();self.cache_bytes=0
    def _shard(self,cid):
        s=self.snapshot;desc=s.shards.get(cid)
        if desc is None:return None
        r=desc['shard'];key=(r['path'],r['sha256'])
        if key in self.cache:self.cache.move_to_end(key);return document(self.cache[key])
        b=s.reader.read(r,MAX_SHARD);x=document(b);c=x.get('context',{})
        need(x.get('schema')=='jope.dl-s1-tail-shard/v1' and c.get('context_sha256')==cid,'native shard identity')
        for k in ['view_id','dataset_hash','fold']:
            need(c.get(k)==s.contexts[cid][k],'native shard/context mismatch:'+k)
        need(x.get('source',{}).get('protocol')=='jope.s1-two-fold/v1' and x['source'].get('vocab_sha256')==s.vocab.sha256,'native shard protocol/vocab')
        need(len(x['records'])==desc['n_records'] and c.get('n_records')==desc['n_records'],'native shard record count')
        ids=[r['program_sha256']for r in x['records']];need(len(ids)==len(set(ids)),'duplicate native program')
        need(all(r['context_sha256']==cid for r in x['records']),'cross-context native record')
        # Do not read any of the historical encodings referenced by the shard.
        self.cache[key]=b;self.cache_bytes+=len(b)
        while self.cache_bytes>MAX_CACHE:
            _,old=self.cache.popitem(last=False);self.cache_bytes-=len(old)
        return x
    def _baseline(self,cid,x):
        s=self.snapshot;desc=s.shards[cid];inp=desc.get('native_input')
        if inp is None:return False
        cr=inp['context_json'];ctx=document(s.reader.read(cr,4*1024**2))
        need(ctx.get('context_sha256')==cid and ctx.get('view_id')==s.contexts[cid]['view_id'],'native input context identity')
        teacher=inp['teacher'];zipped=s.reader.read(teacher,MAX_SHARD)
        need(ctx.get('files',{}).get('teacher.jtt1.gz',{}).get('sha256')==teacher['sha256'],'native context/teacher binding')
        # Decode only the outer-TRAIN prefix. No outer-test X or y is decoded.
        with gzip.GzipFile(fileobj=io.BytesIO(zipped))as f:
            header=f.read(17);need(len(header)==17 and header[:4]==b'JTT1','native teacher header')
            n,nt,p,k=struct.unpack('<IIIB',header[4:]);need(k==2 and 2<=n<=2000 and 1<=p<=4096 and n*(p+1)*8<=64*1024**2,'native train prefix shape/cap')
            fb=f.read(n);tb=f.read(n*(p+1)*8);need(len(fb)==n and len(tb)==n*(p+1)*8,'truncated native train prefix')
        folds=np.frombuffer(fb,dtype=np.uint8);arr=np.frombuffer(tb,dtype='<f8').reshape(n,p+1)
        y=np.asarray(x['context']['y_train'],dtype='<f8')
        need(y.shape==(n,) and np.isfinite(y).all() and np.array_equal(y,arr[:,-1]),'native shard targets differ from exact teacher train prefix')
        need(set(folds.tolist())=={0,1},'native twofold membership')
        vals=[]
        for j in range(2):
            tr=y[folds!=j];va=y[folds==j];need(len(tr)>0 and len(va)>0,'empty native inner fold')
            mean=math.fsum(map(float,tr))/len(tr);vals.append(math.sqrt(math.fsum((float(v)-mean)**2 for v in va)/len(va)))
        for row in x['records']:
            bs=row.get('baseline_rmse_j')
            if bs is not None:need(len(bs)==2 and all(type(a)in(int,float)and math.isfinite(a)and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)for a,b in zip(bs,vals)),'opposite-fold mean baseline mismatch')
        return True
    def get(self,context_ids,*,epoch,batch_index):
        s=self.snapshot;need(type(epoch)is int and epoch>=0 and type(batch_index)is int and batch_index>=0,'nonnegative cursor')
        ids=list(context_ids);need(0<len(ids)<=MAX_BATCH_CONTEXTS and len(ids)==len(set(ids)),'bounded distinct contexts')
        need(all(c in s.contexts for c in ids),'context outside loaded corpus')
        records={};support={};members=[]
        for cid in ids:
            winner=s.winners[cid];x=self._shard(cid);selected=[];refused=[];total=0;baseline=False
            if x is not None:
                rows=sorted(x['records'],key=lambda r:digest({'domain':'native-program-rotation/v1','context':cid,'program':r['program_sha256']}))
                total=len(rows);baseline=self._baseline(cid,x)
                # Label full context before selecting a window: its baseline/ties do not depend on window membership.
                _,all_labs=s.labels.label_context(x['context'],rows)
                labs={a.program_sha256:a for a in all_labs}
                width=min(s.program_window,total);start=(epoch*s.program_window)%total if total else 0
                chosen=[rows[(start+j)%total]for j in range(width)]
                anchor=next((r for r in rows if r['program_sha256']==winner.program_sha256),None)
                if anchor is not None and anchor not in chosen:chosen.append(anchor)
                for row in chosen:
                    ph=sha256(row['program_sha256']);tokens=tuple(row['tokens']);reason=s.token_refusal(tokens)
                    if reason:refused.append({'program_sha256':ph,'reason':reason,'length':len(tokens)});continue
                    dsl=s.tokenizer.detokenize(tokens,s.vocab)
                    need(byte_sha(dsl.encode())==ph,'native token/program SHA mismatch')
                    lab=labs.get(ph);completed=row.get('outcome')=='completed';qok=baseline and lab is not None and lab.q is not None
                    rt=math.expm1(lab.rt_target)if lab is not None and lab.rt_valid else None
                    selected.append(NativeRecord(ph,tokens,row['outcome'],lab.q if qok else None,lab.q_opt if qok else None,lab.q_grid if qok else None,
                        lab.feas_class if lab is not None and lab.feas_valid else (0 if completed else None),rt,lab.rt_censored if lab else False,
                        ph==winner.program_sha256,qok,qok,rt is not None,'native_s1_detailed' if baseline else 'native_s1_baseline_pending',row.get('skeleton_sha256')))
            if winner.tokens and not any(r.program_sha256==winner.program_sha256 for r in selected):selected.append(winner)
            records[cid]=tuple(selected)
            support[cid]={'programs_declared':total,'programs_selected':len(selected),'hwm_winner_ce':bool(winner.tokens),
                'quality_baseline_reconstructed':baseline,'quality_pending_reason':None if baseline else 'native_teacher_train_prefix_not_verified',
                'selected_ce':sum(r.ce_eligible for r in selected),'selected_quality':sum(r.quality_eligible for r in selected),
                'selected_distinct_quality':len({r.q_grid for r in selected if r.q_grid is not None}),'refused_selected':refused,
                'all_programs_exposed':False,'rotation':{'schema':'native-program-rotation/v1','width':s.program_window,'epoch':epoch,'wraps_after_epochs':math.ceil(total/s.program_window)if total else 0}}
            members.append({'context_id':cid,'program_ids':[r.program_sha256 for r in selected],'refused':refused})
        receipt={'schema':'jope.native-program-window/v1','snapshot_sha256':s.identity,'epoch':epoch,'batch_index':batch_index,'members':members,'membership_sha256':digest(members)}
        return NativeBatch({c:s.contexts[c]for c in ids},{c:s.encoding(c)for c in ids},records,support,receipt)


class NativeSnapshot:
    def encoding(self,cid):return self.vectors[self.row_index[cid]]
    def legal_mask_fn(self,tokens):
        key=tuple(tokens)
        if key not in self._mask_cache:
            need(not self.token_refusal(key),'unsupported token sequence')
            table,ids=self.grammar.compile_masks([list(key)],self.automaton.Mode.SYNTAX)
            masks=table[ids[0][1:]].copy();masks.setflags(write=False)
            self._mask_cache[key]=masks
            while len(self._mask_cache)>512:self._mask_cache.popitem(last=False)
        return self._mask_cache[key]
    def token_refusal(self,tokens):
        if not tokens or len(tokens)<2:return 'invalid_sequence'
        if len(tokens)>self.max_seq:return 'over_model_max_seq'
        if any(type(t)is not int or not 0<=t<len(self.vocab.entries) for t in tokens):return 'invalid_token_id'
        if tokens[1]==self.tokenizer.RAW_BEGIN:return 'raw_form'
        if not self.grammar.accepts(list(tokens),self.automaton.Mode.SYNTAX).ok:return 'syntax_rejected'
        return None


def load_snapshot(spec_bytes,expected_sha256,store,*,authority_bytes,now=None):
    need(byte_sha(spec_bytes)==sha256(expected_sha256),'native spec SHA mismatch');spec=document(spec_bytes)
    need(set(spec)=={'schema','components','max_seq','program_window','consumer_source_sha256','vocab_sha256','grammar','authority_pins','bundle','parent_registry','inventory','input_lists','context_json_bytes','tails','prior_ancestry'},'closed native snapshot spec')
    need(spec.get('schema')==SCHEMA and spec.get('components')=={'kumo_tabular_s':DIM},'native KumoS-only spec')
    need(type(spec.get('max_seq'))is int and 16<=spec['max_seq']<=1024,'bounded model sequence length')
    need(type(spec.get('program_window'))is int and 1<=spec['program_window']<=32,'bounded program window')
    now=time.time()if now is None else now;auth=document(authority_bytes)
    need(auth.get('schema')=='jope.native-corpus-current-authority/v1'and auth.get('passed')is True,'fresh native authority proof')
    need(type(auth.get('checked_at_unix'))in(int,float)and 0<=now-auth['checked_at_unix']<=60,'native authority freshness')
    need(auth.get('source_pins')==spec['authority_pins'],'native authority source bindings')
    reader=Reader(store);target=document(reader.read(spec['authority_pins']['target']))
    need(set(spec['authority_pins'])>= {'target','role','layers'},'target/role/layers custody required')
    need(len(target['views'])==target['gp_target'] and len(set(target['views']))==len(target['views']),'native target closure')
    for name,ref in spec['authority_pins'].items():
        if name!='target':reader.read(ref)
    s=NativeSnapshot();s.reader=reader;s.identity=expected_sha256;s.program_window=spec['program_window'];s.max_seq=spec['max_seq'];s._mask_cache=OrderedDict()
    mods={}
    need(set(spec['consumer_source_sha256'])==set(MODULES),'exact native consumer pins')
    for name,pin in spec['consumer_source_sha256'].items():
        mod=importlib.import_module(name);need(byte_sha(Path(mod.__file__).read_bytes())==sha256(pin),'consumer module drift');mods[name]=mod
    s.tokenizer=mods[MODULES[0]];s.labels=mods[MODULES[1]];s.automaton=mods[MODULES[2]];s.vocab=s.tokenizer.Vocab.from_names({})
    need(s.vocab.sha256==spec['vocab_sha256'],'native vocab mismatch')
    gd=document(reader.read(spec['grammar']));need(gd.get('mode')=='SYNTAX'and gd.get('vocab_sha256')==s.vocab.sha256,'frozen syntax grammar')
    need(gd['grammar']['max_seq']==s.max_seq,'grammar/model sequence cap mismatch')
    s.grammar=s.automaton.Grammar(gd['grammar'],gd['signatures'],s.vocab)
    refs=spec['bundle'];manifest=document(reader.read(refs['MANIFEST.json']))
    allowed={'MANIFEST.json','meta.json','hwm_programs.json','blocks_kumo_tabular_s.npy','encoding_sha256.json','gid.npy'}
    need(set(refs)==allowed,'read only selected Kumo bundle components')
    blobs={}
    for name,r in refs.items():
        if name=='MANIFEST.json':continue
        need(manifest['files'][name]==r['sha256'],'bundle manifest member binding');blobs[name]=reader.read(r)
    meta=document(blobs['meta.json']);n=meta['n'];cids=meta['cids'];need(n==len(cids)==len(set(cids)) and 0<n<=20000,'native bundle cardinality')
    z=np.load(io.BytesIO(blobs['blocks_kumo_tabular_s.npy']),allow_pickle=False);need(z.shape==(n,DIM)and z.dtype==np.float32 and np.isfinite(z).all(),'native finite2051 vectors');z.setflags(write=False);s.vectors=z
    gid=np.load(io.BytesIO(blobs['gid.npy']),allow_pickle=False);need(gid.shape==(n,)and gid.dtype.kind in 'iu','retained group identity')
    enc=document(blobs['encoding_sha256.json'])['kumo_tabular_s'];need(set(cids)<=set(enc)and all(len(sha256(enc[c]))==64 for c in cids),'per-context Kumo object pins')
    parents=[document(line)for line in reader.read(spec['parent_registry']).splitlines()if line.strip()];lineages={};parts={}
    for p in parents:
        need(p['bundle_manifest_sha256']==refs['MANIFEST.json']['sha256']and p['gid_sha256']==refs['gid.npy']['sha256'],'retained lineage bundle binding')
        lin=sha256(p['parent_lineage_sha256']);part=partition(lin,p['split_namespace_sha256']);need(part==p['partition'],'retained split drift')
        for cid in p['retained_lineage_context_sha256']:
            need(cid not in lineages,'overlapping retained lineages');lineages[cid]=(lin,p['retained_group']);parts[cid]=part
    need(set(cids)==set(lineages),'complete retained lineage membership')
    need(not(set(lineages)&set(auth.get('protected_context_ids',[]))),'retained lineage intersects protected context')
    inv=document(reader.read(spec['inventory']));inventory={r['view_id']:r for r in inv['rows']};need(len(inventory)==len(inv['rows']),'duplicate inventory view')
    s.contexts={};s.row_index={};s.winners={};s.shards={};pending=[];refused=[];tv=set(target['views'])
    progs=document(blobs['hwm_programs.json'])['programs'];pm={p['cid']:p for p in progs};need(len(pm)==n and set(pm)==set(cids),'HWM program membership')
    for i,cid in enumerate(cids):
        sha256(cid);view=meta['view_id'][i];r=inventory.get(view)
        need(r and r['context_sha256']==cid and r['dataset_hash']==meta['dataset'][i] and r['fold']==meta['fold'][i],'full native identity crosswalk')
        need(lineages[cid][1]==int(gid[i]),'retained group mismatch')
        if view not in tv or view in set(auth.get('protected_views',[])):refused.append({'context_id':cid,'view_id':view,'reason':'current_role_excluded'});continue
        p=pm[cid];ph=sha256(p['program_sha256']);need(ph==meta['program_sha256'][i]and byte_sha(p['dsl'].encode())==ph,'HWM DSL identity')
        need((r.get('gp_index')or{}).get('outcome')=='selection_reached','winner is not published native selection')
        toks=tuple(s.tokenizer.tokenize(p['dsl'],s.vocab));reason=s.token_refusal(toks)
        if reason:refused.append({'context_id':cid,'program_sha256':ph,'reason':reason,'length':len(toks)});toks=()
        need(type(r.get('cut_id',r.get('cut')))is str,'explicit native cut identity required')
        s.contexts[cid]={'context_id':cid,'context_sha256':cid,'view_id':view,'dataset_hash':r['dataset_hash'],'fold':r['fold'],'cut_id':r.get('cut_id',r.get('cut')),'parent_lineage_sha256':lineages[cid][0],'partition':parts[cid],'source_kind':'native_s1_context','observation_dispatch':'pending_native_worker_adapter','gp_hwm_ref':r['gp_index']['main'],'kumo_source_sha256':enc[cid]}
        s.row_index[cid]=i;s.winners[cid]=NativeRecord(ph,toks,'completed',ce_eligible=bool(toks),feasibility=0 if toks else None,skeleton_sha256=s.tokenizer.skeleton_sha256(p['dsl']))
    native_inputs={}
    for ref in spec['input_lists']:
        doc=document(reader.read(ref))
        for row in doc.get('admitted',[]):
            cid=row['context_sha256']
            if cid not in s.contexts:continue
            need(row['view_id']==s.contexts[cid]['view_id'] and row['dataset_hash']==s.contexts[cid]['dataset_hash'],'native input registry identity')
            t=row['teacher'];q={'teacher':{'path':t['path'],'sha256':t['gz_sha256'],'bytes':t['gz_bytes']},
                'context_json':{'path':str(Path(row['context_dir'])/'context.json'),'sha256':row['context_json_sha256'],'bytes':spec['context_json_bytes'][row['context_json_sha256']]},'input_list_sha256':ref['sha256']}
            need(cid not in native_inputs or native_inputs[cid]==q,'conflicting native input registration');native_inputs[cid]=q
    for root in spec['tails']:
        header=document(reader.read(root['header']));need(header.get('vocab_sha256')==s.vocab.sha256 and header.get('shard_schema')=='jope.dl-s1-tail-shard/v1','native tail header')
        for row in [document(b)for b in reader.read(root['index']).splitlines()if b.strip()]:
            cid=row['context_sha256']
            if cid not in s.contexts:continue
            need(cid not in s.shards,'duplicate native tail context')
            need(row['view_id']==s.contexts[cid]['view_id'],'tail view mismatch')
            rel=Path(row['shard']);need(not rel.is_absolute()and '..'not in rel.parts and rel.as_posix()==f'shards/{cid[:2]}/{cid}.json','native shard relative path')
            desc={'shard':{'path':str(Path(root['root'])/rel),'sha256':row['sha256'],'bytes':row['bytes']},'n_records':row['n_records']}
            proof=native_inputs.get(cid)
            if proof:desc['native_input']=proof
            s.shards[cid]=desc
    for view in sorted(tv):
        r=inventory[view];cid=r['context_sha256']
        if cid not in s.contexts:pending.append({'context_id':cid,'view_id':view,'reason':'native_kumo_bundle_join_pending'})
    s.proposal_programs=[{'program_sha256':ph,'tokens':list(w.tokens),'dsl':s.tokenizer.detokenize(w.tokens,s.vocab)}
        for ph,w in sorted({w.program_sha256:w for w in s.winners.values() if w.tokens}.items())]
    ancestry_overrides=[]
    if spec.get('prior_ancestry') is not None:
        old=document(reader.read(spec['prior_ancestry']))
        need(old.get('schema')=='jope.native-prior-model-ancestry/v1' and old.get('verified')is True,'prior model ancestry proof')
        for role,key in [('train','training_lineages'),('validation','validation_lineages')]:
            for lin in old[key]:
                sha256(lin)
                for c in s.contexts.values():
                    if c['parent_lineage_sha256']==lin:
                        need(c['partition']==role,'warmstart ancestry split conflict')
    s.context_loader=NativeContextLoader(s)
    counts={'global_native_views':len(inventory),'current_eligible_views':len(tv),'loaded_kumo_contexts':len(s.contexts),'ce_winner_contexts':sum(bool(w.tokens)for w in s.winners.values()),'native_detail_shard_contexts':len(s.shards),'native_detail_records_declared':sum(d['n_records']for d in s.shards.values()),'baseline_input_contexts':sum('native_input'in d for d in s.shards.values()),'pending_contexts':len(pending),'refused_rows':len(refused),'partitions':dict(Counter(c['partition']for c in s.contexts.values()))}
    s.receipt={'schema':'jope.native-kumos-corpus-loaded/v1','snapshot_sha256':expected_sha256,'authority_sha256':byte_sha(authority_bytes),'counts':counts,'pending':pending,'refused':refused,'membership_sha256':digest(s.contexts),'proposal_programs_sha256':digest(s.proposal_programs),'proposal_programs':len(s.proposal_programs),'limits':['Loaded inventory is not optimizer exposure.','Declared detailed shard rows are not yet read or baseline-qualified until their context window is loaded.','HWM winner quality is unknown; fitness is never a target.','No historical other-model encodings are read.','Research95/5 lineage split is inherited; no untouched holdout claim.']}
    s.batch_manifest={'schema':'jope.native-corpus-batches/v1','snapshot_sha256':expected_sha256,'role_authority_sha256':byte_sha(authority_bytes),'contexts':list(s.contexts.values()),'pending':pending,'refused':refused}
    return s
