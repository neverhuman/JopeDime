"""Exact physical continuation and explicitly bounded recovery checkpoints."""
import json,os,shutil,sqlite3
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import load
from dltrain import model as M
from jope_flywheel.program_io import bound,create,need,sha
from jope_flywheel.protocol import canonical
from neural_storage import checkpoint
from neural_native_stream import BatchStream


def restore(manifest_ref,state_ref,*,expected_parameter_sha256,device):
    m=json.loads(bound(manifest_ref));raw=bound(state_ref,32*1024**2)
    need(sha(raw)==m['state_sha256']and len(raw)==m['state_bytes']and m['optimizer']=='AdamW','exact model/optimizer checkpoint')
    tensors=load(raw);net=M.PolicyLM(M.PolicyConfig(**m['model_config']));params=dict(net.named_parameters())
    saved={k[6:]:v for k,v in tensors.items()if k.startswith('model.')};need(params.keys()==saved.keys(),'all physical tied parameters present')
    with torch.no_grad():
        for name,param in params.items():
            v=saved[name];need(v.shape==param.shape and v.dtype==param.dtype and torch.isfinite(v).all().item(),'exact finite saved parameter');param.copy_(v)
    need(M.parameter_sha256(net)==m['parameter_sha256']==expected_parameter_sha256,'physical origin parameter hash')
    net=net.to(device);opt=torch.optim.AdamW(M.param_groups(net,m['weight_decay']),lr=m['learning_rate'])
    consumed=set()
    for name,param in net.named_parameters():
        prefix='optimizer.'+name+'.';state={k[len(prefix):]:v for k,v in tensors.items()if k.startswith(prefix)}
        if state:
            need(set(state)=={'step','exp_avg','exp_avg_sq'}and state['step'].ndim==0 and state['step'].dtype==torch.float32 and torch.isfinite(state['step']).item() and float(state['step']).is_integer() and 0<float(state['step'])<=m['step'],'exact non-fused AdamW state')
            for key in ('exp_avg','exp_avg_sq'):need(state[key].shape==param.shape and state[key].dtype==param.dtype and torch.isfinite(state[key]).all().item(),'finite AdamW moment shape')
            need((state['exp_avg_sq']>=0).all().item(),'nonnegative AdamW second moment');consumed.update(prefix+k for k in state)
            opt.state[param]={'step':state['step'].clone().cpu(),'exp_avg':state['exp_avg'].clone().to(device),'exp_avg_sq':state['exp_avg_sq'].clone().to(device)}
    rng={'rng.torch_cpu'}|{k for k in tensors if k.startswith('rng.torch_cuda.')and k.removeprefix('rng.torch_cuda.').isdigit()}
    need(consumed and set(tensors)=={'model.'+n for n in params}|consumed|rng,'exact consumed trained optimizer and RNG tensor keys')
    need(all(tensors[k].dtype==torch.uint8 and tensors[k].ndim==1 for k in rng),'byte RNG states')
    torch.set_rng_state(tensors['rng.torch_cpu'])
    if torch.device(device).type=='cuda':
        keys=sorted(k for k in tensors if k.startswith('rng.torch_cuda.'));need(len(keys)==torch.cuda.device_count(),'all saved CUDA RNG states');torch.cuda.set_rng_state_all([tensors[k]for k in keys])
    return net,opt,m


def parent_ancestry(path,expected,checkpoint):
    p=BatchStream.__new__(BatchStream);p.db=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True,isolation_level=None);p.db.row_factory=sqlite3.Row
    try:
        p.db.execute('BEGIN');p.config=p._get('config');proof=p._verify();need(proof==expected and proof['unresolved_batches']==0,'readonly completed parent planner')
        need(p._get('checkpoint')==checkpoint,'parent physical final checkpoint join')
        epochs=p.db.execute("SELECT count(*)FROM epochs WHERE state='complete'").fetchone()[0];need(epochs>0 and p.db.execute('SELECT state FROM epochs WHERE id=?',(p._get('epoch'),)).fetchone()[0]=='complete' and p.db.execute('SELECT count(*) FROM members WHERE epoch=? AND batch IS NULL',(p._get('epoch'),)).fetchone()[0]==0,'completed current parent epoch')
        ids=[r[0]for r in p.db.execute('SELECT DISTINCT run FROM members WHERE batch IS NOT NULL ORDER BY run')]
        result={'planner_ref':str(path),'head':proof,'checkpoint_sha256':checkpoint,'optimizer_steps':p._get('optimizer_step'),
                'completed_epochs':epochs,'fingerprints':[r[0]for r in p.db.execute('SELECT fingerprint FROM batches ORDER BY fingerprint')],
                'exposed_context_ids':ids,'presentations':p.db.execute('SELECT count(*)FROM members WHERE batch IS NOT NULL').fetchone()[0]}
        p.db.execute('COMMIT');return result
    finally:p.db.close()


def normalizer(raw,native):
    p=json.loads(raw);need(p['snapshot_sha256']==native.identity and p['fit_context_ids']==sorted(c for c,r in native.contexts.items()if r['partition']=='train')and p['validation_used_for_fit']is False,'preserved original training-only fit')
    mean=np.asarray(p['mean'],np.float64);scale=np.asarray(p['scale'],np.float64)
    need(mean.shape==scale.shape==(2051,)and np.isfinite(mean).all()and np.isfinite(scale).all()and(scale>0).all()and p['clip']==[-10.,10.],'fixed normalization')
    return lambda cid,z:np.clip((np.asarray(z,np.float64)-mean)/scale,-10.,10.).astype(np.float32)


def append(path,value):
    with Path(path).open('ab')as f:f.write(canonical(value)+b'\n');f.flush();os.fsync(f.fileno())


class Recovery:
    def __init__(self,root):
        self.root=Path(root);self.live=[];self.best=[]
        for rel in ('rolling','best','epochs'):(self.root/rel).mkdir()
    def save(self,net,opt,step,config_sha):
        path=self.root/'rolling'/f'{step:09d}';m=checkpoint(net,opt,path,step=step,config_sha256=config_sha)
        fsyncdir(path);fsyncdir(path.parent);return path,m
    def retire(self,path):
        need(path.parent in(self.root/'rolling',self.root/'best')and path.is_dir()and not path.is_symlink(),'only owned working recovery slots may retire')
        names={p.name for p in path.iterdir()};need(names=={'state.safetensors','manifest.json'},'closed recovery slot files')
        receipt=json.loads((path/'manifest.json').read_bytes());append(self.root/'retired.jsonl',{'path':str(path),'checkpoint':receipt,'retention':'rolling working state; full historical tensor bytes not retained'})
        for name in ('state.safetensors','manifest.json'):(path/name).unlink()
        path.rmdir();fsyncdir(path.parent)
    def committed(self,path):
        self.live.append(path)
        while len(self.live)>2:self.retire(self.live.pop(0))
    def retain(self,path,kind,identity):
        need(kind in('best','epochs'),'registered retained checkpoint class');dest=self.root/kind/identity;dest.mkdir()
        for name in ('state.safetensors','manifest.json'):os.link(path/name,dest/name)
        fd=os.open(dest,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        if kind=='best':
            self.best.append(dest)
            while len(self.best)>2:self.retire(self.best.pop(0))
        fsyncdir(dest.parent);return str(dest)


def fsyncdir(path):
    fd=os.open(path,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
