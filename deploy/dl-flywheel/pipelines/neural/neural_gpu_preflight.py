"""Actual CUDA forward/backward qualification; synthetic input, zero optimizer updates."""
import json,os,sys,time,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(body):return hashlib.sha256(body).hexdigest()
def write(name,obj):
 with (ROOT/name).open('x')as f:json.dump(obj,f,sort_keys=True);f.flush();os.fsync(f.fileno())

def main():
 c=json.loads((ROOT/'config.json').read_text());w=json.loads((ROOT/'window.json').read_text());deadline=w['deadline_unix']
 need(deadline-w['created_unix']==120 and w['created_unix']<=time.time()<deadline,'fixed120s qualification')
 need(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8'and os.environ.get('CUDA_VISIBLE_DEVICES')=='0','one GPU deterministic workspace')
 need(c['seed']==20261009 and c['model']=={'vocab_size':282,'ctx_dim':2051,'d_model':128,'n_layers':2,'n_heads':4,'d_ff':512,'max_seq':128,'dropout':.1,'context_free_head':False,'pad_id':0},'frozen new model origin')
 for ref in c['sources']:
  p=ROOT/ref['relative'];need(sha(p.read_bytes())==ref['sha256'],'source identity')
 group=Path('/sys/fs/cgroup')/next(s[3:]for s in Path('/proc/self/cgroup').read_text().splitlines()if s.startswith('0::')).lstrip('/')
 q,p=(group/'cpu.max').read_text().split();need(q!='max'and int(q)<=8*int(p)and int((group/'memory.max').read_text())<=8*1024**3 and int((group/'memory.swap.max').read_text())==0,'CPU8/8GiB/swap0')
 sys.path.insert(0,str(ROOT/'runtime'))
 import torch,safetensors
 from safetensors.torch import save_file
 from dltrain import model as M,dsl_tokenizer as T
 need(str(torch.__version__)=='2.7.0'and str(safetensors.__version__)=='0.7.0','actual GPU runtime pins')
 M.configure_determinism(1);torch.manual_seed(c['seed']);net=M.PolicyLM(M.PolicyConfig.from_json(c['model']))
 initial=M.parameter_sha256(net);save_file({n:p.detach().contiguous()for n,p in net.named_parameters()},str(ROOT/'initial.safetensors'))
 net=net.cuda();need(torch.cuda.device_count()==1,'one GPU visible')
 vocab=T.Vocab.from_names({});ids=T.tokenize('(pipe X (standardize) (ridge lambda=0.3))',vocab)
 tokens=torch.tensor([ids],dtype=torch.long,device='cuda');lengths=torch.tensor([len(ids)],dtype=torch.long,device='cuda')
 ctx=torch.zeros((1,2051),device='cuda');seq=torch.zeros(1,dtype=torch.long,device='cuda')
 before_rng=torch.cuda.get_rng_state().clone();out=net(tokens,lengths,ctx,seq)
 loss=out.logits.square().mean()+out.q_hat.square().mean()+out.feas_logits.square().mean()+out.rt_hat.square().mean()
 need(torch.isfinite(loss).item(),'finite GPU forward');loss.backward();torch.cuda.synchronize()
 need(any(p.grad is not None for p in net.parameters())and all(p.grad is None or torch.isfinite(p.grad).all().item()for p in net.parameters()),'finite actual GPU backward')
 need(M.parameter_sha256(net)==initial,'preflight performs zero updates')
 net.zero_grad(set_to_none=True);need(time.time()<deadline,'fixed deadline')
 result={'schema':'jope.flywheel-neural-gpu-preflight/v1','state':'passed','config_sha256':sha((ROOT/'config.json').read_bytes()),
  'at':time.time(),'window':w,'parameter_sha256':initial,'parameters':M.count_parameters(net),'optimizer_updates':0,
  'initial_checkpoint':{'path':str(ROOT/'initial.safetensors'),'sha256':sha((ROOT/'initial.safetensors').read_bytes()),'bytes':(ROOT/'initial.safetensors').stat().st_size},
  'fixture':'synthetic zero KumoS-shaped context; actual native model/tokenizer; hardware test only',
  'loss':float(loss.detach().cpu()),'gpu':torch.cuda.get_device_name(0),'cuda':str(torch.version.cuda),
  'torch':str(torch.__version__),'CUBLAS_WORKSPACE_CONFIG':os.environ['CUBLAS_WORKSPACE_CONFIG'],
  'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'memory_peak_bytes':int((group/'memory.peak').read_text())}
 write('result.json',result);print(json.dumps(result))
if __name__=='__main__':
 try:main()
 except BaseException as error:write('failure.json',{'state':'failed','error':type(error).__name__+': '+str(error)[:2000],'at':time.time(),'optimizer_updates':0});raise
