"""Create-only physical model, AdamW and CPU/CUDA RNG checkpoints; no pickle."""
from pathlib import Path
import os,torch
from safetensors.torch import save_file
from dltrain import model as M
from jope_flywheel.protocol import canonical
from jope_flywheel.program_io import need,create,sha

def checkpoint(net,opt,path,*,step,config_sha256):
    """Create-only safetensors, including named AdamW moments; no pickle."""
    path=Path(path);path.mkdir(exist_ok=False);params=dict(net.named_parameters())
    tensors={'model.'+name:p.detach().cpu().contiguous()for name,p in params.items()}
    for name,p in params.items():
        for key,value in opt.state.get(p,{}).items():
            need(isinstance(value,torch.Tensor),'optimizer state is numeric tensor only')
            tensors['optimizer.'+name+'.'+key]=value.detach().cpu().contiguous()
    tensors['rng.torch_cpu']=torch.get_rng_state()
    if next(net.parameters()).is_cuda:
        for index,state in enumerate(torch.cuda.get_rng_state_all()):tensors['rng.torch_cuda.'+str(index)]=state
    save_file(tensors,str(path/'state.safetensors'))
    with (path/'state.safetensors').open('rb')as stream:os.fsync(stream.fileno())
    body=(path/'state.safetensors').read_bytes()
    manifest={'schema':'jope.flywheel-neural-checkpoint/v1','step':step,'config_sha256':config_sha256,
        'parameter_sha256':M.parameter_sha256(net),'state_sha256':sha(body),'state_bytes':len(body),
        'model_config':net.cfg.to_json(),'parameters':M.count_parameters(net),
        'optimizer':'AdamW','learning_rate':opt.param_groups[0]['lr'],'weight_decay':0.01,
        'resume':'No automatic replay. Verify checkpoint, planner reservation, and GP journal before explicit recovery.'}
    create(path/'manifest.json',canonical(manifest));return manifest
