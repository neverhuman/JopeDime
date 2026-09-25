#!/usr/bin/env python3
"""Compile and execute three legacy recipe-refit examples on the frozen synthetic fixture."""
import argparse,hashlib,json,subprocess,struct,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--jope',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
repo=Path(__file__).resolve().parents[4];fixture=repo/'docs/internal/asm-manual/examples';out=a.output.resolve();out.mkdir(exist_ok=True,parents=True)
commands=[[str(a.jope.resolve()),'emit','--genome','(pipe X (standardize) (ridge))','--lang','all','--output',str(out),'--json']]
receipts=[]
def run(argv):
 start=time.monotonic();r=subprocess.run(argv,cwd=out,capture_output=True,timeout=60);i=len(receipts);(out/f'command-{i}.stdout').write_bytes(r.stdout);(out/f'command-{i}.stderr').write_bytes(r.stderr);receipts.append({'argv':argv,'exit_code':r.returncode,'wall_seconds':time.monotonic()-start});r.check_returncode()
run(commands[0])
(out/'program.cpp').write_bytes((out/'model.cpp').read_bytes()+(repo/'deploy/ci/emitters/driver.cpp').read_bytes())
train=str(fixture/'train.csv');predict=str(fixture/'predict.csv')
for argv in [['python3',str(out/'model.py'),train,predict,str(out/'python.txt')],['rustc','-O',str(out/'model.rs'),'-o',str(out/'model-rs')],[str(out/'model-rs'),train,predict,str(out/'rust.txt')],['g++','-std=c++17','-O3','-ffp-contract=off','-fno-fast-math',str(out/'program.cpp'),'-o',str(out/'model-cpp')],[str(out/'model-cpp'),train,predict,str(out/'cpp17.txt')]]:run(argv)
values={name:[float(x) for x in (out/f'{name}.txt').read_text().splitlines()] for name in ('python','rust','cpp17')}
assert len(values['python'])==8
assert all([struct.pack('>d',x) for x in v]==[struct.pack('>d',x) for x in values['python']] for v in values.values())
receipt={'schema':'jope.core-ip-worked-examples/v1','subject':'legacy recipe refit; synthetic16 training rows,8 prediction feature rows; not exact fitted export or GPU measurement','binary':{'path':str(a.jope),'sha256':hashlib.sha256(a.jope.read_bytes()).hexdigest()},'commands':receipts,'exact_bit_parity':True,'predictions':values,'files':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [fixture/'train.csv',fixture/'predict.csv',repo/'deploy/ci/emitters/driver.cpp',*out.glob('model.*')]}}
(out/'verification.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'commands':len(receipts),'exact_bit_parity':True}))
