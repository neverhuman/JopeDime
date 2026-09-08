#!/usr/bin/env python3
"""Export the reference action vocabulary/FSM fixtures for the Rust port."""
import dataclasses,hashlib,json,sys,random
from pathlib import Path
r=Path(__file__).resolve().parents[1];sys.path.insert(0,'/home/ubuntu/jope-prime/src')
from jope.grammars.featmath import GRAMMAR
from jope.policy.tokens import build_action_vocab,decode_tokens
from jope.policy.automaton import DecoderState
vocab=build_action_vocab([GRAMMAR]);data=dataclasses.asdict(vocab);data.pop('op_id_by_name');data['schema']='parallel.action_vocab.v1';data['source_sha256']={str(p.relative_to(Path('/home/ubuntu/jope-prime/src'))):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('/home/ubuntu/jope-prime/src/jope/grammars/featmath.py'),Path('/home/ubuntu/jope-prime/src/jope/policy/tokens.py'),Path('/home/ubuntu/jope-prime/src/jope/policy/automaton.py')]}
out=r/'assets/grammar';out.mkdir(parents=True,exist_ok=True);(out/'featmath.json').write_text(json.dumps(data,indent=2)+'\n');rng=random.Random(20260907);fixtures=[]
for i in range(128):
 state=DecoderState(vocab);tokens=[1];masks=[]
 while not state.is_done():
  legal=state.legal_ids()
  if state.phase=='TASK':legal=[vocab.task_ids['regression']]
  masks.append(sorted(state.legal_ids()));token=rng.choice(legal);tokens.append(token);state.advance(token)
  if len(tokens)>128:break
 if not state.is_done():continue
 engine,actions=decode_tokens(tokens,vocab);fixtures.append(dict(tokens=tokens,masks=masks,actions=actions,engine=engine))
(out/'parity.json').write_text(json.dumps(dict(vocab_sha256=hashlib.sha256((out/'featmath.json').read_bytes()).hexdigest(),fixtures=fixtures),indent=2)+'\n');print(json.dumps(dict(vocabulary=vocab.size,fixtures=len(fixtures),task_ids=vocab.task_ids)))
