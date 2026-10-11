"""anchored_step: the v2.0 training step under owner ruling R23 (D10; amends R8): L6 + weight * KL(policy || v1 anchor).

``run_microbatches_anchored(net, batches, total, *, optimizer, deadline, anchor_kl)`` has three parts.

1. **L6.** This is frozen ``neural_native_train.run_microbatches``, line for line (sha256 e73567dc..., lines 61-91;
   the module runner_v6 imports from pipelines/continuous).  It does:
   - ``zero_grad(set_to_none=True)`` and ``train(True)``;
   - per microbatch: ``tick``, forward and objective, the global-normalized loss and its finite check, backward;
   - numerator and support accounting, and the same result keys.
2. **The anchor KL term**, exactly as the R22 scratch harness revision 4 does it for the sweep 4a ``kl_only`` arm
   (lpi_scratch_train_v4.py sha256 a93631f3..., training step lines 845-853; configs/sweep4a.json: lambda_kl 0.2,
   kl_contexts_per_step 1, max_tokens_per_chunk 16384):
   - ``net.eval()``: no dropout, so the term draws NO random numbers;
   - for each of the batch's first ``contexts_per_step`` contexts, ``kl_entropy(all_nodes(net, z), all_nodes(anchor, z))``.
     The policy pass has gradient on; each trie chunk runs under non-reentrant activation checkpointing.  The anchor
     pass runs under ``no_grad``.  The math is kl_guard's exact trie KL, pinned bitwise to the harness;
   - the mean over those contexts; ``backward(weight * mean)``.
3. **The tail, in run_microbatches' order:** ``net.train()``, the finite-gradient check, ``clip_grad_norm_(1.0)``
   with a finite norm, ``tick``, ``optimizer.step()``.

The result is run_microbatches' dict plus ``anchor_kl`` (the mean KL value), ``anchor_kl_weight``,
``anchor_kl_contexts`` (the context ids) and ``objective_total = loss + weight * anchor_kl``.  **``loss`` stays the
L6 total.**  The loss history, the plateau control and the G32 guard therefore continue on the L6 scale they had
before R23.  The anchor term is reported separately.

The form ``exact_trie_kl_eval_mode_v1`` fixes:
- the deployed trie (runner_v6's train-catalog BatchedTrie);
- temperature 1.0: the harness applies none, and x / 1.0 == x bitwise;
- the policy in eval mode;
- the chain-rule KL over every branching node;
- the per-chunk token budget ``max_tokens_per_chunk`` (16384 in sweep 4a).

A trie with no branching node has KL identically 0 with no parameter dependence.  No forward or backward runs then,
which equals adding an exactly zero gradient.
"""
import math
import time

import torch
from jope_flywheel.program_io import need, tick
from jope_flywheel import timing
from neural_native_batch import objective, OBJECTIVE
from . import kl_guard as KG

FORM = 'exact_trie_kl_eval_mode_v1'
CONFIG_KEYS = ('weight', 'contexts_per_step', 'anchor_parameter_sha256', 'form', 'temperature', 'max_tokens_per_chunk')
IDENTITY_SCHEMA = 'jope.v2-training-objective/v1'
HARNESS = {'file': 'rlfw-takeover/lpi-scratch/lpi_scratch_train_v4.py',
           'sha256': 'a93631f33e437badb72fb011fed0693905924202ba4cfc7fa856493e03ba5d44',
           'config': 'configs/sweep4a.json (arm kl_only: lambda_pi 0, lambda_kl 0.2; kl_contexts_per_step 1; '
                     'max_tokens_per_chunk 16384)'}
FROZEN_RUN_MICROBATCHES = {'file': 'pipelines/continuous/neural_native_train.py', 'lines': '61-91',
                           'sha256': 'e73567dcffde3989c7ac33bdfc9ae2df99dc6c29f8e83baf519ba1b9eac29280'}


def check_config(ak):
    """The closed anchor-KL block ``neural['anchor_kl']`` (refused unless every clause holds).  A weight of exactly 0
    is well-formed: the step then equals frozen run_microbatches.  runner_v6 additionally requires weight > 0 (R23)."""
    need(type(ak) is dict and set(ak) == set(CONFIG_KEYS) and
         type(ak['weight']) is float and math.isfinite(ak['weight']) and ak['weight'] >= 0 and
         type(ak['contexts_per_step']) is int and 1 <= ak['contexts_per_step'] <= 64 and
         ak['form'] == FORM and type(ak['temperature']) is float and ak['temperature'] == 1.0 and
         type(ak['max_tokens_per_chunk']) is int and 640 <= ak['max_tokens_per_chunk'] <= 1 << 20 and
         type(ak['anchor_parameter_sha256']) is str and len(ak['anchor_parameter_sha256']) == 64 and
         all(ch in '0123456789abcdef' for ch in ak['anchor_parameter_sha256']), 'closed anchor-KL policy (R23)')
    return {k: ak[k] for k in CONFIG_KEYS}


def training_objective(ak):
    """The v2.0 training-objective identity (R23): the unchanged L6 objective plus the anchor-KL block."""
    return {'schema': IDENTITY_SCHEMA, 'l6': OBJECTIVE, 'anchor_kl': check_config(ak)}


class AnchorKL:
    """The R23 term on the deployed trie against the shared, frozen, read-only v1 anchor."""

    def __init__(self, trie, anchor, config, *, device):
        self.config = check_config(config)
        need(not anchor.training and not any(p.requires_grad for p in anchor.parameters()), 'frozen eval-mode anchor')
        self.trie, self.anchor, self.device = trie, anchor, device
        self.weight, self.k = self.config['weight'], self.config['contexts_per_step']
        self.ti = None

    def contexts(self, ids, z_of):
        """The step's KL contexts: the batch's first ``contexts_per_step`` ordered context ids, with their inputs."""
        need(len(ids) >= 1, 'a nonempty ordered batch')
        return [(cid, z_of(cid)) for cid in list(ids)[:self.k]]

    def kl(self, net, z, *, deadline):
        """Exact trie KL(net || anchor) at input z.  The net must already be in eval mode.  Returns None for a
        single-program trie, where KL is identically 0."""
        if self.ti is None:
            self.ti = KG.TrieIndex(self.trie, self.device, deadline)
        if not self.ti.B:
            return None
        kw = dict(temperature=self.config['temperature'], deadline=deadline, max_tokens=self.config['max_tokens_per_chunk'])
        lp = KG.all_nodes(net, z, self.ti, self.device, **kw)            # harness argument order: policy pass first
        with torch.no_grad():
            la = KG.all_nodes(self.anchor, z, self.ti, self.device, **kw)
        return KG.kl_entropy(lp, la, self.ti)[0]


def run_microbatches_anchored(net, batches, total, *, optimizer, deadline, anchor_kl):
 term, contexts = anchor_kl
 need(optimizer is not None and isinstance(term, AnchorKL) and type(contexts) is list and 1 <= len(contexts) <= term.k,
      'an optimizer step with its anchor-KL contexts')
 # ---- L6: frozen run_microbatches (e73567dc, lines 61-91), line for line ------------------------------------------
 previous=net.training;training=optimizer is not None;net.train(training);device=next(net.parameters()).device
 totals={k:0. for k in total};support={};forward_seconds=0.;backward_seconds=0.
 if training:optimizer.zero_grad(set_to_none=True)
 try:
  for cpu in batches:
   tick(deadline);b=cpu.to(device);start=time.monotonic()
   with torch.set_grad_enabled(training):
    with timing.span('model.forward_objective'):
     out=net(b.tokens,b.lengths,b.ctx_z,b.seq_ctx);_,report,parts=objective(out,b)
    if device.type=='cuda':torch.cuda.synchronize(device)
    forward_seconds+=time.monotonic()-start
    loss=sum(OBJECTIVE['coefficients'][k]*n/total[k] for k,(n,d)in parts.items()if total[k])
    need(torch.isfinite(loss).item(),'finite global-normalized microbatch loss')
    if training:
     start=time.monotonic()
     with timing.span('model.backward'):loss.backward()
     if device.type=='cuda':torch.cuda.synchronize(device)
     backward_seconds+=time.monotonic()-start
   for k,v in report['numerators'].items():totals[k]+=v
   for k,v in report['support'].items():support[k]=support.get(k,0)+v
   del out,b,parts,loss
  terms={k:totals[k]/total[k]if total[k]else 0. for k in total};result={'loss':sum(OBJECTIVE['coefficients'][k]*v for k,v in terms.items()),'terms':terms,'numerators':totals,'denominators':total,'support':support,'dropout':training,'forward_seconds':forward_seconds,'backward_seconds':backward_seconds}
  # ---- R23 anchor KL: harness v4 lines 845-853 (eval mode: no dropout, no RNG draw) --------------------------
  with timing.span('model.anchor_kl'):
   net.eval();kls=[]
   for cid,z in contexts:
    tick(deadline);value=term.kl(net,z,deadline=deadline)
    if value is not None:kls.append(value)
   if kls:
    kl_t=torch.stack(kls).mean();(term.weight*kl_t).backward();kl=float(kl_t.detach())
   else:kl=0.                     # single-program trie: KL identically 0, no parameter dependence, no backward
   need(math.isfinite(kl) and kl>=KG.KL_FLOOR,'finite nonnegative anchor KL')
   net.train()
  result.update(anchor_kl=kl,anchor_kl_weight=term.weight,anchor_kl_contexts=[cid for cid,_ in contexts],objective_total=result['loss']+term.weight*kl)
  # ---- tail: run_microbatches' order (finite check, clip 1.0, tick, step) --------------------------------------
  if training:
   need(all(p.grad is None or torch.isfinite(p.grad).all().item()for p in net.parameters()),'finite actual optimizer gradients')
   norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.);need(torch.isfinite(norm).item(),'finite clipping norm');tick(deadline)
   with timing.span('model.optimizer_step'):
    optimizer.step()
    if device.type=='cuda':torch.cuda.synchronize(device)
   result['gradient_norm_before_clip']=float(norm)
  return result
 finally:net.train(previous)
