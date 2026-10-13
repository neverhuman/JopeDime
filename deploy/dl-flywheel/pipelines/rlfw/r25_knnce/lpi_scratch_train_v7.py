"""R22 scratch-only, non-qualifying L_PI train + eval, harness revision 7 (lane harness; not part of any PR tree).

Revision 7 = revision 6 (140c0a47, kept unchanged) + the R25-kNN-CE term signed off by ENG:watcher 2026-10-13T07:42:55Z:
- an arm may carry a fourth weight lambda_knn; its loss adds lambda_knn * CE_kNN, where CE_kNN is the mean over the
  batch contexts c of the mean trie-children negative log-likelihood, under c's own context, of the frozen native HWM
  teacher programs of c's k nearest neighbours (knn_table.py: cosine on the frozen-normalised encodings; neighbours
  only from the L6 train roster, never the anchor's parent lineage, never a held-out, validation or other non-roster
  context; the table is built once before any update and its sha256 is in the receipt);
- config max_updates: training stops when every active arm has made exactly that many updates, with the same partial
  epoch evaluation and checkpoint as a deadline stop (stop_reason max_updates).
Nothing else changes: L6, KL, L_PI, data, evaluation and paired rows are revision 6's code.

Revision 6 = revision 5 (40956bdf, kept) + config l6_roster "full_minus_heldout_lineages" (L6 batches over the native
train roster minus every context of a held-out parent lineage; L_PI targets still only on detail contexts in the
training split) and arm_max_epochs (per-arm epoch budget, recorded as frozen_reason). Revision 5 notes follow.


Revision 5 = revision 4 (a93631f3, kept) + config device cpu|cuda (CPU smoke with CUDA hidden), per-context paired
rows on full validation (paired_sets ["a"]), an optional pin of the lineage split to earlier shas, and smoke caps
on the train L_PI evaluation. Revision 4 = revision 3 + host-configurable inputs (lpi_scratch_data_v4) and the
lineage holdout with in-run paired rows. Revision 3 notes follow.

Revision 3 is revision 2 (3300e591, kept unchanged) plus knobs for sweep 3:
- torch.cuda.set_per_process_memory_fraction(config cuda_memory_fraction) so another lane can share the GPU;
- config max_tokens_per_chunk, and activation checkpointing of each trie-likelihood chunk on gradient paths
  (the exact training KL graph was the sweep-2 peak; values are unchanged, the chunk is recomputed in backward);
- wall_seconds up to 7020 s, and the scope's RuntimeMaxUSec read back must be at most 7200 s and at least
  wall_seconds + 120 s;
- a checkpoint per arm per epoch (model, AdamW moments, RNG) under RUN/checkpoints/, an index file
  RUN/checkpoints-index.json and a per-epoch RUN/progress.json, each replaced atomically after every epoch.

Revision 2 notes follow.

Revision 2 answers CX-REV's r22-scratch review (trainer copy 4338a29c; run 20261010T174612Z used 0733d38b, whose
bytes are kept unchanged as lpi_scratch_train.py):
- every startup cross-check has a numeric tolerance; any mismatch is fatal before the first update, and the
  cross-checks run before the full target build so a failure costs about one minute;
- every loop (anchor/target construction, trie likelihood chunks, training steps and microbatches, KL, evaluation,
  probes, saving) checks the hard deadline; training stops at the training deadline (hard deadline minus the
  final reserve) and the receipt is always written, including after a deadline or cross-check refusal;
- the run refuses to start unless its own systemd scope reports a finite RuntimeMaxUSec of at most one hour
  (the external timeout), read back from inside the scope by launch_v2.sh;
- every update journals the arm's parameter sha256 before and after; the receipt carries initial, per-epoch and
  final parameter shas and changed-parameter statistics.
Settings come from a JSON config file whose bytes and sha256 are recorded in the receipt.

Arms run interleaved on identical batches, identical per-step dropout seeds and identical update counts. L6 is the
repository's pinned six-term objective through the v1 microbatch code; L_PI targets come from the rebased
rlfw.exit_targets (set form, clip centre at h_c) on native S1 detail-shard measured programs that are trie leaves.
Validation contexts are evaluation-only. Guards bypassed for R22 are listed in the receipt (BYPASSED_GUARDS).
"""
import copy
import hashlib
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lpi_scratch_data_v4 as D  # noqa: E402  (revision 4: host-configurable roots)
import knn_table as KNN  # noqa: E402  (revision 7: R25-kNN-CE neighbour table)

MAX_TOKENS_PER_CHUNK = 49152
TOLERANCE = {'trie_logprob_max_abs_diff': 1e-5, 'policy_imitation_abs_diff': 1e-5,
             'hidden_vs_forward_logits_max_abs_diff': 1e-6, 'exact_trie_kl_abs_diff': 1e-7,
             'exact_trie_kl_rel_diff': 1e-4, 'leaf_probability_closure_abs_diff': 1e-9}
HARD_DEADLINE = None


class DeadlineExceeded(RuntimeError):
    pass


class CrosscheckFailed(RuntimeError):
    pass


def check(where):
    if time.time() >= HARD_DEADLINE:
        raise DeadlineExceeded(where)


def digest(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def runtime_seconds(text):
    """Parse systemd's timespan rendering (for example '1h', '55min 30s'); None for infinity/unparsable."""
    units = {'us': 1e-6, 'ms': 1e-3, 's': 1, 'min': 60, 'h': 3600, 'd': 86400}
    total = 0.0
    for part in text.split():
        num = ''.join(ch for ch in part if ch.isdigit() or ch == '.')
        unit = part[len(num):]
        if not num or unit not in units:
            return None
        total += float(num) * units[unit]
    return total if text.strip() else None


class TrieIndex:
    """Branching nodes of the train-winner trie with a covering leaf for each node's next-token logits."""
    def __init__(self, trie, torch, device):
        nodes = trie.nodes
        self.leaves = sorted(trie.programs)
        first, depth, parent, paths = {}, {}, {}, []
        for li, toks in enumerate(self.leaves):
            if li % 512 == 0:
                check('trie index')
            node, last, path = 0, None, []
            for pos, tok in enumerate(toks[1:], 1):
                first.setdefault(node, li)
                depth[node] = pos
                if len(nodes[node]) >= 2:
                    parent.setdefault(node, last)
                    path.append((node, tok))
                    last = (node, tok)
                node = nodes[node][tok]
            paths.append(path)
        self.branch = sorted(parent, key=lambda n: (depth[n], n))
        index = {n: i for i, n in enumerate(self.branch)}
        self.children = [sorted(nodes[n]) for n in self.branch]
        cmax = max(len(c) for c in self.children)
        self.cmax = cmax
        cover = sorted({first[n] for n in self.branch})
        self.cover = cover
        cpos = {li: i for i, li in enumerate(cover)}
        B = len(self.branch)
        child_tok = [[0] * cmax for _ in range(B)]
        mask = [[False] * cmax for _ in range(B)]
        for i, ch in enumerate(self.children):
            for j, t in enumerate(ch):
                child_tok[i][j], mask[i][j] = t, True
        anc = []
        for n in self.branch:
            chain, cur = [], parent[n]
            while cur is not None:
                a, tok = cur
                chain.append(index[a] * cmax + self.children[index[a]].index(tok))
                cur = parent[a]
            anc.append(chain)
        dmax = max(1, max(len(a) for a in anc))
        self.child_tok = torch.tensor(child_tok, device=device)
        self.mask = torch.tensor(mask, device=device)
        self.anc = torch.tensor([a + [0] * (dmax - len(a)) for a in anc], device=device)
        self.anc_mask = torch.tensor([[True] * len(a) + [False] * (dmax - len(a)) for a in anc], device=device)
        self.node_cover = [cpos[first[n]] for n in self.branch]
        self.node_row = [depth[n] - 1 for n in self.branch]
        self.leaf_edges = [[index[a] * cmax + self.children[index[a]].index(t) for a, t in p] for p in paths]
        self.leaf_index = {toks: i for i, toks in enumerate(self.leaves)}
        self.path_nodes = [[(index[a], self.children[index[a]].index(t)) for a, t in p] for p in paths]
        self.prefix_len = {n: depth[n] for n in self.branch}
        self.B = B


def hidden(net, tokens, lengths, ctx_z, seq_ctx, torch):
    """PolicyLM.forward up to the final LayerNorm (identical ops; logits only where needed)."""
    n, t = tokens.shape
    ctx_emb = net.ctx_mlp(ctx_z)[seq_ctx]
    pos = net.pos_emb.weight
    prefix = (ctx_emb + pos[0]).unsqueeze(1)
    body = net.tok_emb(tokens) + pos[1:t + 1]
    x = net.emb_drop(torch.cat([prefix, body], dim=1))
    for block in net.blocks:
        x = block(x)
    return net.ln_f(x)


def node_logp(net, z, seqs, queries, ti, torch, device):
    """Log-probabilities over trie children at queried (seq, row, node) positions; float64, -inf outside children."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    by_seq = {}
    for qi, (si, row, b) in enumerate(queries):
        by_seq.setdefault(si, []).append((qi, row, b))
    outs, qorder = [], []
    i = 0
    while i < len(order):
        check('trie likelihood chunk')
        width = len(seqs[order[i]])
        j = i
        while j < len(order) and (j - i + 1) * max(width, len(seqs[order[j]])) <= MAX_TOKENS_PER_CHUNK:
            width = max(width, len(seqs[order[j]]))
            j += 1
        chunk = order[i:j]
        i = j
        qs = [(qi, k, row, b) for k, si in enumerate(chunk) for qi, row, b in by_seq.get(si, [])]
        if not qs:
            continue
        tok = torch.zeros((len(chunk), width), dtype=torch.long)
        for k, si in enumerate(chunk):
            tok[k, :len(seqs[si])] = torch.tensor(seqs[si])
        tok = tok.to(device)
        lengths = torch.tensor([len(seqs[si]) for si in chunk], device=device)
        ctx = torch.as_tensor(z, dtype=torch.float32, device=device).reshape(1, -1)
        width_t = tok.shape[1] + 1
        flat_pos = torch.tensor([k * width_t + row + 1 for _, k, row, _ in qs], device=device)
        bs = torch.tensor([b for _, _, _, b in qs], device=device)
        seq0 = torch.zeros(len(chunk), dtype=torch.long, device=device)

        def chunk_logits(tok_, lengths_, ctx_, seq_, pos_):
            h_ = hidden(net, tok_, lengths_, ctx_, seq_, torch)
            return net.lm_head(h_.reshape(-1, h_.shape[-1]).index_select(0, pos_))

        if torch.is_grad_enabled():
            # Keep only the selected logits; recompute the chunk's activations in backward (dropout is off).
            logits = torch.utils.checkpoint.checkpoint(chunk_logits, tok, lengths, ctx, seq0, flat_pos,
                                                       use_reentrant=False).double()
        else:
            logits = chunk_logits(tok, lengths, ctx, seq0, flat_pos).double()
        values = logits.gather(1, ti.child_tok[bs]).masked_fill(~ti.mask[bs], float('-inf'))
        outs.append(torch.log_softmax(values, dim=1))
        qorder.extend(qi for qi, _, _, _ in qs)
    stacked = torch.cat(outs)
    inverse = [0] * len(qorder)
    for pos, qi in enumerate(qorder):
        inverse[qi] = pos
    return stacked.index_select(0, torch.tensor(inverse, device=device))


def all_nodes(net, z, ti, torch, device):
    seqs = [ti.leaves[li] for li in ti.cover]
    queries = [(c, r, b) for b, (c, r) in enumerate(zip(ti.node_cover, ti.node_row))]
    return node_logp(net, z, seqs, queries, ti, torch, device)


def kl_entropy(lp, la, ti, torch):
    """Exact trie KL(policy || anchor) and policy entropy by the chain rule (nats per program)."""
    flat = lp.reshape(-1)
    safe = flat.masked_fill(~ti.mask.reshape(-1), 0.0)
    gathered = safe.index_select(0, ti.anc.reshape(-1)).reshape(ti.anc.shape)
    logreach = gathered.masked_fill(~ti.anc_mask, 0.0).sum(1)
    p = lp.exp().masked_fill(~ti.mask, 0.0)
    diff = (lp - la).masked_fill(~ti.mask, 0.0)
    plogp = p * lp.masked_fill(~ti.mask, 0.0)
    reach = logreach.exp()
    return (reach * (p * diff).sum(1)).sum(), -(reach * plogp.sum(1)).sum(), logreach


def program_logprob(net, z, programs, ti, torch, device):
    """Summed EOS-inclusive trie-children log-likelihood of trie leaves (single-child steps contribute 0)."""
    queries, owners = [], []
    for si, toks in enumerate(programs):
        for b, c in ti.path_nodes[ti.leaf_index[tuple(toks)]]:
            queries.append((si, ti.prefix_len[ti.branch[b]] - 1, b))
            owners.append((si, c))
    lp = node_logp(net, z, [tuple(p) for p in programs], queries, ti, torch, device)
    picked = lp.gather(1, torch.tensor([[c] for _, c in owners], device=device)).squeeze(1)
    out, pos = [], 0
    for si, toks in enumerate(programs):
        k = len(ti.path_nodes[ti.leaf_index[tuple(toks)]])
        out.append(picked[pos:pos + k].sum() if k else picked.new_zeros(()))
        pos += k
    return torch.stack(out)


def scope_properties(out, wall_seconds):
    path = Path(str(out) + '.scope-properties.txt')
    D.need(path.is_file(), 'launch_v2.sh scope property readback missing')
    raw = path.read_bytes()
    props = dict(line.split('=', 1) for line in raw.decode().splitlines() if '=' in line)
    seconds = runtime_seconds(props.get('RuntimeMaxUSec', ''))
    unit = os.environ.get('LPI_UNIT', '')
    D.need(props.get('Id') == unit + '.scope' and props.get('ActiveState') == 'active', 'readback is this active scope')
    D.need(seconds is not None and wall_seconds + 120 <= seconds <= 7200,
           'external timeout RuntimeMaxSec <= 7200 and >= harness wall + 120 s required')
    return {'path': str(path), 'sha256': D.sha(raw), 'properties': props, 'runtime_max_seconds': seconds}


def main(config_path, out_dir):
    global HARD_DEADLINE
    started = time.time()
    cfg_raw = Path(config_path).read_bytes()
    cfg = json.loads(cfg_raw)
    D.need(cfg.get('schema') == 'jope.r22-lpi-scratch-config/v7' and 0 < cfg['wall_seconds'] <= 7020
           and 0 < cfg['cuda_memory_fraction'] <= 1 and cfg['max_tokens_per_chunk'] >= 640,
           'bounded scratch config')
    HARD_DEADLINE = started + cfg['wall_seconds']
    train_end = HARD_DEADLINE - cfg['final_reserve_seconds']
    out = Path(out_dir)
    out.mkdir(exist_ok=False)
    metrics = (out / 'metrics.jsonl').open('x')

    def log(kind, **kw):
        metrics.write(json.dumps({'kind': kind, 't': round(time.time() - started, 3), **kw}, sort_keys=True) + '\n')
        metrics.flush()

    R = {'schema': 'jope.r22-lpi-scratch-train/v7', **D.R22, 'non_qualifying': True, 'benchmark_valid': False,
         'promotion_input': False, 'state': 'started', 'started_unix': started,
         'config': {'path': str(config_path), 'sha256': D.sha(cfg_raw), 'body': cfg},
         'tolerances': TOLERANCE, 'harness_sha256': {Path(f).name: D.sha(Path(f).read_bytes())
                                                     for f in (__file__, D.__file__, KNN.__file__)},
         'injected_mismatch': os.environ.get('LPI_SCRATCH_INJECT_MISMATCH') == '1'}
    exit_code = 1
    try:
        R['scope'] = scope_properties(out, cfg['wall_seconds'])
        log('scope', **R['scope'])
        body(cfg, out, log, R, started, train_end)
        R['state'] = 'completed'
        exit_code = 0
    except CrosscheckFailed as e:
        R['state'] = 'crosscheck_failed_before_training'
        R['error'] = str(e)
        exit_code = 3
    except (DeadlineExceeded, TimeoutError) as e:  # TimeoutError: tick() inside the imported v1 L6 helpers
        R['state'] = 'hard_deadline_exceeded'
        R['error'] = 'deadline reached in: ' + repr(e)
        exit_code = 4
    except Exception as e:  # noqa: BLE001 - recorded, then the process exits nonzero
        R['state'] = 'error'
        R['error'] = repr(e)
        R['traceback'] = traceback.format_exc()
    finally:
        R['finished_unix'] = time.time()
        R['wall_seconds'] = R['finished_unix'] - started
        metrics.close()
        raw = (json.dumps(R, indent=1, sort_keys=True, default=str) + '\n').encode()
        with (out / 'receipt.json').open('xb') as f:
            f.write(raw)
        print(json.dumps({'receipt': str(out / 'receipt.json'), 'sha256': D.sha(raw), 'state': R['state'],
                          'epochs': R.get('epochs_completed'), 'updates': R.get('updates_per_arm')},
                         sort_keys=True))
    return exit_code


def body(cfg, out, log, R, started, train_end):
    D.setup_paths()
    import torch
    from safetensors.torch import load as st_load, save_file
    from dltrain import model as M
    from jope_flywheel.program_io import bind_campaign_deadline
    from neural_native_batch import OBJECTIVE, objective as l6_objective
    from neural_native_train import prepare_microbatches, run_microbatches, evaluate_full_validation
    from continuous_state import normalizer
    from rlfw.exit_targets import TargetConfig, build_targets, tilt_weights
    from rlfw.exit_objective import TensorPolicy, trie_logprob, policy_imitation, exact_trie_kl
    from rlfw.arena.game_spec import digest as rdigest

    need = D.need
    bind_campaign_deadline(HARD_DEADLINE)
    global MAX_TOKENS_PER_CHUNK
    MAX_TOKENS_PER_CHUNK = cfg['max_tokens_per_chunk']
    use_cuda = cfg.get('device', 'cuda') == 'cuda'
    if use_cuda:
        device = torch.device('cuda:0')
        need(torch.cuda.is_available() and os.environ.get('CUDA_VISIBLE_DEVICES') == '0', 'CUDA device 0 only')
        R['determinism'] = M.configure_determinism(1)
        R['device'] = torch.cuda.get_device_name(0)
        torch.cuda.set_per_process_memory_fraction(cfg['cuda_memory_fraction'], 0)
        R['cuda'] = {'memory_fraction': cfg['cuda_memory_fraction'], 'max_tokens_per_chunk': MAX_TOKENS_PER_CHUNK,
                     'total_bytes': torch.cuda.get_device_properties(0).total_memory,
                     'grad_chunk_activation_checkpointing': True}
    else:
        # Revision 5: CPU (smoke while the GPU is held by another job). CUDA must be hidden from this process.
        device = torch.device('cpu')
        need(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not torch.cuda.is_available(), 'CPU run with CUDA hidden')
        R['determinism'] = M.configure_determinism(cfg.get('cpu_threads', 4))
        R['device'] = 'cpu'
        R['cuda'] = None
    smoke = cfg.get('smoke_steps')
    R['smoke'] = smoke is not None

    # Inputs (every read is sha256-checked).
    check('inputs')
    s = D.load_native()
    trie = D.train_trie(s)
    norm = normalizer(D.read_pinned(D.PREPROCESSING, D.PREPROCESSING_SHA), s)
    cov_raw = (D.LANE / 'run/coverage.json').read_bytes()
    cov = json.loads(cov_raw)
    need(cov['trie']['sha256'] == trie.identity, 'coverage built on this trie')
    sel_raw = (D.LANE / 'run/select.json').read_bytes()
    sel = json.loads(sel_raw)
    manifest_raw = D.mapped(D.CHECKPOINT + '/manifest.json').read_bytes()
    manifest = json.loads(manifest_raw)
    state_raw = D.read_pinned(D.CHECKPOINT + '/state.safetensors', D.CHECKPOINT_STATE_SHA)
    need(manifest['state_sha256'] == D.CHECKPOINT_STATE_SHA and manifest['parameter_sha256'] == D.CHECKPOINT_PARAMETER_SHA,
         'v1 final checkpoint identity')
    tensors = st_load(state_raw)
    base = M.PolicyLM(M.PolicyConfig(**manifest['model_config']))
    with torch.no_grad():
        for name, p in base.named_parameters():
            p.copy_(tensors['model.' + name])
    need(M.parameter_sha256(base) == D.CHECKPOINT_PARAMETER_SHA, 'restored v1 parameter sha')
    v2cfg_raw = D.mapped(D.V2_CONFIG).read_bytes()
    v2cfg = json.loads(v2cfg_raw)
    need(v2cfg['preprocessing']['sha256'] == D.PREPROCESSING_SHA and v2cfg['snapshot']['sha256'] == D.SPEC_SHA
         and v2cfg['objective_sha256'] == rdigest(OBJECTIVE), 'same normalizer, snapshot and L6 as the v2 build')
    R['inputs'] = {'checkpoint': {'path': D.CHECKPOINT, 'state_sha256': D.CHECKPOINT_STATE_SHA,
                                  'parameter_sha256': D.CHECKPOINT_PARAMETER_SHA,
                                  'manifest_sha256': D.sha(manifest_raw), 'step': manifest['step']},
                   'snapshot_spec_sha256': D.SPEC_SHA, 'preprocessing_sha256': D.PREPROCESSING_SHA,
                   'v2_config_sha256': D.sha(v2cfg_raw), 'l6_objective_sha256': rdigest(OBJECTIVE),
                   'l6_coefficients': OBJECTIVE['coefficients'], 'coverage_sha256': D.sha(cov_raw),
                   'select_sha256': D.sha(sel_raw), 'trie': {'sha256': trie.identity, 'programs': len(trie.programs)},
                   'rebased_rlfw_sha256': {f: D.sha((D.PIPELINES / 'rlfw' / f).read_bytes())
                                           for f in ('exit_targets.py', 'exit_objective.py')}}

    anchor = copy.deepcopy(base).to(device).eval()
    anchor.requires_grad_(False)
    arms = {}
    for spec_arm in cfg['arms']:
        need(len(spec_arm) in (3, 4), 'arm = [name, lambda_pi, lambda_kl] or [name, lambda_pi, lambda_kl, lambda_knn]')
        name, lam_pi, lam_kl = spec_arm[:3]
        lam_knn = spec_arm[3] if len(spec_arm) == 4 else 0.0
        net = copy.deepcopy(base).to(device)
        opt = torch.optim.AdamW(M.param_groups(net, manifest['weight_decay']), lr=cfg['learning_rate'],
                                weight_decay=manifest['weight_decay'])
        restored = 0
        if cfg['resume_v1_moments']:
            for pname, p in net.named_parameters():
                prefix = 'optimizer.' + pname + '.'
                st = {k[len(prefix):]: v for k, v in tensors.items() if k.startswith(prefix)}
                if st:
                    need(set(st) == {'step', 'exp_avg', 'exp_avg_sq'} and float(st['step']) <= manifest['step'],
                         'exact non-fused v1 AdamW state')
                    opt.state[p] = {'step': st['step'].clone(), 'exp_avg': st['exp_avg'].to(device).clone(),
                                    'exp_avg_sq': st['exp_avg_sq'].to(device).clone()}
                    restored += 1
        arms[name] = {'net': net, 'opt': opt, 'lambda_pi': lam_pi, 'lambda_kl': lam_kl, 'lambda_knn': lam_knn, 'steps': 0,
                      'frozen_at_epoch': None, 'restored_moments': restored,
                      'parameter_sha256': {'initial': M.parameter_sha256(net)}}
    ti = TrieIndex(trie, torch, device)
    R['inputs']['trie'].update(branching_nodes=ti.B, cover_leaves=len(ti.cover))

    contexts = s.contexts
    train_ids = sorted(c for c, r in cov['contexts'].items() if r['partition'] == 'train')
    val_ids = sorted(c for c, r in cov['contexts'].items() if r['partition'] == 'validation')
    need(all(contexts[c]['partition'] == 'train' for c in train_ids), 'train partition only for targets')
    need(all(contexts[c]['partition'] == 'validation' for c in val_ids), 'validation partition for evaluation')
    detail_train_all = list(train_ids)
    heldout_ids = []
    holdout = cfg.get('holdout')
    if holdout:
        # Revision 4: hold out whole parent lineages of the detail-shard train contexts, in a fixed digest order,
        # until at least max(min_contexts, ceil(fraction * total)) contexts are held out. Held-out contexts are
        # evaluation-only: no gradient ever uses them.
        lineage = {c: cov['contexts'][c]['lineage'] for c in train_ids}
        per = {}
        for c, lin in lineage.items():
            per[lin] = per.get(lin, 0) + 1
        want = max(holdout['min_contexts'], math.ceil(holdout['fraction'] * len(train_ids)))
        chosen, count = set(), 0
        for lin in sorted(per, key=lambda x: digest({'seed': D.SEED, 'holdout-lineage': x})):
            if count >= want:
                break
            chosen.add(lin)
            count += per[lin]
        heldout_ids = sorted(c for c in train_ids if lineage[c] in chosen)
        train_ids = sorted(c for c in train_ids if lineage[c] not in chosen)
        need(len(heldout_ids) >= holdout['min_contexts'] and not set(heldout_ids) & set(train_ids)
             and not {lineage[c] for c in heldout_ids} & {lineage[c] for c in train_ids}, 'lineage-disjoint holdout')
        R['holdout'] = {'method': 'whole parent lineages, digest order seed 20261009 key holdout-lineage',
                        'fraction': holdout['fraction'], 'min_contexts': holdout['min_contexts'], 'target': want,
                        'heldout_contexts': len(heldout_ids), 'heldout_lineages': len(chosen),
                        'train_contexts': len(train_ids), 'train_lineages': len(set(per) - chosen),
                        'heldout_sha256': digest(heldout_ids), 'train_sha256': digest(train_ids)}
        for key in ('train_sha256', 'heldout_sha256'):  # revision 5: optional pin to an earlier sweep's split
            if holdout.get('expect_' + key):
                need(R['holdout'][key] == holdout['expect_' + key], 'holdout split differs from the pinned split: ' + key)
    unused_b = sorted((c for c, r in contexts.items() if r['partition'] == 'train' and c not in set(detail_train_all)),
                      key=lambda c: digest({'seed': D.SEED, 'paired-set-b': c}))[:1000] if holdout else []
    # Revision 6: L6 may train on the full native train roster. "full_minus_heldout_lineages" drops every context of
    # a held-out parent lineage (not only the held-out detail contexts), so the held-out set stays lineage-held-out.
    # L_PI targets still come only from detail contexts in the training split (train_ids).
    l6_roster = cfg.get('l6_roster', 'detail')
    if l6_roster == 'full_minus_heldout_lineages':
        need(bool(heldout_ids), 'full-roster L6 requires the lineage holdout')
        held_lineages = {cov['contexts'][c]['lineage'] for c in heldout_ids}
        roster = sorted(c for c, r in contexts.items() if r['partition'] == 'train')
        l6_train_ids = sorted(c for c in roster if contexts[c]['parent_lineage_sha256'] not in held_lineages)
        need(set(train_ids) <= set(l6_train_ids) and not set(heldout_ids) & set(l6_train_ids), 'full-roster split')
        R['l6_roster'] = {'mode': l6_roster, 'native_train_roster': len(roster),
                          'excluded_heldout_lineage_contexts': len(roster) - len(l6_train_ids),
                          'excluded_non_detail_siblings': len(roster) - len(l6_train_ids) - len(heldout_ids),
                          'l6_train_contexts': len(l6_train_ids), 'l6_train_sha256': digest(l6_train_ids),
                          'lpi_target_train_contexts': len(train_ids),
                          'sample_b_trained': sum(1 for c in unused_b if c in set(l6_train_ids)),
                          'sample_b_in_heldout_lineages': sum(1 for c in unused_b if c not in set(l6_train_ids))}
    else:
        need(l6_roster == 'detail', 'known l6_roster')
        l6_train_ids = list(train_ids)
    zvec = lambda cid: norm(cid, s.encoding(cid))

    # Revision 7: R25-kNN-CE neighbour table, built once before any update, only when an arm uses it.
    knn, knn_teacher = None, {}
    if any(a['lambda_knn'] > 0 for a in arms.values()):
        check('knn table')
        kcfg = cfg['knn']
        need(kcfg.get('metric') == 'cosine_frozen_normalised_encoding' and int(kcfg['k']) >= 1, 'knn config')
        roster = set(l6_train_ids)
        excluded = set(contexts) - roster  # held-out lineages, validation and every other non-roster context
        need(not excluded & roster and set(val_ids) <= excluded and set(heldout_ids) <= excluded,
             'knn exclusions cover held-out lineages and validation')
        need(all(contexts[c]['partition'] == 'train' for c in roster), 'knn anchors are train contexts')
        t0 = time.time()
        # Neighbours need a frozen native HWM teacher that is a trie leaf; every roster context stays an anchor.
        teacher_of = {}
        for c in l6_train_ids:
            w = s.winners.get(c)
            toks = tuple(w.tokens) if w is not None and w.tokens else None
            if toks is not None and toks in ti.leaf_index:
                teacher_of[c] = list(toks)
        candidates = sorted(teacher_of)
        lineage_of = {c: contexts[c]['parent_lineage_sha256'] for c in roster}
        knn = KNN.build_knn_table(l6_train_ids, {c: zvec(c) for c in l6_train_ids}, lineage_of, excluded,
                                  int(kcfg['k']), torch, candidate_ids=candidates)
        KNN.verify_knn_table(knn, l6_train_ids, lineage_of, excluded, int(kcfg['k']), candidates)
        used = sorted({n for nbrs in knn.values() for n in nbrs})
        knn_teacher.update({c: teacher_of[c] for c in used})
        table_raw = (json.dumps({'schema': 'jope.r25-knn-table/v1', 'k': int(kcfg['k']), 'metric': kcfg['metric'],
                                 'anchors': len(knn), 'table': knn}, sort_keys=True, indent=1) + '\n').encode()
        (out / 'knn-table.json').write_bytes(table_raw)
        R['knn'] = {'k': int(kcfg['k']), 'metric': kcfg['metric'], 'anchors': len(knn),
                    'candidates_with_trie_leaf_teacher': len(candidates),
                    'roster_without_trie_leaf_teacher': len(l6_train_ids) - len(candidates),
                    'distinct_neighbours': len(used), 'table_digest': KNN.table_digest(knn),
                    'file_sha256': D.sha(table_raw), 'excluded_contexts': len(excluded),
                    'validation_excluded': len(val_ids), 'heldout_excluded': len(heldout_ids),
                    'teacher_programs': len({tuple(t) for t in knn_teacher.values()}),
                    'seconds': time.time() - t0}
        log('knn_table', **R['knn'])
        expect = kcfg.get('expect_table_digest')
        if expect:
            need(R['knn']['table_digest'] == expect, 'knn table differs from the pre-registered digest')

    def knn_value(net, cid):
        """CE_kNN(c): mean trie-children NLL, under c's context, of c's neighbours' HWM teacher programs."""
        ll = program_logprob(net, zvec(cid), [knn_teacher[n] for n in knn[cid]], ti, torch, device)
        return -ll.mean()

    spec = json.loads(D.read_pinned(D.SPEC, D.SPEC_SHA))
    role_sha = spec['authority_pins']['role']['sha256']
    evaluator = spec['tails'][0]['header']['sha256']
    canonicalizer = spec['grammar']['sha256']
    tc = TargetConfig(D.CHECKPOINT_PARAMETER_SHA, trie.identity, s.vocab.sha256, D.PREPROCESSING_SHA, role_sha,
                      evaluator, canonicalizer, temperature=cfg['tau'], alpha=cfg['alpha'],
                      advantage_clip=cfg['advantage_clip'], form='set', clip_center='hwm')
    anchor_lp = {}

    def anchor_logprobs(cid, programs):
        check('anchor likelihood')
        key = (cid, tuple(tuple(p) for p in programs))
        if key not in anchor_lp:
            with torch.no_grad():
                anchor_lp[key] = [float(v) for v in program_logprob(anchor, zvec(cid), programs, ti, torch, device)]
        return anchor_lp[key]

    def hwm_key(cid):
        c = contexts[cid]
        return rdigest({'context': cid, 'dataset_hash': c['dataset_hash'], 'cut_id': c['cut_id'], 'fold': c['fold'],
                        'protocol': tc.protocol, 'evaluator': evaluator, 'vocabulary': s.vocab.sha256})

    def rows_for(cid):
        check('target rows')
        c = cov['contexts'][cid]
        return [{'context_id': cid, 'lineage_id': c['lineage'], 'role': 'train', 'source': 'measured',
                 'outcome_class': 'qualified_complete', 'evaluator_sha256': evaluator,
                 'canonicalizer_sha256': canonicalizer, 'protocol': tc.protocol, 'role_map_sha256': role_sha,
                 'hwm_key_sha256': hwm_key(cid),
                 'receipt_sha256': rdigest({'shard': c['shard_sha256'], 'program': m['program_sha256']}),
                 'q': m['q'], 'q_opt': m['q_opt'], 'q_grid': m['q_grid'],
                 'program': {'dsl': trie.programs[tuple(m['tokens'])]['dsl'], 'program_sha256': m['program_sha256'],
                             'tokens': m['tokens']}} for m in c['measured_in_trie']]

    def scratch_verifier(row):
        c = cov['contexts'][row['context_id']]
        need(row['receipt_sha256'] == rdigest({'shard': c['shard_sha256'], 'program': row['program']['program_sha256']}),
             'row not from its sha-verified detail shard')
        return {**{k: row[k] for k in ('context_id', 'lineage_id', 'hwm_key_sha256', 'receipt_sha256', 'q', 'q_opt',
                                       'q_grid')}, 'program_sha256': row['program']['program_sha256'],
                'qualified': True, 'closure_verified': True, 'parity_verified': True, 'parity_tolerance': 1e-9}

    def incumbent(cid, key):
        inc = cov['contexts'][cid]['incumbent']
        if inc is None:
            return None
        return {'context_id': cid, 'hwm_key_sha256': key, 'receipt_sha256': rdigest({'incumbent': cid,
                'shard': cov['contexts'][cid]['shard_sha256']}), 'q': inc['q'], 'q_opt': inc['q_opt'],
                'q_grid': inc['q_grid'], 'closure_verified': True}

    def role_check(cid):
        return {'role': contexts[cid]['partition'], 'role_map_sha256': role_sha}

    def build(cids):
        rows = [r for cid in cids if cov['contexts'][cid]['counts']['eligible'] for r in rows_for(cid)]
        return build_targets(rows, tc, trie=trie, role_check=role_check, verify_receipt=scratch_verifier,
                             anchor_logprob=lambda cid, toks: anchor_logprobs(cid, [list(toks)])[0],
                             incumbent=incumbent)

    def as_target(c):
        return {'programs': [p['program']['tokens'] for p in c['programs']],
                'weights': [p['weight'] for p in c['programs']], 'clip_center_source': c['clip_center_source']}

    def lpi_value(net, cid, target, grad):
        """L_PI(c) = -sum_p rho_c(p) * l_theta(c, p) over the target's measured trie leaves."""
        with torch.set_grad_enabled(grad):
            ll = program_logprob(net, zvec(cid), target['programs'], ti, torch, device)
            return -(torch.tensor(target['weights'], dtype=torch.float64, device=device) * ll).sum()

    # Fatal cross-checks against the rebased exit_objective functions, before any target build or update.
    t0 = time.time()
    xc_ids = [c for c in train_ids if cov['contexts'][c]['counts']['eligible']][:2]
    xc_built = build(xc_ids)
    xc = {c['context_id']: c for c in xc_built['manifest']['contexts'] if c['eligible']}
    need(len(xc) == 2, 'two eligible cross-check contexts')
    probe_net = arms[next(iter(arms))]['net']
    probe_net.eval()
    tp = TensorPolicy(type('P', (), {'net': probe_net, 'trie': trie, 'torch': torch})())
    inject = 1e-3 if R['injected_mismatch'] else 0.0
    cross = {'contexts': xc_ids}
    with torch.no_grad():
        diffs, pis = [], []
        for cid in xc_ids:
            check('cross-check likelihood')
            progs = as_target(xc[cid])['programs'][:3]
            batched = program_logprob(probe_net, zvec(cid), progs, ti, torch, device)
            ref = [float(trie_logprob(tp, zvec(cid), p, trie)[0]) + inject for p in progs]
            diffs.extend(abs(float(a) - b) for a, b in zip(batched, ref))
            pi_ref = float(policy_imitation(tp, {cid: zvec(cid)}, [xc[cid]], trie, check_train=lambda c: None))
            pis.append(abs(float(lpi_value(probe_net, cid, as_target(xc[cid]), False)) - pi_ref))
        cross['trie_logprob_max_abs_diff'] = max(diffs)
        cross['policy_imitation_abs_diff'] = max(pis)
        cid = xc_ids[0]
        progs = as_target(xc[cid])['programs'][:1]
        toks = torch.tensor(progs, device=device)
        lens = torch.tensor([len(progs[0])], device=device)
        ctx = torch.as_tensor(zvec(cid), device=device).reshape(1, -1)
        seq = torch.zeros(1, dtype=torch.long, device=device)
        h = hidden(probe_net, toks, lens, ctx, seq, torch)
        full = probe_net(toks, lens, ctx, seq).logits
        cross['hidden_vs_forward_logits_max_abs_diff'] = float((probe_net.lm_head(h[:, 1:len(progs[0])]) - full).abs().max())
        perturbed = copy.deepcopy(probe_net)
        for p in perturbed.parameters():
            p.add_(1e-3 * torch.sin(torch.arange(p.numel(), device=device, dtype=p.dtype).reshape(p.shape)))
        ppol = TensorPolicy(type('P', (), {'net': perturbed, 'trie': trie, 'torch': torch})())
        apol = TensorPolicy(type('P', (), {'net': anchor, 'trie': trie, 'torch': torch})())
        exact_ref = float(exact_trie_kl(ppol, apol, {cid: zvec(cid)}, trie, deadline=HARD_DEADLINE,
                                        check_train=lambda c: None))
        lp = all_nodes(perturbed, zvec(cid), ti, torch, device)
        la = all_nodes(anchor, zvec(cid), ti, torch, device)
        kl, ent, _ = kl_entropy(lp, la, ti, torch)
        flat = lp.reshape(-1).masked_fill(~ti.mask.reshape(-1), 0.0)
        mass = 0.0
        for e in ti.leaf_edges:
            mass += math.exp(float(flat.index_select(0, torch.tensor(e, device=device)).sum())) if e else 1.0
        check('cross-check closure')
        cross.update(exact_trie_kl_batched=float(kl), exact_trie_kl_rebased=exact_ref,
                     exact_trie_kl_abs_diff=abs(float(kl) - exact_ref),
                     exact_trie_kl_rel_diff=abs(float(kl) - exact_ref) / max(abs(exact_ref), 1e-300),
                     leaf_probability_closure=mass, leaf_probability_closure_abs_diff=abs(mass - 1.0),
                     entropy_nats=float(ent), seconds=time.time() - t0)
        del perturbed
    probe_net.train()
    failures = {k: cross[k] for k, tol in TOLERANCE.items()
                if not (math.isfinite(cross[k]) and cross[k] <= tol)}
    if failures.get('exact_trie_kl_rel_diff') is not None and cross['exact_trie_kl_abs_diff'] <= TOLERANCE['exact_trie_kl_abs_diff']:
        failures.pop('exact_trie_kl_rel_diff')  # either the absolute or the relative bound suffices
    if failures.get('exact_trie_kl_abs_diff') is not None and cross['exact_trie_kl_rel_diff'] <= TOLERANCE['exact_trie_kl_rel_diff']:
        failures.pop('exact_trie_kl_abs_diff')
    cross['passed'] = not failures
    cross['failures'] = failures
    R['crosschecks'] = cross
    log('crosschecks', **cross)
    if failures:
        raise CrosscheckFailed(json.dumps(failures, sort_keys=True))

    # Targets: train via rebased build_targets; validation with the same tilt, evaluation only.
    t0 = time.time()
    built = build(train_ids)
    targets = {c['context_id']: as_target(c) for c in built['manifest']['contexts'] if c['eligible']}
    heldout_targets = {}
    if heldout_ids:  # evaluation-only tilt targets on held-out lineages (same builder; never a gradient)
        held_built = build(heldout_ids)
        heldout_targets = {c['context_id']: as_target(c) for c in held_built['manifest']['contexts'] if c['eligible']}
        need(not set(heldout_targets) & set(targets), 'held-out targets disjoint from training targets')
        R['holdout']['heldout_target_sha256'] = held_built['target_sha256']
    val_targets = {}
    for cid in val_ids:
        check('validation targets')
        c = cov['contexts'][cid]
        if not c['counts']['eligible']:
            continue
        ms = c['measured_in_trie']
        progs = [m['tokens'] for m in ms]
        w, _ = tilt_weights([m['q'] for m in ms], [m['q_opt'] for m in ms], anchor_logprobs(cid, progs), tc,
                            center=None if c['incumbent'] is None else c['incumbent']['q_opt'])
        val_targets[cid] = {'programs': progs, 'weights': list(w)}
    R['targets'] = {'train_eligible': len(targets), 'validation_eligible_eval_only': len(val_targets),
                    'target_sha256': built['target_sha256'], 'target_config': built['manifest']['config'],
                    'clip_center_sources': {k: sum(1 for c in targets.values() if c['clip_center_source'] == k)
                                            for k in ('comparable_hwm', 'measured_mean_null_hwm')},
                    'seconds': time.time() - t0}
    R['coverage_totals'] = cov['totals']
    log('targets', **{k: v for k, v in R['targets'].items() if k != 'target_config'})

    probes = cfg['probe_contexts']
    probe_ids = sorted(train_ids, key=lambda c: digest({'seed': D.SEED, 'probe': c}))[:probes]
    anchor_nodes = {}

    def anchor_all(cid):
        if cid in anchor_nodes:
            return anchor_nodes[cid]
        with torch.no_grad():
            value = all_nodes(anchor, zvec(cid), ti, torch, device)
        if cid in probe_ids:
            anchor_nodes[cid] = value
        return value

    def probe(net):
        net.eval()
        kls, ents = [], []
        with torch.no_grad():
            for cid in probe_ids:
                check('probe')
                kl, ent, _ = kl_entropy(all_nodes(net, zvec(cid), ti, torch, device), anchor_all(cid), ti, torch)
                kls.append(float(kl))
                ents.append(float(ent))
        net.train()
        return {'kl_mean': sum(kls) / len(kls), 'kl_max': max(kls), 'entropy_mean': sum(ents) / len(ents),
                'contexts': len(kls)}

    train_panel, dev_panel = sel['panels']['train'], sel['panels']['validation']
    l6cfg = {'batch_size': 16, 'max_padded_tokens': 8192}
    check('panels')
    panels = {role: prepare_microbatches(s, ids, epoch=0, batch_index=0, normalizer=norm, max_padded_tokens=8192)[:2]
              for role, ids in (('train_panel', train_panel), ('validation_panel', dev_panel))}
    dev_all = sorted(c for c, r in contexts.items() if r['partition'] == 'validation')
    if cfg.get('smoke_dev_contexts'):
        dev_all = dev_all[:cfg['smoke_dev_contexts']]
    heldout_eval = heldout_ids[:cfg['smoke_eval_contexts']] if cfg.get('smoke_eval_contexts') else heldout_ids
    unused_eval = unused_b[:cfg['smoke_eval_contexts']] if cfg.get('smoke_eval_contexts') else unused_b
    # Revision 5: optional per-context rows on full validation (set a, the same contexts as l6_full_validation).
    val_eval = (dev_all[:cfg['smoke_eval_contexts']] if cfg.get('smoke_eval_contexts') else dev_all) \
        if heldout_ids and 'a' in cfg.get('paired_sets', []) else []
    per_context = {}
    for cid in heldout_eval + unused_eval + val_eval:
        check('prepare per-context eval')
        per_context[cid] = prepare_microbatches(s, [cid], epoch=0, batch_index=0, normalizer=norm,
                                                max_padded_tokens=8192)[:2]
    paired_rows = (out / 'paired-rows.jsonl').open('x') if heldout_ids else None
    if heldout_ids:
        replace_index = {'schema': 'jope.r22-lpi-scratch-paired-index/v1', **D.R22, 'run': str(out),
                         'row_format': 'paired_eval.py rows: checkpoint, arm, epoch, updates, parameter_sha256, set, '
                                       'set_list, rank, context_id, l6_loss, terms, numerators, denominators, lpi, '
                                       'lpi_programs',
                         'sets': {'heldout': {'contexts': len(heldout_eval), 'sha256': digest(heldout_eval),
                                              'definition': 'lineage-held-out detail-shard train contexts; L6 and '
                                                            'L_PI per context; evaluation-only'},
                                  'b': {'contexts': len(unused_eval), 'sha256': digest(unused_eval),
                                        'definition': ('trained full-roster contexts sample: the same first 1,000 '
                                                       'of paired_eval set b; with full-roster L6 these are trained '
                                                       'except those in held-out lineages (see receipt l6_roster); '
                                                       'L6 per context')
                                        if l6_roster != 'detail' else
                                        'first 1,000 of paired_eval set b (scratch-unused train, seen '
                                        'by the v1 base); L6 per context; paired differences only'},
                                  **({'a': {'contexts': len(val_eval), 'sha256': digest(val_eval),
                                            'definition': 'all native validation contexts (full validation); L6 per '
                                                          'context, plus L_PI where a validation target exists; '
                                                          'evaluation-only'}} if val_eval else {})},
                         'checkpoint_index': 'checkpoints-index.json'}
        (out / 'paired-index.json').write_text(json.dumps(replace_index, indent=1, sort_keys=True) + '\n')
        for name, ids in (('heldout', heldout_eval), ('b', unused_eval), ('a', val_eval)):
            (out / f'contexts-{name}.json').write_text(json.dumps(ids) + '\n')

    def per_context_eval(net, name, epoch, res):
        """Rows per context on the held-out and unused-train sets, plus macro aggregates (no gradient)."""
        from neural_native_batch import OBJECTIVE
        meta = {'checkpoint': 'start' if epoch == 0 else f'epoch_{epoch:03d}', 'arm': 'all' if epoch == 0 else name,
                'epoch': epoch, 'updates': arms[name]['steps'], 'parameter_sha256': M.parameter_sha256(net)}
        for set_name, ids in (('heldout', heldout_eval), ('b', unused_eval), ('a', val_eval)):
            if not ids:
                continue
            num, den, lpis = {}, {}, []
            for rank, cid in enumerate(ids):
                check('per-context eval ' + set_name)
                batches, total = per_context[cid]
                r = run_microbatches(net, batches, total, deadline=HARD_DEADLINE)
                row = {**meta, 'set': set_name, 'set_list': set_name, 'rank': rank, 'context_id': cid,
                       'l6_loss': r['loss'], 'terms': r['terms'], 'numerators': r['numerators'],
                       'denominators': r['denominators']}
                for k, v in r['numerators'].items():
                    num[k] = num.get(k, 0.0) + v
                    den[k] = den.get(k, 0) + r['denominators'][k]
                eval_targets = {'heldout': heldout_targets, 'a': val_targets}.get(set_name, {})
                if cid in eval_targets:  # evaluation-only L_PI (held-out lineages; validation targets on set a)
                    net.eval()
                    with torch.no_grad():
                        v = float(lpi_value(net, cid, eval_targets[cid], False))
                    row.update(lpi=v, lpi_programs=len(eval_targets[cid]['programs']))
                    lpis.append(v)
                paired_rows.write(json.dumps(row, sort_keys=True) + '\n')
            terms = {k: num[k] / den[k] if den[k] else 0.0 for k in num}
            res[f'{set_name}_l6'] = {'loss': sum(OBJECTIVE['coefficients'][k] * v for k, v in terms.items()),
                                     'terms': terms, 'contexts': len(ids)}
            if set_name in ('heldout', 'a'):
                res[f'{set_name}_lpi'] = {'mean': sum(lpis) / len(lpis) if lpis else None, 'contexts': len(lpis)}
        paired_rows.flush()

    def evaluate(name, epoch):
        net = arms[name]['net']
        t = time.time()
        res = {'arm': name, 'epoch': epoch, 'updates': arms[name]['steps']}
        for role, (batches, total) in panels.items():
            check('eval panel')
            r = run_microbatches(net, batches, total, deadline=HARD_DEADLINE)
            res['l6_' + role] = {'loss': r['loss'], 'terms': r['terms']}
        check('eval full validation')
        full = evaluate_full_validation(s, net, dev_all, epoch=0, normalizer=norm, config=l6cfg, deadline=HARD_DEADLINE)
        res['l6_full_validation'] = {'loss': full['loss'], 'terms': full['terms'], 'contexts': full['context_count']}
        net.eval()
        vals, trs = [], []
        with torch.no_grad():
            for cid, t_ in sorted(val_targets.items()):
                check('eval L_PI validation')
                vals.append(float(lpi_value(net, cid, t_, False)))
            for i, (cid, t_) in enumerate(sorted(targets.items())):
                if cfg.get('smoke_eval_contexts') and i >= cfg['smoke_eval_contexts']:
                    break
                check('eval L_PI train')
                trs.append(float(lpi_value(net, cid, t_, False)))
        net.train()
        res['lpi_validation_eval_only'] = {'mean': sum(vals) / len(vals) if vals else None, 'contexts': len(vals)}
        res['lpi_train_targets'] = {'mean': sum(trs) / len(trs) if trs else None, 'contexts': len(trs)}
        if heldout_ids:
            per_context_eval(net, name, epoch, res)
        res['probe'] = probe(net)
        res['kl_alarm'] = res['probe']['kl_mean'] >= cfg['kl_alarm']
        res['kl_disqualify'] = res['probe']['kl_mean'] >= cfg['kl_disqualify']
        res['parameter_sha256'] = M.parameter_sha256(net)
        res['seconds'] = time.time() - t
        log('eval', **res)
        return res

    R['initial_eval'] = [evaluate(next(iter(arms)), 0)]  # all arms start from the identical restored state

    def replace_json(path, value):
        tmp = path.with_name(path.name + '.pending')
        tmp.write_bytes((json.dumps(value, indent=1, sort_keys=True, default=str) + '\n').encode())
        os.replace(tmp, path)

    ck_index = {'schema': 'jope.r22-lpi-scratch-checkpoint-index/v1', **D.R22, 'run': str(out),
                'read_only_for_other_lanes': True,
                'origin': {'path': D.CHECKPOINT, 'state_sha256': D.CHECKPOINT_STATE_SHA,
                           'parameter_sha256': D.CHECKPOINT_PARAMETER_SHA, 'updates': 0,
                           'note': 'epoch 0 for every arm; the sealed v1 final checkpoint'},
                'model_config': manifest['model_config'],
                'data_manifest': {'path': 'frozen/run-20261010T174612Z/inputs-xbabe2.sha256 (lane) and '
                                          'xbabe3:' + str(D.INPUTS), 'sha256': cfg.get('data_manifest_sha256')},
                'tensor_keys': 'model.<param>, optimizer.<param>.{step,exp_avg,exp_avg_sq}, rng.torch_cpu, '
                               'rng.torch_cuda.0', 'checkpoints': []}

    def save_checkpoint(name, tag, epoch):
        check('checkpoint')
        arm = arms[name]
        d = out / 'checkpoints' / tag / name
        d.mkdir(parents=True, exist_ok=False)
        state = {'model.' + k: v.detach().cpu().contiguous() for k, v in arm['net'].named_parameters()}
        for pname, p in arm['net'].named_parameters():
            for key, value in arm['opt'].state.get(p, {}).items():
                state[f'optimizer.{pname}.{key}'] = value.detach().cpu().contiguous()
        state['rng.torch_cpu'] = torch.get_rng_state()
        if use_cuda:
            state['rng.torch_cuda.0'] = torch.cuda.get_rng_state(0)
        save_file(state, str(d / 'state.safetensors'))
        psha = M.parameter_sha256(arm['net'])
        meta = {'schema': 'jope.r22-lpi-scratch-checkpoint/v1', **D.R22, 'arm': name, 'epoch': epoch, 'tag': tag,
                'updates': arm['steps'], 'lambda_pi': arm['lambda_pi'], 'lambda_kl': arm['lambda_kl'],
                'learning_rate': cfg['learning_rate'], 'weight_decay': manifest['weight_decay'],
                'model_config': manifest['model_config'], 'parameter_sha256': psha,
                'state_sha256': D.sha((d / 'state.safetensors').read_bytes()),
                'origin_parameter_sha256': D.CHECKPOINT_PARAMETER_SHA, 'run_config_sha256': R['config']['sha256'],
                'frozen_at_epoch': arm['frozen_at_epoch']}
        (d / 'manifest.json').write_text(json.dumps(meta, indent=1, sort_keys=True) + '\n')
        pins = ''.join(D.sha(p.read_bytes()) + '  ' + p.name + '\n' for p in sorted(d.iterdir()))
        (d / 'MANIFEST.sha256').write_text(pins)
        for p in d.iterdir():
            p.chmod(0o444)
        ck_index['checkpoints'].append({'arm': name, 'epoch': epoch, 'tag': tag, 'updates': arm['steps'],
                                        'path': str(d), 'parameter_sha256': psha, 'state_sha256': meta['state_sha256'],
                                        'manifest_sha256': D.sha((d / 'manifest.json').read_bytes())})
        replace_json(out / 'checkpoints-index.json', ck_index)

    def write_progress():
        rows = [{'epoch': 0, 'arm': 'all', 'updates': 0, **{k: v for k, v in table_row(R['initial_eval'][0]).items()}}]
        for ep in R['epochs']:
            for e in ep['eval']:
                rows.append({'epoch': ep['epoch'], 'arm': e['arm'], 'updates': e['updates'],
                             'epoch_complete': ep['complete'], **table_row(e),
                             'early_stop_kl': e['kl_disqualify']})
        replace_json(out / 'progress.json', {'schema': 'jope.r22-lpi-scratch-progress/v1', **D.R22,
                                             'updates_per_epoch_per_arm': steps_per_epoch, 'rows': rows,
                                             'frozen': {n: a['frozen_at_epoch'] for n, a in arms.items()},
                                             'updated_unix': time.time()})

    def table_row(e):
        return {'lpi_validation': e['lpi_validation_eval_only']['mean'], 'lpi_train': e['lpi_train_targets']['mean'],
                'l6_validation_panel': e['l6_validation_panel']['loss'], 'l6_train_panel': e['l6_train_panel']['loss'],
                'l6_full_validation_263': e['l6_full_validation']['loss'], 'kl_mean': e['probe']['kl_mean'],
                'kl_max': e['probe']['kl_max'], 'entropy': e['probe']['entropy_mean'],
                **({'heldout_lpi': e['heldout_lpi']['mean'], 'heldout_l6': e['heldout_l6']['loss'],
                    'unused_train_1000_l6': e['b_l6']['loss']} if 'heldout_lpi' in e else {}),
                **({'full_validation_per_context_l6': e['a_l6']['loss']} if 'a_l6' in e else {})}

    replace_json(out / 'checkpoints-index.json', ck_index)
    batch = cfg['batch_contexts']
    steps_per_epoch = smoke or math.ceil(len(l6_train_ids) / batch)
    R.update(steps_per_epoch=steps_per_epoch, selected_train_contexts=len(l6_train_ids),
             lpi_target_train_contexts=len(train_ids),
             validation_contexts_eval_only=len(val_ids), l6_full_validation_contexts=len(dev_all), epochs=[],
             stop_reason='max_epochs')
    for epoch in range(1, cfg['max_epochs'] + 1):
        e_start = time.time()
        order = sorted(l6_train_ids, key=lambda c: digest({'seed': D.SEED, 'epoch': epoch, 'context': c}))
        sums = {n: {'l6': 0.0, 'terms': {k: 0.0 for k in OBJECTIVE['coefficients']}, 'lpi': 0.0, 'lpi_steps': 0,
                    'kl': 0.0, 'kl_steps': 0, 'knn': 0.0, 'knn_steps': 0, 'grad_norm': 0.0, 'steps': 0, 'ratio': [],
                    'cosine': []} for n in arms}
        stopped = False
        for b in range(steps_per_epoch):
            if time.time() >= train_end:
                R['stop_reason'] = 'training_deadline_mid_epoch'
                stopped = True
                break
            cap = cfg.get('max_updates')  # revision 7: stop at an exact update count (matched-control length)
            if cap and all(a['steps'] >= cap for a in arms.values() if a['frozen_at_epoch'] is None):
                R['stop_reason'] = 'max_updates'
                stopped = True
                break
            check('training step')
            ids = order[b * batch:(b + 1) * batch]
            batches, total = prepare_microbatches(s, ids, epoch=epoch - 1, batch_index=b, normalizer=norm,
                                                  max_padded_tokens=8192)[:2]
            step_seed = D.SEED * 1000 + (epoch - 1) * steps_per_epoch + b
            for name, arm in arms.items():
                if arm['frozen_at_epoch'] is not None:
                    continue
                if cfg.get('max_updates') and arm['steps'] >= cfg['max_updates']:
                    continue
                check('training arm')
                net, opt = arm['net'], arm['opt']
                before = M.parameter_sha256(net)
                net.train()
                opt.zero_grad(set_to_none=True)
                torch.manual_seed(step_seed)
                numer = {k: 0.0 for k in total}
                for cpu in batches:
                    check('training microbatch')
                    bb = cpu.to(device)
                    o = net(bb.tokens, bb.lengths, bb.ctx_z, bb.seq_ctx)
                    _, report, parts = l6_objective(o, bb)
                    loss = sum(OBJECTIVE['coefficients'][k] * n / total[k] for k, (n, d) in parts.items() if total[k])
                    loss.backward()
                    for k, v in report['numerators'].items():
                        numer[k] += v
                terms = {k: numer[k] / total[k] if total[k] else 0.0 for k in total}
                l6 = sum(OBJECTIVE['coefficients'][k] * v for k, v in terms.items())
                sel_t = [(cid, targets[cid]) for cid in ids if cid in targets]
                actor = arm['lambda_pi'] > 0 or arm['lambda_kl'] > 0 or arm['lambda_knn'] > 0
                if actor:
                    l6_grads = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                                for p in net.parameters()]
                net.eval()
                lpi = None
                if sel_t:
                    grad = arm['lambda_pi'] > 0
                    with torch.set_grad_enabled(grad):
                        vals = []
                        for cid, t_ in sel_t:
                            check('training L_PI')
                            vals.append(lpi_value(net, cid, t_, grad))
                        lpi_t = torch.stack(vals).mean()
                    if grad:
                        (arm['lambda_pi'] * lpi_t).backward()
                    lpi = float(lpi_t.detach())
                kl = None
                if arm['lambda_kl'] > 0:
                    kls = []
                    for cid in ids[:cfg['kl_contexts_per_step']]:
                        check('training KL')
                        k_, _, _ = kl_entropy(all_nodes(net, zvec(cid), ti, torch, device), anchor_all(cid), ti, torch)
                        kls.append(k_)
                    kl_t = torch.stack(kls).mean()
                    (arm['lambda_kl'] * kl_t).backward()
                    kl = float(kl_t.detach())
                knn_v = None
                if arm['lambda_knn'] > 0:  # revision 7: R25-kNN-CE over every batch context (all are roster anchors)
                    vals = []
                    for cid in ids:
                        check('training kNN-CE')
                        vals.append(knn_value(net, cid))
                    knn_t = torch.stack(vals).mean()
                    (arm['lambda_knn'] * knn_t).backward()
                    knn_v = float(knn_t.detach())
                net.train()
                need(all(p.grad is None or torch.isfinite(p.grad).all().item() for p in net.parameters()),
                     'finite gradients')
                ratio = cosine = None
                if actor:
                    total_g = [p.grad if p.grad is not None else torch.zeros_like(p) for p in net.parameters()]
                    l6n = math.sqrt(sum(float((g.double() ** 2).sum()) for g in l6_grads))
                    act = [t_ - g for t_, g in zip(total_g, l6_grads)]
                    actn = math.sqrt(sum(float((a.double() ** 2).sum()) for a in act))
                    dot = sum(float((a.double() * g.double()).sum()) for a, g in zip(act, l6_grads))
                    ratio = actn / l6n if l6n else None
                    cosine = dot / (actn * l6n) if actn and l6n else None
                    del l6_grads, act
                gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0))
                opt.step()
                arm['steps'] += 1
                after = M.parameter_sha256(net)
                sm = sums[name]
                sm['l6'] += l6
                for k in terms:
                    sm['terms'][k] += terms[k]
                if lpi is not None:
                    sm['lpi'] += lpi
                    sm['lpi_steps'] += 1
                if kl is not None:
                    sm['kl'] += kl
                    sm['kl_steps'] += 1
                if knn_v is not None:
                    sm['knn'] += knn_v
                    sm['knn_steps'] += 1
                sm['grad_norm'] += gn
                sm['steps'] += 1
                if ratio is not None:
                    sm['ratio'].append(ratio)
                if cosine is not None:  # undefined when the actor gradient is exactly zero (e.g. KL at the anchor)
                    sm['cosine'].append(cosine)
                elif ratio is not None:
                    sm['cosine_undefined_steps'] = sm.get('cosine_undefined_steps', 0) + 1
                total_loss = (l6 + arm['lambda_pi'] * (lpi or 0.0) + arm['lambda_kl'] * (kl or 0.0)
                              + arm['lambda_knn'] * (knn_v or 0.0))
                log('step', arm=name, epoch=epoch, update=arm['steps'], l6=l6, terms=terms, lpi=lpi,
                    lpi_contexts=len(sel_t), kl=kl, knn_ce=knn_v, total=total_loss, grad_norm_before_clip=gn,
                    actor_to_l6_grad_norm_ratio=ratio, actor_l6_grad_cosine=cosine,
                    parameters_before=before, parameters_after=after, parameters_changed=before != after)
        epoch_seconds = time.time() - e_start
        summary = {}
        for name, sm in sums.items():
            n = sm['steps']
            if not n:
                summary[name] = {'updates': arms[name]['steps'], 'frozen_at_epoch': arms[name]['frozen_at_epoch']}
                continue
            summary[name] = {'updates': arms[name]['steps'], 'updates_this_epoch': n, 'train_l6_mean': sm['l6'] / n,
                             'train_terms_mean': {k: v / n for k, v in sm['terms'].items()},
                             'train_lpi_mean': sm['lpi'] / sm['lpi_steps'] if sm['lpi_steps'] else None,
                             'lpi_steps': sm['lpi_steps'],
                             'train_kl_mean': sm['kl'] / sm['kl_steps'] if sm['kl_steps'] else None,
                             'train_knn_ce_mean': sm['knn'] / sm['knn_steps'] if sm['knn_steps'] else None,
                             'train_objective_mean': sm['l6'] / n + arms[name]['lambda_pi'] * (
                                 sm['lpi'] / sm['lpi_steps'] if sm['lpi_steps'] else 0.0) + arms[name]['lambda_kl'] * (
                                 sm['kl'] / sm['kl_steps'] if sm['kl_steps'] else 0.0) + arms[name]['lambda_knn'] * (
                                 sm['knn'] / sm['knn_steps'] if sm['knn_steps'] else 0.0),
                             'grad_norm_mean': sm['grad_norm'] / n,
                             'actor_to_l6_grad_norm_ratio_mean': sum(sm['ratio']) / len(sm['ratio']) if sm['ratio'] else None,
                             'actor_l6_grad_cosine_mean': sum(sm['cosine']) / len(sm['cosine']) if sm['cosine'] else None,
                             'actor_l6_grad_cosine_undefined_steps': sm.get('cosine_undefined_steps', 0)}
        ev = [evaluate(name, epoch) for name, arm in arms.items() if arm['frozen_at_epoch'] is None]
        for e in ev:
            arms[e['arm']]['parameter_sha256'][f'epoch_{epoch}'] = e['parameter_sha256']
        R['epochs'].append({'epoch': epoch, 'complete': not stopped, 'seconds': epoch_seconds, 'train': summary,
                            'eval': ev})
        R['epochs_completed'] = sum(1 for e in R['epochs'] if e['complete'])
        R['updates_per_arm'] = {n: a['steps'] for n, a in arms.items()}
        log('epoch', epoch=epoch, seconds=epoch_seconds, complete=not stopped, train=summary)
        for e in ev:
            if e['kl_disqualify'] and arms[e['arm']]['frozen_at_epoch'] is None:
                arms[e['arm']]['frozen_at_epoch'] = epoch  # P5 rule: probe KL >= 0.05 disqualifies; arm stops
                arms[e['arm']]['frozen_reason'] = 'kl_stop'
        for name, cap in cfg.get('arm_max_epochs', {}).items():  # revision 6: per-arm epoch budget
            if name in arms and arms[name]['frozen_at_epoch'] is None and epoch >= cap:
                arms[name]['frozen_at_epoch'] = epoch
                arms[name]['frozen_reason'] = 'arm_max_epochs'
        for e in ev:
            save_checkpoint(e['arm'], f'epoch_{epoch:03d}' if not stopped else f'partial_epoch_{epoch:03d}', epoch)
        write_progress()
        if stopped:
            break
        if all(a['frozen_at_epoch'] is not None for n, a in arms.items()
               if a['lambda_pi'] > 0 or a['lambda_kl'] > 0 or a['lambda_knn'] > 0):
            R['stop_reason'] = 'kl_disqualify_all_treatment_arms'
            break
    states = {}
    for name, arm in arms.items():
        check('save')
        path = out / f'final-{name}.safetensors'
        save_file({'model.' + k: v.detach().cpu().contiguous() for k, v in arm['net'].named_parameters()}, str(path))
        final = {k: v.detach().cpu() for k, v in arm['net'].named_parameters()}
        init = {k[6:]: v for k, v in tensors.items() if k.startswith('model.')}
        d2 = sum(float(((final[k].double() - init[k].double()) ** 2).sum()) for k in init)
        i2 = sum(float((init[k].double() ** 2).sum()) for k in init)
        arm['parameter_sha256']['final'] = M.parameter_sha256_tensors(final)
        states[name] = {'path': str(path), 'sha256': D.sha(path.read_bytes()), 'parameter_sha256': arm['parameter_sha256'],
                        'tensors_changed': sum(1 for k in init if not torch.equal(init[k], final[k])),
                        'tensors': len(init), 'relative_delta_l2': (d2 / i2) ** 0.5,
                        'max_abs_delta': max(float((final[k] - init[k]).abs().max()) for k in init),
                        'restored_v1_moments': arm['restored_moments']}
    R['final_states'] = states
    if paired_rows is not None:
        paired_rows.flush()
        os.fsync(paired_rows.fileno())
        paired_rows.close()
        R['paired_rows'] = {'path': str(out / 'paired-rows.jsonl'),
                            'sha256': D.sha((out / 'paired-rows.jsonl').read_bytes()),
                            'rows': sum(1 for _ in (out / 'paired-rows.jsonl').open()),
                            'index_sha256': D.sha((out / 'paired-index.json').read_bytes())}
    idx_raw = (out / 'checkpoints-index.json').read_bytes()
    R['checkpoints_index'] = {'path': str(out / 'checkpoints-index.json'), 'sha256': D.sha(idx_raw),
                              'entries': len(ck_index['checkpoints'])}
    R['progress'] = {'path': str(out / 'progress.json'),
                     'sha256': D.sha((out / 'progress.json').read_bytes()) if (out / 'progress.json').exists() else None}
    R['arms'] = {n: {'lambda_pi': a['lambda_pi'], 'lambda_kl': a['lambda_kl'], 'lambda_knn': a['lambda_knn'],
                     'frozen_at_epoch': a['frozen_at_epoch'],
                     'frozen_reason': a.get('frozen_reason'),
                     'updates': a['steps']} for n, a in arms.items()}
    R['control_kl_drift_per_epoch'] = [{'epoch': ep['epoch'], 'kl_mean': e['probe']['kl_mean'],
                                        'kl_max': e['probe']['kl_max'], 'entropy_mean': e['probe']['entropy_mean']}
                                       for ep in R['epochs'] for e in ep['eval'] if e['arm'] == 'control']
    R['kl_alarm_events'] = [{'arm': e['arm'], 'epoch': e['epoch'], 'kl_mean': e['probe']['kl_mean'],
                             'disqualify': e['kl_disqualify']} for ep in R['epochs'] for e in ep['eval'] if e['kl_alarm']]
    R['gpu_memory'] = {'max_allocated_bytes': torch.cuda.max_memory_allocated(0),
                       'max_reserved_bytes': torch.cuda.max_memory_reserved(0)} if use_cuda else None
    R['bypassed_guards'] = D.BYPASSED_GUARDS + [
        {'guard': 'rlfw.exit_offline_train CLI/TrainConfig (owner window manifest, verify_runtime_scope, health '
                  'baseline, 64-probe adapter contract)', 'replacement': 'this harness: config-bounded wall, '
                  'deadline checks in every loop, external RuntimeMaxSec read back and required',
         'why': 'R22 scratch run; no owner window manifest or adapter exists'},
        {'guard': 'build_targets verify_receipt (seven-artifact native/Rust closure, 1e-9 parity)',
         'replacement': 'scratch verifier: row must come from its sha-verified detail shard',
         'why': 'R22: native detail shards have no Rust closure'},
        {'guard': 'P5 trainer: fresh AdamW, never resume the service optimizer',
         'replacement': 'v1 final AdamW moments restored from the sealed byte copy; no service state touched',
         'why': 'a fresh AdamW made sign-sized first steps (run-1 smoke)'},
        {'guard': 'outcome_class qualified_complete and incumbent closure_verified',
         'replacement': 'asserted for completed native rows with a loader-verified outer-train baseline',
         'why': 'R22 non-qualifying native measurements'}]
    R['limits'] = ['Scratch-only and non-qualifying (R22): directional, never qualifying evidence or promotion input.',
                   'No live learner, drain, waves, S3, program execution or chat post.',
                   'Validation contexts are evaluation-only; no outer-test value is decoded.',
                   'L_PI acts only on measured programs that are trie leaves.']


if __name__ == '__main__':
    sys.exit(main(sys.argv[1], sys.argv[2]))
