"""R22 scratch-only, non-qualifying L_PI harness: native snapshot access and target coverage (xbabe3 only).

Owner ruling R22 (receipt rulings/owner-askuserquestion-receipt-R22-v1.json sha256 55d9452c...): train L_PI on
native S1 detail-shard measured programs, labelled scratch-only and non-qualifying (no seven-artifact Rust closure).
This module never writes into the rebased source tree. It imports the rebased rlfw math and the repository's pinned
v1 policy/L6 code read-only, and reads byte-pinned copies of the v1/v2 native inputs under the lane's inputs/ root
(each copy keeps its original absolute path below that root; every read is size- and sha256-checked by the v1 Reader).

Guards replaced here (recorded in every receipt, see BYPASSED_GUARDS):
- native role authority freshness helper: recomputed from byte-pinned copies of the same ten authority files,
  because the helper only runs on xbabe2 and xbabe2 may not run Python in this lane;
- everything else in this module uses the v1 loader unchanged (train-prefix-only teacher decode, label rules).
"""
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import os  # noqa: E402  (revision 4: host-configurable roots; defaults are the xbabe3 lane)
# LPI_LANE holds run/coverage.json and run/select.json; LPI_INPUTS is the root under which every original absolute
# input path is found ('/' on xbabe2, where the originals live; the lane copy on xbabe3); LPI_TREE is the rebased
# tree. Every byte read is still size- and sha256-checked against the spec pins.
LANE = Path(os.environ.get('LPI_LANE', '/home/ubuntu/claude-rlfw-lane/rlfw-takeover/lpi-scratch'))
INPUTS = Path(os.environ.get('LPI_INPUTS', str(LANE / 'inputs')))
TREE = Path(os.environ.get('LPI_TREE', '/home/ubuntu/claude-rlfw-lane/rlfw-takeover/lpi/tree'))
PIPELINES = TREE / 'deploy/dl-flywheel/pipelines'
CODE = PIPELINES / 'continuous'
SPEC = '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_native_full_20261009/spec.json'
SPEC_SHA = '05dc9558e36461068832b449cedff46123376e3567b9c011e7d74f9b8a8018ee'
PREPROCESSING = '/home/ubuntu/jopedime_probe_exp/ofm-20261008/dl_final/flywheel_neural_native_full_20261009_v1/training/preprocessing.json'
PREPROCESSING_SHA = '4f093f59af6cb4522c8fda77cce357ecd72d0561e3816ac261ec1c91808ca62d'
CHECKPOINT = '/home/ubuntu/jopedime_flywheel_v2/v1-final-capture/20261010T011722Z/capture/checkpoint'
CHECKPOINT_PARAMETER_SHA = 'ea078456690b0665592652b2ed5b04a0deb39b9e604de97a513dc885e467f9fb'
CHECKPOINT_STATE_SHA = '3cdd73d8cef758e4c1895fc2ab7ea811c0c5c69b05709aca012eafc8e3492b19'
V2_CONFIG = '/home/ubuntu/jopedime_flywheel_v2/v2.0.0-campaign-r2-20261010T140801Z/config.json'
SEED = 20261009
R22 = {'ruling': 'R22', 'receipt': 'rulings/owner-askuserquestion-receipt-R22-v1.json',
       'receipt_sha256': '55d9452c9e87c9fb49727de9e5b6fce28d6117758a453d99407673ac8ca8f118',
       'label': 'R22 scratch-only, non-qualifying'}
BYPASSED_GUARDS = [
    {'guard': 'native_corpus_authority.check (helper freshness + latest-pointer globbing on xbabe2)',
     'replacement': 'same closure recomputed from sha256-verified copies of the ten pinned authority files; '
                    'latest-pointer check done separately on xbabe2 with ls (pinned target v4 and layers v4 are the '
                    'newest files)',
     'why': 'the helper runs only on xbabe2 and this lane may not run Python on xbabe2'},
]


def sha(b):
    return hashlib.sha256(b).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def mapped(path):
    return INPUTS / path.lstrip('/')


def read_pinned(path, expected):
    raw = mapped(path).read_bytes()
    need(sha(raw) == expected, 'pinned input drift: ' + path)
    return raw


def setup_paths():
    sys.dont_write_bytecode = True
    for p in (str(PIPELINES), str(CODE), str(CODE / 'consumer'), str(CODE / 'runtime')):
        if p not in sys.path:
            sys.path.insert(0, p)


class MappedStore:
    """The v1 Reader checks size and sha256 of every byte read through this store."""
    def read(self, ref):
        p = mapped(ref['path'])
        need(p.is_file() and not p.is_symlink(), 'pinned copy missing: ' + ref['path'])
        with p.open('rb') as f:
            return f.read(ref['bytes'] + 1)


def authority(spec):
    from jope_flywheel.protocol import canonical
    pins = spec['authority_pins']
    docs = {}
    for name, r in pins.items():
        raw = read_pinned(r['path'], r['sha256'])
        need(len(raw) == r['bytes'], 'authority size:' + name)
        docs[name] = json.loads(raw)
    target, reg, role = docs['target'], docs['layers'], docs['role']
    need(target.get('schema') == 'jope.dltrain-gp-target-after-layers/v1'
         and role.get('schema') == 'jope.dltrain-one-holdout-role-map/v1', 'authority schema')
    need(target['registry']['path'] == pins['layers']['path']
         and target['registry']['sha256'] == pins['layers']['sha256'], 'target registry binding')
    need(target['inputs'].get(pins['role']['path']) == pins['role']['sha256'], 'target role binding')
    bypath = {r['path']: (r, docs[k]) for k, r in pins.items()}
    protected, views = set(), set()
    for r in reg['layers']:
        need(r['path'] in bypath and bypath[r['path']][0]['sha256'] == r['sha256'], 'missing pinned exclusion layer')
        d = bypath[r['path']][1]
        if d.get('schema') == 'jope.dltrain-training-exclusion-layer/v1':
            for row in d['entries']:
                protected.add(row['context_sha256'])
                views.update(row.get('view_ids', []))
        else:
            need(d.get('schema') in {'jope.dltrain-one-holdout-role-map/v1', 'jope.dl-p3-exclusion-list/v1',
                                     'jope.gp-hwm-excluded-by-ruling/v1', 'jope.dltrain-gp-target/v1'},
                 'unknown exclusion authority schema')
    need(len(target['views']) == len(set(target['views'])) == target['gp_target'], 'target counts')
    doc = {'schema': 'jope.native-corpus-current-authority/v1', 'passed': True, 'checked_at_unix': time.time(),
           'source_pins': pins, 'helper_sha256': sha(Path(__file__).read_bytes()),
           'eligible_views_sha256': sha(canonical(sorted(target['views']))), 'eligible_views': len(target['views']),
           'protected_context_ids': sorted(protected), 'protected_views': sorted(views),
           'research_split_only': True, 'untouched_holdout_claim': False}
    return canonical(doc)


def load_native():
    setup_paths()
    from jope_flywheel import training_snapshot_native as N
    spec_bytes = read_pinned(SPEC, SPEC_SHA)
    spec = json.loads(spec_bytes)
    started = time.time()
    s = N.load_snapshot(spec_bytes, SPEC_SHA, MappedStore(), authority_bytes=authority(spec))
    s.load_seconds = time.time() - started
    return s


def train_trie(s):
    from neural_decode import BatchedTrie
    train = sorted(c for c, r in s.contexts.items() if r['partition'] == 'train')
    programs = {s.winners[c].program_sha256 for c in train}
    catalog = [p for p in s.proposal_programs if p['program_sha256'] in programs]
    return BatchedTrie(catalog, s.vocab, s.max_seq)


def panels(s):
    from jope_flywheel.protocol import digest
    train = sorted(c for c, r in s.contexts.items() if r['partition'] == 'train')
    dev = sorted(c for c, r in s.contexts.items() if r['partition'] == 'validation')
    return (sorted(train, key=lambda x: digest({'seed': SEED, 'panel': x}))[:16],
            sorted(dev, key=lambda x: digest({'seed': SEED, 'panel': x}))[:16])


def select(out):
    s = load_native()
    train_panel, dev_panel = panels(s)
    rows, files = {}, set()
    for cid, desc in s.shards.items():
        part = s.contexts[cid]['partition']
        has_input = 'native_input' in desc
        rows[cid] = {'partition': part, 'baseline_input': has_input, 'n_records': desc['n_records']}
        if has_input or cid in train_panel or cid in dev_panel:
            files.add(desc['shard']['path'])
            if has_input:
                files.add(desc['native_input']['teacher']['path'])
                files.add(desc['native_input']['context_json']['path'])
    count = lambda part, inp: sum(1 for r in rows.values() if r['partition'] == part and r['baseline_input'] >= inp)
    doc = {'schema': 'jope.r22-lpi-scratch-select/v1', **R22, 'snapshot_counts': s.receipt['counts'],
           'load_seconds': s.load_seconds, 'shard_contexts': {'train': count('train', 0), 'validation':
           count('validation', 0), 'train_with_baseline_input': count('train', 1),
           'validation_with_baseline_input': count('validation', 1)},
           'panels': {'train': train_panel, 'validation': dev_panel}, 'copy_files': sorted(files),
           'bypassed_guards': BYPASSED_GUARDS}
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True) + '\n')
    print(json.dumps({k: doc[k] for k in ('snapshot_counts', 'shard_contexts', 'load_seconds')}, sort_keys=True))
    print('copy_files', len(files))


def coverage(out):
    s = load_native()
    trie = train_trie(s)
    loader = s.context_loader
    started = time.time()
    contexts, totals = {}, {}
    for cid, desc in sorted(s.shards.items()):
        part = s.contexts[cid]['partition']
        if 'native_input' not in desc or part not in ('train', 'validation'):
            continue
        x = loader._shard(cid)
        baseline = loader._baseline(cid, x)
        _, labs = s.labels.label_context(x['context'], x['records'])
        labs = {a.program_sha256: a for a in labs}
        winner = s.winners[cid].program_sha256
        c = {'records': 0, 'tokenizable': 0, 'in_trie': 0, 'q_known': 0, 'in_trie_q_known': 0}
        measured, incumbent = [], None
        for row in x['records']:
            ph, tokens = row['program_sha256'], tuple(row['tokens'])
            c['records'] += 1
            lab = labs.get(ph)
            known = baseline and lab is not None and lab.q is not None
            c['q_known'] += known
            if s.token_refusal(tokens):
                continue
            c['tokenizable'] += 1
            if known and ph == winner:
                incumbent = {'q': lab.q, 'q_opt': lab.q_opt, 'q_grid': lab.q_grid}
            if tokens not in trie.programs:
                continue
            c['in_trie'] += 1
            if known:
                c['in_trie_q_known'] += 1
                measured.append({'program_sha256': ph, 'tokens': list(tokens), 'q': lab.q, 'q_opt': lab.q_opt,
                                 'q_grid': lab.q_grid, 'outcome': row.get('outcome')})
        distinct = len({m['q_grid'] for m in measured})
        c['eligible'] = len(measured) >= 2 and distinct >= 2
        for k, v in c.items():
            totals.setdefault(part, {}).setdefault(k, 0)
            totals[part][k] += int(v)
        totals[part]['contexts'] = totals[part].get('contexts', 0) + 1
        contexts[cid] = {'partition': part, 'counts': c, 'baseline': baseline, 'measured_in_trie': measured,
                         'incumbent': incumbent, 'shard_sha256': s.shards[cid]['shard']['sha256'],
                         'lineage': s.contexts[cid]['parent_lineage_sha256']}
    doc = {'schema': 'jope.r22-lpi-scratch-coverage/v1', **R22, 'trie': {'sha256': trie.identity,
           'programs': len(trie.programs)}, 'seconds': time.time() - started, 'totals': totals,
           'contexts': contexts, 'bypassed_guards': BYPASSED_GUARDS,
           'scope': 'native S1 detail-shard measured programs; q only where the v1 loader verified the outer-train '
                    'prefix baseline; actor support only for programs that are leaves of the train-winner trie'}
    Path(out).write_text(json.dumps(doc, sort_keys=True) + '\n')
    print(json.dumps({'trie': doc['trie'], 'totals': totals, 'seconds': doc['seconds']}, sort_keys=True))


if __name__ == '__main__':
    mode = sys.argv[1]
    {'select': select, 'coverage': coverage}[mode](sys.argv[2])
