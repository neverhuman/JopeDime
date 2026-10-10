"""Private fixture world and a minimal executor that drives the C10b revision-2 API (tests only; not a runner).

World: five native train contexts plus one native validation context with CE-teacher records, and one composed
flywheel variant context (A).  B and C append verified train results through C10a revision 2.  The executor follows
the runner's per-update order (reserve -> prepare -> optimizer step -> physical checkpoint -> batch proof -> planner
commit) and calls ``Inbox.reconcile``/``adopt_next`` only at completed epochs.  Role proofs are fresh per call.
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import test_native_variant_snapshot as variant
from dltrain import dsl_tokenizer as T
from dltrain import labels
from dltrain import model as M
from jope_flywheel import native_variant_snapshot as V
from jope_flywheel import training_snapshot_native as N
from jope_flywheel.program_io import sha
from jope_flywheel.protocol import canonical, digest
from neural_native_stream import BatchStream, Waiting
from neural_native_train import fit_preprocessing, prepare_microbatches, run_microbatches
from neural_storage import checkpoint
from continuous_state import restore
from pipelines.generated_target_preparation_v2 import preparation as C
from pipelines.generated_target_preparation_v2.tests import world as W
from pipelines.boundary_admission_v2 import boundary as Bd

CAMPAIGN = 'e' * 64


class World:
    def __init__(self, base):
        self.base = Path(base)
        f, n, spec, nsraw, registry, _, auth = variant.VariantTests().fixture()
        self.f, self.native, self.nsraw, self.registry = f, n, nsraw, registry
        self.store = W.AuditedStore(f.store)
        self._auth = auth
        lineage = spec['tasks'][0]['parent_lineage_sha256']
        dsl = '(pipe X (ridge_gcv))'
        tokens = tuple(T.tokenize(dsl, n.vocab))
        teacher = N.NativeRecord(sha(dsl.encode()), tokens, 'completed', ce_eligible=True)
        contexts = {}
        roster = [(lineage, 'train')] + [(digest({'fixture-lineage': k}), 'train') for k in range(1, 5)]
        roster.append((digest({'fixture-lineage': 'dev'}), 'validation'))
        for i, (lin, role) in enumerate(roster):
            c = digest({'native-context': i})
            contexts[c] = {'context_id': c, 'context_sha256': c, 'view_id': f'view{i}', 'dataset_hash': 'a' * 16,
                           'fold': i, 'cut_id': f'cut{i}', 'parent_lineage_sha256': lin, 'partition': role,
                           'source_kind': 'native_s1_context', 'observation_dispatch': 'pending_native_worker_adapter',
                           'gp_hwm_ref': {'fixture': True}, 'kumo_source_sha256': digest({'kumo': i})}
        n.contexts = contexts
        n.proposal_programs = [{'program_sha256': teacher.program_sha256, 'tokens': list(tokens), 'dsl': dsl}]
        n.legal_mask_fn = lambda t: np.ones((len(t) - 1, 282), dtype=np.bool_)
        n.context_loader = type('Loader', (), {'get': staticmethod(
            lambda ids, **kw: N.NativeBatch({c: contexts[c] for c in ids}, {c: n.encoding(c) for c in ids},
                                            {c: (teacher,) for c in ids}, {c: {} for c in ids}, {'fixture': True}))})()
        train = sorted(c for c, r in contexts.items() if r['partition'] == 'train')
        self.pre = canonical({'schema': 'jope.native-kumos-preprocessing/v1', 'snapshot_sha256': n.identity,
                              'fit_context_ids': train, 'validation_used_for_fit': False, 'mean': [0.] * 2051,
                              'scale': [1.] * 2051, 'clip': [-10., 10.]})
        spec['preprocessing_sha256'] = V.sha(self.pre)
        self.A = spec
        self.contract = W.contract(self.pre)
        self.parents = {p['parent_lineage_sha256']: p for p in map(json.loads, registry.splitlines())}
        self.genesis = self.compose(self.A)
        (self.base / 'acceptances').mkdir()
        self.acceptances = {}

    def auth(self):
        return canonical({**json.loads(self._auth()), 'checked_at_unix': time.time()})

    def role_check(self, raw, proof=None):
        V.check_authority(json.loads(raw), json.loads(self.nsraw), proof or self.auth(), self.parents, time.time())

    def compose(self, spec):
        b = canonical(spec)
        s = V.compose(self.native, b, V.sha(b), self.f.store, native_spec_bytes=self.nsraw,
                      registry_bytes=self.registry, preprocessing_bytes=self.pre, authority_callback=self.auth,
                      deadline=time.time() + 30)
        s.extension_raw = b
        return s

    def append(self, spec, *programs):
        import copy
        new = copy.deepcopy(spec)
        refs = [W.add_program(self.f, new['tasks'][0], dsl, rmse) for dsl, rmse in programs]
        new['tasks'][0]['program_results'].extend(refs)
        return new, refs

    def root(self, old, new, refs, name):
        raw, pred = W.x1_manifest(self.f, old, new, refs, pre=self.pre)
        receipt = C.prepare_root(raw, expected_manifest_sha256=sha(raw), predecessor_raw=pred, native=self.native,
                              store=self.store, native_spec_bytes=self.nsraw, registry_bytes=self.registry,
                              preprocessing_bytes=self.pre, objective_bytes=W.OBJECTIVE_BYTES, contract=self.contract,
                              export_provenance=dict(W.PROVENANCE), authority_callback=self.auth,
                              root=self.base / name, deadline=time.time() + 60, labels=labels)
        # r7: the operator accepts the root after the in-process worker returned (exact returned receipt pinned).
        self.acceptances[receipt['root']] = C.accept_root(
            receipt['root'], acceptance_path=self.base / 'acceptances' / (name + '.json'),
            physical={'kind': 'in_process_return', 'returned_sha256': sha(canonical(receipt))})
        return receipt

    def descriptor(self, receipt):
        return C.candidate_descriptor(receipt['root'], campaign_sha256=CAMPAIGN,
                                      acceptance=self.acceptances[receipt['root']])

    def variant(self, receipt):
        """Inbox.publish keyword arguments for one accepted root: the closed descriptor plus its acceptance pin."""
        return {'descriptor': self.descriptor(receipt), 'acceptance': self.acceptances[receipt['root']]}

    def load(self, inbox, record):
        return Bd.load_adopted(inbox, record, self.native, genesis=self.genesis,
                               preprocessing_sha256=self.A['preprocessing_sha256'],
                               registry_sha256=self.A['parent_registry_sha256'],
                               authority_check=lambda raw: self.role_check(raw))


class Executor:
    def __init__(self, world, root, inbox, *, snapshot=None, restored=None):
        self.w, self.root, self.inbox = world, Path(root), inbox
        self.root.mkdir(exist_ok=True)
        (self.root / 'ckpt').mkdir(exist_ok=True)
        self.snapshot = snapshot or world.genesis
        self.norm = fit_preprocessing(world.genesis)[0]
        if restored is None:
            torch.manual_seed(4)
            self.net = M.PolicyLM(M.PolicyConfig(282, 2051, 32, 1, 4, 64, 640, dropout=.2))
            self.opt = torch.optim.AdamW(M.param_groups(self.net, .01), lr=3e-4)
            self.step = 0
            m = checkpoint(self.net, self.opt, self.root / 'ckpt' / '000000', step=0, config_sha256='d' * 64)
            self.ckpt = m['parameter_sha256']
            raw = canonical(self.snapshot.batch_manifest)
            self.planner = BatchStream(self.root / 'batches.sqlite', raw, sha(raw), seed=4, batch_size=2,
                                       initial_checkpoint_sha256=self.ckpt)
        else:
            self.net, self.opt, self.step, self.ckpt, self.planner = restored

    @classmethod
    def restart(cls, world, root, inbox, *, step):
        """A fresh process: restore physical state, reopen the planner on its ACTIVE manifest, resolve the snapshot."""
        root = Path(root)
        path = root / 'ckpt' / f'{step:06d}'
        ref = lambda name: {'path': str(path / name), 'sha256': sha((path / name).read_bytes())}
        manifest = json.loads((path / 'manifest.json').read_bytes())
        net, opt, _ = restore(ref('manifest.json'), ref('state.safetensors'),
                              expected_parameter_sha256=manifest['parameter_sha256'], device='cpu')
        raw, active = Bd.active_manifest(root / 'batches.sqlite')
        planner = BatchStream(root / 'batches.sqlite', raw, active, seed=4, batch_size=2,
                              initial_checkpoint_sha256=json.loads((root / 'ckpt' / '000000' / 'manifest.json')
                                                                   .read_bytes())['parameter_sha256'])
        try:
            state = inbox.reconcile(planner)
            snapshot = world.genesis
            if state['records'] and not state['holds']:
                snapshot = world.load(inbox, state['records'][-1])
        except BaseException:
            planner.close()
            raise
        need_ckpt = planner.summary()['checkpoint_sha256']
        assert need_ckpt == manifest['parameter_sha256'], 'restored checkpoint is the planner checkpoint'
        return cls(world, root, inbox, snapshot=snapshot, restored=(net, opt, step, need_ckpt, planner))

    def epoch_complete(self):
        epoch = self.planner._get('epoch')
        row = self.planner.db.execute('SELECT state FROM epochs WHERE id=?', (epoch,)).fetchone()
        return epoch > 0 and row is not None and row[0] == 'complete'

    def update(self):
        b = self.planner.reserve(self.ckpt)
        if isinstance(b, Waiting):
            return None
        ids = list(b.ordered_data_run_ids)
        batches, total, selection = prepare_microbatches(self.snapshot, ids, epoch=b.epoch - 1, batch_index=b.index,
                                                         normalizer=self.norm, max_padded_tokens=4096)
        result = run_microbatches(self.net, batches, total, optimizer=self.opt, deadline=time.time() + 60)
        self.step += 1
        m = checkpoint(self.net, self.opt, self.root / 'ckpt' / f'{self.step:06d}', step=self.step,
                       config_sha256='d' * 64)
        proof = {'schema': 'jope.continuous-neural-batch-proof/v1', 'absolute_optimizer_step': self.step,
                 'batch': asdict(b), 'selection': selection, 'objective': result, 'checkpoint': m,
                 'snapshot_sha256': self.snapshot.identity, 'fresh_generated_rewards_used': False}
        raw = canonical(proof)
        (self.root / f'batch-{self.step:09d}.json').write_bytes(raw)
        self.planner.commit(b.batch_id, checkpoint_before=self.ckpt, checkpoint_after=m['parameter_sha256'],
                            optimizer_step=self.step, ordered_data_run_ids=ids,
                            evidence={'origin': 'native_optimizer', 'event_sha256': sha(raw)})
        self.ckpt = m['parameter_sha256']
        return proof

    def run_epoch(self):
        """Train until the current epoch completes (at least one update)."""
        done = 0
        while True:
            if self.update() is None:
                raise AssertionError('planner waiting inside an epoch')
            done += 1
            if self.epoch_complete():
                return done

    def boundary(self, *, fresh_authority=None, authority_check=None):
        assert self.epoch_complete(), 'boundary only at a completed epoch'
        replacement = self.inbox.adopt_next(self.planner, self.snapshot, self.w.native, campaign_sha256=CAMPAIGN,
                                            preprocessing_sha256=self.w.A['preprocessing_sha256'],
                                            registry_sha256=self.w.A['parent_registry_sha256'],
                                            checkpoint_sha256=self.ckpt,
                                            fresh_authority=fresh_authority or self.w.auth,
                                            authority_check=authority_check or self.w.role_check)
        if replacement is not None:
            self.snapshot = replacement
        return replacement

    def proofs(self):
        return sorted(self.root.glob('batch-*.json'))

    def close(self):
        self.planner.close()


def new_inbox(world, path, **kwargs):
    raw = canonical(world.genesis.batch_manifest)
    return Bd.Inbox.create(path, genesis_snapshot_sha256=world.genesis.identity, genesis_manifest_sha256=sha(raw),
                           campaign_sha256=CAMPAIGN, contract=world.contract, **kwargs)
