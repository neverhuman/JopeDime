"""W2 private CPU fixtures: GET-only exact-version prefetch into a closed cache, then zero-network preparation.

The remote is a counting fake over the portable fixture store; nothing here touches S3, a planner, an optimizer,
a GPU or a GP executor.
"""
import copy
import json
import os
import tempfile
import time
import types
import unittest
from pathlib import Path

import test_native_variant_snapshot as variant
from dltrain import labels
from jope_flywheel.protocol import canonical
from pipelines.generated_target_admission_v1.common import canonical as xcanonical, sha
from pipelines.generated_target_preparation_v2 import preparation as C
from pipelines.generated_target_preparation_v2 import worker as Wk
from pipelines.generated_target_preparation_v2.tests import world as W

NAMESPACE = ('fixture', '')


class Remote:
    """Counting fake remote: every get is a network GET; ``wrong_version`` simulates a bad VersionId readback."""
    nativecanonical_json = staticmethod(canonical)

    def __init__(self, inner):
        self.inner, self.gets, self.bytes, self.wrong_version = inner, 0, 0, False

    def get_version(self, ref):
        self.gets += 1
        body, version = self.inner.get_version(ref)
        self.bytes += len(body)
        return body, ('wrong' if self.wrong_version else version)


class WorkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        variant.VariantTests.setUpClass()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        f, n, spec, nsraw, registry, pre, auth = variant.VariantTests().fixture()
        self.f, self.native, self.A, self.nsraw, self.registry, self.pre = f, n, spec, nsraw, registry, pre
        proof = canonical({**json.loads(auth()), 'checked_at_unix': time.time()})
        self.auth = lambda: proof          # one pinned proof: prepared bytes bind the composing proof digest
        (self.base / 'acceptances').mkdir()
        self.acceptances = {}

    def accept(self, name, proof, *, physical=None, tag=None):
        """Operator acceptance after the in-process prefetch returned (r7): pins the returned completion proof."""
        ref = Wk.accept_cache(self.base / name, acceptance_path=self.base / 'acceptances' / ((tag or name) + '.json'),
                              physical=physical or {'kind': 'in_process_return',
                                                    'returned_sha256': sha(xcanonical(proof))})
        self.acceptances.setdefault(name, ref)
        return ref

    def append(self, spec, *programs):
        new = copy.deepcopy(spec)
        refs = [W.add_program(self.f, new['tasks'][0], dsl, rmse) for dsl, rmse in programs]
        new['tasks'][0]['program_results'].extend(refs)
        return new, refs

    def common(self, old, new, refs):
        raw, pred = W.x1_manifest(self.f, old, new, refs, pre=self.pre)
        return raw, dict(expected_manifest_sha256=sha(raw), predecessor_raw=pred, native=self.native,
                         native_spec_bytes=self.nsraw, registry_bytes=self.registry, preprocessing_bytes=self.pre,
                         objective_bytes=W.OBJECTIVE_BYTES, contract=W.contract(self.pre),
                         export_provenance=dict(W.PROVENANCE), authority_callback=self.auth, labels=labels)

    def prefetch(self, raw, kw, name, remote=None, previous=None):
        remote = remote or Remote(self.f.store)
        proof = Wk.prefetch(raw, remote=remote, cache_root=self.base / name, deadline=time.time() + 60,
                            namespace=NAMESPACE, previous_cache=previous,
                            previous_cache_acceptance=self.acceptances[previous.name] if previous else None, **kw)
        self.accept(name, proof)
        return proof, remote

    def closed_prepare(self, raw, kw, cache, root):
        return Wk.prepare(raw, cache_root=self.base / cache, cache_acceptance=self.acceptances[cache],
                          canonicalizer=canonical, namespace=NAMESPACE,
                          root=self.base / root, deadline=time.time() + 60, **kw)

    def test_w2_prefetch_then_closed_prepare_equals_in_memory_root(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015), ('(pipe X (ridge alpha=4.0))', .02))
        raw, kw = self.common(self.A, B, refs)
        proof, remote = self.prefetch(raw, kw, 'cache-b')
        self.assertEqual(proof['network_gets'], proof['objects'])
        self.assertEqual(remote.gets, proof['objects'], 'one GET per exact object (deduplicated)')
        closed = self.closed_prepare(raw, kw, 'cache-b', 'root-closed')
        self.assertEqual(closed['store_audit']['delta']['network_gets'], 0)
        self.assertGreater(closed['store_audit']['delta']['local_reads'], 0)
        direct = C.prepare_root(raw, store=W.AuditedStore(self.f.store), root=self.base / 'root-direct',
                                deadline=time.time() + 60, **kw)
        self.assertEqual(closed['prepared']['sha256'], direct['prepared']['sha256'])
        self.assertEqual(closed['rows'], direct['rows'])
        self.assertEqual(remote.gets, proof['objects'], 'closed preparation performed no remote read')

    def test_w2_closed_cache_refuses_undeclared_and_corrupted_objects(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, kw = self.common(self.A, B, refs)
        self.prefetch(raw, kw, 'cache')
        cache = Wk.ClosedCache(self.base / 'cache', NAMESPACE, canonicalizer=canonical, acceptance=self.acceptances['cache'])
        stranger = self.f.store.put('fixture/never-declared', b'not in the roster')
        with self.assertRaisesRegex(ValueError, 'not in the closed exact roster'):
            cache.get_version(stranger)
        manifest = json.loads((self.base / 'cache' / 'manifest.json').read_bytes())
        victim = manifest['refs'][0]
        path = self.base / 'cache' / (sha(canonical(victim)) + '.bin')
        body = path.read_bytes()
        os.unlink(path)
        path.write_bytes(bytes([body[0] ^ 1]) + body[1:])
        with self.assertRaisesRegex(ValueError, 'cached bytes corrupted'):
            self.closed_prepare(raw, kw, 'cache', 'root-corrupt')
        self.assertFalse((self.base / 'root-corrupt').exists())

    def test_w2_chained_cache_fetches_only_new_objects(self):
        B, refs_b = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw_b, kw_b = self.common(self.A, B, refs_b)
        proof_b, _ = self.prefetch(raw_b, kw_b, 'cache-b')
        Cspec, refs_c = self.append(B, ('(pipe X (ridge alpha=5.0))', .004))
        raw_c, kw_c = self.common(B, Cspec, refs_c)
        proof_c, remote_c = self.prefetch(raw_c, kw_c, 'cache-c', previous=self.base / 'cache-b')
        self.assertGreater(proof_c['previous_cache_hits'], 0)
        self.assertEqual(proof_c['network_gets'] + proof_c['previous_cache_hits'], proof_c['objects'])
        self.assertLess(proof_c['network_gets'], proof_c['objects'])
        self.assertEqual(remote_c.gets, proof_c['network_gets'])
        receipt = self.closed_prepare(raw_c, kw_c, 'cache-c', 'root-c')
        self.assertEqual(receipt['dispositions'], {'record': 1})
        manifest = json.loads((self.base / 'cache-c' / 'manifest.json').read_bytes())
        self.assertEqual(manifest['previous_cache_manifest_sha256'],
                         sha((self.base / 'cache-b' / 'manifest.json').read_bytes()))

    def test_w2_bad_remote_readback_or_expired_clock_leaves_no_cache(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, kw = self.common(self.A, B, refs)
        remote = Remote(self.f.store)
        remote.wrong_version = True
        with self.assertRaisesRegex(ValueError, 'VersionId'):
            self.prefetch(raw, kw, 'cache-bad', remote=remote)
        self.assertFalse((self.base / 'cache-bad').exists())
        with self.assertRaisesRegex(ValueError, 'deadline'):
            Wk.prefetch(raw, remote=Remote(self.f.store), cache_root=self.base / 'cache-late',
                        deadline=time.time() - 1, namespace=NAMESPACE, **kw)
        self.assertFalse((self.base / 'cache-late').exists())
        with self.assertRaises((ValueError, OSError)):
            Wk.ClosedCache(self.base / 'cache-late', NAMESPACE, canonicalizer=canonical, acceptance=None)

    def test_w2_namespace_allow_list_refuses_foreign_objects(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, kw = self.common(self.A, B, refs)
        with self.assertRaisesRegex(ValueError, 'allowed namespace'):
            Wk.prefetch(raw, remote=Remote(self.f.store), cache_root=self.base / 'cache-prod-ns',
                        deadline=time.time() + 60, namespace=Wk.PRODUCTION_NAMESPACE, **kw)
        self.assertFalse((self.base / 'cache-prod-ns').exists())

    # ------------------------------------------------------------------ r4: bounded streaming recorder
    def test_w2r4_recorder_streams_bodies_and_keeps_only_refs_in_memory(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        entry = self.f.store.doc(refs[0])
        wanted = [refs[0]] + [entry['artifacts'][k] for k in ('program', 'evaluation')]
        rec = Wk.Recorder(Remote(self.f.store), NAMESPACE, root=self.base / 'stream')
        first = [rec.get_version(r)[0] for r in wanted]
        self.assertTrue(all(type(v) is dict and 'sha256' in v for v in rec.objects.values()), 'refs only, no bodies')
        self.assertEqual(rec.declared, sum(r['bytes'] for r in wanted))
        for r, body in zip(wanted, first):
            path = self.base / 'stream' / (sha(canonical(Wk.K.normalized_ref(r))) + '.bin')
            self.assertEqual(path.read_bytes(), body)
        self.assertEqual([rec.get_version(r)[0] for r in wanted], first, 'repeat reads come from the streamed files')
        victim = self.base / 'stream' / (sha(canonical(Wk.K.normalized_ref(wanted[1]))) + '.bin')
        body = victim.read_bytes()
        os.unlink(victim)
        victim.write_bytes(bytes([body[0] ^ 1]) + body[1:])
        with self.assertRaisesRegex(ValueError, 'recorded bytes corrupted'):
            rec.get_version(wanted[1])

    def test_w2r4_cache_cap_refuses_before_overshoot_and_leaves_no_complete_cache(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, kw = self.common(self.A, B, refs)
        full, _ = self.prefetch(raw, kw, 'cache-full')
        for cap in (full['bytes'] - 1, full['bytes'] // 3):
            with self.subTest(cap=cap):
                remote = Remote(self.f.store)
                name = 'cache-cap-' + str(cap)
                with self.assertRaisesRegex(ValueError, 'closed cache cap would be exceeded'):
                    Wk.prefetch(raw, remote=remote, cache_root=self.base / name, deadline=time.time() + 60,
                                namespace=NAMESPACE, cache_cap_bytes=cap, **kw)
                self.assertLessEqual(remote.bytes, cap, 'no fetch beyond the cap')
                self.assertFalse((self.base / name / 'complete.json').exists())
                with self.assertRaises((ValueError, OSError)):
                    Wk.ClosedCache(self.base / name, NAMESPACE, canonicalizer=canonical, acceptance=None)

    def test_w2r5_expiry_after_durable_completion_leaves_cache_unqualified(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, kw = self.common(self.A, B, refs)
        deadline, state = time.time() + 60, {'expired': False}
        real_time, real_create = time.time, Wk._create
        C.time = types.SimpleNamespace(time=lambda: deadline + 1 if state['expired'] else real_time(),
                                       monotonic=time.monotonic, time_ns=time.time_ns)

        def create(path, body):
            out = real_create(path, body)
            state['expired'] |= Path(path).name == 'complete.json'
            return out
        Wk._create = create
        try:
            with self.assertRaisesRegex(ValueError, 'after durable completion marker'):
                Wk.prefetch(raw, remote=Remote(self.f.store), cache_root=self.base / 'cache-late-q',
                            deadline=deadline, namespace=NAMESPACE, **kw)
        finally:
            C.time, Wk._create = time, real_create
        self.assertTrue((self.base / 'cache-late-q' / 'complete.json').exists())
        self.assertFalse((self.base / 'cache-late-q' / 'qualified.json').exists())
        with self.assertRaises((ValueError, OSError)):
            Wk.ClosedCache(self.base / 'cache-late-q', NAMESPACE, canonicalizer=canonical, acceptance=None)


    # ------------------------------------------------------------------ r7: bound authority and final publication (cache)
    def swap(self, path, body):
        os.unlink(path)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        os.write(fd, body)
        os.close(fd)

    def test_w2r7_replaced_qualified_bound_refused_and_acceptance_required(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, kw = self.common(self.A, B, refs)
        proof, _ = self.prefetch(raw, kw, 'cache-r7')
        root, accepted = self.base / 'cache-r7', self.acceptances['cache-r7']
        qpath = root / 'qualified.json'
        original = qpath.read_bytes()
        q = json.loads(original)
        b = q['bound']
        extended = {**q, 'bound': {**b, 'wall_deadline_unix': b['wall_deadline_unix'] + 3600,
                                   'monotonic_deadline': b['monotonic_deadline'] + 3600},
                    'qualified_after_durable_completion': {'boot_id': b['boot_id'],
                                                           'wall_unix': b['wall_deadline_unix'] + 10,
                                                           'monotonic': b['monotonic_deadline'] + 10}}
        self.swap(qpath, xcanonical(extended))
        with self.assertRaisesRegex(ValueError, 'qualified bound is the sealed completion clock bound'):
            Wk.ClosedCache(root, NAMESPACE, canonicalizer=canonical, acceptance=accepted)
        other = json.loads(original)
        other['qualified_after_durable_completion']['wall_unix'] -= 1e-3
        self.swap(qpath, xcanonical(other))
        with self.assertRaisesRegex(ValueError, 'acceptance pins the exact final marker bytes'):
            Wk.ClosedCache(root, NAMESPACE, canonicalizer=canonical, acceptance=accepted)
        self.swap(qpath, original)
        with self.assertRaisesRegex(ValueError, 'exact bounded reference'):
            Wk.ClosedCache(root, NAMESPACE, canonicalizer=canonical, acceptance=None)
        self.prefetch(raw, kw, 'cache-r7-other')
        foreign = self.acceptances['cache-r7-other']
        with self.assertRaisesRegex(ValueError, 'accepted bounded terminal envelope for this subject'):
            Wk.ClosedCache(root, NAMESPACE, canonicalizer=canonical, acceptance=foreign)
        self.assertEqual(Wk.ClosedCache(root, NAMESPACE, canonicalizer=canonical, acceptance=accepted).manifest_sha256,
                         proof['manifest_sha256'])

    def test_w2r7_expiry_during_final_qualified_publication_is_refused(self):
        B, refs = self.append(self.A, ('(pipe X (ridge alpha=3.0))', .015))
        raw, kw = self.common(self.A, B, refs)
        deadline, state = time.time() + 60, {'expired': False}
        real_time, real_create = time.time, Wk._create
        C.time = types.SimpleNamespace(time=lambda: deadline + 1 if state['expired'] else real_time(),
                                       monotonic=time.monotonic, time_ns=time.time_ns)

        def create(path, body):
            out = real_create(path, body)
            state['expired'] |= Path(path).name == 'qualified.json'     # the deadline passes during the final write
            return out
        Wk._create = create
        root = self.base / 'cache-r7-late'
        try:
            with self.assertRaisesRegex(ValueError, 'after final qualified publication'):
                Wk.prefetch(raw, remote=Remote(self.f.store), cache_root=root, deadline=deadline, namespace=NAMESPACE,
                            **kw)
            done = (root / 'complete.json').read_bytes()
            with self.assertRaisesRegex(ValueError, 'acceptance refused'):
                Wk.accept_cache(root, acceptance_path=self.base / 'acceptances' / 'cache-r7-late.json',
                                physical={'kind': 'in_process_return', 'returned_sha256': sha(done)})
        finally:
            C.time, Wk._create = time, real_create
        self.assertTrue((root / 'qualified.json').exists(), 'the late marker is preserved as custody')
        refused = self.base / 'acceptances' / 'cache-r7-late.json'
        self.assertEqual(json.loads(refused.read_bytes())['verdict'], 'refused')
        ref = {'path': str(refused), 'sha256': sha(refused.read_bytes()), 'bytes': len(refused.read_bytes())}
        with self.assertRaisesRegex(ValueError, 'accepted bounded terminal envelope'):
            Wk.ClosedCache(root, NAMESPACE, canonicalizer=canonical, acceptance=ref)


if __name__ == '__main__':
    unittest.main()
