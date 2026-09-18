import tempfile
import unittest
from pathlib import Path

import checkpoint_ledger as cl
from resume_expired_cycle import atomic, sha

RUN = 'runs/r'


class CheckpointLedger(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.mirror = Path(self.tmp.name)
        base = self.mirror / 'learning-receipts'
        files = {}
        def put(rel, value, stub=False, full=None):
            path = base / rel.replace(RUN + '/', '', 1)
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic(path, value)
            files[rel] = dict(sha256=sha(path), bytes=path.stat().st_size, mtime=1.0, stub=stub, full_sha256=full, mirrored=str(path.relative_to(base)), tail_lines=None)
            return sha(path)
        # cycle 0: steps 1000/2000/3000; 3000 monitored. cycle 1: step 6000 unmonitored (orphan), as a stub with a full digest
        put(f'{RUN}/learning/cycle-000000/train-2/checkpoints/step-1000/state.json', dict(step=1000, model_sha256='m1'))
        put(f'{RUN}/learning/cycle-000000/train-2/checkpoints/step-2000/state.json', dict(step=2000, model_sha256='m2'))
        s3 = put(f'{RUN}/learning/cycle-000000/train-2/checkpoints/step-3000/state.json', dict(step=3000, model_sha256='m3'))
        put(f'{RUN}/learning/cycle-000001/train-2/checkpoints/step-6000/state.json', dict(schema='parallel.receipt_stub.v1', fields=dict(step=6000, model_sha256='m6')), stub=True, full='f' * 64)
        put(f'{RUN}/learning/monitoring/p-{s3}.json', dict(schema='parallel.frozen_monitoring.v1', binding=dict(checkpoint_sha256=s3), ranking_bt=0.51, sequence_nll=1.39, evaluated_step=3000))
        atomic(base / 'MANIFEST.json', dict(schema='parallel.learning_receipt_mirror.v1', run=RUN, files=files))
        self.s3 = s3

    def test_classification_orphans_and_immutable_receipt(self):
        result = cl.ledger(self.mirror, RUN, verdicts={self.s3: 'diagnostic_accepted'})
        self.assertEqual(result['counts'], dict(total=4, block_end=2, intermediate=2, real=1, orphans=1, monitor_receipts=1, monitored_identities=1))
        self.assertEqual(result['orphans'], [dict(cycle='cycle-000001', step=6000, state_sha256='f' * 64)])
        real = next(r for r in result['rows'] if r['real'])
        self.assertEqual((real['step'], real['verdict'], real['model_sha256'], real['monitor_receipts'][0]['ranking_bt']), (3000, 'diagnostic_accepted', 'm3', 0.51))
        output = self.mirror / 'ledger.json'
        cl.main(['--mirror', str(self.mirror), '--run', RUN, '--output', str(output)])
        digest = sha(output)
        cl.main(['--mirror', str(self.mirror), '--run', RUN, '--output', str(output)])
        self.assertEqual(sha(output), digest)


if __name__ == '__main__':
    unittest.main()
