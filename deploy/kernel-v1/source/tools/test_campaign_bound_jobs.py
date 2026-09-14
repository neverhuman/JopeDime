import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import campaign_bound_jobs


class CampaignBoundJobsTests(unittest.TestCase):
    def test_lists_unused_campaign_cuts_and_does_not_enqueue(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            campaign = {'schema': 'parallel.context_index.v1', 'encodings': [
                {'dataset': 'in', 'cut': 0, 'prepared': 'in', 'prepared_sha256': 'aa'},
                {'dataset': 'unused', 'cut': 1, 'prepared': 'unused', 'prepared_sha256': 'bb'},
            ]}
            union = {'schema': 'parallel.context_index.v1', 'encodings': campaign['encodings'] + [
                {'dataset': 'extra', 'cut': 9, 'prepared': 'extra', 'prepared_sha256': 'cc'},
            ]}
            (root / 'campaign.json').write_text(json.dumps(campaign))
            (root / 'union.json').write_text(json.dumps(union))
            (root / 'gp' / 'in' / 'cut-0').mkdir(parents=True)
            (root / 'gp' / 'extra' / 'cut-9').mkdir(parents=True)
            output = root / 'jobs.json'
            argv = ['campaign_bound_jobs.py', '--campaign', str(root / 'campaign.json'),
                    '--union', str(root / 'union.json'), '--output', str(output),
                    '--gp-root', str(root / 'gp')]
            with patch('sys.argv', argv):
                campaign_bound_jobs.main()
            report = json.loads(output.read_text())
            self.assertEqual(report['unused_campaign_cuts'], 1)
            self.assertEqual(report['wasted_gp'], 1)
            self.assertEqual(report['jobs'][0]['dataset'], 'unused')
            self.assertFalse(report['enqueued'])
            self.assertFalse(report['final_test_targets_opened'])


if __name__ == '__main__':
    unittest.main()
