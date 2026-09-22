#!/usr/bin/env python3
import datetime
import unittest
from unittest.mock import patch
import status


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.run = dict(id=99, run_attempt=2, head_sha='a' * 40, status='queued', conclusion=None, html_url='https://example.invalid/99', created_at='2026-09-22T00:00:00Z')
        self.jobs = [dict(id=4, name='test', status='queued', conclusion=None, labels=['self-hosted', 'jope-ci'], created_at='2026-09-22T00:01:00Z')]
        self.now = datetime.datetime.fromisoformat('2026-09-22T00:12:00+00:00')

    def test_queue_labels_online_matching_and_exact_run(self):
        runners = [dict(name='matching', status='online', busy=True, labels=[{'name':'SELF-HOSTED'}, {'name':'jope-ci'}]), dict(name='wrong', status='online', busy=False, labels=[{'name':'jope-gpu'}])]
        result = status.describe(self.run, self.jobs, runners, self.now)
        self.assertEqual(99, result['run_id'])
        self.assertEqual(2, result['attempt'])
        queued = result['queued_jobs'][0]
        self.assertTrue(queued['over_ten_minutes'])
        self.assertEqual([{'name':'matching','busy':True}], queued['matching_online_runners'])
        self.assertEqual(660, queued['queued_seconds'])

    def test_under_threshold_does_not_report_ten_minutes(self):
        self.jobs[0]['created_at'] = '2026-09-22T00:05:00Z'
        self.assertFalse(status.describe(self.run, self.jobs, [], self.now)['queued_jobs'][0]['over_ten_minutes'])

    def test_unobservable_service_is_unavailable(self):
        with patch('status.subprocess.run', side_effect=status.subprocess.TimeoutExpired('ssh', 30)):
            self.assertEqual('unavailable', status.probe('xbabe2')['status'])
        with self.assertRaises(ValueError):
            status.probe('-oBadOption')


if __name__ == '__main__':
    unittest.main()
