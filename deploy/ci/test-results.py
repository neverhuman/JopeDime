#!/usr/bin/env python3
from pathlib import Path
import tempfile
import unittest
import results


class ResultsTests(unittest.TestCase):
    def test_nextest_omitted_junit_skips_use_execution_summary(self):
        with tempfile.TemporaryDirectory() as scratch:
            report = Path(scratch) / 'junit.xml'
            report.write_text('<testsuites><testsuite tests="2"><testcase/><testcase/></testsuite></testsuites>')
            self.assertIsNone(results.junit_counts(report)['skipped'])
            self.assertEqual(results.counts(2, 0, 3), results.summarize('Summary [1s] 2 tests run: 2 passed, 3 skipped', 'nextest', report))
            with self.assertRaises(ValueError):
                results.summarize('Summary [1s] 3 tests run: 3 passed, 3 skipped', 'nextest', report)

    def test_pytest_deselection_and_skip_are_distinct(self):
        actual = results.summarize('207 passed, 2 skipped, 15 deselected in 5.39s', 'pytest')
        self.assertEqual(results.counts(207, 0, 2, 15), actual)

    def test_collection_error_zero_and_partial_are_not_success_counts(self):
        for backend, log in [('pytest', 'ERROR collecting test.py'), ('nextest', 'no tests run'), ('unittest', 'Ran 2 tests in 0.1s\n\nFAILED (failures=1)'), ('rust', 'could not compile')]:
            with self.subTest(backend=backend), self.assertRaises(ValueError):
                results.summarize(log, backend)
        self.assertEqual(0, results.summarize('Summary [1s] 0 tests run: 0 passed', 'nextest')['executed'])

    def test_multiple_harnesses_reconcile(self):
        log = 'test result: ok. 2 passed; 0 failed; 1 ignored; 0 measured; 3 filtered out\ntest result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out'
        self.assertEqual(results.counts(3, 0, 1, 3), results.summarize(log, 'rust'))
        self.assertEqual(results.counts(4, 0, 1), results.summarize('Ran 2 tests in 0.1s\n\nOK\nRan 3 tests in 0.1s\n\nOK (skipped=1)', 'unittest'))

    def test_malformed_and_inconsistent_junit_fail(self):
        with tempfile.TemporaryDirectory() as scratch:
            report = Path(scratch) / 'junit.xml'
            for data in ('<broken', '<testsuites><testsuite skipped="2"><testcase/></testsuite></testsuites>'):
                report.write_text(data)
                with self.assertRaises(ValueError):
                    results.junit_counts(report)


if __name__ == '__main__':
    unittest.main()
