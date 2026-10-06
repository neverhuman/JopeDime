"""Offline P5 tests (PLAN §5 P5, §6.1, §6.3, §6.6): corpus schemas, the build, the poison harness and lane hygiene.

Synthetic fixtures live in tests_p5/fixtures_p5.py. No test touches S3 or the network, and every write goes under
TMPDIR (RUN/tmp) through tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']). Run only inside the lane slice:
  systemd-run --user --scope --quiet --slice=jopedime-dltrain.slice ... python -m unittest discover -s tests_p5 -t . -v

`unittest discover -s tests_p5` calls load_tests below: it discovers every test module of this package and then
appends one final check that the fixtures' audit-hook write guard recorded zero writes outside TMPDIR across the
whole suite (tests_p5.test_corpus_hygiene checks that the hook is live).
"""
import os
import unittest


def _zero_writes_outside_tmpdir() -> None:
    from tests_p5 import fixtures_p5 as fx
    bad = fx.violations()
    if bad:
        raise AssertionError(f"{len(bad)} audited writes outside TMPDIR during the suite: {bad[:10]}")


def load_tests(loader, standard_tests, pattern):
    here = os.path.dirname(os.path.abspath(__file__))
    suite = unittest.TestSuite()
    suite.addTests(standard_tests)
    suite.addTests(loader.discover(start_dir=here, pattern=pattern or "test*.py"))
    suite.addTest(unittest.FunctionTestCase(
        _zero_writes_outside_tmpdir,
        description="tests_p5 suite end: the audit-hook write guard recorded zero writes outside TMPDIR"))
    return suite
