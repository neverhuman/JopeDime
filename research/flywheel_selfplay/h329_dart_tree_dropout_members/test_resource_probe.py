import unittest
from resource_probe import eligible_from_report,ASSIGNED


class ResourceTests(unittest.TestCase):
    def report(self,values):
        return {'sysstat':{'hosts':[{'statistics':[{'cpu-load':[{'cpu':str(c),'idle':v} for c,v in values.items()]}]}]}}
    def test_both_siblings_and_exact_threshold(self):
        values={c:95 for c in (*ASSIGNED,*(c+64 for c in ASSIGNED))}
        values[80]=84.9;values[17]=85
        cores,idle=eligible_from_report(self.report(values))
        self.assertNotIn(16,cores);self.assertIn(17,cores);self.assertEqual(idle[16],84.9)
    def test_missing_sibling_fails_closed(self):
        with self.assertRaises(KeyError):eligible_from_report(self.report({c:95 for c in ASSIGNED}))
    def test_multiple_intervals_not_silently_minimized(self):
        values={c:95 for c in (*ASSIGNED,*(c+64 for c in ASSIGNED))};r=self.report(values)
        r['sysstat']['hosts'][0]['statistics']*=2
        with self.assertRaises(AssertionError):eligible_from_report(r)
