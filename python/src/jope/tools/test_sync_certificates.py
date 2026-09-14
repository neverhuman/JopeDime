import unittest

import sync_certificates as s


class SyncCertificateTests(unittest.TestCase):
    def test_presence_hashes_and_lag_numbers_never_replace_original_verification(self):
        for digest in (None, '', 'a' * 64):
            cert = s.certify({name: dict(present=True, sha256=digest, lag_seconds=0)
                              for name in s.BOUNDARIES})
            self.assertFalse(cert['certified'])
            self.assertFalse(cert['transfer_prerequisite'])
            self.assertFalse(cert['activation_allowed'])

    def test_unknown_or_malformed_capacity_never_grants_backfill(self):
        cap = s.certify_capacity(free_bytes=800*s.GIB, filesystem_bytes=1000*s.GIB)
        self.assertFalse(cap['backfill_allowed'])
        self.assertIsNone(cap['required_bytes'])
        for value in (-1, True, float('nan')):
            with self.assertRaises(ValueError):
                s.certify_capacity(800*s.GIB, 1000*s.GIB, missing_unique_bytes=value)

    def test_missing_declared_root_fails(self):
        cert = s.certify({
            "source": {"present": True, "sha256": "aa"},
            "durable_data": {"present": False},
        })
        self.assertFalse(cert["certified"])
        self.assertIn("durable_data", cert["missing_roots"])
        self.assertFalse(cert["one_synced_bit"])

    def test_capacity_blocks_when_short(self):
        cap = s.certify_capacity(free_bytes=10 * s.GIB, filesystem_bytes=100 * s.GIB)
        self.assertFalse(cap["backfill_allowed"])
        ok = s.certify_capacity(free_bytes=800 * s.GIB, filesystem_bytes=1000 * s.GIB,
                                missing_unique_bytes=10*s.GIB, staging_bytes=2*s.GIB,
                                snapshots_bytes=0, materialization_bytes=0, expected_growth_bytes=0, metadata_bytes=0)
        self.assertTrue(ok["backfill_allowed"])


if __name__ == "__main__":
    unittest.main()
