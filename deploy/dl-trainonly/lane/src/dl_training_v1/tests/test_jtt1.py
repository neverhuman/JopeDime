import gzip
import hashlib
import os
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dltrain import jtt1


def make_raw(n_train=5, n_test=3, p=2, magic=b"JTT1", seed=0):
    rng = np.random.default_rng(seed)
    labels = (np.arange(n_train) % 2).astype(np.uint8)
    matrix = rng.normal(size=(n_train + n_test, p + 1)).astype("<f8")
    raw = magic + struct.pack("<IIIB", n_train, n_test, p, 2) + labels.tobytes() + matrix.tobytes(order="C")
    return raw, labels, matrix


def ref_for(raw):
    gz = gzip.compress(raw, compresslevel=3, mtime=0)
    return gz, {"bucket": "b", "key": "k", "version_id": "v", "sha256": hashlib.sha256(gz).hexdigest(), "bytes": len(gz),
                "raw_sha256": hashlib.sha256(raw).hexdigest(), "raw_bytes": len(raw)}


class Parse(unittest.TestCase):
    def test_train_view_decodes_train_rows_only(self):
        raw, labels, matrix = make_raw()
        h, lab, X, y = jtt1.train_view(raw)
        self.assertEqual((h["n_train"], h["n_test"], h["p"], h["inner_folds"]), (5, 3, 2, 2))
        np.testing.assert_array_equal(lab, labels)
        np.testing.assert_array_equal(X, matrix[:5, :-1])
        np.testing.assert_array_equal(y, matrix[:5, -1])

    def test_counterfactual_test_row_change(self):
        raw, _, _ = make_raw()
        h = jtt1.parse_header(raw)
        start, stop = jtt1.train_range(h)
        mutated = bytearray(raw)
        for i in range(stop, len(raw)):
            mutated[i] ^= 0xFF
        mutated = bytes(mutated)
        a, b = jtt1.train_view(raw), jtt1.train_view(mutated)
        for x, z in zip(a[1:], b[1:]):
            np.testing.assert_array_equal(x, z)
        self.assertEqual(jtt1.train_rows_sha256(raw), jtt1.train_rows_sha256(mutated))
        flipped = bytearray(raw)
        flipped[stop - 1] ^= 0x01
        self.assertNotEqual(jtt1.train_rows_sha256(raw), jtt1.train_rows_sha256(bytes(flipped)))

    def test_refusals(self):
        raw, _, _ = make_raw()
        with self.assertRaises(jtt1.JTTError):
            jtt1.parse_header(make_raw(magic=b"JTT2")[0])
        with self.assertRaises(jtt1.JTTError):
            jtt1.parse_header(raw[:-1])
        with self.assertRaises(jtt1.JTTError):
            jtt1.parse_header(raw + b"\0")
        with self.assertRaises(jtt1.JTTError):
            jtt1.parse_header(raw[:10])


class Cache(unittest.TestCase):
    def test_cache_roundtrip_and_tamper(self):
        raw, _, _ = make_raw()
        gz, ref = ref_for(raw)

        class Fake:
            calls = 0

        fake = Fake()

        def fake_get(client, key, version_id, bucket=None):
            fake.calls += 1
            self.assertEqual((key, version_id, bucket), ("k", "v", "b"))
            return gz, {}

        import dltrain.s3io as s3io
        orig = s3io.get_exact
        s3io.get_exact = fake_get
        try:
            with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
                ledger = {}
                self.assertEqual(jtt1.load_raw(ref, tmp, client=object(), ledger=ledger), raw)
                self.assertEqual(ledger, {"gets": 1, "bytes": len(gz)})
                self.assertEqual(jtt1.load_raw(ref, tmp), raw)  # from cache, no client
                self.assertEqual(fake.calls, 1)
                self.assertEqual([p.name for p in Path(tmp).iterdir()], [f"{ref['raw_sha256']}.jtt1.gz"])
                bad = dict(ref, sha256="0" * 64)
                with self.assertRaises(jtt1.JTTError):
                    jtt1.load_raw(bad, tmp)
                with self.assertRaises(jtt1.JTTError):
                    jtt1.load_raw(dict(ref, raw_sha256="1" * 64), tmp, client=object())
        finally:
            s3io.get_exact = orig

    def test_not_cached_without_client(self):
        raw, _, _ = make_raw()
        _, ref = ref_for(raw)
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
            with self.assertRaises(jtt1.JTTError):
                jtt1.load_raw(ref, tmp)


if __name__ == "__main__":
    unittest.main()
