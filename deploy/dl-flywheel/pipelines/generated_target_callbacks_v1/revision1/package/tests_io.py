"""Offline fake-transport tests for the bounded, read-only artifact reader.

No fixture creates an SDK client, opens a socket, writes an artifact or runs GP.
The production runner owns qualification; this module has no standalone runner.
"""
import hashlib
import unittest
from unittest.mock import patch

from . import readonly_io as rio


BUCKET = 'veox-jopedime-learning-use2'
ROOT = 'experiments/gp-policy-dl/v2-trainonly/'
PREFIXES = (ROOT + 'datasets/flywheel/v1/',
            ROOT + 'receipts/flywheel-datasets/v1/',
            ROOT + 'program-results/flywheel/v1/')


class _Clock:
    def __init__(self):
        self.wall = 1000.0
        self.mono = 2000.0

    def time(self):
        return self.wall

    def monotonic(self):
        return self.mono


class _Body:
    """A size-respecting byte stream which records every actual read."""
    def __init__(self, data, *, chunk_size=None, on_eof=None, fail_read=None):
        self.data = data
        self.chunk_size = chunk_size
        self.on_eof = on_eof
        self.fail_read = fail_read
        self.position = 0
        self.requests = []
        self.bytes_returned = 0
        self.closed = False

    def read(self, size=-1):
        if self.closed:
            raise AssertionError('read after close')
        self.requests.append(size)
        if self.fail_read == len(self.requests):
            raise OSError('synthetic stream failure')
        if self.position == len(self.data):
            if self.on_eof is not None:
                self.on_eof()
            return b''
        if size <= 0:
            raise AssertionError('unbounded or empty stream read')
        take = min(size, len(self.data) - self.position)
        if self.chunk_size is not None:
            take = min(take, self.chunk_size)
        block = self.data[self.position:self.position + take]
        self.position += len(block)
        self.bytes_returned += len(block)
        return block

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


class _Client:
    def __init__(self, response=None, *, failure=None):
        self.response = response
        self.failure = failure
        self.gets = []
        self.heads = []
        self.puts = []

    def get_object(self, **kwargs):
        self.gets.append(kwargs)
        if self.failure is not None:
            raise self.failure
        return self.response() if callable(self.response) else self.response

    def head_object(self, **kwargs):
        self.heads.append(kwargs)
        raise AssertionError('HEAD is outside the read-only exact-version API')

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        raise AssertionError('PUT is outside the read-only exact-version API')


def _ref(data, prefix=PREFIXES[0]):
    return {'bucket': BUCKET, 'key': prefix + 'a' * 64 + '/artifact',
            'version_id': 'exact-version-1',
            'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def _response(body, ref):
    return {'Body': body, 'VersionId': ref['version_id'],
            'ContentLength': ref['bytes']}


class ReaderIOTests(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        self.wall_patch = patch.object(rio.time, 'time', self.clock.time)
        self.mono_patch = patch.object(rio.time, 'monotonic', self.clock.monotonic)
        self.wall_patch.start()
        self.mono_patch.start()
        self.addCleanup(self.mono_patch.stop)
        self.addCleanup(self.wall_patch.stop)

    def window(self, *, max_gets=8, max_bytes=1024 * 1024):
        return rio.Window(deadline_unix=self.clock.wall + 10,
                          max_gets=max_gets, max_bytes=max_bytes)

    def assert_read_only(self, client):
        self.assertEqual(client.heads, [])
        self.assertEqual(client.puts, [])

    def test_exact_streaming_read_counts_actual_bytes_and_closes(self):
        data = b'payload' * 10000
        ref = _ref(data)
        body = _Body(data)
        client = _Client(_response(body, ref))
        window = self.window()
        self.assertEqual(rio.ExactReader(client, window).get_version(ref),
                         (data, ref['version_id']))
        self.assertEqual(client.gets, [{'Bucket': ref['bucket'], 'Key': ref['key'],
                                       'VersionId': ref['version_id']}])
        self.assertTrue(body.closed)
        self.assertGreaterEqual(len(body.requests), 2)
        self.assertTrue(all(0 < size <= 64 * 1024 for size in body.requests))
        self.assertEqual(window.counters()['gets'], 1)
        self.assertEqual(window.counters()['bytes'], len(data))
        self.assert_read_only(client)

    def test_all_three_allowed_prefixes_use_get_only(self):
        for prefix in PREFIXES:
            with self.subTest(prefix=prefix):
                ref = _ref(b'data', prefix)
                body = _Body(b'data')
                client = _Client(_response(body, ref))
                self.assertEqual(rio.ExactReader(client, self.window()).get_version(ref),
                                 (b'data', ref['version_id']))
                self.assertEqual(len(client.gets), 1)
                self.assertTrue(body.closed)
                self.assert_read_only(client)

    def test_invalid_closed_references_and_namespaces_refuse_before_get(self):
        good = _ref(b'data')
        cases = [dict(good, extra='unapproved'),
                 {k: v for k, v in good.items() if k != 'version_id'},
                 dict(good, bucket='another-bucket'),
                 dict(good, key=ROOT + 'other/record.json'),
                 dict(good, key='foreign/' + good['key']),
                 dict(good, key=ROOT + 'datasets/flywheel/v10/artifact'),
                 dict(good, version_id=''), dict(good, version_id='null'),
                 dict(good, version_id=None), dict(good, sha256='not-a-sha'),
                 dict(good, bytes=0), dict(good, bytes=-1),
                 dict(good, bytes=True), dict(good, bytes=4.0)]
        for index, ref in enumerate(cases):
            with self.subTest(index=index):
                client = _Client(failure=AssertionError('GET must not happen'))
                window = self.window()
                with self.assertRaises(rio.Failure):
                    rio.ExactReader(client, window).get_version(ref)
                self.assertEqual(client.gets, [])
                self.assertEqual(window.counters()['gets'], 0)
                self.assertEqual(window.counters()['bytes'], 0)
                self.assert_read_only(client)

    def test_declared_object_and_total_byte_caps_refuse_before_get(self):
        good = _ref(b'data')
        for size, budget in [(32 * 1024**2 + 1, 64 * 1024**2), (4, 3)]:
            with self.subTest(size=size, budget=budget):
                client = _Client(failure=AssertionError('GET must not happen'))
                window = self.window(max_bytes=budget)
                with self.assertRaises(rio.Failure):
                    rio.ExactReader(client, window).get_version(dict(good, bytes=size))
                self.assertEqual(client.gets, [])
                self.assertEqual(window.counters()['gets'], 0)
                self.assertEqual(window.counters()['bytes'], 0)

    def test_version_and_length_header_mismatch_close_without_retry(self):
        ref = _ref(b'data')
        cases = [('VersionId', 'different-version'), ('VersionId', None),
                 ('ContentLength', 3), ('ContentLength', 5)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                body = _Body(b'data')
                response = _response(body, ref)
                response[key] = value
                client = _Client(response)
                window = self.window()
                with self.assertRaises(rio.Failure):
                    rio.ExactReader(client, window).get_version(ref)
                self.assertTrue(body.closed)
                self.assertEqual(len(client.gets), 1)
                self.assertEqual(window.counters()['gets'], 1)
                self.assertEqual(window.counters()['bytes'], body.bytes_returned)
                self.assert_read_only(client)

    def test_hash_truncation_and_stream_overflow_fail_with_actual_byte_counts(self):
        ref = _ref(b'data')
        for data in (b'xxxx', b'dat', b'data-extra'):
            with self.subTest(data=data):
                body = _Body(data, chunk_size=2)
                client = _Client(_response(body, ref))
                window = self.window()
                with self.assertRaises(rio.Failure):
                    rio.ExactReader(client, window).get_version(ref)
                self.assertTrue(body.closed)
                self.assertEqual(len(client.gets), 1)
                self.assertEqual(window.counters()['gets'], 1)
                self.assertEqual(window.counters()['bytes'], body.bytes_returned)
                self.assertGreater(body.bytes_returned, 0)
                self.assertTrue(all(0 < size <= 64 * 1024 for size in body.requests))
                self.assert_read_only(client)

    def test_transport_error_charges_one_attempt_and_never_retries(self):
        client = _Client(failure=OSError('synthetic transport failure'))
        window = self.window()
        with self.assertRaises((rio.Failure, OSError)):
            rio.ExactReader(client, window).get_version(_ref(b'data'))
        self.assertEqual(len(client.gets), 1)
        self.assertEqual(window.counters()['gets'], 1)
        self.assertEqual(window.counters()['bytes'], 0)
        self.assert_read_only(client)

    def test_stream_error_closes_and_retains_already_read_byte_count(self):
        ref = _ref(b'data')
        body = _Body(b'data', chunk_size=2, fail_read=2)
        client = _Client(_response(body, ref))
        window = self.window()
        with self.assertRaises((rio.Failure, OSError)):
            rio.ExactReader(client, window).get_version(ref)
        self.assertTrue(body.closed)
        self.assertEqual(len(client.gets), 1)
        self.assertEqual(window.counters()['gets'], 1)
        self.assertEqual(window.counters()['bytes'], 2)
        self.assert_read_only(client)

    def test_get_attempt_limit_is_checked_before_second_request(self):
        ref = _ref(b'data')
        client = _Client(lambda: _response(_Body(b'data'), ref))
        window = self.window(max_gets=1)
        reader = rio.ExactReader(client, window)
        reader.get_version(ref)
        with self.assertRaises(rio.Failure):
            reader.get_version(ref)
        self.assertEqual(len(client.gets), 1)
        self.assertEqual(window.counters()['gets'], 1)
        self.assertEqual(window.counters()['bytes'], 4)

    def test_consumed_byte_budget_is_checked_before_second_request(self):
        ref = _ref(b'data')
        client = _Client(lambda: _response(_Body(b'data'), ref))
        window = self.window(max_bytes=7)
        reader = rio.ExactReader(client, window)
        reader.get_version(ref)
        with self.assertRaises(rio.Failure):
            reader.get_version(ref)
        self.assertEqual(len(client.gets), 1)
        self.assertEqual(window.counters()['gets'], 1)
        self.assertEqual(window.counters()['bytes'], 4)

    def test_expired_deadline_refuses_before_get(self):
        client = _Client(failure=AssertionError('GET must not happen'))
        window = self.window()
        self.clock.wall += 11
        self.clock.mono += 11
        with self.assertRaises((rio.Failure, TimeoutError)):
            rio.ExactReader(client, window).get_version(_ref(b'data'))
        self.assertEqual(client.gets, [])
        self.assertEqual(window.counters()['gets'], 0)
        self.assertEqual(window.counters()['bytes'], 0)

    def test_deadline_crossed_by_final_eof_cannot_return_success(self):
        for expired_clock in ('wall', 'mono'):
            with self.subTest(expired_clock=expired_clock):
                self.clock.wall, self.clock.mono = 1000.0, 2000.0
                window = self.window()
                def late_eof():
                    setattr(self.clock, expired_clock,
                            getattr(self.clock, expired_clock) + 11)
                ref = _ref(b'data')
                body = _Body(b'data', on_eof=late_eof)
                client = _Client(_response(body, ref))
                with self.assertRaises((rio.Failure, TimeoutError)):
                    rio.ExactReader(client, window).get_version(ref)
                self.assertTrue(body.closed)
                self.assertEqual(len(client.gets), 1)
                self.assertEqual(window.counters()['gets'], 1)
                self.assertEqual(window.counters()['bytes'], 4)
                self.assert_read_only(client)

    def test_identical_version_reads_are_not_cached(self):
        ref = _ref(b'data')
        bodies = []
        def response():
            body = _Body(b'data')
            bodies.append(body)
            return _response(body, ref)
        client = _Client(response)
        window = self.window()
        reader = rio.ExactReader(client, window)
        self.assertEqual(reader.get_version(ref), (b'data', ref['version_id']))
        self.assertEqual(reader.get_version(ref), (b'data', ref['version_id']))
        self.assertEqual(len(client.gets), 2)
        self.assertTrue(all(body.closed for body in bodies))
        self.assertEqual(window.counters()['gets'], 2)
        self.assertEqual(window.counters()['bytes'], 8)
        self.assert_read_only(client)
