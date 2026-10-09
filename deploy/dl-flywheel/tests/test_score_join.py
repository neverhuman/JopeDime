from dataclasses import replace
import hashlib
import json
import unittest
from jope_flywheel.protocol import Context
from jope_flywheel.score_join import build_records, validate_records, inner_cut, native_bytes


def native_fixture(value):
    # Test double with serde_json's e-7 spelling. Production supplies the pinned
    # native executable; there is deliberately no production Python fallback.
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).replace('e-07', 'e-7').encode()


class ScoreJoinTests(unittest.TestCase):
    def setUp(self):
        self.context = Context('regression', 'a'*64, 'b'*64, 'c'*64, 'd'*64, 'e'*64, 'f'*64, '1'*64,
                               'rmse', 'prepared_T1_input_units', 'minimize')
        self.args = (self.context, '2'*64, '3'*64, [1e-7, 0.5], '4'*64, '5'*40, 1234, native_fixture)

    def test_roundtrip_two_rows_and_native_float_spelling(self):
        body = build_records(*self.args)
        self.assertIn(b'1e-7', body)
        self.assertNotIn(b'1e-07', body)
        rows = validate_records(body, *self.args)
        self.assertNotEqual(rows[0]['cut_id'], rows[1]['cut_id'])
        self.assertEqual([r['value'] for r in rows], [1e-7, 0.5])

    def test_hash_blanks_field_instead_of_removing_it(self):
        rows = json.loads(build_records(*self.args))
        blank = dict(rows[0], row_sha256='')
        self.assertEqual(rows[0]['row_sha256'], hashlib.sha256(native_fixture(blank)).hexdigest())
        del blank['row_sha256']
        self.assertNotEqual(rows[0]['row_sha256'], hashlib.sha256(native_fixture(blank)).hexdigest())

    def test_no_native_callback_no_admission(self):
        with self.assertRaisesRegex(ValueError, 'native canonical'):
            build_records(*self.args[:-1], None)

    def test_callback_cannot_mutate_json(self):
        with self.assertRaisesRegex(ValueError, 'changed'):
            native_bytes({'a': 1}, lambda value: b'{"a":2}')

    def test_noncanonical_array_and_duplicate_rows_rejected(self):
        rows = json.loads(build_records(*self.args))
        with self.assertRaisesRegex(ValueError, 'native canonical'):
            validate_records(json.dumps(rows, indent=2).encode(), *self.args)
        with self.assertRaises(ValueError):
            validate_records(native_fixture([rows[0], rows[0]]), *self.args)

    def test_forged_value_context_and_raw_q_are_rejected(self):
        for field, value in [('value', 0.1), ('metric', 'q'), ('evaluator_sha256', '0'*64),
                             ('benchmark_valid', True), ('protocol', 'other'), ('method_id', 'other')]:
            rows = json.loads(build_records(*self.args))
            rows[0][field] = value
            rows[0]['row_sha256'] = hashlib.sha256(native_fixture(dict(rows[0], row_sha256=''))).hexdigest()
            with self.assertRaises(ValueError):
                validate_records(native_fixture(rows), *self.args)

    def test_changed_split_produces_new_cut(self):
        first = inner_cut(self.context, 0, native_fixture)
        second = inner_cut(replace(self.context, split_sha256='9'*64), 0, native_fixture)
        self.assertNotEqual(first, second)

    def test_nonfinite_and_negative_rmse_rejected(self):
        for values in ([float('nan'), 1.], [-1., 1.], [True, 1.]):
            args = list(self.args)
            args[3] = values
            with self.assertRaises(ValueError):
                build_records(*args)


if __name__ == '__main__':
    unittest.main()
