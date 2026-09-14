import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import original_learning_selection as selection
import qualify_regression_gain as trial


TEXTS = {
    'seed-selection': '- Four conditions × three matched training seeds.\n'
                      '- Seeds `20260909`, `20260910`, and `20260911`.\n',
    'condition-selection': '\n'.join((
        '| Condition | Support | Replay |', '|---|---|---|',
        '| A | Strictly admitted narrow support | Family-balanced broad |',
        '| B | Strictly admitted narrow support | 50/50 fresh/broad |',
        '| C | All eligible broader support at freeze | Family-balanced broad |',
        '| D | All eligible broader support at freeze | 50/50 fresh/broad |')),
    'governing-constraints': '1,000 new committed steps; 300-second task caps. '
        'Do not add a reset ablation. No eligible new support means an unavailable support condition.',
}


def session(path, text, identity='fixture', role='user', source='cli', cwd=selection.REPOSITORY):
    header = dict(type='session_meta', payload=dict(id=identity, source=source, cwd=cwd))
    record = dict(type='response_item', timestamp='2026-09-10T18:17:00.191Z',
                  payload=dict(type='message', role=role, content=[dict(type='input_text', text=text)]))
    lines = [header] + [dict(type='unrelated_fixture', payload='must not be copied') for _ in range(7)] + [record]
    path.write_bytes(b''.join(json.dumps(value).encode() + b'\n' for value in lines))
    return header, record, hashlib.sha256(text.encode()).hexdigest()


def custody(root):
    expected = {}
    for name, text in TEXTS.items():
        path = root / (name + '.jsonl')
        _, record, digest = session(path, text, identity=name)
        selection.capture_user_message(path, root / 'custody' / name, digest)
        expected[name] = dict(session_id=name, timestamp=record['timestamp'], message_sha256=digest)
    return expected


class OriginalLearningSelectionTests(unittest.TestCase):
    def test_fifo_is_rejected_without_waiting_for_a_writer(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metadata.json'
            os.mkfifo(path)
            with self.assertRaisesRegex(ValueError, 'regular'):
                selection.custody_json(path, 1024)

    def test_duplicate_and_nonfinite_custody_json_reject(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'metadata.json'
            for data in ('{"seed":1,"seed":2}', '{"value":NaN}', '{"value":1e999}'):
                path.write_text(data)
                with self.assertRaises(ValueError):
                    selection.custody_json(path, 1024)

    def test_capture_preserves_only_original_user_bytes_and_source_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            original = root / 'session.jsonl'
            _, record, digest = session(original, 'explicit selected seed identities')
            result = selection.capture_user_message(original, root / 'copy', digest)
            self.assertEqual(result['source_sha256'], hashlib.sha256(original.read_bytes()).hexdigest())
            self.assertEqual((root / 'copy/user-record.json').read_bytes(), original.read_bytes().splitlines(keepends=True)[8])
            self.assertEqual((root / 'copy/user-message.txt').read_text(), 'explicit selected seed identities')
            self.assertEqual(set(p.name for p in (root / 'copy').iterdir()),
                             {'session-header.json', 'user-record.json', 'user-message.txt', 'provenance.json'})
            self.assertFalse(result['funding_balance_verified'])
            self.assertFalse(result['numerical_work_authorized'])

    def test_assistant_subagent_unrelated_or_unpinned_source_cannot_select(self):
        changes = [dict(role='assistant'), dict(source={'subagent': 'review'}),
                   dict(cwd='/unrelated'), dict()]
        for change in changes:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                _, _, digest = session(root / 'source.jsonl', 'selected seeds', **change)
                with self.assertRaises(ValueError):
                    selection.capture_user_message(root / 'source.jsonl', root / 'copy',
                                                   '0' * 64 if not change else digest)
                self.assertFalse((root / 'copy').exists())

    def test_source_and_destination_symlinks_cannot_alias_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, _, digest = session(root / 'source.jsonl', 'selection')
            (root / 'alias.jsonl').symlink_to(root / 'source.jsonl')
            with self.assertRaisesRegex(ValueError, 'symlink'):
                selection.capture_user_message(root / 'alias.jsonl', root / 'copy', digest)
            (root / 'destination').mkdir()
            (root / 'copy').symlink_to(root / 'destination', target_is_directory=True)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                selection.capture_user_message(root / 'source.jsonl', root / 'copy', digest)
            self.assertEqual(list((root / 'destination').iterdir()), [])

    def test_original_selection_is_recovered_without_reset_or_funding(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = custody(root)
            with patch.dict(selection.ORIGINALS, expected, clear=True):
                value = selection.recover_selection(root / 'custody')
                status = trial.required_trial_status(root / 'custody')
            self.assertEqual(value['seeds'], [20260909, 20260910, 20260911])
            self.assertEqual([row['identity'] for row in value['conditions']], ['A', 'B', 'C', 'D'])
            self.assertEqual([row['replay'] for row in value['conditions']],
                             ['broad', 'fresh_balanced', 'broad', 'fresh_balanced'])
            self.assertEqual(value['constraints']['new_committed_steps'], 1000)
            self.assertEqual(value['constraints']['training_ceiling_seconds'], 300)
            self.assertFalse(value['constraints']['optimizer_reset_ablation_allowed'])
            self.assertTrue(status['original_selection_recovered'])
            self.assertEqual(len(status['selected_branches']), 12)
            self.assertTrue(all(not row['launch_allowed'] for row in status['selected_branches']))
            self.assertFalse(status['launch_allowed'])
            self.assertFalse(status['references_opened'])
            self.assertFalse(status['numerical_work_started'])

    def test_newly_self_consistent_user_record_does_not_replace_pinned_original(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = custody(root)
            saved = root / 'custody/seed-selection/user-record.json'
            value = json.loads(saved.read_bytes())
            value['payload']['content'][0]['text'] = TEXTS['seed-selection'].replace('20260911', '20260912')
            saved.chmod(0o644)
            saved.write_text(json.dumps(value))
            receipt = root / 'custody/seed-selection/provenance.json'
            claim = json.loads(receipt.read_bytes())
            claim['files']['user-record.json']['sha256'] = hashlib.sha256(saved.read_bytes()).hexdigest()
            claim['original_selection_verified'] = True
            receipt.write_text(json.dumps(claim))
            with patch.dict(selection.ORIGINALS, expected, clear=True):
                with self.assertRaisesRegex(ValueError, 'instruction changed'):
                    selection.recover_selection(root / 'custody')

    def test_missing_original_is_not_replaced_by_a_caller_seed_list(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'required-selection.json').write_text(json.dumps(dict(
                seeds=[20260909, 20260910, 20260911], original_selection_verified=True)))
            with self.assertRaises(FileNotFoundError):
                selection.recover_selection(root)


if __name__ == '__main__':
    unittest.main()
