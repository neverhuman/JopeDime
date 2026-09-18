import importlib.util
from pathlib import Path

import jope.research
import tempfile
import unittest


SCRIPT = Path(jope.research.__file__).parent / "append_realtime_agent_chat.py"
SPEC = importlib.util.spec_from_file_location("append_realtime_agent_chat", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class AppendRealtimeAgentChatTests(unittest.TestCase):
    def test_appends_one_normalized_seven_field_line(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = Path(directory) / "REALTIME_AGENT_CHAT.md"
            chat.write_text("existing\n", encoding="utf-8")
            line = MODULE.append_event(
                chat,
                actor=" codex ",
                hypothesis="CLAIM_REQUEST  lease=L1 task=U4  paths=search",
                evidence=" latest 20 read ",
                next_action=" implement  contracts ",
                blocker="none",
                handoff="reviewer=root",
                timestamp="2026-08-06T06:00:00Z",
            )
            self.assertEqual(chat.read_text(encoding="utf-8"), "existing\n" + line)
            self.assertEqual(len(line.rstrip("\n").split(" | ")), 7)
            self.assertNotIn("  ", line)

    def test_rejects_separator_and_unregistered_event(self):
        with tempfile.TemporaryDirectory() as directory:
            chat = Path(directory) / "REALTIME_AGENT_CHAT.md"
            chat.touch()
            with self.assertRaises(ValueError):
                MODULE.append_event(
                    chat,
                    actor="codex",
                    hypothesis="CLAIM_REQUEST lease=L1 task=U4",
                    evidence="bad | evidence",
                    next_action="stop",
                    blocker="none",
                    handoff="none",
                )
            with self.assertRaises(ValueError):
                MODULE.append_event(
                    chat,
                    actor="codex",
                    hypothesis="OTHER lease=L1 task=U4",
                    evidence="none",
                    next_action="stop",
                    blocker="none",
                    handoff="none",
                )


if __name__ == "__main__":
    unittest.main()
