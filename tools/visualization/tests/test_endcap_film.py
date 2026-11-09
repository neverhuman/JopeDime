"""Verify the VEOX retime, moving handoff, and original PCM preservation."""

from pathlib import Path
import sys
import tempfile
import unittest
import wave

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import endcap_film as endcap
import intro_film as intro
from intelligence_wave import scroll_offsets


def source_receipt():
    cases = []
    for index in range(10):
        start = index * intro.SOURCE_CASE_COUNT
        cases.append({"case_id": f"case-{index}", "dataset_name": f"Dataset {index}", "start": start,
                      "end": start + intro.SOURCE_CASE_COUNT, "visual_rows": 90, "line_height_px": 19,
                      "scroll_offsets_px": scroll_offsets(90, 19, 845),
                      "files": [{"filename": "model.rs", "source_sha256": "source", "visual_start": 0,
                                 "visual_rows": 90}], "table_row_orders": [[]] * 15,
                      "cues": [{"case_frame": frame, "rows": [0, 89]} for frame in (93, 94, 95)]})
    return {"width": 1920, "height": 1080, "source_viewport": [1319, 133, 563, 845], "cases": cases}


class EndcapFilmTests(unittest.TestCase):
    def test_exact_schedule_keeps_original_handoff_source_rows_and_one_frame_cues(self):
        source = source_receipt()
        approved = intro.plan_retime(source)
        timeline = endcap.plan_timeline(source, {"retime": approved})
        selected = timeline["selected_source_indices"]
        self.assertEqual((len(selected), selected[0], selected[-1]), (840, 0, 1259))
        self.assertEqual(selected[:31], approved["selected_source_indices"][:31])
        self.assertEqual(selected[29:31], [36, 37])
        self.assertEqual(timeline["chapters"][-1], {"title": "VEOX", "start_frame": 1170, "end_frame": 1350})
        self.assertEqual(timeline["visual_rows_fully_visible"], 900)
        self.assertEqual(timeline["table_shuffle_count"], 150)
        self.assertIn(94 + 126, selected)
        self.assertNotIn(94, endcap.distribute(0, 125, 83))
        self.assertIn(1, timeline["code_coverage"][1]["approved_displayed_cue_indices"])
        self.assertIn(1, timeline["code_coverage"][1]["displayed_cue_indices"])

    def test_rejects_changed_approved_retime(self):
        source = source_receipt()
        approved = intro.plan_retime(source)
        approved["selected_source_indices"][29] += 1
        with self.assertRaisesRegex(ValueError, "Approved cut"):
            endcap.plan_timeline(source, {"retime": approved})

    def test_audio_keeps_exact_approved_prefix_and_ends_silent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            approved, master, tail = (root / name for name in ("old.wav", "master.wav", "tail.wav"))
            total = intro.MASTER_COUNT * endcap.SAMPLES_PER_FRAME
            prefix_frames = endcap.ENDCAP_START * endcap.SAMPLES_PER_FRAME
            first_sample = b"\x03\0\x05\0"
            with wave.open(str(approved), "wb") as original:
                original.setparams((2, 2, intro.AUDIO_RATE, total, "NONE", "not compressed"))
                original.writeframes(first_sample + bytes((total - 1) * 4))
            report = endcap.make_audio(approved, master, tail)
            self.assertEqual(report["prefix_samples_per_channel"], prefix_frames)
            with wave.open(str(master), "rb") as merged, wave.open(str(tail), "rb") as ending:
                self.assertEqual(merged.getnframes(), total)
                self.assertEqual(merged.readframes(prefix_frames), first_sample + bytes((prefix_frames - 1) * 4))
                tail_pcm = ending.readframes(ending.getnframes())
                self.assertEqual(merged.readframes(total - prefix_frames), tail_pcm)
                self.assertGreater(max(tail_pcm), 0)
                self.assertFalse(any(tail_pcm[-endcap.SAMPLES_PER_FRAME * 4:]))

    def test_crossfade_applies_only_to_the_start_of_the_master_endcap(self):
        old, new = Image.new("RGB", (2, 2), "white"), Image.new("RGB", (2, 2), "black")
        middle = endcap.master_tail_frame(new, old, endcap.CROSSFADE_FRAMES // 2)
        self.assertEqual(middle.getpixel((0, 0)), (127, 127, 127))
        self.assertIs(endcap.master_tail_frame(new, old, endcap.CROSSFADE_FRAMES), new)
        self.assertEqual(new.getpixel((0, 0)), (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
