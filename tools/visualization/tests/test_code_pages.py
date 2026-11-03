"""Complete-source raster controls; these fixtures are not emitted models."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image

spec = importlib.util.spec_from_file_location("code_pages", Path(__file__).parents[1] / "code_pages.py")
code_pages = importlib.util.module_from_spec(spec)
spec.loader.exec_module(code_pages)


class CompleteCodePages(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).parents[3] / "target" / "code-page-controls"
        target.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="case-", dir=target)
        self.root = Path(self.scratch.name)

    def tearDown(self):
        self.scratch.cleanup()

    def test_all_bytes_survive_wrapping_tabs_crlf_and_syntax(self):
        raw = ('use std::collections::BTreeMap;\r\n'
               '\tpub fn predict<T>() -> BTreeMap<T, f64> {\r\n'
               '    let λ = r##"literal :: no rejoined spaces"##;\r\n'
               '\tlet values = [1.25e-4, -3.0, 42usize];\n'
               '    /* nested /* comment */ still comment */\n'
               '    std::collections::BTreeMap::new()\r\n}\n').encode("utf-8")
        source = self.root / "predictor.rs"
        source.write_bytes(raw)
        output = self.root / "pages"
        manifest = code_pages.make_pages(source, output, columns=16, rows=2)
        persisted = json.loads((output / "manifest.json").read_text())
        self.assertEqual(code_pages.reconstruct_source(persisted), raw)
        text = raw.decode("utf-8")
        fallback = code_pages._fallback_spans(text)
        self.assertEqual("".join(text[a:b] for a, b, _ in fallback), text)
        self.assertTrue(any(text[a:b] == "pub" and category == "keyword" for a, b, category in fallback))
        self.assertGreater(manifest["page_count"], 4)
        self.assertTrue(any(row["continuation"] for page in manifest["pages"] for row in page["rows"]))
        self.assertTrue(all(row["visual_cells"] <= 16 for page in manifest["pages"] for row in page["rows"]))
        corrupted = copy.deepcopy(persisted)
        corrupted["pages"][0]["rows"][0]["text"] = corrupted["pages"][0]["rows"][0]["text"].replace("::", ":", 1)
        with self.assertRaisesRegex(ValueError, "coverage|identity"):
            code_pages.reconstruct_source(corrupted)

    def test_pages_are_opaque_nonempty_and_existing_assets_refused(self):
        source = self.root / "predictor.rs"
        source.write_text("pub fn predict(x: f64) -> f64 { x * 1.25 }\n")
        output = self.root / "pages"
        manifest = code_pages.make_pages(source, output)
        self.assertEqual((manifest["width"], manifest["height"]), (684, 660))
        self.assertEqual(manifest["page_count"], 1)
        page = output / manifest["pages"][0]["file"]
        before = page.read_bytes()
        with Image.open(page) as image:
            self.assertEqual(image.mode, "RGBA")
            self.assertEqual(image.getchannel("A").getextrema(), (255, 255))
            self.assertGreater(len(image.getcolors(image.width * image.height)), 3)
        with self.assertRaises(FileExistsError):
            code_pages.make_pages(source, output)
        self.assertEqual(page.read_bytes(), before)

    def test_source_and_page_limits_fail_before_output_creation(self):
        source = self.root / "large.rs"
        source.write_bytes(b" " * (code_pages.MAX_SOURCE_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "4 MiB"):
            code_pages.make_pages(source, self.root / "large-pages")
        self.assertFalse((self.root / "large-pages").exists())
        source.write_text("\n" * (code_pages.MAX_PAGES * 2 + 1))
        with self.assertRaisesRegex(ValueError, "page limit"):
            code_pages.make_pages(source, self.root / "too-many-pages", rows=2)
        self.assertFalse((self.root / "too-many-pages").exists())


if __name__ == "__main__":
    unittest.main()
