"""Syntax coloring preserves the source bytes across supported display languages."""
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pygments.lexers import get_lexer_by_name
from pygments.util import ClassNotFound

sys.path.insert(0, str(Path(__file__).parents[1]))
import cinematic_pages
import code_pages


class MultilingualCodePages(unittest.TestCase):
    def setUp(self):
        target = Path(__file__).parents[3] / "target" / "code-page-controls"
        target.mkdir(parents=True, exist_ok=True)
        self.scratch = tempfile.TemporaryDirectory(prefix="language-", dir=target)
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)

    def test_language_lexers_and_lossless_cinematic_pages(self):
        fixtures = {
            "python": ("model.py", "PythonLexer", '"""λ predictor."""\r\ndef predict(x):\r\n\treturn x + 3\r\n'),
            "cpp": ("model.cpp", "CppLexer", "// λ predictor\r\ndouble predict(double x) {\r\n\treturn x + 3;\r\n}\r\n"),
            "c": ("model.h", "CLexer", "/* λ predictor */\r\ndouble predict(const double *x);\r\n"),
            "javascript": ("model.mjs", "JavascriptLexer", "// λ predictor\r\nexport const predict = (x) => x + 3;\r\n"),
            "typescript": ("model.ts", "TypeScriptLexer", "// λ predictor\r\nexport const predict = (x: number): number => x + 3;\r\n"),
            "go": ("model.go", "GoLexer", "package model\r\n// λ predictor\r\nfunc Predict(x float64) float64 {\r\n\treturn x + 3\r\n}\r\n"),
            "cuda": ("model.cu", None, "// λ predictor\r\n__global__ void predict(float *x) {\r\n\tx[0] += 3;\r\n}\r\n"),
        }
        for language, (filename, lexer, text) in fixtures.items():
            with self.subTest(language=language):
                source = self.root / filename
                source.write_bytes(text.encode("utf-8"))
                with cinematic_pages.presentation(language):
                    manifest = code_pages.make_pages(
                        source, self.root / language, columns=16, rows=2, language=language
                    )
                self.assertEqual(manifest["language"], language)
                self.assertEqual(manifest["palette"], cinematic_pages.PALETTE)
                if lexer is not None:
                    self.assertIn(lexer, manifest["lexer"])
                else:
                    self.assertRegex(manifest["lexer"], r"CudaLexer|CppLexer \(CUDA syntax fallback\)")
                self.assertEqual(code_pages.reconstruct_source(manifest), source.read_bytes())

        python_text = fixtures["python"][2]
        spans, _ = code_pages._language_spans(python_text, "python")
        self.assertTrue(any("λ predictor" in python_text[a:b] and category == "string"
                            for a, b, category in spans))

    def test_cuda_fallback_is_labeled(self):
        def without_cuda(alias, **options):
            if alias == "cuda":
                raise ClassNotFound("CUDA alias unavailable")
            return get_lexer_by_name(alias, **options)

        source = self.root / "model.cu"
        source.write_text("__global__ void predict(float *x) { x[0] += 3; }\n")
        with patch("pygments.lexers.get_lexer_by_name", side_effect=without_cuda):
            manifest = code_pages.make_pages(source, self.root / "fallback", language="cuda")
        self.assertIn("CppLexer (CUDA syntax fallback)", manifest["lexer"])
        self.assertEqual(manifest["language"], "cuda")
        self.assertEqual(code_pages.reconstruct_source(manifest), source.read_bytes())

    def test_default_and_explicit_rust_keep_same_manifest_and_raster(self):
        source = self.root / "model.rs"
        source.write_bytes(b"/// Prediction\r\npub fn predict(x: f64) -> f64 { x + 3.0 }\r\n")
        with cinematic_pages.presentation():
            default = code_pages.make_pages(source, self.root / "default")
        with cinematic_pages.presentation("rust"):
            explicit = code_pages.make_pages(source, self.root / "explicit", language="rust")
        self.assertNotIn("language", default)
        self.assertEqual({**default, "output_directory": explicit["output_directory"]}, explicit)


if __name__ == "__main__":
    unittest.main()
