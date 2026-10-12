"""Real rustc checks for the reproducible executable cache. No network."""
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score_cache import (
    ExecutableCache,
    compile_once,
    historical_argv,
    repro_argv,
    run_batch,
    sha256_bytes,
)

SOURCE = b"""fn main() {
    let marker = file!();
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 1 {
        std::process::exit(2);
    }
    std::fs::write(&args[0], marker).unwrap();
}
"""


class CacheTests(unittest.TestCase):
    def test_historical_paths_differ_and_remap_matches(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            left = root / "a"
            right = root / "b"
            for directory in (left, right):
                directory.mkdir()
                (directory / "model.rs").write_bytes(SOURCE)
            historical = []
            remapped = []
            for directory in (left, right):
                out = directory / "historical"
                body = compile_once(
                    "rustc",
                    historical_argv("rustc", directory / "model.rs", out),
                    directory,
                )
                historical.append(sha256_bytes(body))
                out = directory / "model"
                argv = repro_argv("rustc", directory / "model.rs", out, directory)
                body = compile_once("rustc", argv, directory)
                remapped.append(sha256_bytes(body))
                self.assertNotIn(str(directory).encode(), body)
                self.assertIn(b"/src/model.rs", body)
            self.assertNotEqual(historical[0], historical[1])
            self.assertEqual(remapped[0], remapped[1])

    def test_cache_hit_miss_and_batch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
            root = Path(tmp)
            cache = ExecutableCache(root / "cache")
            first = cache.ensure(SOURCE)
            self.assertFalse(first.hit)
            self.assertEqual(cache.compiles, 1)
            second = cache.ensure(SOURCE)
            self.assertTrue(second.hit)
            self.assertEqual(second.sha256, first.sha256)
            self.assertEqual(cache.compiles, 1)
            changed = SOURCE.replace(b"marker", b"marked")
            third = cache.ensure(changed)
            self.assertNotEqual(third.key, first.key)
            self.assertEqual(cache.compiles, 2)
            side = json_side(first.path.parent / "cache.json")
            binary = first.path.read_bytes()
            first.path.write_bytes(binary + b"\x00")
            with self.assertRaises(ValueError):
                cache.ensure(SOURCE)
            first.path.write_bytes(binary)
            (first.path.parent / "cache.json").write_text(side)
            out_a = root / "a.txt"
            out_b = root / "b.txt"
            results = run_batch(cache, [(SOURCE, [str(out_a)]), (SOURCE, [str(out_b)])])
            self.assertEqual([item.returncode for item in results], [0, 0])
            self.assertEqual(out_a.read_text(), "/src/model.rs")
            self.assertEqual(out_b.read_text(), "/src/model.rs")
            self.assertEqual(cache.compiles, 2)


def json_side(path: Path) -> str:
    return path.read_text()


if __name__ == "__main__":
    unittest.main()
