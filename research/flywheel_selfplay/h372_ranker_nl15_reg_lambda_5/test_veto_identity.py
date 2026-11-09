import hashlib, tempfile, unittest
from pathlib import Path
import veto_identity as v


class VetoIdentityTests(unittest.TestCase):
    def setup_files(self, root):
        td = root / "P0/fold_0/root_30001"
        (td / "refit").mkdir(parents=True)
        inv = {}
        for i in range(3):
            p = td / "refit" / f"m{i}-veto.txt"
            p.write_bytes(str(i).encode())
            inv["run/" + str(p.relative_to(root))] = dict(
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()
            )
        return td, inv

    def parent(self):
        p = Path(__file__).resolve().parent.parent / "veto-test-fixtures"
        p.mkdir(exist_ok=True)
        return p

    def test_all_three_require_exact_BASE_bytes(self):
        with tempfile.TemporaryDirectory(dir=self.parent()) as d:
            td, inv = self.setup_files(Path(d))
            self.assertEqual(v.verify(td, inv, Path(d))["matched"], 3)

    def test_any_changed_veto_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.parent()) as d:
            td, inv = self.setup_files(Path(d))
            (td / "refit/m1-veto.txt").write_bytes(b"changed")
            with self.assertRaises(AssertionError):
                v.verify(td, inv, Path(d))

    def test_missing_veto_rejected(self):
        with tempfile.TemporaryDirectory(dir=self.parent()) as d:
            td, inv = self.setup_files(Path(d))
            (td / "refit/m1-veto.txt").unlink()
            with self.assertRaises(AssertionError):
                v.verify(td, inv, Path(d))


if __name__ == "__main__":
    unittest.main()
