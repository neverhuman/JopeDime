import ast, subprocess, unittest
from pathlib import Path
import leaves_patch as p

W = Path("/home/ubuntu/JopeDime")
PIN = "213b30ad52e1ce48927bba960ed609bd98ab4dfe"
PREFIX = "research/flywheel_selfplay/h287_bagswitch_seed_ensemble_export/"


def source(file, fun):
    raw = subprocess.check_output(
        ["git", "-C", str(W), "show", PIN + ":" + PREFIX + file], text=True
    )
    node = next(
        n
        for n in ast.parse(raw).body
        if isinstance(n, ast.FunctionDef) and n.name == fun
    )
    return ast.get_source_segment(raw, node)


def kw(t):
    return {
        k.arg: ast.dump(k.value, include_attributes=False)
        for k in p.calls(t)[0].keywords
    }


class LeavesTests(unittest.TestCase):
    def test_both_full_BASE_asts_restore(self):
        for file, fun in [("topk.py", "fit_ranker"), ("bagged.py", "fit_ranker_n")]:
            old, new = p.transform(source(file, fun))
            self.assertEqual(
                ast.dump(old, include_attributes=False),
                ast.dump(p.strip(new), include_attributes=False),
            )

    def test_only_existing_num_leaves_value_changes(self):
        for file, fun in [("topk.py", "fit_ranker"), ("bagged.py", "fit_ranker_n")]:
            old, new = p.transform(source(file, fun))
            a, b = kw(old), kw(new)
            self.assertEqual(set(a), set(b))
            self.assertEqual(
                {k: v for k, v in a.items() if k != "num_leaves"},
                {k: v for k, v in b.items() if k != "num_leaves"},
            )
            self.assertEqual(
                b["num_leaves"], ast.dump(ast.Constant(15), include_attributes=False)
            )

    def test_regularization_depth_and_minchild_BASE(self):
        _, new = p.transform(source("topk.py", "fit_ranker"))
        v = {k.arg: k.value for k in p.calls(new)[0].keywords}
        for k, w in [
            ("reg_lambda", 1.0),
            ("max_depth", 3),
            ("min_child_samples", 20),
            ("metric", "ndcg"),
        ]:
            self.assertEqual(ast.literal_eval(v[k]), w)
        for k in [
            "reg_alpha",
            "objective_seed",
            "sigmoid",
            "min_split_gain",
            "min_sum_hessian_in_leaf",
        ]:
            self.assertNotIn(k, v)

    def test_second_override_rejected(self):
        _, new = p.transform(source("topk.py", "fit_ranker"))
        with self.assertRaises(AssertionError):
            p.transform(ast.unparse(new))

    def test_wrong_BASE_l2_rejected(self):
        with self.assertRaises(AssertionError):
            p.transform(
                source("topk.py", "fit_ranker").replace(
                    "reg_lambda=1.0", "reg_lambda=5.0"
                )
            )

    def test_classifier_not_patched(self):
        with self.assertRaises(AssertionError):
            p.transform("def f(): return lgb.LGBMClassifier(num_leaves=7)")


if __name__ == "__main__":
    unittest.main()
