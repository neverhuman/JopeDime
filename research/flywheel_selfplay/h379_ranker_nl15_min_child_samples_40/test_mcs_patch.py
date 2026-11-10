import ast, unittest
from leaves_patch import transform as leaves, calls
from mcs_patch import transform, restore

SOURCE = """def f():
 return lgb.LGBMRanker(objective="lambdarank", boosting_type="gbdt", num_leaves=7,max_depth=3,min_child_samples=20,reg_lambda=1.0,n_estimators=100,learning_rate=.1)
"""


class PatchTests(unittest.TestCase):
    def test_both_real_pinned_H352_function_ASTs_restore(self):
        from test_leaves import source

        for file, fun in [("topk.py", "fit_ranker"), ("bagged.py", "fit_ranker_n")]:
            h352, new = transform(source(file, fun))
            self.assertEqual(ast.dump(restore(new)), ast.dump(h352))

    def test_entire_H352_AST_inverse(self):
        _, want = leaves(SOURCE)
        h352, new = transform(SOURCE)
        self.assertEqual(ast.dump(want), ast.dump(h352))
        self.assertEqual(ast.dump(restore(new)), ast.dump(h352))
        kws = {k.arg: ast.literal_eval(k.value) for k in calls(new)[0].keywords}
        old = {k.arg: ast.literal_eval(k.value) for k in calls(h352)[0].keywords}
        self.assertEqual(
            {k: (old[k], v) for k, v in kws.items() if v != old[k]},
            {"min_child_samples": (20, 40)},
        )

    def test_unexpected_input_rejected(self):
        for source in [
            SOURCE.replace("min_child_samples=20", "min_child_samples=40"),
            SOURCE.replace("max_depth=3", "max_depth=4"),
            SOURCE.replace("reg_lambda=1.0", "reg_lambda=5.0"),
        ]:
            with self.assertRaises(AssertionError):
                transform(source)


if __name__ == "__main__":
    unittest.main()
