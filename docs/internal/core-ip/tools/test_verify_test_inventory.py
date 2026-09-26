import copy
import json
from pathlib import Path
import tempfile
import unittest

from verify_test_inventory import INVENTORY, verify


class InventoryTests(unittest.TestCase):
    def test_committed_inventory(self):
        result = verify()
        self.assertEqual(result["baseline_failures"], 33)
        for subsystem in ("preparation", "encoders", "policy", "gp", "fitting",
                          "teaching", "training", "storage", "export"):
            self.assertGreater(result["executing_regression_subsystems"].get(subsystem, 0), 0)

    def test_mutations_are_rejected(self):
        document = json.loads(INVENTORY.read_text())
        regression = next(k for k, v in document["items"].items() if v["role"] == "regression")
        historical = next(k for k, v in document["items"].items() if v["role"] == "historical")
        fence = next(i for i, r in enumerate(document["baseline_triage"])
                     if r["triage"] == "fit-authority-fence")
        fixed = next(i for i, r in enumerate(document["baseline_triage"])
                     if r["status"] == "fixed-locally")

        def only_subsystem(doc, name):
            # Remove every regression claim for one required subsystem.
            for item in doc["items"].values():
                if item["role"] == "regression" and name in item["subsystems"]:
                    item["subsystems"].remove(name)
                    if not item["subsystems"]:
                        item["subsystems"].append("orchestration")

        mutations = {
            "unclassified item": (lambda d: d["items"].pop(regression), "unclassified"),
            "stale item": (lambda d: d["items"].__setitem__("target:ghost/test/none",
                                                             {"role": "research", "reason": "x"}), "stale"),
            "unknown role": (lambda d: d["items"][regression].update(role="passing"), ": role"),
            "unknown subsystem": (lambda d: d["items"][regression].update(subsystems=["magic"]),
                                  ": subsystems"),
            "regression without subsystem": (lambda d: d["items"][regression].update(subsystems=[]),
                                             ": subsystems"),
            "non-regression without reason": (lambda d: d["items"][historical].pop("reason"),
                                              ": reason"),
            "uncovered subsystem": (lambda d: only_subsystem(d, "teaching"),
                                    "no executing regression item for \\['teaching'\\]"),
            "untriaged failure": (lambda d: d["baseline_triage"].pop(0), "untriaged baseline"),
            "duplicate triage": (lambda d: d["baseline_triage"].append(
                copy.deepcopy(d["baseline_triage"][0])), "duplicate triage"),
            "invented failure": (lambda d: d["baseline_triage"][0].update(test="imaginary"),
                                 "unknown baseline failure"),
            "fence marked fixed": (lambda d: d["baseline_triage"][fence].update(
                status="fixed-locally", evidence=["x"]), "cannot be marked fixed"),
            "fixed without evidence": (lambda d: d["baseline_triage"][fixed].pop("evidence"),
                                       "without evidence"),
            "unknown status": (lambda d: d["baseline_triage"][0].update(status="passed"), ": status"),
            "subsystem without package": (lambda d: d["subsystems"]["gp"].update(work_packages=[]),
                                          "subsystem gp definition"),
        }
        for label, (mutate, reason) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                changed = copy.deepcopy(document)
                mutate(changed)
                path = Path(directory) / "inventory.json"
                path.write_text(json.dumps(changed))
                with self.assertRaisesRegex(ValueError, reason):
                    verify(path)


if __name__ == "__main__":
    unittest.main()
