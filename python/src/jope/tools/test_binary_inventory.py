import struct
import tempfile
import unittest
from pathlib import Path

import binary_verify as b
import experiment_stages as stages
from experiment_runtime import write


def fixture(root):
    output = root / "runs/binary-experiment"
    write(output / "aliases.json", {"families": [{"dataset": "d", "family": "f", "partition": "meta_train"}]})
    source = root / "source.csv"
    source.write_text("x,y\n" + "".join(f"{i},{i%2}\n" for i in range(10)))
    cuts = []
    for index in range(5):
        recipe = output / f"bindings/d/cut-{index}/preprocessing.json"
        write(recipe, {"labels": ["0", "1"]})
        cuts.append({"index": index, "train_indices": [i for i in range(10) if i // 2 != index],
            "validation_indices": [2*index, 2*index+1], "training_sha256": f"training-{index}",
            "validation_sha256": f"validation-{index}", "preprocessing_sha256": b.sha(recipe)})
    prepared = {"dataset": "d", "family": "f", "task": "binary", "qualification_only": True, "meta_partition": "meta_train",
        "binary_aliases_sha256": b.sha(output / "aliases.json"), "selected_train_indices": list(range(10)), "cuts": cuts,
        "storage": {"source": "source.csv", "source_sha256": b.sha(source), "original_training_row_map": list(range(10)), "feature_columns": ["x"], "target_column": "y"}}
    write(output / "bindings/d/prepared.json", prepared)
    jobs = [{"prepared": str((output / "bindings/d").relative_to(root)), "prepared_sha256": b.sha(output / "bindings/d/prepared.json"), "cut": i} for i in range(5)]
    write(output / "jobs.json", {"jobs": jobs})
    write(output / "jobs-initial.json", {"jobs": jobs})
    spec_paths = {name: output / "sampling" / f"{name}-spec.json" for name in ("target", "foundation")}
    spec_paths["hyperion"] = root / "manifests/hyperion-reviewed4-safe-spec.json"
    for name, path in spec_paths.items():
        spec = {}
        for field in (("binary", "weights", "config", "source_manifest", "loader", "runtime_manifest", "persistent_binary") if name == "hyperion" else ("module", "sampling", "parity")):
            artifact = root / f"{name}-{field}.json"
            write(artifact, {"files": [{"destination": spec["loader"], "sha256": spec["loader_sha256"], "bytes": (root / spec["loader"]).stat().st_size}]} if field == "runtime_manifest" else {"frozen": True})
            spec[field] = artifact.name
            spec[field + "_sha256"] = b.sha(artifact)
        write(path, spec)
    write(root / "transforms.json", {"frozen": True})
    write(root / "assembly.json", {"transforms_sha256": b.sha(root / "transforms.json")})
    write(output / "encoding-run.json", {"jobs_sha256": b.sha(output / "jobs.json"), "aliases_sha256": b.sha(output / "aliases.json"), "hyperion_spec_sha256": b.sha(spec_paths["hyperion"]), "projection_sha256": b.sha(root / "assembly.json")})
    write(output / "contexts/application.json", {"jobs_sha256": b.sha(output / "jobs.json"), "projection_refitted": False, "transforms": "transforms.json", "transforms_sha256": b.sha(root / "transforms.json")})
    write(output / "contexts/reference-parity.json", {"passed": True})
    rows = []
    for index, (job, cut) in enumerate(zip(jobs, cuts)):
        bindings = {}
        for name, (filename, dimensions) in b.COMPONENTS.items():
            directory = output / "components" / name / "d" / f"cut-{index}"
            directory.mkdir(parents=True)
            payload, receipt = directory / filename, directory / f"{name}.json"
            payload.write_bytes(struct.pack("<f", 1.0) * dimensions)
            write(receipt, {"dataset": "d", "family": "f", "cut": index, "prepared_sha256": job["prepared_sha256"],
                "training_sha256": cut["training_sha256"], "preprocessing_sha256": cut["preprocessing_sha256"],
                "signature_sha256": b.sha(payload), "shape": [dimensions], "final_test_targets_used": False, "spec_sha256": b.sha(spec_paths[name])})
            bindings[name] = {"receipt": str(receipt.relative_to(root)), "receipt_sha256": b.sha(receipt), "signature": str(payload.relative_to(root)), "signature_sha256": b.sha(payload), "spec_sha256": b.sha(spec_paths[name])}
        payload = output / "contexts/d" / f"cut-{index}.f32"
        payload.parent.mkdir(exist_ok=True)
        payload.write_bytes(struct.pack("<f", 1.0) * 2048)
        metadata = {"task": "binary", "qualification": True, "binary_aliases_sha256": b.sha(output / "aliases.json"), "dataset": "d", "family": "f", "cut": index, "partition": "meta_train",
            "prepared_sha256": job["prepared_sha256"], "training_sha256": cut["training_sha256"], "preprocessing_sha256": cut["preprocessing_sha256"],
            "encoder_spec_sha256": b.sha(root / "assembly.json"), "application_sha256": b.sha(output / "contexts/application.json"), "selected_training_rows_sha256": b.compact_sha(cut["train_indices"]),
            "dataset_dimensions": 1536, "target_dimensions": 512, "output_sha256": b.sha(payload), "components": bindings, "components_sha256": b.compact_sha(bindings)}
        write(payload.with_suffix(".json"), metadata)
        rows.append(dict(job, dataset="d", family="f", partition="meta_train", path=str(payload.relative_to(root)), sha256=b.sha(payload), receipt_sha256=b.sha(payload.with_suffix(".json")), components_sha256=metadata["components_sha256"]))
    index = {"task": "binary", "qualification": True, "dimensions": 2048, "encodings": rows, "application_sha256": b.sha(output / "contexts/application.json"), "assembly": "assembly.json", "assembly_sha256": b.sha(root / "assembly.json"), "reference_parity_sha256": b.sha(output / "contexts/reference-parity.json")}
    write(output / "contexts/index.json", index)
    write(output / "contexts-initial/index.json", index)
    report = {"admitted_contexts": 5, "completed_contexts": 5, "initial_contexts": 5, "admitted_datasets": 1, "aliases_sha256": b.sha(output / "aliases.json"), "jobs_sha256": b.sha(output / "jobs.json"), "context_index": str((output / "contexts/index.json").relative_to(root)), "context_index_sha256": b.sha(output / "contexts/index.json")}
    write(output / "report.json", report)
    return output


class BinaryInventoryTests(unittest.TestCase):
    def test_exact_inventory_and_normalized_optimization_behavior(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = fixture(root)
            result = b.verify(root, output)
            self.assertEqual(result["context_hashes"], 5)
            self.assertEqual(result["component_hashes"], 15)

    def test_missing_duplicate_malformed_extra_and_wrong_binding_rejected(self):
        for mutation in ("missing", "duplicate", "shape", "nan", "extra", "binding", "context-omission", "overlap", "runtime"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                output = fixture(root)
                payload = output / "components/target/d/cut-0/target.f32"
                if mutation == "runtime":
                    (root / "hyperion-loader.json").write_text("changed")
                elif mutation == "missing":
                    payload.unlink()
                elif mutation == "shape":
                    payload.write_bytes(b"1234")
                elif mutation == "nan":
                    payload.write_bytes(struct.pack("<f", float("nan")) * 512)
                elif mutation == "extra":
                    payload.with_name("foreign.f32").write_bytes(payload.read_bytes())
                elif mutation == "binding":
                    path = payload.with_suffix(".json")
                    value = b.read(path)
                    value["training_sha256"] = "foreign"
                    write(path, value)
                elif mutation == "overlap":
                    path = output / "contexts-initial/index.json"
                    value = b.read(path)
                    value["encodings"][0]["sha256"] = "foreign"
                    write(path, value)
                else:
                    path = output / "contexts/index.json"
                    value = b.read(path)
                    if mutation == "duplicate":
                        value["encodings"][1] = value["encodings"][0]
                    else:
                        value["encodings"].pop()
                    write(path, value)
                    report = b.read(output / "report.json")
                    report["context_index_sha256"] = b.sha(path)
                    write(output / "report.json", report)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    b.verify(root, output)

    def test_completed_stage_rechecks_payloads_before_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = fixture(root)
            commands = [["binary", "target-rank", "--output", str(output / "components/target"), "--jobs", str(output / "jobs.json"), "--spec", str(output / "sampling/target-spec.json"), "--rank", "0", "--world", "1"]]
            receipt = {"complete": True, "outputs": stages.inventory(root, commands)}
            stages.verify(root, commands, receipt)
            (output / "components/target/d/cut-4/target.f32").unlink()
            with self.assertRaises(FileNotFoundError):
                stages.verify(root, commands, receipt)
            with self.assertRaisesRegex(ValueError, "legacy stage"):
                stages.verify(root, commands, {"complete": True})


if __name__ == "__main__":
    unittest.main()
