"""Fresh exact-version H352 NL15 reference recovery; zero fits or predictions."""

from pathlib import Path
import hashlib, json, tarfile
import boto3
from nl15_native_reference import parse, compare_native

S = Path(__file__).resolve().parent
D = S.parent
pins = json.loads((S / "h352_reference_pins.json").read_text())
s3 = boto3.Session(profile_name="veox-jepson-prod").client("s3")


class Reader:
    def __init__(self, body):
        self.body = body
        self.h = hashlib.sha256()
        self.n = 0

    def read(self, n=-1):
        raw = self.body.read(n)
        self.h.update(raw)
        self.n += len(raw)
        return raw


def main():
    receipts = []
    saved = {}
    for part in pins["reference_parts"]:
        response = s3.get_object(
            Bucket=part["bucket"], Key=part["key"], VersionId=part["version_id"]
        )
        assert response["VersionId"] == part["version_id"]
        reader = Reader(response["Body"])
        got = {}
        with tarfile.open(fileobj=reader, mode="r|gz") as tar:
            for item in tar:
                if item.name not in part["members"]:
                    continue
                assert item.isfile() and ".." not in Path(item.name).parts
                raw = tar.extractfile(item).read()
                meta = part["members"][item.name]
                assert (
                    len(raw) == meta["bytes"]
                    and hashlib.sha256(raw).hexdigest() == meta["sha256"]
                )
                rel = item.name.removeprefix("h352-jrb01/")
                assert rel != item.name
                path = (
                    D
                    / (
                        "nl15_native"
                        if "/refit/" in rel or rel.endswith("/native_model_review.json")
                        else "nl15_reference"
                    )
                    / rel
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.exists():
                    assert path.read_bytes() == raw, (
                        "Existing reference copy differs; retain and stop"
                    )
                else:
                    path.write_bytes(raw)
                    path.chmod(0o444)
                got[item.name] = meta
                saved[rel] = dict(path=str(path), **meta)
        while reader.read(1024 * 1024):
            pass
        response["Body"].close()
        assert (
            reader.h.hexdigest() == part["sha256"]
            and reader.n == part["bytes"]
            and got == part["members"]
        )
        receipts.append(
            dict(
                bucket=part["bucket"],
                key=part["key"],
                version_id=part["version_id"],
                sha256=reader.h.hexdigest(),
                bytes=reader.n,
                members=got,
            )
        )
        print(
            json.dumps(dict(reference_part=len(receipts), selected_members=len(got))),
            flush=True,
        )
    locks = sorted((D / "nl15_reference").glob("P*/fold_*/root_*/fit_lock.json"))
    assert len(locks) == 36
    expected_veto = json.loads((S / "base_veto_sha256.json").read_text())["members"]
    rankers = vetos = 0
    models = []
    for lp in locks:
        task = lp.parent.relative_to(D / "nl15_reference")
        lock = json.loads(lp.read_text())
        marker = json.loads((lp.parent / "score_once.json").read_text())
        assert marker["state"] == "COMPLETE"
        review = json.loads(
            (D / "nl15_native" / task / "native_model_review.json").read_text()
        )
        records = (
            review["models"]
            if isinstance(review, dict) and "models" in review
            else review
        )
        assert isinstance(records, list), (type(review), list(review)[:5])
        for path in sorted((D / "nl15_native" / task / "refit").glob("*.txt")):
            raw = path.read_bytes()
            sha = hashlib.sha256(raw).hexdigest()
            relative = "refit/" + path.name
            # Find the model-bound full constructor/native review from H352.
            matches = [r for r in records if r.get("sha256") == sha]
            assert len(matches) == 1, (str(path), len(matches))
            row = matches[0]
            native = parse(raw.decode())
            assert native == row["native_parameters"]
            kind = "LGBMClassifier" if path.name.endswith("-veto.txt") else "LGBMRanker"
            if kind == "LGBMRanker" and float(native["lambda_l2"]) != 1.0:
                (D / "INCOMPLETE_SPEC_MISMATCH.json").write_text(
                    json.dumps(
                        dict(
                            status="INCOMPLETE_SPEC_MISMATCH",
                            reason="Archived H352 native lambda_l2 must equal1 before any treatment fit",
                            file=str(path),
                            actual=native["lambda_l2"],
                            optimizer_fits=0,
                        )
                    )
                    + "\n"
                )
                raise SystemExit(3)
            compare_native(row["constructor_parameters"], native, kind, True)
            assert sha in json.dumps(lock), str(path)
            if kind == "LGBMClassifier":
                assert (
                    sha
                    == expected_veto["run/" + str(task) + "/refit/" + path.name][
                        "sha256"
                    ]
                )
                vetos += 1
            else:
                rankers += 1
            models.append(
                dict(
                    task=str(task),
                    file=relative,
                    kind=kind,
                    sha256=sha,
                    num_leaves=int(native["num_leaves"]),
                    lambda_l2=float(native["lambda_l2"]),
                    min_data_in_leaf=int(native["min_data_in_leaf"]),
                    max_depth=int(native["max_depth"]),
                )
            )
    assert rankers == 972 and vetos == 108
    receipt = dict(
        status="EXACT_VERSION_GET_AND_SHA_PASS",
        source_commit=pins["source_commit"],
        parts=receipts,
        members=saved,
        locks=36,
        scores=36,
        optimizer_fits=0,
        new_predictions=0,
    )
    (D / "nl15-fetch-receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    (D / "nl15-native-config.json").write_text(
        json.dumps(
            dict(
                status="PASS",
                rankers=rankers,
                veto_SHA_identities_verified=vetos,
                models=models,
                optimizer_fits=0,
                new_predictions=0,
            ),
            indent=1,
        )
        + "\n"
    )
    print(
        json.dumps(
            dict(status="PASS", locks=36, scores=36, rankers=rankers, vetos=vetos)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
