"""H312 exact-version recovery and utility-only regression; no raw datasets/models."""

from pathlib import Path
import hashlib, json, tarfile, boto3
import numpy as np

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h314-astra")
KEY = "experiments/rd_breakthrough_20261030/H312/astra-20261104T093953Z/h312_evidence.tar.gz"
VERSION = "IWjt3KdZCfYSgh6.XeNPrKrBXQW_MNqQ"
SHA = "d8bbd501469baaeb777ce93e5d1bf49d657a7a4aa11e8f71a008f3d0df29ed15"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert json.loads((P / "prereg-published.json").read_text())["published"]
    s = boto3.Session(profile_name="veox-jepson-prod")
    assert s.client("sts").get_caller_identity()["Account"] == "714044927654"
    client = s.client("s3", region_name="us-east-2")
    ref = D / "references"
    ref.mkdir(parents=True, exist_ok=True)
    dest = ref / "h312_evidence.tar.gz"
    reply = client.get_object(
        Bucket="veox-jopedime-learning-use2", Key=KEY, VersionId=VERSION
    )
    assert reply["VersionId"] == VERSION
    with dest.open("xb") as f:
        for b in reply["Body"].iter_chunks(1024 * 1024):
            f.write(b)
    reply["Body"].close()
    assert sha(dest) == SHA
    wanted = [
        "outputs/out/all_seeds.npz",
        "outputs/out/family_gains.npz",
        "source/metrics.json",
        "source/score_receipt.json",
    ]
    members = []
    with tarfile.open(dest) as tf:
        for name in wanted:
            out = ref / Path(name).name
            raw = tf.extractfile(name).read()
            out.write_bytes(raw)
            members.append(
                dict(
                    path=str(out.relative_to(D)),
                    archive_member=name,
                    sha256=sha(out),
                    bytes=len(raw),
                )
            )
    inp = dict(
        status="PASS",
        objects=[
            dict(
                path=str(dest.relative_to(D)),
                bucket="veox-jopedime-learning-use2",
                key=KEY,
                version_id=VERSION,
                sha256=SHA,
                exact_version_get_verified=True,
            )
        ],
        members=members,
    )
    (P / "source/input_receipt.json").write_text(json.dumps(inp, indent=2) + "\n")
    with np.load(ref / "all_seeds.npz", allow_pickle=False) as z:
        u = z["u"]
        fam = z["families"]
        roots = z["roots"]
    d = np.array(
        [
            [
                u[s, fam == f, 1].mean() - u[s, fam == f, 0].mean()
                for f in sorted(set(fam))
            ]
            for s in range(6)
        ]
    )
    old = json.loads((ref / "metrics.json").read_text())
    expected = [x["gain"] for x in old["per_seed"]]
    with np.load(ref / "family_gains.npz", allow_pickle=False) as z:
        exact = bool(np.array_equal(d, z["gains"]))
    rec = dict(
        status="PASS"
        if d.mean(axis=1).tolist() == expected
        and float(u[0, :, 0].mean()) == 1.1389572403748425
        and exact
        else "REGRESSION_FAIL",
        roots=roots.tolist(),
        gains=d.mean(axis=1).tolist(),
        expected_gains=expected,
        baseline_mean=float(u[0, :, 0].mean()),
        family_vectors_exact=exact,
        prereg_commit=json.loads((P / "prereg-published.json").read_text())["head"],
        archive_sha256=SHA,
        raw_feature_or_label_reads=0,
        new_model_scores=0,
    )
    (P / "source/regression_receipt.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec), flush=True)
    assert rec["status"] == "PASS"
    (P / "source/regression_gate.json").write_text(
        json.dumps(
            dict(
                status="PASS", receipt_sha256=sha(P / "source/regression_receipt.json")
            )
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
