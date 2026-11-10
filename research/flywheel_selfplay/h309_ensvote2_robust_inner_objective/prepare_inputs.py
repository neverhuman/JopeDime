"""Pinned source export and exact-version H303 reference download; no feature reads."""

from pathlib import Path
import hashlib
import json
import subprocess
import tarfile
import boto3

ROOT = Path(__file__).resolve().parent.parent
SCRATCH = Path("/home/ubuntu/jope-data/h309-astra")
REPO = Path("/home/ubuntu/JopeDime")
H303 = "h303_ensvote2_fully_nested_hparams"
H302 = "h302_ensvote2_trainseed_nested_cv"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    pins = {}
    vendor = ROOT / "vendor"
    vendor.mkdir(exist_ok=True)
    for slug, files in [
        (H303, ["run_h303.py", "evaluate_h303.py", "manifest.json"]),
        (H302, ["run_h302.py", "analysis_h302.py"]),
    ]:
        d = vendor / slug
        d.mkdir(exist_ok=True)
        for name in files:
            path = f"research/flywheel_selfplay/{slug}/{name}"
            raw = subprocess.check_output(
                ["git", "show", "7bef8870f7a9ffdb6565c5277c59f4af6c27c93d:" + path],
                cwd=REPO,
            )
            (d / name).write_bytes(raw)
            pins[slug + "/" + name] = hashlib.sha256(raw).hexdigest()
    assert (
        pins[H303 + "/run_h303.py"]
        == "fa393b0a512efe0b471d8efc7f0a927b72f1e2334c3980941ab39b9afe20e067"
    )
    for name, digest in json.loads((vendor / H303 / "manifest.json").read_text())[
        "h302_sources"
    ].items():
        assert pins[H302 + "/" + name] == digest
    raw = subprocess.check_output(
        [
            "git",
            "show",
            "2e2a22eda5:research/flywheel_selfplay/h308_ensvote2_trainonly_repartition/trainonly_loader.py",
        ],
        cwd=REPO,
    )
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == "65a86b6d636461761eaca9bdba413ba5dec34140cd49d4e0ca020aa849a26c6d"
    (ROOT / "source/trainonly_loader.py").write_bytes(raw)
    (ROOT / "source/dependency_pins.json").write_text(
        json.dumps(
            dict(
                h303_commit="7bef8870f7a9ffdb6565c5277c59f4af6c27c93d",
                vendor=pins,
                loader_commit=subprocess.check_output(
                    ["git", "rev-parse", "2e2a22eda5"], cwd=REPO, text=True
                ).strip(),
                loader_sha256=digest,
            ),
            indent=2,
        )
        + "\n"
    )
    SCRATCH.mkdir(exist_ok=True, parents=True)
    sess = boto3.Session(profile_name="veox-jepson-prod")
    assert sess.client("sts").get_caller_identity()["Account"] == "714044927654"
    s3 = sess.client("s3", region_name="us-east-2")
    key = (
        "experiments/rd_breakthrough_20261030/H303/h303_models_locks_and_outputs.tar.gz"
    )
    ver = "hshnLPUZuvpHewykQNTh..mWAGu9KGHo"
    expected = "8957cf0e0b55c21231fd3d40d56ce5bdf27aa915dc97786bcbc00dbee782857d"
    p = SCRATCH / "h303_reference.tar.gz"
    r = s3.get_object(Bucket="veox-jopedime-learning-use2", Key=key, VersionId=ver)
    assert r["VersionId"] == ver
    h = hashlib.sha256()
    with p.open("xb") as f:
        for b in r["Body"].iter_chunks(1024 * 1024):
            h.update(b)
            f.write(b)
    r["Body"].close()
    assert h.hexdigest() == expected
    dest = SCRATCH / "h303_reference"
    dest.mkdir()
    members = []
    with tarfile.open(p, "r:gz") as tf:
        for m in tf.getmembers():
            name = m.name.removeprefix("./")
            parts = Path(name).parts
            if m.isfile() and (
                (name.startswith("scratch/fold_") and name.endswith("/fit_lock.json"))
                or name in {"out/picks.npz", "out/metrics_full.json"}
            ):
                assert not Path(name).is_absolute() and ".." not in parts
                raw = tf.extractfile(m).read()
                out = dest / name
                out.parent.mkdir(exist_ok=True, parents=True)
                out.write_bytes(raw)
                members.append(
                    dict(
                        path=name,
                        bytes=len(raw),
                        sha256=hashlib.sha256(raw).hexdigest(),
                    )
                )
    assert len(list(dest.glob("scratch/fold_*/root_*/fit_lock.json"))) == 12
    assert (dest / "out/picks.npz").is_file() and (
        dest / "out/metrics_full.json"
    ).is_file()
    rec = dict(
        bucket="veox-jopedime-learning-use2",
        key=key,
        version_id=ver,
        sha256=expected,
        bytes=p.stat().st_size,
        exact_version_get_verified=True,
        members=members,
    )
    (ROOT / "source/input_receipt.json").write_text(json.dumps(rec, indent=2) + "\n")
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
