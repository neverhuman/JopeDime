"""Fresh roots through byte-pinned H310/H303; no development outcomes enter this process."""

from pathlib import Path
import sys, os, json, hashlib, datetime
from concurrent.futures import ProcessPoolExecutor

P = Path(__file__).resolve().parent.parent
D = Path("/home/ubuntu/jope-data/h312-astra")
sys.path.insert(0, str(P / "vendor/h310_ensvote2_dev91_vs_r3175"))
import run_h310 as h

ROOTS = (30004, 30005, 30006)
COUNTERS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def durable(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


def task(root):
    result = h.lock_task(root)
    import trainonly_loader as tl

    lp = D / "lock/fold_4" / f"root_{root}" / "fit_lock.json"
    rec = dict(
        root_seed=root,
        fit_lock_sha256=sha(lp),
        counters=dict(
            COUNTERS,
            **{
                "worker_dev_guard_denials": h.DEV_GUARD["denied"],
                "worker_forbidden_packet_denials": h.LOCK_CTX["r303"].G["P8"].denied,
            },
        ),
        dev91_rows_read_before_lock=tl.COUNTERS["dlval_feature_rows_read"],
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        result=result,
    )
    assert rec["dev91_rows_read_before_lock"] == 0 and h.DEV_GUARD["denied"] == 0
    durable(lp.parent / "lock_sha_receipt.json", rec)
    return rec


def main():
    assert json.loads((P / "prereg-published.json").read_text())["published"]
    assert (
        json.loads((P / "source/regression_gate.json").read_text())["status"] == "PASS"
    )
    # Read only the status, never historical outcomes or choices, inside a fresh process.
    pins = json.loads((P / "source/dependency_pins.json").read_text())
    for name, digest in pins["vendor"].items():
        if name.endswith(".py") or Path(name).name == "manifest.json":
            assert sha(P / "vendor" / name) == digest
    h.DATA = D
    h.check_pins()
    dev, _ = h.dev_cids()
    h.install_dev_guard(dev)

    # Block own historical outcomes as well as the frozen H310 filename guard.
    def guard(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            text = os.fsdecode(args[0])
            assert not any(
                x in text
                for x in [
                    "/references/",
                    "/out/",
                    "regression_receipt.json",
                    "metrics_full.json",
                ]
            ), text

    sys.addaudithook(guard)
    r, counters = h.load_train_side()
    COUNTERS.update(counters)
    import trainonly_loader as tl

    r.G["P8"] = tl.GuardedPackets(r.G["P8"], set(dev))
    a = r.G["a302"]
    core = r.G["core"]
    for root, expected in a.EXPECTED_MEMBER_SEEDS.items():
        assert (
            tuple(core.stream_seed(root, "H300_ENS_VOTE2_MEMBER", m) for m in range(3))
            == expected
        )
    derived = {
        root: tuple(
            core.stream_seed(root, "H300_ENS_VOTE2_MEMBER", m) for m in range(3)
        )
        for root in ROOTS
    }
    a.EXPECTED_MEMBER_SEEDS.update(derived)
    h.LOCK_CTX["r303"] = r
    with ProcessPoolExecutor(max_workers=3) as ex:
        receipts = list(ex.map(task, ROOTS))
    durable(
        P / "source/lock_receipt.json",
        dict(
            status="PASS",
            roots=ROOTS,
            member_seeds=derived,
            receipts=receipts,
            dev91_rows_read_before_lock=0,
            source_sha256=sha(__file__),
        ),
    )
    print("ALL THREE FRESH LOCKS FSYNCED", flush=True)


if __name__ == "__main__":
    main()
