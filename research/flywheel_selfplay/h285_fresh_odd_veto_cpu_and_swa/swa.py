"""Outcome-blind parameter de-duplication and uniform adapter-only averaging; no forward."""

from pathlib import Path
import hashlib
import json
import importlib.util
from dependencies import ROOT
import h124
import torch

ROUNDS = (2750, 2795, 3100, 3175, 4330)
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"


def tensor_sha(t):
    a = t.detach().cpu().contiguous().numpy()
    h = hashlib.sha256()
    h.update(f"{a.dtype.str}:{list(a.shape)}:".encode())
    h.update(a.tobytes())
    return h.hexdigest()


def state_sha(state, keys):
    return hashlib.sha256(
        "".join(f"{k}={tensor_sha(state[k])}\n" for k in sorted(keys)).encode()
    ).hexdigest()


def average(states, keys):
    result = {}
    for k in keys:
        values = [s[k] for s in states]
        assert all(
            v.shape == values[0].shape and v.dtype == values[0].dtype for v in values
        )
        if values[0].is_floating_point():
            result[k] = (
                torch.stack([v.double() for v in values]).mean(0).to(values[0].dtype)
            )
        else:
            assert all(torch.equal(v, values[0]) for v in values)
            result[k] = values[0].clone()
    return result


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    h124.REPO = Path("/home/ubuntu/JopeDime")
    guard = h124.source_module(
        "python/src/jope/north_star/dope_freeze_guard.py", "h285_guard"
    )
    src = Path("/home/ubuntu/jope-data/h50/input/cpu_exact_fused.py")
    assert (
        hashlib.sha256(src.read_bytes()).hexdigest()
        == "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
    )
    spec = importlib.util.spec_from_file_location("h285_fused", src)
    fused = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    guard.lock_dope_weights_and_eval_mode(net)
    net.eval()

    def forbid_forward(*args):
        raise RuntimeError("No forward allowed during SWA preparation")

    net.register_forward_pre_hook(forbid_forward)
    reference = torch.load(
        ROOT / "inputs/checkpoints/round-3175.pt", map_location="cpu", weights_only=True
    )
    ref = reference["state_dict"]
    keys = sorted(k for k in ref if k.startswith("adapter_10tok."))
    other = sorted(set(ref) - set(keys))
    assert keys
    receipts = []
    unique = {}
    states = []
    for n in ROUNDS:
        path = ROOT / "inputs/checkpoints" / f"round-{n}.pt"
        raw = path.read_bytes()
        ck = torch.load(path, map_location="cpu", weights_only=True)
        state = ck["state_dict"]
        assert ck["dope_sha256"] == DOPE and set(state) == set(ref)
        assert all(torch.equal(state[k], ref[k]) for k in other), (
            f"non-adapter state differs in r{n}"
        )
        net.load_state_dict(state, strict=True)
        snapshot = guard.snapshot_dope_weights(net)
        assert snapshot.composite_sha256 == DOPE
        digest = state_sha(state, keys)
        duplicate = unique.get(digest)
        if duplicate is None:
            unique[digest] = n
            states.append(state)
        receipts.append(
            dict(
                round=n,
                checkpoint_sha256=hashlib.sha256(raw).hexdigest(),
                adapter_sha256=digest,
                duplicate_of=duplicate,
                retained=duplicate is None,
                dope_sha256=snapshot.composite_sha256,
                non_adapter_bit_identical_to_r3175=True,
                adapter_tensor_sha256={k: tensor_sha(state[k]) for k in keys},
            )
        )
    merged = {k: v.clone() for k, v in ref.items()}
    merged.update(average(states, keys))
    assert all(torch.equal(merged[k], ref[k]) for k in other)
    net.load_state_dict(merged, strict=True)
    after = guard.snapshot_dope_weights(net).composite_sha256
    assert after == DOPE and not torch.cuda.is_initialized()
    out = ROOT / "inputs/checkpoints/swa.pt"
    assert not out.exists()
    torch.save(
        dict(
            state_dict=merged,
            dope_sha256=DOPE,
            evaluation_identity=reference["evaluation_identity"],
            round=-285,
            origin="H285 deterministic uniform distinct-adapter average; exploratory",
            optimizer_updates=0,
            members=list(unique.values()),
        ),
        out,
    )
    out.chmod(0o444)
    record = dict(
        status="PASS",
        round_order=list(ROUNDS),
        members=list(unique.values()),
        unique_adapters=len(states),
        deduplication="sorted adapter state tensor names, dtype+shape+bytes sha256; retain first round",
        averaging="uniform float64 sum/mean then cast each floating tensor to original dtype; nonfloating tensors must match",
        member_receipts=receipts,
        swa_adapter_sha256=state_sha(merged, keys),
        swa_checkpoint_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        non_adapter_sha256=state_sha(ref, other),
        dope_before=DOPE,
        dope_after=after,
        adapter_tensor_count=len(keys),
        adapter_elements=sum(ref[k].numel() for k in keys),
        forward_calls=0,
        optimizer_updates=0,
        cuda_initialized=False,
    )
    (ROOT / "source/swa-members.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: v for k, v in record.items() if k != "member_receipts"}))


if __name__ == "__main__":
    main()
