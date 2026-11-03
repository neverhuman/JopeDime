"""H282 eligibility (xbabe1 CPU, CUDA never initialised; decided before any utility is read for scoring).

Eligible = the read-only checkpoint copy loads STRICTLY (state-dict keys and shapes) into the pinned fused source
b9f1468a with adapter h40b, its round/admission/DOPE fields are consistent, AND a completed round-N.json exists with
dev_utility.mean (plus 91 per-context values and selected programs for the exact regression). No forward pass."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

INPUT = Path("/home/ubuntu/jope-data/h50/input")
SOURCE_SHA256 = "b9f1468a61aa4dc888c8f2931f9c759f7d4f057793a77ccb05eed6300fb4f9ab"
DOPE = "c51a56d031a4af40cb2b2661e6230109db684fc5cf80743dcad5c2fc770ba814"
ROUNDS = (782, 785, 859, 2750, 2795, 3100, 4330)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(ckdir):
    torch.set_num_threads(4)
    sys.path.insert(0, "/home/ubuntu/JopeDimeMicro/python")
    source = INPUT / "cpu_exact_fused.py"
    if sha(source) != SOURCE_SHA256:
        raise RuntimeError("source drift")
    fs = importlib.util.spec_from_file_location("cpu_exact_fused", source)
    fused = importlib.util.module_from_spec(fs)
    fs.loader.exec_module(fused)
    net = fused.FusedJopeNetwork(device=torch.device("cpu"), adapter_variant="h40b")
    out = []
    for n in ROUNDS:
        pt, js = ckdir / f"round-{n}.pt", ckdir / f"round-{n}.json"
        row = dict(round=n, pt_sha256=sha(pt), json_sha256=sha(js))
        try:
            state = torch.load(pt, map_location="cpu", weights_only=True)
            row.update(
                ckpt_round=int(state["round"]),
                admission=state.get("admission"),
                dope_sha256=state.get("dope_sha256"),
                evaluation_identity=state.get("evaluation_identity"),
            )
            missing, unexpected = net.load_state_dict(state["state_dict"], strict=True)
            row["strict_load"] = True
        except Exception as error:  # recorded, not raised: eligibility is a list
            row.update(strict_load=False, load_error=repr(error)[:300])
        meta = json.loads(js.read_text())
        du = meta.get("dev_utility") or {}
        row.update(
            json_round=meta.get("round"),
            json_evaluation_identity=meta.get("evaluation_identity"),
            has_dev_mean=isinstance(du.get("mean"), float),
            n_per_context=len(du.get("per_context") or []),
            n_selected_programs=len(du.get("selected_programs") or []),
        )
        row["eligible"] = bool(
            row["strict_load"]
            and row.get("ckpt_round") == n
            and row.get("admission") != "rejected"
            and row.get("dope_sha256") == DOPE
            and row["json_round"] == n
            and row["has_dev_mean"]
            and row["n_per_context"] == 91
            and row["n_selected_programs"] == 91
        )
        out.append(row)
    print(
        json.dumps(dict(cuda_initialized=torch.cuda.is_initialized(), checkpoints=out))
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]))
