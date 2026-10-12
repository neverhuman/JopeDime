"""Load a PolicyLM copy from a checkpoint directory. The source tree is not written.

Optimizer moments are dropped. lm_head stays tied to tok_emb. The parameter
hash is dltrain.model.parameter_sha256.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _import_dltrain():
    root = Path(__file__).resolve().parents[5]
    src = root / "deploy/dl-trainonly/lane/src/dl_training_v1"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from dltrain.model import PolicyConfig, PolicyLM, assert_tied, parameter_sha256

    return PolicyConfig, PolicyLM, assert_tied, parameter_sha256


def state_dict_from_checkpoint_tensors(raw):
    """Keep model weights. Refuse a stored lm_head and drop Adam moments."""
    state = {}
    for key, value in raw.items():
        if key.startswith(("optimizer.", "rng.")):
            continue
        if "lm_head.weight" in key:
            raise ValueError("lm_head.weight is tied and must not be stored")
        if not key.startswith("model."):
            raise ValueError(f"unexpected checkpoint tensor {key}")
        state[key[len("model.") :]] = value
    if "tok_emb.weight" not in state:
        raise ValueError("checkpoint is missing tok_emb.weight")
    return state


def load_weights(model, raw, expected_sha256):
    """Load ``raw`` tensors, retie the head, and require the parameter hash."""
    _, _, assert_tied, parameter_sha256 = _import_dltrain()
    state = state_dict_from_checkpoint_tensors(raw)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise ValueError(f"unexpected model keys: {unexpected}")
    if set(missing) != {"lm_head.weight"}:
        raise ValueError(f"unexpected missing keys: {missing}")
    model.lm_head.weight = model.tok_emb.weight
    assert_tied(model)
    digest = parameter_sha256(model)
    if digest != expected_sha256:
        raise ValueError("parameter_sha256 does not match the checkpoint")
    model.eval()
    return digest


def load_policy_copy(checkpoint_dir):
    """Read a copied checkpoint. Does not write ``checkpoint_dir``."""
    checkpoint_dir = Path(checkpoint_dir)
    PolicyConfig, PolicyLM, _, _ = _import_dltrain()
    manifest = json.loads((checkpoint_dir / "manifest.json").read_text())
    model = PolicyLM(PolicyConfig.from_json(manifest["model_config"]))
    from safetensors.torch import load_file

    raw = load_file(str(checkpoint_dir / "state.safetensors"))
    digest = load_weights(model, raw, manifest["parameter_sha256"])
    return model, {
        "parameter_sha256": digest,
        "step": manifest["step"],
        "parameters": manifest["parameters"],
        "state_sha256": manifest["state_sha256"],
        "promoted": False,
    }
