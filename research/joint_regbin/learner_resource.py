"""Explicit, separately qualified resource-policy amendment to frozen learner558.

The base learner file, numerical implementation and plan stay unchanged. This
entry point replaces exactly its resource callback: CPU dispatch belongs to the
five-sample outer gate; an in-flight unit keeps running through CPU spikes.
GPU memory safety remains immediate. A separate immutable sidecar binds every
new response to this full source hash and its CPU numerical-equivalence proof.
"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.joint_regbin import driver, learner, protocol  # noqa: E402

BASE_SHA = "558780bb5d91318864991ca398b9c63c4b31a12d1c8f3dd04ae26db7655b9918"
AUTHORITY = Path("/home/ubuntu/nudges/lanes_t0430_rulings.md")
AUTHORITY_SHA = "9f37306a25cafb0d213625d999c1d255bd8139425794ab2e5e82db15583da038"


def resource_check(device):
    """CPU spikes cannot abort a running unit; retain the exact GPU guard."""
    if device.type == "cuda":
        free, total = learner.torch.cuda.mem_get_info(device)
        if free < 0.1 * total:
            raise RuntimeError("GPU memory exceeds 90% budget")


def install():
    # Deliberate reviewed execution amendment, never a hidden hash substitution.
    # The new callback does no model, gradient, optimizer or RNG computation.
    learner.resource_check = resource_check


def qualification(path, expected_sha):
    if protocol.file_sha(path) != expected_sha:
        raise ValueError("resource qualification hash differs")
    evidence = json.loads(Path(path).read_text())
    if (
        evidence.get("status") != "PASS"
        or evidence.get("base_learner_sha256") != BASE_SHA
        or evidence.get("wrapper_sha256") != protocol.file_sha(__file__)
        or evidence.get("authority_sha256") != AUTHORITY_SHA
        or evidence.get("numerical_equivalence") is not True
        or evidence.get("finite_gradients") is not True
        or evidence.get("actual_optimizer_updates") != 18
        or evidence.get("arms") != list(protocol.ARMS)
        or evidence.get("dope_sha256") != protocol.DOPE
    ):
        raise ValueError("resource amendment lacks its qualified source binding")
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "response", "packet", "initial-checkpoint"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if protocol.file_sha(AUTHORITY) != AUTHORITY_SHA:
        raise ValueError("resource amendment authority changed")
    if protocol.file_sha(learner.__file__) != BASE_SHA:
        raise ValueError("qualified base learner changed")
    learner.verify_sources()
    receipt_sha = os.environ["JOPE_JRB_RESOURCE_QUALIFICATION_SHA256"]
    qualification(Path(os.environ["JOPE_JRB_RESOURCE_QUALIFICATION"]), receipt_sha)
    request_path = protocol.safe_path(args.request.parent, args.request.name)
    request = json.loads(request_path.read_text())
    install()
    learner.main()
    driver.write_new(
        args.response.with_suffix(".resource-amendment.json"),
        dict(
            schema="jope.jrb01.resource-amendment-response/v1",
            base_learner_sha256=BASE_SHA,
            wrapper_sha256=protocol.file_sha(__file__),
            qualification_sha256=receipt_sha,
            authority_sha256=AUTHORITY_SHA,
            request_sha256=protocol.file_sha(request_path),
            response_sha256=protocol.file_sha(args.response),
            plan_sha256=request["plan_sha256"],
            job_id=request["job_id"],
            phase=request["phase"],
        ),
    )


if __name__ == "__main__":
    main()
