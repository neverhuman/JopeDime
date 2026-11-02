"""Resolve two immutable qualified-model source inputs without another checkout.

The reserved plan identifies the exact548 learner and its dc355 candidate
packer. Main subsequently changed that dependency pin. Import the two original
Git blobs as read-only, content-addressed model inputs; all other dependencies
must still match the original full source hashes. No model or optimizer runs.
"""

import importlib.util
import hashlib
import os
from pathlib import Path
import subprocess
import sys

from . import runtime

ROOT = Path(__file__).resolve().parents[2]
COMMIT = "f3818eed408adf50368d14f7284c1b17af4ea896"
CACHE = Path("/home/ubuntu/jope_plan/evidence/JRB-01/sol-jrb01-t1245/source-blobs")
SOURCES = {
    "candidate_builder": "dc3550d134615045e7baa3353670cdee0f2bbc7c8cb0a4acc76e1b0a17d7a6d9",
    "learner": "558780bb5d91318864991ca398b9c63c4b31a12d1c8f3dd04ae26db7655b9918",
}


def source_blob(name):
    expected = SOURCES[name]
    CACHE.mkdir(parents=True, exist_ok=True)
    if CACHE.is_symlink():
        raise ValueError("source-input directory cannot be a symlink")
    path = CACHE / (expected + ".py")
    if not path.exists():
        raw = subprocess.check_output(
            ["git", "show", COMMIT + ":research/joint_regbin/" + name + ".py"],
            cwd=ROOT,
        )
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("qualified Git source blob differs")
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        path.chmod(0o444)
    runtime.verify_file_pin(path, expected)
    return path


def load():
    """Import verified source bytes; retain ordinary file-hash request checks."""
    package = sys.modules[__package__]
    current = getattr(package, "learner", None)
    if getattr(current, "qualified_source_commit", None) == COMMIT:
        current.verify_sources()
        return current
    sys.dont_write_bytecode = True
    for name in SOURCES:
        path = source_blob(name)
        full = __package__ + "." + name
        spec = importlib.util.spec_from_file_location(full, path)
        if spec is None or spec.loader is None:
            raise ValueError("qualified source input cannot be imported")
        module = importlib.util.module_from_spec(spec)
        sys.modules[full] = module
        setattr(package, name, module)
        spec.loader.exec_module(module)
    learner = package.learner

    def verify_sources():
        runtime.verify_file_pin(Path(learner.__file__), SOURCES["learner"])
        for filename, expected in learner.SOURCE_PINS.items():
            name = Path(filename).stem
            active = sys.modules[__package__ + "." + name]
            if getattr(package, name) is not active:
                raise ValueError("ambiguous qualified dependency module")
            runtime.verify_file_pin(Path(active.__file__), expected)

    # Source-location verification is explicitly adapted to the actual imported
    # module files. Their byte hashes and all numerical functions are unchanged.
    learner.verify_sources = verify_sources
    learner.qualified_source_commit = COMMIT
    verify_sources()
    return learner
