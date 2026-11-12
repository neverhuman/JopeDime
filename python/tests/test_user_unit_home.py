"""A relative or empty HOME must not expand the systemd user unit directory."""

import importlib.util
import sys
from pathlib import Path

import pytest

_RESEARCH = Path(__file__).resolve().parents[1] / "test-support" / "jope" / "research"
sys.path.insert(0, str(_RESEARCH))
_SPEC = importlib.util.spec_from_file_location(
    "deploy_selfplay_bundle",
    _RESEARCH / "deploy_selfplay_bundle.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
user_unit_dir = _MODULE.user_unit_dir


def test_a_relative_home_does_not_expand_the_user_unit_dir():
    assert user_unit_dir("/users/alice") == Path("/users/alice/.config/systemd/user")
    with pytest.raises(ValueError):
        user_unit_dir("relative/home")
    with pytest.raises(ValueError):
        user_unit_dir("")
