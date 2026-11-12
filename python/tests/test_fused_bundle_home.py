"""A relative or empty HOME must not expand a fused bundle path."""

from pathlib import Path

import pytest

from jope.target_maker.fused import expand_fused_bundle


def test_a_relative_home_does_not_expand_a_fused_bundle():
    assert expand_fused_bundle("~/bundle", "/users/alice") == Path("/users/alice/bundle")
    with pytest.raises(ValueError):
        expand_fused_bundle("~/bundle", "relative/home")
    with pytest.raises(ValueError):
        expand_fused_bundle("~/bundle", "")
    assert expand_fused_bundle("bundle", "relative/home") == Path("bundle")
