"""A relative or empty HOME must not expand a target-maker path."""

from pathlib import Path

import pytest

from jope.target_maker.data import expand_target_path


def test_a_relative_home_does_not_expand_a_target_path():
    assert expand_target_path("~/cache", "/users/alice") == Path("/users/alice/cache")
    with pytest.raises(ValueError):
        expand_target_path("~/cache", "relative/home")
    with pytest.raises(ValueError):
        expand_target_path("~/cache", "")
    assert expand_target_path("cache", "relative/home") == Path("cache")
