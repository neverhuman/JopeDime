"""A relative or empty HOME must not expand a Prime config tilde."""

from pathlib import Path

import pytest

from jope.prime_config import expand_config_home


def test_a_relative_home_does_not_expand_a_config_tilde():
    assert expand_config_home("~/prime.toml", "/users/alice") == Path("/users/alice/prime.toml")
    with pytest.raises(ValueError):
        expand_config_home("~/prime.toml", "relative/home")
    with pytest.raises(ValueError):
        expand_config_home("~/prime.toml", "")
    assert expand_config_home("prime.toml", "relative/home") == Path("prime.toml")
