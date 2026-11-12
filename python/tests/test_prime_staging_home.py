"""A relative or empty HOME must not expand a Prime staging tilde."""

from pathlib import Path

import pytest

from jope.prime_staging import expand_staged_path


def test_a_relative_home_does_not_expand_a_staging_tilde():
    assert expand_staged_path("~/data", "/users/alice") == Path("/users/alice/data")
    with pytest.raises(ValueError):
        expand_staged_path("~/data", "relative/home")
    with pytest.raises(ValueError):
        expand_staged_path("~/data", "")
    assert expand_staged_path("data", "relative/home") == Path("data")
