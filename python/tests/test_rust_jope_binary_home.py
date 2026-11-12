"""A relative or empty HOME must not expand JOPE_RUST_BIN."""

from pathlib import Path

import pytest

from jope.cli import rust_jope_binary


def test_a_relative_home_does_not_expand_the_rust_binary():
    assert rust_jope_binary("~/bin/jope", "/users/alice") == Path("/users/alice/bin/jope")
    with pytest.raises(ValueError):
        rust_jope_binary("~/bin/jope", "relative/home")
    with pytest.raises(ValueError):
        rust_jope_binary("~/bin/jope", "")
    assert rust_jope_binary("bin/jope", "relative/home") == Path("bin/jope")
