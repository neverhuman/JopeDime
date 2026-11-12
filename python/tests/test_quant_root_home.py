"""A relative or empty HOME must not expand the quant dataset root."""

from pathlib import Path

import pytest

from jope.target_maker.data import quant_base


def test_a_relative_home_does_not_expand_the_quant_root():
    assert quant_base("/users/alice") == Path("/users/alice/quant")
    with pytest.raises(ValueError):
        quant_base("relative/home")
    with pytest.raises(ValueError):
        quant_base("")
