"""A relative or empty HOME must not expand a signature dataset tilde."""

from pathlib import Path

import pytest

from jope.signatures import expand_signature_dataset


def test_a_relative_home_does_not_expand_a_signature_dataset():
    assert expand_signature_dataset("~/dataset", "/users/alice") == Path("/users/alice/dataset")
    with pytest.raises(ValueError):
        expand_signature_dataset("~/dataset", "relative/home")
    with pytest.raises(ValueError):
        expand_signature_dataset("~/dataset", "")
    assert expand_signature_dataset("dataset", "relative/home") == Path("dataset")
