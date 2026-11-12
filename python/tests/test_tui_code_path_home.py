"""A relative or empty HOME must not expand a code-path tilde."""

from pathlib import Path

from jope.tui_helpers import _expand_home_in


def test_a_relative_home_does_not_expand_a_tilde():
    assert _expand_home_in("~/models/a.py", "/users/alice") == Path("/users/alice/models/a.py")
    assert _expand_home_in("~/models/a.py", "relative/home") == Path("~/models/a.py")
    assert _expand_home_in("~/models/a.py", "") == Path("~/models/a.py")
    assert _expand_home_in("~", "") == Path("~")
    assert _expand_home_in("plain/a.py", "relative/home") == Path("plain/a.py")
