"""Balanced, bounded dataset expansion with immutable provenance and backpressure.

This service generates prepared research variants. Encoding, GP scoring and
training admission are separate measured stages; merely creating data does not
advance those counters. Children always inherit their parent's DL partition.
"""
from . import evergreen_expansion as _expansion
from . import evergreen_recipes as _recipes
from . import evergreen_source as _source

_MODULES = (_source, _recipes, _expansion)
for _module in _MODULES:
    for _name, _value in vars(_module).items():
        if not _name.startswith("_"):
            globals()[_name] = _value
        if getattr(_value, "__module__", None) == _module.__name__:
            _value.__module__ = __name__
            if isinstance(_value, type):
                for _member in vars(_value).values():
                    if getattr(_member, "__module__", None) == _module.__name__:
                        _member.__module__ = __name__


def __getattr__(name):
    for _module in _MODULES:
        try:
            return getattr(_module, name)
        except AttributeError:
            pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()).union(*(set(dir(_module)) for _module in _MODULES)))


if __name__ == "__main__":
    main()
