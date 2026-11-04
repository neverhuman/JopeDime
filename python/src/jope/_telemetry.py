"""`jope._telemetry` is the Rust telemetry extension from `crates/jope-telemetry`.

The compiled module ships in the separate, non-editable `jope-native` distribution
(`python/native`) so builds never place a native library in this source tree. This file only
aliases it: after import, `sys.modules["jope._telemetry"]` is the extension module itself, so
`__file__` points at the installed `.so` and every class and exception keeps its identity.
"""

import sys

from jope_native import _telemetry as _native

sys.modules[__name__] = _native
