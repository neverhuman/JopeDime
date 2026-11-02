"""Compatibility facade for the continuous consumer label implementation."""
from . import labels_core as _core

_EXPORTS = {'_check_eps': '_impl_check_eps', 'LabelTable': '_impl_LabelTable', 'midrank_pcts': '_impl_midrank_pcts', 'label_context': '_impl_label_context', '_is_pair': '_impl_is_pair', 'robust_scale': '_impl_robust_scale', 'epsilon': '_impl_epsilon', 'is_detailed': '_impl_is_detailed', 'LabelError': '_impl_LabelError', 'label_corpus': '_impl_label_corpus', 'clip_q': '_impl_clip_q', 'ContextLabel': '_impl_ContextLabel', 'normalize_outcome': '_impl_normalize_outcome', '_y_array': '_impl_y_array', '_req_str': '_impl_req_str', 'RecordLabel': '_impl_RecordLabel', '_real': '_impl_real', 'ranking_pairs': '_impl_ranking_pairs', 'q_cap': '_impl_q_cap', '_valid_baseline': '_impl_valid_baseline', 'q_from_detail': '_impl_q_from_detail', '_wall_us': '_impl_wall_us', 'q_grid': '_impl_q_grid'}

for _name, _internal in _EXPORTS.items():
    _value = getattr(_core, _internal)
    if hasattr(_value, "__module__") and getattr(_value, "__module__", None) == _core.__name__:
        _value.__module__ = __name__
    if hasattr(_value, "__name__") and getattr(_value, "__module__", None) == __name__:
        _value.__name__ = _name
        _value.__qualname__ = _name
    globals()[_name] = _value

__all__ = ['LabelError', 'robust_scale', 'epsilon', 'is_detailed', 'normalize_outcome', 'q_from_detail', 'q_cap', 'clip_q', 'q_grid', 'midrank_pcts', 'RecordLabel', 'ContextLabel', 'LabelTable', 'label_context', 'label_corpus', 'ranking_pairs', 'math', 'numbers', 'np', 'dataclass', 'Any', 'Mapping', 'Sequence', 'COMPLETED', 'TIMEOUT_OUTCOME', 'TIMEOUT_S', 'NONFINITE', 'FAILED_PREFIX', 'ABANDONED', 'MISSING', 'LOCAL_BACKENDS', 'Q_GRID_SCALE', 'Q_CLIP', 'GATE_RATIO', 'FEAS_COMPLETED', 'FEAS_FAILED', 'FEAS_TIMEOUT', 'CONTEXT_STATUSES', 'Q_STATUSES', 'COUNT_KEYS', 'RANK_WEIGHT_CAP', 'RANK_BOOST_THRESHOLD', 'RANK_BOOST_FACTOR']
for _public_name in __all__:
    if _public_name not in globals():
        globals()[_public_name] = getattr(_core, _public_name)
for _public_name, _internal_name in _EXPORTS.items():
    globals()[_internal_name] = globals()[_public_name]

def __getattr__(name):
    if name in _EXPORTS:
        return getattr(_core, _EXPORTS[name])
    return getattr(_core, name)
