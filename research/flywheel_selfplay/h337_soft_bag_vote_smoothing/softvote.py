"""One comparison-node treatment; BASE integer threshold semantics are retained."""
import ast
import copy
from contextlib import contextmanager
import hashlib
import inspect
import math

import numpy as np
from scipy.special import expit

_CURRENT = None
_TRACE = None


def indicator(delta, margin):
    assert _CURRENT is not None and np.shape(delta) == (8,)
    assert math.isfinite(margin)
    values = expit((np.asarray(delta, dtype=np.float64) - margin) / _CURRENT)
    assert np.isfinite(values).all() and np.all((values >= 0) & (values <= 1))
    _TRACE.append(float(np.sum(values)))
    return values


@contextmanager
def temperatures(values):
    global _CURRENT, _TRACE
    assert _CURRENT is None
    values = np.asarray(values, dtype=np.float64)
    assert values.shape == (8,) and np.isfinite(values).all() and np.all(values > 0)
    trace = []
    _CURRENT, _TRACE = values.copy(), trace
    try:
        yield trace
    finally:
        _CURRENT = _TRACE = None


def transform(source):
    tree = ast.parse(source)
    expected = ast.parse('(bag_scores[:, i, j_int] - bag_scores[:, i, 0]) > margin', mode='eval').body
    matches = [n for n in ast.walk(tree) if ast.dump(n, include_attributes=False) == ast.dump(expected, include_attributes=False)]
    assert len(matches) == 1, 'Expected one thresholded bag margin; no alternate arm'
    target = matches[0]
    replacement = ast.Call(func=ast.Name(id='_h337_indicator', ctx=ast.Load()),
                           args=[copy.deepcopy(target.left), copy.deepcopy(target.comparators[0])], keywords=[])

    class Replace(ast.NodeTransformer):
        def visit_Compare(self, node):
            return ast.copy_location(replacement, node) if node is target else self.generic_visit(node)

    changed = ast.fix_missing_locations(Replace().visit(tree))
    inverse = copy.deepcopy(changed)

    class Undo(ast.NodeTransformer):
        def visit_Call(self, node):
            if isinstance(node.func, ast.Name) and node.func.id == '_h337_indicator':
                return ast.copy_location(copy.deepcopy(expected), node)
            return self.generic_visit(node)

    inverse = Undo().visit(inverse)
    assert ast.dump(inverse, include_attributes=False) == ast.dump(ast.parse(source), include_attributes=False)
    return changed, dict(original_sha256=hashlib.sha256(source.encode()).hexdigest(),
                         original_ast_sha256=hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode()).hexdigest(),
                         treatment_ast_sha256=hashlib.sha256(ast.dump(changed, include_attributes=False).encode()).hexdigest(),
                         replacement_nodes=1, inverse_exact=True, original_comparison=ast.unparse(expected),
                         replacement=ast.unparse(replacement), source=ast.unparse(changed))


def install(bagged):
    original = bagged.vote_counts
    tree, receipt = transform(inspect.getsource(original))
    namespace = dict(original.__globals__, _h337_indicator=indicator)
    exec(compile(tree, '<H337-single-vote-comparison>', 'exec'), namespace)
    bagged.vote_counts = namespace['vote_counts']
    return receipt


def temperature_from_deltas(deltas):
    values = np.asarray(deltas, dtype=np.float64)
    assert values.ndim == 2 and values.shape[1] == 7 and values.shape[0] > 0
    assert np.isfinite(values).all()
    value = .5 * float(np.std(values, ddof=0, dtype=np.float64))
    assert math.isfinite(value) and value > 0, 'INCOMPLETE_DEGENERATE_TEMPERATURE'
    return value
