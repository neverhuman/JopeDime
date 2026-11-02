"""Pure append-only contract, independent of any optimizer/admission API."""
from .common import digest, need


def validate_extension_transition(previous, new):
    """Preserve every prior task and ordered ref; permit only unique tail refs."""
    need(set(previous) == set(new) and all(previous[k] == new[k] for k in previous if k != 'tasks'),
         'extension contract changed')
    need(len(previous['tasks']) == len(new['tasks']), 'prior task removal or new context')
    for before, after in zip(previous['tasks'], new['tasks']):
        need(set(before) == set(after) and
             all(before[k] == after[k] for k in before if k != 'program_results'), 'prior task identity changed')
        a, b = before['program_results'], after['program_results']
        need(b[:len(a)] == a, 'prior result reference removed/reordered/replaced')
        need(len({digest(r) for r in b}) == len(b), 'duplicate result weight')
    return True
